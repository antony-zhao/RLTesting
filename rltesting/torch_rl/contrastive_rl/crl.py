import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rltesting.torch_rl.models import MLP, IMPALACNN, reparameterize_normal
from rltesting.torch_rl.buffers import PerEnvTrajectoryBuffer
from torch.optim import Adam
from torch.distributions import Normal, Categorical        

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=64, hidden_dim=256, depth=4, act=nn.GELU):
        # specify input_dim as (size, size, channels)
        # hidden_dim for vec encoder
        super().__init__()
        self.obs_dim = input_dim
        self.is_image = type(input_dim) is tuple and len(input_dim) > 1
        self.output_dim = output_dim
        if self.is_image:
            self.conv_enc = IMPALACNN(input_dim[-1], depth-1, input_dim[0], act=act)
            self.out = nn.Linear(self.conv_enc.output_dim, output_dim)
        else:
            self.out = MLP(input_dim[0], output_dim, hidden_dim, num_hiddens=depth, act=act)
            
    def forward(self, x):
        if self.is_image:
            x = self.conv_enc(x).flatten(-2)
        return self.out(x)

class MultiEncoder(nn.Module):
    def __init__(self, input_dims, depths, output_dim=64, hidden_dim=256, act=nn.GELU):
        super().__init__()
        self.num = len(input_dims)
        self.encoders = nn.ModuleList(
            [Encoder(input_dims[i], output_dim, hidden_dim, depths[i] - 1) for i in range(self.num)]
        )
        self.act = act()
        self.out = nn.Linear(output_dim * self.num, output_dim)
    
    def forward(self, xs):
        xs = [self.encoders[i](xs[i]) for i in range(self.num)]
        x = torch.cat(xs, dim=-1)
        x = self.out(self.act(x))
        return x
    
class PolicyModel(nn.Module):
    def __init__(self, obs_dim, action_dim, action_type="continuous", hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(obs_dim, hidden_dim, hidden_dim)
        self.goal_encoder = Encoder(obs_dim, hidden_dim, hidden_dim)
        self.action_type = action_type
        
        if action_type == "continuous":
            self.policy_mu = MLP(hidden_dim * 2, action_dim[0])
            self.policy_logstd = nn.Parameter(torch.zeros(1, action_dim[0]))
        else:
            self.policy = MLP(hidden_dim * 2, action_dim[0])
    
    def policy_dist(self, obs, goal=None):
        x = self.encoder(obs)
        if goal is not None:
            goal_enc = self.goal_encoder(goal)
        else:
            goal_enc = torch.zeros_like(x).to(x.device)
        x = torch.cat([x, goal_enc], dim=-1)
        if self.action_type == "continuous":
            mu = self.policy_mu(x) # TODO STRAIGHT THROUGH GRADIENTS FOR BOTH
            return Normal(mu, self.policy_logstd.exp())
        else:
            act_logits = self.policy(x)
            return Categorical(logits=act_logits)
            
    def forward(self, obs, goal=None):
        dist = self.policy_dist(obs, goal)
        action = dist.rsample()
        return action
        

class CRLAgent:
    def __init__(
        self, obs_dim, action_dim, num_envs, device, repr_dim=64, action_type="continuous", 
        encoder_depth=4, buffer_size=1_000_000
    ):
        self.device = device
        self.sa_encoder = MultiEncoder(
            (obs_dim, action_dim), depths=(encoder_depth, encoder_depth), output_dim=repr_dim
        ).to(device)
        self.g_encoder = Encoder(obs_dim, output_dim=repr_dim, depth=encoder_depth).to(device)
        self.policy = PolicyModel(obs_dim, action_dim, action_type).to(device)
        self.sa_encoder_optim = Adam(self.sa_encoder.parameters(), 3e-4)
        self.g_encoder_optim = Adam(self.g_encoder.parameters(), 3e-4)
        self.policy_optim = Adam(self.policy.parameters(), 3e-4)
        buffer_shapes = [obs_dim, action_dim, (1, ), obs_dim, (1, )]
        is_image = type(obs_dim) is tuple and len(obs_dim) > 1
        obs_type = np.float32 if not is_image else np.uint8
        dtypes = [obs_type, np.float32, np.float32, obs_type, np.bool]
        self.buffer = PerEnvTrajectoryBuffer(num_envs, buffer_shapes, dtypes, buffer_size)
    
    def critic_loss(self, obs, action, obs_f):
        sa_repr = self.sa_encoder([obs, action])
        g_repr = self.g_encoder(obs_f)
        logits = torch.einsum('ik, jk -> ij', sa_repr, g_repr)
        return F.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[0]).to(self.device), reduction='mean')
    
    def actor_loss(self, obs, goal):
        obs = torch.cat([obs, obs], dim=0)
        random_goals = torch.roll(goal, shifts=1, dims=0)
        goal = torch.cat([goal, random_goals], dim=0)
        action = self.policy(obs, goal)
        sa_repr = self.sa_encoder([obs, action])
        g_repr = self.g_encoder(goal)
        logits = torch.einsum('ik, ik -> i', sa_repr, g_repr)
        return -logits.mean()
    
    def train(self):
        sample, future_states = self.buffer.sample_with_goals_as_tensors(self.device, batch_size=256)
        obs, action, reward, true_goals, done = sample
        
        self.policy_optim.zero_grad()
        actor_loss = self.actor_loss(obs, true_goals)
        actor_loss.backward()
        self.policy_optim.step()
        
        self.sa_encoder_optim.zero_grad()
        self.g_encoder_optim.zero_grad()
        critic_loss = self.critic_loss(obs, action, future_states)
        critic_loss.backward()
        self.sa_encoder_optim.step()
        self.g_encoder_optim.step()

        return critic_loss, actor_loss
    
    def act(self, obs, goal=None):
        return self.policy(obs, goal)

    def store_sample(self, obs, action, reward, goal, done):
        self.buffer.add_sample([obs, action, reward, goal, done])
        