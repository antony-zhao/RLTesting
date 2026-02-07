import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rltesting.torch_rl.models import MLP, IMPALACNN
from torch.optim import Adam
from torch.distributions import Normal, Categorical

class TrajectoryBuffer:
    def __init__(self, buffer_size=1_000_000, num_envs=16):
        self.action_buffer = np.empty()
        self.state_buffer = np.empty()
        self.done_buffer = np.empty()
    
    def sample_triplet(self, batch_size=256):
        # returns the (s_t, a_t, s_f+) triplet
        pass
    
    def sample_goals(self, batch_size=256):
        # return random states
        pass

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=64, hidden_dim=256, depth=4, act=nn.GELU):
        # specify input_dim as (size, size, channels)
        # hidden_dim for vec encoder
        super().__init__()
        self.obs_dim = input_dim
        self.is_image = len(input_dim) > 1
        self.output_dim = output_dim
        if self.is_image:
            self.conv_enc = IMPALACNN(input_dim[0], depth-1, input_dim[-1], act=act)
            self.out = nn.Linear(self.conv_enc.output_dim, output_dim)
        else:
            self.out = MLP(input_dim, output_dim, hidden_dim, num_hiddens=depth, act=act)
            
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
        x = torch.cat(x, dim=-1)
        x = self.out(self.act(x))
        return x
    
class PolicyModel(nn.Module):
    def __init__(self, obs_dim, action_dim, action_type="continuous", hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(obs_dim, hidden_dim, hidden_dim)
        
        if action_type == "continuous":
            self.policy_mu = MLP(hidden_dim, action_dim)
            self.policy_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.policy = MLP(hidden_dim, action_dim)
    
    def policy_dist(self, obs):
        x = self.encoder(obs)
        if self.action_type == "continuous":
            mu = self.policy_mu(x)
            return Normal(mu, self.policy_logstd.exp())
        else:
            act_logits = self.policy(x)
            return Categorical(logits=act_logits)
            
    def forward(self, obs):
        dist = self.policy_dist(obs)
        action = dist.sample()
        return action, dist.log_prob(action), dist
        

class CRLAgent:
    def __init__(
        self, obs_dim, action_dim, repr_dim=64, action_type="continuous", 
        encoder_depth=4, buffer_size=1_000_000
    ):
        self.sa_encoder = MultiEncoder(
            (obs_dim, action_dim), depths=(encoder_depth, encoder_depth), output_dim=repr_dim
        )
        self.g_encoder = Encoder(obs_dim, output_dim=repr_dim, depth=encoder_depth)
        self.policy = PolicyModel(obs_dim, action_dim, action_type)
    
    def critic_loss(self, obs, action, obs_f):
        sa_repr = self.sa_encoder(obs, action)
        g_repr = self.g_encoder(obs_f)
        logits = sa_repr @ g_repr.T
        return F.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[0]))
    
    def actor_loss(self, obs, goal):
        action, _, _ = self.policy(obs)
        self.sa_encoder.eval()
        self.g_encoder.eval()
        sa_repr = self.sa_encoder(obs, action)
        g_repr = self.g_encoder(goal)
        self.sa_encoder.train()
        self.g_encoder.train()
        logits = (sa_repr @ g_repr.T).sum(-1)
        return -logits