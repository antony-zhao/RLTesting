import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rltesting.torch_rl.models import MLP, IMPALACNN, reparameterize_normal
from rltesting.torch_rl.buffers import PerEnvTrajectoryBuffer
from rltesting.torch_rl.distributions import SafeTanhNormal
from rltesting.torch_rl.utils import to_numpy
from torch.optim import Adam
from torch.distributions import Normal, Categorical

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim=256, act=nn.SiLU):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), act()]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x) + x
    
class ScalingMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=4, hidden_dim=256, act=nn.SiLU):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            *[ResidualBlock(hidden_dim, act) for _ in range(num_blocks)],
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.blocks(x)
        

class  Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=64, hidden_dim=256, depth=2, act=nn.SiLU):
        # specify input_dim as (size, size, channels) for image
        # hidden_dim for vec encoder
        super().__init__()
        self.obs_dim = input_dim
        self.output_dim = output_dim
        self.is_image = type(input_dim) is tuple and len(input_dim) > 1
        if self.is_image:
            self.conv_enc = IMPALACNN(input_dim[-1], depth-1, input_dim[0], act=act)
            self.out = nn.Linear(self.conv_enc.output_dim, output_dim)
        else:
            self.out = ScalingMLP(input_dim[0], output_dim, depth, hidden_dim, act=act)
            
    def forward(self, x):
        if self.is_image:
            x = self.conv_enc(x).flatten(-2)
        return self.out(x)

class MultiEncoder(nn.Module):
    def __init__(self, input_dims, depths, num_blocks=4, output_dim=64, hidden_dim=256, act=nn.SiLU):
        super().__init__()
        self.num = len(input_dims)
        self.encoders = nn.ModuleList(
            [
                Encoder(input_dims[i], hidden_dim, depths[i] - 1, act) 
                if type(input_dims[i]) is tuple and len(input_dims[i]) > 1 
                else nn.Identity() for i in range(self.num)
            ]
        )
        temp_dim = np.sum([
                            hidden_dim if 
                            type(input_dims[i]) is tuple and len(input_dims[i]) > 1 else 
                            input_dims[i] for i in range(self.num)
                        ]
        )
        self.out = ScalingMLP(temp_dim, output_dim, num_blocks, hidden_dim, act)
    
    def forward(self, xs):
        xs = [self.encoders[i](xs[i]) for i in range(self.num)]
        x = torch.cat(xs, dim=-1)
        x = self.out(x)
        return x
    
class PolicyModel(nn.Module):
    def __init__(self, obs_dim, action_dim, action_type="continuous", hidden_dim=256, num_blocks=4, act=nn.SiLU, upscale=1.0):
        super().__init__()
        is_image = type(obs_dim) is tuple and len(obs_dim) > 1
        self.encoder = Encoder(obs_dim, hidden_dim) if is_image else nn.Identity()
        self.goal_encoder = Encoder(obs_dim, hidden_dim) if is_image else nn.Identity()
        mlp_dim = hidden_dim * 2 if is_image else obs_dim[0] * 2
        self.action_type = action_type
        self.upscale = upscale
        
        if action_type == "continuous":
            self.policy_mu = ScalingMLP(mlp_dim, action_dim[0], num_blocks, hidden_dim, act)
            self.policy_logstd = nn.Parameter(torch.zeros(1, action_dim[0]))
        else:
            self.policy = ScalingMLP(mlp_dim, action_dim[0], num_blocks, hidden_dim, act)
    
    def policy_dist(self, obs, goal=None):
        x = self.encoder(obs)
        if goal is not None:
            goal_enc = self.goal_encoder(goal)
        else:
            goal_enc = torch.zeros_like(x).to(x.device)
        x = torch.cat([x, goal_enc], dim=-1)
        if self.action_type == "continuous":
            mu = self.policy_mu(x)
            log_std = torch.clamp(self.policy_logstd, -20, 2)
            dist = SafeTanhNormal(mu, log_std.exp())
            return dist
        else:
            act_logits = self.policy(x)
            return Categorical(logits=act_logits)
            
    def forward(self, obs, goal=None):
        dist = self.policy_dist(obs, goal)
        if self.action_type == "continuous":
            action = dist.rsample()
        else:
            action = dist.sample()
        return action
        

class CRLAgent:
    def __init__(
        self, obs_dim, action_dim, num_envs, device, obs_to_goal, repr_dim=64, action_type="continuous", 
        obs_encoder_depth=1, action_encoder_depth=1, depth=2, buffer_size=1_000_000, use_alpha=True, penalty=0.1, batch_size=256
    ):
        self.device, self.obs_to_goal, self.use_alpha, self.penalty, self.batch_size, self.action_type, self.num_actions = (
            device, obs_to_goal, use_alpha, penalty, batch_size, action_type, action_dim[0]
        )

        self.sa_encoder = MultiEncoder(
            (obs_dim, action_dim), depths=(obs_encoder_depth, action_encoder_depth), output_dim=repr_dim, num_blocks=depth
        ).to(device)
        self.g_encoder = Encoder(obs_dim, output_dim=repr_dim, depth=depth).to(device)
        self.policy = PolicyModel(obs_dim, action_dim, action_type, num_blocks=depth).to(device)
        self.critic_optim = Adam(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()), 
            3e-4
        )
        self.policy_optim = Adam(self.policy.parameters(), 3e-4)
        buffer_shapes = [obs_dim, action_dim, (1, ), obs_dim, (1, )]
        is_image = type(obs_dim) is tuple and len(obs_dim) > 1
        obs_type = np.float32 if not is_image else np.uint8
        dtypes = [obs_type, np.float32, np.float32, obs_type, np.bool]
        self.target_entropy = -action_dim[0] * 0.5 if action_type == "continuous" else np.log(action_dim[0])
        self.log_alpha = nn.Parameter(torch.tensor(0.0).to(device))
        self.alpha_optim = Adam([self.log_alpha], 3e-4)
        self.buffer = PerEnvTrajectoryBuffer(num_envs, buffer_size=buffer_size)
    
    def critic_fn(self, obs, action, obs_f):
        sa_repr = self.sa_encoder([obs, action])
        g_repr = self.g_encoder(obs_f)
        logits = -torch.sqrt(torch.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, dim=-1))
        return logits
    
    def critic_loss(self, obs, action, obs_f):
        logits = self.critic_fn(obs, action, obs_f)
        logits_pos = logits.diag().mean()
        n = logits.shape[0]
        logits_neg = (logits.sum() - logits.diag().sum()) / (n * n - n)
        loss = -torch.diag(logits) + torch.logsumexp(logits, dim=1)
        loss = loss.mean()
        logsumexp_reg = self.penalty * torch.logsumexp(logits + 1e-6, dim=1).pow(2).mean()
        return loss + logsumexp_reg, {
            "logits_pos": to_numpy(logits_pos),
            "logits_neg": to_numpy(logits_neg)
        }
    
    def actor_loss(self, obs, goal):
        obs = torch.cat([obs, obs], dim=0)
        random_goals = torch.roll(goal, shifts=1, dims=0)
        goal = torch.cat([goal, random_goals], dim=0)
        B = obs.shape[0]
        
        action_dist = self.policy.policy_dist(obs, goal)
        alpha = torch.exp(self.log_alpha).detach()
        g_repr = self.g_encoder(goal).detach()
        
        if self.action_type == "continuous":
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1)
            sa_repr = self.sa_encoder([obs, action])
            logits = -torch.sqrt(torch.sum((sa_repr - g_repr) ** 2, dim=-1))
            if self.use_alpha:
                loss = (-logits + alpha * log_prob).mean()
            else:
                loss = -logits.mean()
        else:
            probs = action_dist.probs
            log_probs = action_dist.logits
            obs_expanded = obs.repeat_interleave(self.num_actions, dim=0)
            one_hots = torch.eye(self.num_actions, device=obs.device).repeat(B, 1)

            sa_reprs = self.sa_encoder([obs_expanded, one_hots])
            sa_reprs = sa_reprs.view(B, self.num_actions, -1)
            q_values = -torch.sqrt(torch.sum((sa_reprs - g_repr[:, None, :]) ** 2, dim=-1)).detach()
            if self.use_alpha:
                alpha = torch.exp(self.log_alpha).detach()
                loss = (probs * (alpha * log_probs - q_values)).sum(-1).mean()
            else:
                loss = (probs * -q_values).sum(-1).mean()
                
        return loss, {"alpha": to_numpy(alpha)}
    
    def alpha_loss(self, obs, goal):
        action_dist = self.policy.policy_dist(obs, goal)
        if self.action_type == "continuous":
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1)
        else:
            log_prob = (action_dist.probs * action_dist.logits).sum(-1)
        alpha = torch.exp(self.log_alpha)
        alpha_loss = alpha * (-log_prob - self.target_entropy).detach()
        return (alpha_loss).mean()
    
    def train(self):
        sample, future_states, traj_ids = self.buffer.sample_with_goals_as_tensors(self.device, batch_size=self.batch_size)
        obs, action, reward, true_goals, done = sample
        if self.action_type == "discrete":
            action = F.one_hot(action.long(), self.num_actions)
        obs_goal = to_numpy(future_states)
        obs_goal = self.obs_to_goal(obs_goal)
        obs_goal = torch.as_tensor(obs_goal).to(self.device)
        
        self.policy_optim.zero_grad()
        actor_loss, actor_metrics = self.actor_loss(obs, obs_goal)
        actor_loss.backward()
        self.policy_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss, critic_metrics = self.critic_loss(obs, action, obs_goal)
        critic_loss.backward()
        self.critic_optim.step()
        
        if self.use_alpha:
            self.alpha_optim.zero_grad()
            alpha_loss = self.alpha_loss(obs, obs_goal)
            alpha_loss.backward()
            self.alpha_optim.step()

        return critic_loss, actor_loss, actor_metrics | critic_metrics
    
    def act(self, obs, goal=None):
        return self.policy(obs, goal)

    def store_sample(self, obs, action, reward, goal, done):
        self.buffer.add_sample([obs, action, reward, goal, done])
        