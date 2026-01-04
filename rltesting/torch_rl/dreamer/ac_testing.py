from torch import nn
import torch.nn.functional as F
import torch
from dreamer import DreamerMLP, unimix, probs_to_logits, Categorical, Normal, WeightedAverageOverBins, TargetNetwork, to_numpy
from rltesting.torch_rl.ppo.ppo import Rollout
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import compute_lambda_returns, compute_lambda_values

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DreamerMLP(6, 3, 256, 2)
    
    def policy_dist(self, x):
        logits = self.mlp(x)
        probs = F.softmax(logits, -1)
        unimixed_probs = unimix(probs, 3)
        action_dist = Categorical(probs=unimixed_probs)
        return action_dist
    
    def sample_action(self, dist):
        if self.action_type == "discrete":
            action = dist.sample()
        else:
            action = dist.rsample()
        return action
    
    def policy_fn(self, x, det=False):
        # actually choosing the action, returns the actual action as well as the log prob of the action and entropy
        action_dist = self.policy_dist(x)
        if not det:
            action = self.sample_action(action_dist)
        else:
            return action_dist.mode
        return action.detach()
        
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DreamerMLP(6, 1, 256, 2)
    
    def forward(self, x):
        return self.mlp(x)

def make_env():
    def thunk():
        env = gym.make(f"Acrobot-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def train_actor(returns, values, action_log_probs, entropy, entropy_coef=3e-4):
        actor_loss = -torch.mean((returns - values) * action_log_probs + entropy * entropy_coef)
        return actor_loss, entropy.mean()
    
def train_critic(critic, critic_target, states, rewards, dones, gamma=0.99, lambda_=0.95):
    critic_target.update()
    values = critic(states)
    value_target = critic_target(states)
    continues = (1 - dones)
    returns = compute_lambda_values(values, rewards, continues, gamma, lambda_)
    loss = -F.mse_loss(values, returns.detach())
    loss -= F.mse_loss(values, value_target)
    return loss, returns.detach(), values.detach()

if __name__ == "__main__":
    env = make_vec_env(make_env(), 16, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor = Actor().to(device)
    critic = Critic().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), 3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), 3e-4)
    critic_target = TargetNetwork(critic, 0.01)
    rollout_length = 16
    rollout = Rollout(rollout_length, 16, env.observation_space, env.action_space)
    
    obs = env.reset()
    for r in range(1000):
        log_probs = []
        entropies = []
        total_reward = 0
        num_completed = 0
        for i in range(rollout_length):
            action_dist = actor.policy_dist(torch.tensor(obs, device=device))
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action = to_numpy(action.unsqueeze(1))
            log_probs.append(log_prob)
            entropies.append(action_dist.entropy())
            log_prob = to_numpy(log_prob)
            next_obs, reward, done, infos = env.step(action[:, 0])
            rollout.add(obs, action, reward, done, log_prob)
            obs = next_obs
            for info in infos:
                if 'episode' in info.keys():
                    total_reward += info['episode']['r']
                    num_completed += 1
        if num_completed > 0:
            print(f"Rollout: {r}, Reward: {total_reward / max(num_completed, 1)}, Completed: {num_completed}")
        rollout.last_obs = obs
        obs, action, reward, done, log_prob, last_obs = rollout.unpack()
        obs = torch.tensor(obs, device=device)
        action = torch.tensor(action, device=device)
        reward = torch.tensor(reward, device=device).unsqueeze(-1)
        done = torch.tensor(done, device=device).unsqueeze(-1)
        log_prob = torch.stack(log_probs).unsqueeze(-1)
        entropy = torch.stack(entropies).unsqueeze(-1)
        critic_loss, returns, values = train_critic(critic, critic_target, obs, reward, done)
        critic_loss.backward()
        critic_opt.step()
        critic_opt.zero_grad()
        actor_loss, entropy = train_actor(returns, values, log_prob, entropy)
        actor_loss.backward()
        actor_opt.step()
        actor_opt.zero_grad()
        obs = rollout.last_obs
        rollout.reset()
