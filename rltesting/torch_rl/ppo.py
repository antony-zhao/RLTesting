import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, TanhTransform
import numpy as np

from models import MLP, AtariConv
from utils import to_numpy, get_action_dim, get_obs_shape

class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    last_obs: np.ndarray
    
    def __init__(self, rollout_length, num_envs, obs_space, action_space):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs_shape = get_obs_shape(obs_space)
        self.action_dim = get_action_dim(action_space)
        self.ind = 0
        self.reset()
    
    def reset(self):
        self.obs = np.zeros((self.rollout_length, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.rollout_length, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
    
    def add(self, obs, action, reward, done, log_prob):
        self.obs[self.ind] = obs
        self.actions[self.ind] = action
        self.rewards[self.ind] = reward
        self.dones[self.ind] = done
        self.log_probs[self.ind] = log_prob
        self.ind += 1
        
    def unpack(self):
        return self.obs, self.actions, self.rewards, self.dones, self.log_probs, self.last_obs

class AtariPolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariPolicyNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x):
        x = self.conv(x / 255.0)
        x = x.reshape(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x
    
    def policy_dist(self, x):
        logits = self(x)
        action_dist = Categorical(logits=logits)
        return action_dist
    
    def policy_fn(self, x, det=False):
        # actually choosing the action
        if not det:
            action_dist = self.policy_dist(x)
        if det:
            action_logits = self(x)
            return torch.argmax(action_logits, dim=-1), None
        action = action_dist.sample()
        return action, action_dist.log_prob(action) # [B, 1]

class AtariValueNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariValueNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, num_actions)
    
    def forward(self, x):
        x = self.conv(x / 255.0)
        x = x.view(-1, self.conv.output_dim)
        x = self.mlp(x)
        return x # [B, 1]

class PPONetwork(nn.Module):
    def __init__(self, policy_network, value_network):
        super(PPONetwork, self).__init__()
        self.policy_network = policy_network
        self.value_network = value_network

def compute_gae(rewards, values, dones, discount=0.99, lam=0.95):
    # from here https://github.com/zplizzi/pytorch-ppo/blob/master/gae.py
    # values is T+1, since it needs to include the value for the next_obs, also purely numpy
    N = rewards.shape[1]
    T = rewards.shape[0]
    next_value = values[-1]
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[t]
        if t == T - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t + 1]
        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + discount * lam * nextnonterminal * lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns

def compute_ppo_loss(ppo_network: PPONetwork, obs, actions, old_log_prob, ret, adv,
                     clip=0.2, ent_coef=0, val_coef=1):
    # compute_gae is numpy only
    values = ppo_network.value_network(obs)
    val_loss = F.mse_loss(ret, values.squeeze(1))
    
    action_dist = ppo_network.policy_network.policy_dist(obs)
    action_log_prob = action_dist.log_prob(actions.squeeze(1))
    ratio = torch.exp(action_log_prob - old_log_prob)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    policy_loss1 = adv * ratio
    policy_loss2 = adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
    
    ent_loss = -action_dist.entropy().mean()
    
    loss = ent_loss * ent_coef + policy_loss + val_loss * val_coef
    return loss, {'policy_loss': policy_loss, 'entropy_loss': ent_loss, 'value_loss': val_loss}

def collect_rollout(env, ppo_network, rollout_length, obs):
    rollout = Rollout(rollout_length, env.num_envs, env.observation_space, env.action_space)
    total_reward = 0
    num_completed = 0
    for _ in range(rollout_length):
        obs_tensor = torch.as_tensor(obs).float()
        action, log_prob = ppo_network.policy_network.policy_fn(obs_tensor)
        action = to_numpy(action)
        log_prob = to_numpy(log_prob)
        next_obs, reward, done, infos = env.step(action)
        next_obs = next_obs
        for info in infos:
            if 'episode' in info.keys():
                total_reward += info['episode']['r']
                num_completed += 1
        action = action[:, np.newaxis]
        rollout.add(obs, action, reward, done, log_prob)
        obs = next_obs
    rollout.last_obs = obs
    return rollout, obs

def train_ppo(rollout, ppo_network, ppo_optim, num_minibatches, num_epochs, device, max_grad_norm=0.5,
              clip=0.2, discount=0.99, lam=0.95, ent_coef=1e-3, val_coef=1):
    with torch.device(device):
        obs, actions, rewards, dones, log_probs, last_obs = rollout.unpack()
        last_obs = np.expand_dims(last_obs, 0)
        all_obs = torch.as_tensor(np.concatenate([obs, last_obs])).float()
        all_obs = all_obs.reshape(-1, *all_obs.shape[2:])
        values = ppo_network.value_network(all_obs)
        
        val_np = to_numpy(values).reshape(-1, rewards.shape[1])
        adv, ret = compute_gae(rewards, val_np, dones, discount, lam)
        
        num_samples = obs.shape[0] * obs.shape[1]
        obs = obs.reshape(num_samples, *obs.shape[2:])
        actions = actions.reshape(num_samples, *actions.shape[2:])
        log_probs = log_probs.reshape(num_samples, *log_probs.shape[2:])
        ret = ret.reshape(num_samples, *ret.shape[2:])
        adv = adv.reshape(num_samples, *adv.shape[2:])
        
        metrics = {}
        minibatch_size = num_samples // num_minibatches
        for i in range(num_epochs):
            indices = np.random.permutation(num_samples)
            start_ind = 0
            while start_ind < num_samples:
                batch_inds = indices[start_ind: min(num_samples, start_ind + minibatch_size)]
                obs_batch = torch.as_tensor(obs[batch_inds])
                actions_batch = torch.as_tensor(actions[batch_inds])
                log_probs_batch = torch.as_tensor(log_probs[batch_inds])
                ret_batch = torch.as_tensor(ret[batch_inds])
                adv_batch = torch.as_tensor(adv[batch_inds])
                
                ppo_loss, metric = compute_ppo_loss(ppo_network, obs_batch, actions_batch, log_probs_batch, ret_batch, adv_batch,
                                            clip, ent_coef, val_coef)
                for key in metric.keys():
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += to_numpy(metric[key]) / (num_samples * num_epochs / minibatch_size)
                ppo_optim.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_network.parameters(), max_grad_norm)
                ppo_optim.step()
                start_ind += minibatch_size
        return metrics