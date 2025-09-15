import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models import MLP, AtariConv, layer_init
from ppo import compute_gae, PPONetwork, AtariPolicyNetwork
from utils import to_numpy, get_action_dim, get_obs_shape

class RND_Rollout:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    last_obs: np.ndarray
    returns: np.ndarray
    
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
        self.intrinsic_rewards = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        self.returns = 0
    
    def add(self, obs, action, reward, int_rew, done, log_prob):
        self.obs[self.ind] = obs
        self.actions[self.ind] = action
        self.rewards[self.ind] = reward
        self.intrinsic_rewards[self.ind] = int_rew
        self.dones[self.ind] = done
        self.log_probs[self.ind] = log_prob
        self.ind += 1
        
    def unpack(self):
        return self.obs, self.actions, self.rewards, self.intrinsic_rewards, self.dones, self.log_probs, self.last_obs

    def returns_step(self, rewards, discount=0.99):
        self.returns = self.returns * discount + rewards
        return self.returns

class RND(nn.Module):
    def __init__(self, feature_model, target_model, obs_channels=1):
        super(RND, self).__init__()
        self.feature_model = feature_model
        self.feature_model_target = target_model
        self.obs_channels = obs_channels
    
    def compute_intrinsic(self, next_obs, sample_prob=1):
        next_obs = torch.clamp(next_obs[:, -self.obs_channels:], -5, 5)
        features = self.feature_model(next_obs)
        with torch.no_grad():
            target_features = self.feature_model_target(next_obs)

        error = ((features - target_features) ** 2)
        intrinsic_reward = to_numpy(error.sum(-1))
        mask = (torch.rand(error.shape[0]) < sample_prob).float().to(next_obs.device)
        error = error.mean(-1)
        loss = (error * mask).sum() / torch.max(torch.sum(mask), torch.tensor(1).to(next_obs.device))
        return intrinsic_reward, loss
    
class AtariFeatureModel(nn.Module):
    def __init__(self, feature_dim=512, num_hidden=1):
        super(AtariFeatureModel, self).__init__()
        self.conv = AtariConv(input_channels=1)
        self.mlp = MLP(self.conv.output_dim, feature_dim, num_hiddens=num_hidden)
    
    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(-1, self.conv.output_dim)
        return self.mlp(x)
    
class RunningMeanStd(object):
    # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon
        self.samples = []
        self.update_after = 32
        
    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class AtariRNDValueNetwork(nn.Module):
    def __init__(self, act=nn.SiLU):
        super(AtariRNDValueNetwork, self).__init__()
        self.conv = AtariConv()
        self.mlp = MLP(self.conv.output_dim, 256)
        self.val1 = nn.Linear(256, 1)
        self.val2 = nn.Linear(256, 1)
        self.act = act()
    
    def forward(self, x):
        x = self.conv(x / 255)
        x = x.view(-1, self.conv.output_dim)
        x = self.act(self.mlp(x))
        v1 = self.val1(x)
        v2 = self.val2(x)
        return v1, v2

def compute_rnd_loss(ppo_network: PPONetwork, obs, actions, old_log_prob, reg_ret, rnd_ret, reg_adv, rnd_adv,
                     clip=0.2, ent_coef=0, val_coef=1, ext_coef=2, rnd_coef=1):
    # compute_gae is numpy only
    values, rnd_values = ppo_network.value_network(obs)
    val_loss = F.mse_loss(reg_ret, values.squeeze(1)) + F.mse_loss(rnd_ret, rnd_values.squeeze(1)) * rnd_coef
    
    action_dist = ppo_network.policy_network.policy_dist(obs)
    action_log_prob = action_dist.log_prob(actions.squeeze(1))
    if len(action_log_prob.shape) > 1:
        action_log_prob = action_log_prob.sum(-1)
    ratio = torch.exp(action_log_prob - old_log_prob)
    reg_adv = (reg_adv - reg_adv.mean()) / (reg_adv.std() + 1e-8)
    rnd_adv = (rnd_adv - rnd_adv.mean()) / (rnd_adv.std() + 1e-8)
    adv = reg_adv * ext_coef + rnd_adv * rnd_coef
    policy_loss1 = adv * ratio
    policy_loss2 = adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
    
    ent_loss = -action_dist.entropy().mean()
    
    loss = ent_loss * ent_coef + policy_loss + val_loss * val_coef
    return loss, {'policy_loss': policy_loss, 'entropy_loss': ent_loss, 'value_loss': val_loss}

def collect_rollout(env, ppo_network, rnd, rollout_length, obs, rms):
    rollout = RND_Rollout(rollout_length, env.num_envs, env.observation_space, env.action_space)
    total_reward = 0
    num_completed = 0
    for _ in range(rollout_length):
        obs_tensor = torch.as_tensor(obs).float()
        action, log_prob = ppo_network.policy_network.policy_fn(obs_tensor)
        action = to_numpy(action)
        log_prob = to_numpy(log_prob)
        next_obs, reward, done, infos = env.step(action)
        intrinsic_reward, _ = rnd.compute_intrinsic(torch.as_tensor((next_obs - rms.mean) / (np.sqrt(rms.var) + 1e-8)).float())
        next_obs = next_obs
        for info in infos:
            if 'episode' in info.keys():
                total_reward += info['episode']['r']
                num_completed += 1
        action = action[:, np.newaxis]
        rollout.add(obs, action, reward, intrinsic_reward, done, log_prob)
        obs = next_obs
    rollout.last_obs = obs
    return rollout, obs
    
def train_rnd(rollout, ppo_network, rnd, obs_rms, rnd_rms, ppo_optim, rnd_optim, num_minibatches, num_epochs, device, max_grad_norm=0.5,
              clip=0.2, discount=0.999, rnd_discount=0.99, lam=0.95, ent_coef=1e-3, val_coef=1, sample_prob=1, ext_coef=2, rnd_coef=1):
    with torch.device(device):
        obs, actions, rewards, intrinsic_rewards, dones, log_probs, last_obs = rollout.unpack()
        obs_rms.update(obs.reshape(-1, *obs.shape[2:]))
        last_obs = np.expand_dims(last_obs, 0)
        all_obs = torch.as_tensor(np.concatenate([obs, last_obs])).float()
        intrinsic_returns = np.array([rollout.returns_step(intrinsic_reward) for intrinsic_reward in intrinsic_rewards[::-1]])
        rnd_rms.update(intrinsic_returns.flatten())
        intrinsic_rewards = intrinsic_rewards / (np.sqrt(rnd_rms.var) + 1e-8)
        all_obs = all_obs.reshape(-1, *all_obs.shape[2:])
        values, rnd_values = ppo_network.value_network(all_obs)
        
        val_np = to_numpy(values).reshape(-1, rewards.shape[1])
        rnd_val_np = to_numpy(rnd_values).reshape(-1, rewards.shape[1])
        reg_adv, reg_ret = compute_gae(rewards, val_np, dones, discount, lam)
        rnd_adv, rnd_ret = compute_gae(intrinsic_rewards, rnd_val_np, np.zeros_like(dones), rnd_discount, lam)
        
        num_samples = obs.shape[0] * obs.shape[1]
        obs = obs.reshape(num_samples, *obs.shape[2:])
        actions = actions.reshape(num_samples, *actions.shape[2:])
        log_probs = log_probs.reshape(num_samples, *log_probs.shape[2:])
        reg_ret = reg_ret.reshape(num_samples, *reg_ret.shape[2:])
        rnd_ret = rnd_ret.reshape(num_samples, *rnd_ret.shape[2:])
        reg_adv = reg_adv.reshape(num_samples, *reg_adv.shape[2:])
        rnd_adv = rnd_adv.reshape(num_samples, *rnd_adv.shape[2:])
        minibatch_size = num_samples // num_minibatches
        
        metrics = {'intrinsic_loss': 0, 'intrinsic_reward': 0}
        for i in range(num_epochs):
            indices = np.random.permutation(num_samples)
            start_ind = 0
            while start_ind < num_samples:
                batch_inds = indices[start_ind: min(num_samples, start_ind + minibatch_size)]
                obs_batch = torch.as_tensor(obs[batch_inds])
                actions_batch = torch.as_tensor(actions[batch_inds])
                log_probs_batch = torch.as_tensor(log_probs[batch_inds])
                reg_ret_batch = torch.as_tensor(reg_ret[batch_inds])
                rnd_ret_batch = torch.as_tensor(rnd_ret[batch_inds])
                reg_adv_batch = torch.as_tensor(reg_adv[batch_inds])
                rnd_adv_batch = torch.as_tensor(rnd_adv[batch_inds])
                if start_ind == 0:
                    _, int_loss = rnd.compute_intrinsic(torch.as_tensor((to_numpy(obs_batch) - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8)).float(), sample_prob)
                    metrics['intrinsic_loss'] += to_numpy(int_loss) / (num_samples * num_epochs / minibatch_size)
                    metrics['intrinsic_reward'] += np.sum(intrinsic_rewards) / (num_samples * num_epochs / minibatch_size)
                    rnd_optim.zero_grad()
                    int_loss.backward()
                    rnd_optim.step()
                
                ppo_loss, metric = compute_rnd_loss(ppo_network, obs_batch, actions_batch, log_probs_batch, reg_ret_batch, rnd_ret_batch, reg_adv_batch, rnd_adv_batch,
                                            clip, ent_coef, val_coef, ext_coef, rnd_coef)
                for key in metric.keys():
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += to_numpy(metric[key]) / (num_samples * num_epochs / minibatch_size)
                ppo_optim.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_network.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(rnd.parameters(), max_grad_norm)
                ppo_optim.step()
                start_ind += minibatch_size
        return metrics
            
            