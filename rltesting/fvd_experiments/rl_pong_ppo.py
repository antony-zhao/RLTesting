import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ale_py
import imageio

# Existing PPO logic from your library
from rltesting.torch_rl.ppo.ppo import PPONetwork, compute_gae, compute_ppo_loss, Rollout
from rltesting.torch_rl.utils import load_config, simple_process_config
from rltesting.torch_rl.buffers import ReplayBuffer
from rltesting.utils.torch_utils import to_numpy
from rltesting.utils.logger import Logger

# Autoencoder/Environment imports
from fvd_pretrain import load_or_create_model
from fvd_models import Encoder, Decoder, AEVAE
from rltesting.torch_rl.models import MLP

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv,
)

# --- Latent Heads (Bridge AE to PPO) ---

class LatentPolicyNetwork(nn.Module):
    def __init__(self, latent_dim, num_actions):
        super().__init__()
        self.mlp = MLP(latent_dim, num_actions, skip_connections=False)
    
    def forward(self, x):
        return self.mlp(x)
    
    def policy_dist(self, x):
        return torch.distributions.Categorical(logits=self(x))
    
    def policy_fn(self, x, det=False):
        dist = self.policy_dist(x)
        if det: return torch.argmax(dist.logits, dim=-1), None
        action = dist.sample()
        return action, dist.log_prob(action)

class LatentValueNetwork(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = MLP(latent_dim, 1, skip_connections=False)
    
    def forward(self, x):
        return self.mlp(x)

# --- Helper Logic ---

def make_env(gym_id, seed, time_limit=4500, eval=False):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4) # Standard Atari skip
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if not eval:
            env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 2)
        if not eval:
            env = gym.wrappers.TimeLimit(env, time_limit)
        env.action_space.seed(seed)
        return env
    return thunk

def anneal_lr(optimizer, lr, update, num_updates):
    frac = 1.0 - (update / num_updates)
    for param_group in optimizer.param_groups:
        param_group["lr"] = frac * lr

def main(args):
    gym.register_envs(ale_py)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load AE Config and Models
    config = simple_process_config(load_config("rltesting/fvd_experiments/configs/atari/default.yaml"))
    encoder = Encoder(config.framestack, config.latent_dim, 64, 1).to(device)
    decoder = Decoder(encoder.conv_dim, config.framestack, config.latent_dim, 1).to(device)
    double_ae = load_or_create_model(config, encoder, decoder, AEVAE)
    dtypes = [np.uint8]
    buffer_shapes = [(2, 64, 64)]
    buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=1_000_000)

    
    # 2. Setup Environments
    env = make_vec_env(make_env(f'ALE/{args.env}-v5', args.seed), args.num_envs, args.seed, 
                       vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    eval_env = make_vec_env(make_env(f'ALE/{args.env}-v5', args.seed, eval=True), 1, args.seed)
    
    # 3. Setup PPO Network
    policy = LatentPolicyNetwork(double_ae.intrinsic_dim, env.action_space.n)
    value = LatentValueNetwork(double_ae.intrinsic_dim)
    ppo_net = PPONetwork(policy, value).to(device)
    
    ppo_opt = optim.Adam(ppo_net.parameters(), lr=args.lr, eps=1e-5)
    ae_opt = optim.Adam(double_ae.parameters(), lr=args.lr)
    
    logger = Logger(f'logs/{args.env}_latent')
    rollout = Rollout(args.rollout_length, args.num_envs, env.observation_space, env.action_space)
    
    obs = env.reset()
    total_updates = args.timesteps // (args.num_envs * args.rollout_length)

    for i in range(total_updates):
        # --- Collection ---
        mb_values = []
        for t in range(args.rollout_length):
            obs_t = (torch.as_tensor(obs).float() / 255.0).to(device)
            with torch.no_grad():
                latents = double_ae.double_encode(obs_t)
                action, log_prob = ppo_net.policy_network.policy_fn(latents)
                val = ppo_net.value_network(latents)
            
            next_obs, reward, done, infos = env.step(to_numpy(action))
            rollout.add(obs, action.cpu().numpy()[:, None], reward, done, log_prob.cpu().numpy())
            mb_values.append(to_numpy(val).flatten())
            obs = next_obs
            for info in infos:
                total_reward = 0
                num_completed = 0
                if 'episode' in info.keys():
                    total_reward += info['episode']['r']
                    num_completed += 1
                if num_completed > 0:
                    logger.add_scalar("ep_rew", total_reward)
                    logger.write((i * args.rollout_length + t) * args.num_envs)
            # for j in range(args.num_envs):
            #     buffer.add_sample([obs[j]])

        # --- Advantages ---
        rollout.last_obs = obs
        with torch.no_grad():
            last_obs_t = (torch.as_tensor(obs).float() / 255.0).to(device)
            last_val = to_numpy(ppo_net.value_network(double_ae.double_encode(last_obs_t))).flatten()
        
        mb_values.append(last_val)
        r_obs, r_actions, r_rewards, r_dones, r_logprobs, _ = rollout.unpack()
        advs, rets = compute_gae(r_rewards, np.asarray(mb_values), r_dones, args.discount)

        # Flatten
        b_obs = r_obs.reshape(-1, *env.observation_space.shape)
        b_actions = r_actions.reshape(-1)
        b_logprobs = r_logprobs.reshape(-1)
        b_rets = rets.reshape(-1)
        b_advs = advs.reshape(-1)

        # --- Training ---
        indices = np.arange(len(b_obs))
        for epoch in range(args.num_epochs):
            # for _ in range(args.num_ae_steps):
                # # Update AE
                # obs_sample = torch.tensor(buffer.sample(256)[0]).to(device) / 255.0
                # ae_loss = double_ae.reconstruction_loss(obs_sample)
                # ae_opt.zero_grad()
                # ae_loss.backward()
                # ae_opt.step()
            np.random.shuffle(indices)
            for start in range(0, len(b_obs), len(b_obs)//args.num_minibatches):
                idx = indices[start:start + len(b_obs)//args.num_minibatches]
                
                t_obs = (torch.as_tensor(b_obs[idx]).float() / 255.0).to(device)
                # Update AE
                ae_loss = double_ae.reconstruction_loss(t_obs)
                ae_opt.zero_grad()
                ae_loss.backward()
                ae_opt.step()

                # Update PPO
                curr_latents = double_ae.double_encode(t_obs).detach()
                loss, metrics = compute_ppo_loss(
                    ppo_net, curr_latents, torch.as_tensor(b_actions[idx]).to(device)[:, None], 
                    torch.as_tensor(b_logprobs[idx]).to(device), torch.as_tensor(b_rets[idx]).to(device), 
                    torch.as_tensor(b_advs[idx]).to(device), clip=args.clip, ent_coef=args.ent_coef
                )
                
                ppo_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ppo_net.parameters(), args.max_grad_norm)
                ppo_opt.step()

        rollout.reset()
        anneal_lr(ppo_opt, args.lr, i, total_updates)
        
        if i % 10 == 0:
            print(f"Update {i}/{total_updates} | AE Loss: {ae_loss.item():.4f}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Boxing') #MontezumaRevenge
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--val-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=50_000_000)
    parser.add_argument('--log-every', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=16)
    parser.add_argument('--num-ae-steps', type=int, default=64)
    parser.add_argument('--num-minibatches', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
    