"""
Tiny DQN on learned intrinsic representation.
~4K param MLP Q-network on frozen encoder output.

Usage: python train_tiny_dqn.py --env boxing
       python train_tiny_dqn.py --env acrobot --mode pixels
       python train_tiny_dqn.py --env acrobot --mode raw
"""
import argparse
import copy
import json
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from config import get_config, make_env, add_env_arg, to_numpy, env_path
from rltesting.fvd_experiments.fvd_models import (
    Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
)


class TinyQ(nn.Module):
    """Small MLP Q-network."""
    def __init__(self, input_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Simple circular replay buffer."""
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[indices]).cuda(),
            torch.tensor(self.actions[indices]).cuda(),
            torch.tensor(self.rewards[indices]).cuda(),
            torch.tensor(self.next_states[indices]).cuda(),
            torch.tensor(self.dones[indices]).cuda(),
        )


def load_autoencoder(cfg):
    """Load a trained autoencoder checkpoint."""
    path = env_path(cfg, "autoencoder.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint at {path}. Run train_autoencoder.py first.")

    ckpt = torch.load(path, weights_only=False)
    intrinsic_dim = ckpt['intrinsic_dim']
    latent_dim = ckpt['latent_dim']
    hidden_dim = ckpt.get('hidden_dim', 128)
    filter_base = ckpt.get('filter_base', 16)
    encoder_type = ckpt.get('encoder_type', 'conv')
    img_ch = ckpt.get('image_channels', cfg.get('image_channels', 1))

    if encoder_type == 'impala':
        encoder = IMPALAEncoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                                image_channels=img_ch, filter_base=filter_base).cuda()
        decoder = IMPALADecoder(encoder.conv_dim, cfg['framestack'], latent_dim,
                                image_channels=img_ch, filter_base=filter_base).cuda()
    else:
        encoder = Encoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                          image_channels=img_ch, filter_base=filter_base).cuda()
        decoder = Decoder(encoder.conv_dim, cfg['framestack'], latent_dim,
                          image_channels=img_ch, filter_base=filter_base).cuda()

    model = DoubleAutoEncoder(encoder, decoder, latent_dim,
                              intrinsic_dim, hidden_dim=hidden_dim).cuda()
    model.load_state_dict(ckpt['weights'])
    model.eval()
    print(f"Loaded autoencoder ({encoder_type}): intrinsic_dim={intrinsic_dim}")
    return model, intrinsic_dim


def compute_norm_stats(env, model, n_episodes=20):
    all_z = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_obs = [np.array(obs)]
        while not done:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            ep_obs.append(np.array(obs))
            done = term or trunc
        with torch.no_grad():
            batch = np.stack(ep_obs)
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            raw = to_numpy(model.double_encode(t))
            all_z.append(raw)
    all_z = np.concatenate(all_z, axis=0)
    means = all_z.mean(axis=0)
    stds = all_z.std(axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def encode_single(obs, model, means, stds):
    with torch.no_grad():
        t = (torch.tensor(np.array(obs)).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
        raw = to_numpy(model.double_encode(t)).squeeze()
        return (raw - means) / stds


def main():
    parser = argparse.ArgumentParser(description='Tiny DQN on learned representation')
    add_env_arg(parser)
    parser.add_argument('--mode', type=str, default=None, choices=['raw', 'pixels'],
                        help='raw=state vector, pixels=learned repr. '
                             'Default: raw for classic, pixels for atari')
    parser.add_argument('--max-steps', type=int, default=None, help='Override max timesteps')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=50000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--target-update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--train-start', type=int, default=1000, help='Steps before training starts')
    args = parser.parse_args()

    cfg = get_config(args.env)

    # Determine mode
    if args.mode is not None:
        mode = args.mode
    elif cfg.get('env_type') == 'classic':
        mode = 'raw'
    else:
        mode = 'pixels'

    max_steps = args.max_steps or cfg['sarsa_max_timesteps']
    gamma = cfg['gamma']
    num_actions = cfg['num_actions']

    np.random.seed(42)
    torch.manual_seed(0)

    # Setup encoder or raw state
    if mode == 'pixels':
        env_mode = 'pixels' if cfg.get('env_type') == 'classic' else 'default'
        env = make_env(cfg, mode=env_mode)
        ae_model, intrinsic_dim = load_autoencoder(cfg)
        ae_model = copy.deepcopy(ae_model)
        ae_model.eval()
        means, stds = compute_norm_stats(env, ae_model)
        state_dim = intrinsic_dim

        def get_state(obs):
            return encode_single(obs, ae_model, means, stds)
    else:
        env = make_env(cfg, mode='raw')
        state_dim = cfg.get('state_dim', env.observation_space.shape[0])

        def get_state(obs):
            return np.array(obs, dtype=np.float32)

    # Q-networks
    q_net = TinyQ(state_dim, num_actions, hidden=args.hidden).cuda()
    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    q_params = sum(p.numel() for p in q_net.parameters())
    print(f"\n=== Tiny DQN ({mode}) — {cfg['env_name']} ===")
    print(f"  state_dim={state_dim}, actions={num_actions}, hidden={args.hidden}")
    print(f"  Q-network params: {q_params:,}")
    print(f"  lr={args.lr}, gamma={gamma}, buffer={args.buffer_size}")
    print(f"  target_update={args.target_update}, train_start={args.train_start}")
    print(f"  max_steps={max_steps}")

    # Replay buffer
    buffer = ReplayBuffer(args.buffer_size, state_dim)

    # Training
    returns = []
    timestep_checkpoints = []
    total_steps = 0
    ep = 0
    t_start = time.time()

    while total_steps < max_steps:
        epsilon = max(0.05, 1.0 - total_steps / (max_steps * 0.3))

        obs, _ = env.reset()
        state = get_state(obs)
        ep_reward = 0

        while True:
            # Epsilon-greedy
            if np.random.random() < epsilon or total_steps < args.train_start:
                action = np.random.randint(num_actions)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(state).float().unsqueeze(0).cuda())
                    action = q_vals.argmax(dim=1).item()

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_state(obs_next) if not done else np.zeros(state_dim, dtype=np.float32)

            buffer.add(state, action, reward, next_state, float(done))
            ep_reward += reward
            total_steps += 1

            # Train
            if total_steps >= args.train_start and buffer.size >= args.batch_size:
                s, a, r, s_next, d = buffer.sample(args.batch_size)

                with torch.no_grad():
                    q_next = target_net(s_next).max(dim=1).values
                    targets = r + gamma * q_next * (1 - d)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Target update
            if total_steps % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

            state = next_state

        returns.append(ep_reward)
        timestep_checkpoints.append(total_steps)
        ep += 1

        if ep % 25 == 0:
            elapsed = time.time() - t_start
            avg = np.mean(returns[-25:])
            print(f"Steps: {total_steps:>8d} | Ep {ep:4d} | "
                  f"Avg(25): {avg:7.1f} | Eps: {epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")

    # Plot
    window = 25
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    smoothed_steps = timestep_checkpoints[window - 1:]

    plt.figure(figsize=(10, 6))
    plt.plot(timestep_checkpoints, returns, alpha=0.3, label='Raw')
    plt.plot(smoothed_steps, smoothed, label=f'{window}-ep avg')
    plt.xlabel('Timesteps')
    plt.ylabel('Return')
    plt.title(f"Tiny DQN ({mode}) — {cfg['env_name'].title()} ({q_params:,} params)")
    plt.legend()
    plt.savefig(env_path(cfg, f"tiny_dqn_{mode}.png"))
    plt.close()

    # Save
    results = {
        'env': cfg['env_name'],
        'mode': mode,
        'method': 'tiny_dqn',
        'q_params': q_params,
        'hidden': args.hidden,
        'lr': args.lr,
        'returns': [float(r) for r in returns],
        'timesteps': [int(t) for t in timestep_checkpoints],
        'final_avg_25': float(np.mean(returns[-25:])),
        'final_avg_100': float(np.mean(returns[-100:])) if len(returns) >= 100 else None,
        'total_steps': total_steps,
        'config': {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
    }
    if mode == 'pixels':
        results['intrinsic_dim'] = state_dim

    save_path = env_path(cfg, f"tiny_dqn_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Q-network params: {q_params:,}")
    print(f"Final avg (last 25):  {np.mean(returns[-25:]):.1f}")
    if len(returns) >= 100:
        print(f"Final avg (last 100): {np.mean(returns[-100:]):.1f}")
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
