"""
SARSA(λ) with hashed tile coding.

Supports two modes:
  - pixels: learned representation from autoencoder (Atari, or classic control with --mode pixels)
  - raw:    tile coding directly on state vector (classic control)

Usage: python train_sarsa.py --env boxing                    # Atari (pixels, needs autoencoder)
       python train_sarsa.py --env cartpole --mode raw       # classic control on raw state
       python train_sarsa.py --env acrobot --mode pixels     # classic control on pixels
       python train_sarsa.py --env boxing --finetune-once
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
import torch.nn.functional as F
from matplotlib import pyplot as plt

from config import get_config, make_env, add_env_arg, to_numpy, env_path
from rltesting.fvd_experiments.fvd_models import (
    Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
)
from rltesting.fvd_experiments.tiles import tiles


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
    img_ch = ckpt.get('image_channels', 3)

    if encoder_type == 'impala':
        encoder = IMPALAEncoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                                image_channels=img_ch,
                                filter_base=filter_base).cuda()
        decoder = IMPALADecoder(encoder.conv_dim, cfg['framestack'], latent_dim,
                                image_channels=img_ch,
                                filter_base=filter_base).cuda()
    else:
        encoder = Encoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                          image_channels=img_ch,
                          filter_base=filter_base).cuda()
        decoder = Decoder(encoder.conv_dim, cfg['framestack'], latent_dim,
                          image_channels=img_ch,
                          filter_base=filter_base).cuda()

    model = DoubleAutoEncoder(encoder, decoder, latent_dim,
                              intrinsic_dim, hidden_dim=hidden_dim).cuda()
    model.load_state_dict(ckpt['weights'])
    model.eval()
    print(f"Loaded autoencoder ({encoder_type}): intrinsic_dim={intrinsic_dim}, latent_dim={latent_dim}")
    return model, intrinsic_dim


def compute_norm_stats(env, model, n_episodes=20):
    """Compute mean/std of intrinsic representations from random episodes."""
    print("Computing normalization stats...")
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
    print(f"  Stats from {len(all_z)} observations")
    return means, stds


def encode_single(obs, model, means, stds):
    """Encode a single observation to normalized intrinsic space."""
    with torch.no_grad():
        t = (torch.tensor(np.array(obs)).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
        raw = to_numpy(model.double_encode(t)).squeeze()
        return (raw - means) / stds


def batch_encode(obs_array, model, means, stds, batch_size=128):
    """Encode a batch of observations to normalized intrinsic space."""
    with torch.no_grad():
        results = []
        for i in range(0, len(obs_array), batch_size):
            batch = obs_array[i:i + batch_size]
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            raw = to_numpy(model.double_encode(t))
            results.append(raw)
        return (np.concatenate(results, axis=0) - means) / stds


def finetune_model(model, recent_episodes, ft_opt, steps=50, batch_size=256):
    """Fine-tune autoencoder on recent on-policy observations."""
    all_obs = []
    for ep_obs in recent_episodes:
        all_obs.extend(ep_obs)
    all_obs = np.stack(all_obs)
    n = len(all_obs)

    model.train()
    total_loss = 0.0
    for _ in range(steps):
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        batch = (torch.tensor(all_obs[idx]).float().transpose(-3, -1) / 255).cuda()
        intrinsic = model.double_encode(batch)
        reconstruction = model.double_decode(intrinsic)
        loss = F.mse_loss(reconstruction, batch)
        loss.backward()
        ft_opt.step()
        ft_opt.zero_grad()
        total_loss += loss.item()
    model.eval()
    return total_loss / steps


def main():
    parser = argparse.ArgumentParser(description='SARSA(λ) with hashed tile coding')
    add_env_arg(parser)
    parser.add_argument('--mode', type=str, default=None, choices=['raw', 'pixels'],
                        help='raw=tile code on state vector, pixels=use autoencoder. '
                             'Default: raw for classic control, pixels for atari')
    parser.add_argument('--finetune', action='store_true', help='Enable delayed fine-tuning')
    parser.add_argument('--finetune-once', action='store_true',
                        help='Fine-tune once at finetune_at episode, then resume SARSA')
    parser.add_argument('--max-steps', type=int, default=None, help='Override max timesteps')
    args = parser.parse_args()

    cfg = get_config(args.env)

    # Determine mode
    if args.mode is not None:
        mode = args.mode
    elif cfg.get('env_type') == 'classic':
        mode = 'raw'
    else:
        mode = 'pixels'

    if args.finetune:
        cfg['finetune'] = True
    if args.finetune_once:
        cfg['finetune_once'] = True
    if args.max_steps:
        cfg['sarsa_max_timesteps'] = args.max_steps

    np.random.seed(42)
    torch.manual_seed(0)

    if mode == 'raw':
        _run_raw(cfg)
    else:
        _run_pixels(cfg)


def _run_raw(cfg):
    """SARSA(λ) with tile coding directly on raw state vector."""
    env = make_env(cfg, mode='raw')
    num_actions = env.action_space.n
    state_dim = cfg.get('state_dim', env.observation_space.shape[0])

    # Tile coding setup
    hash_size = cfg['hash_size']
    num_tilings = cfg['num_tilings']
    num_tiles_per_dim = cfg['num_tiles_per_dim']
    alpha = cfg['alpha']
    lam = cfg['lam']
    gamma = cfg['gamma']

    w = np.zeros(hash_size)

    # Scaling: map observation bounds to tile grid
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    # Clip infinite bounds to reasonable range
    obs_low = np.clip(obs_low, -10.0, None)
    obs_high = np.clip(obs_high, None, 10.0)
    obs_range = obs_high - obs_low
    obs_range[obs_range == 0] = 1.0
    scale = np.full(state_dim, num_tiles_per_dim) / obs_range

    def get_tiles(state, action):
        scaled = ((state - obs_low) * scale).tolist()
        return tiles(hash_size, num_tilings, scaled, ints=[action])

    def Q_all(state):
        qs = np.zeros(num_actions)
        for a in range(num_actions):
            qs[a] = np.sum(w[get_tiles(state, a)])
        return qs

    def pick_action(state, eps):
        if np.random.random() < eps:
            return np.random.randint(num_actions)
        return np.argmax(Q_all(state))

    print(f"\n=== SARSA(λ) Raw State: {cfg['env_name']} ===")
    print(f"  state_dim={state_dim}, tilings={num_tilings}, "
          f"tiles_per_dim={num_tiles_per_dim}, hash_size={hash_size}")
    print(f"  alpha={alpha:.5f}, lambda={lam}, gamma={gamma}")

    returns = []
    timestep_checkpoints = []
    total_steps = 0
    ep = 0
    t_start = time.time()

    while total_steps < cfg['sarsa_max_timesteps'] and ep < cfg['sarsa_max_episodes']:
        epsilon = max(cfg['epsilon_min'], 1.0 - ep / cfg['epsilon_decay_episodes'])

        obs, _ = env.reset()
        S = np.array(obs, dtype=np.float64)
        A = pick_action(S, epsilon)

        trace_indices = set()
        e = np.zeros(hash_size)
        ep_reward = 0

        while True:
            obs_next, R, terminated, truncated, _ = env.step(A)
            done = terminated or truncated
            total_steps += 1
            ep_reward += R

            active = get_tiles(S, A)
            q_sa = np.sum(w[active])

            if done:
                td_error = R - q_sa
            else:
                S_next = np.array(obs_next, dtype=np.float64)
                A_next = pick_action(S_next, epsilon)
                q_next = np.sum(w[get_tiles(S_next, A_next)])
                td_error = R + gamma * q_next - q_sa

            # Decay traces
            if trace_indices:
                trace_list = np.array(list(trace_indices))
                e[trace_list] *= gamma * lam
                dead = trace_list[np.abs(e[trace_list]) < 1e-4]
                e[dead] = 0.0
                trace_indices -= set(dead)

            # Replacing traces
            for i in active:
                e[i] = 1.0
                trace_indices.add(i)

            # Update
            if trace_indices:
                trace_list = np.array(list(trace_indices))
                w[trace_list] += alpha * td_error * e[trace_list]

            if done:
                break
            S = S_next
            A = A_next

        returns.append(ep_reward)
        timestep_checkpoints.append(total_steps)
        ep += 1

        if ep % 25 == 0:
            elapsed = time.time() - t_start
            avg = np.mean(returns[-25:])
            print(f"Steps: {total_steps:>8d} | Ep {ep:4d} | "
                  f"Avg(25): {avg:7.1f} | Eps: {epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")

    _save_and_plot(cfg, returns, timestep_checkpoints, total_steps, mode='raw')


def _run_pixels(cfg):
    """SARSA(λ) with tile coding on learned intrinsic representation."""
    env_mode = 'pixels' if cfg.get('env_type') == 'classic' else 'default'
    env = make_env(cfg, mode=env_mode)
    num_actions = env.action_space.n
    model, intrinsic_dim = load_autoencoder(cfg)
    model = copy.deepcopy(model)
    model.eval()

    means, stds = compute_norm_stats(env, model)

    # Tile coding setup
    hash_size = cfg['hash_size']
    num_tilings = cfg['num_tilings']
    num_tiles_per_dim = cfg['num_tiles_per_dim']
    alpha = cfg['alpha']
    lam = cfg['lam']
    gamma = cfg['gamma']

    w = np.zeros(hash_size)
    mins = np.full(intrinsic_dim, -3.0)
    scale = np.full(intrinsic_dim, num_tiles_per_dim / 6.0)

    def get_tiles(state, action):
        scaled = ((state - mins) * scale).tolist()
        return tiles(hash_size, num_tilings, scaled, ints=[action])

    def Q_all(state):
        qs = np.zeros(num_actions)
        for a in range(num_actions):
            qs[a] = np.sum(w[get_tiles(state, a)])
        return qs

    def pick_action(state, eps):
        if np.random.random() < eps:
            return np.random.randint(num_actions)
        return np.argmax(Q_all(state))

    # Fine-tuning setup
    ft_opt = None
    recent_episodes = deque(maxlen=max(30, cfg.get('finetune_once_buffer_size', 50)))
    finetune_count = 0
    finetune_once_done = False
    if cfg['finetune'] or cfg.get('finetune_once'):
        ft_opt = torch.optim.Adam(model.parameters(), cfg['finetune_lr'])

    # Training loop
    ft_mode = 'ONCE' if cfg.get('finetune_once') else ('ON' if cfg['finetune'] else 'OFF')
    print(f"\n=== SARSA(λ) Pixels: {cfg['env_name']} ===")
    print(f"  intrinsic_dim={intrinsic_dim}, tilings={num_tilings}, "
          f"tiles_per_dim={num_tiles_per_dim}, hash_size={hash_size}")
    print(f"  alpha={alpha:.5f}, lambda={lam}, gamma={gamma}")
    print(f"  Fine-tuning: {ft_mode}")
    if cfg.get('finetune_once'):
        print(f"  Fine-tune once at episode {cfg['finetune_once_at']}, "
              f"{cfg['finetune_once_steps']} steps, "
              f"reset_weights={cfg['finetune_once_reset_weights']}")

    time_collect = 0
    time_encode = 0
    time_sarsa = 0
    time_finetune = 0

    returns = []
    timestep_checkpoints = []
    total_steps = 0
    ep = 0
    t_start = time.time()

    while total_steps < cfg['sarsa_max_timesteps'] and ep < cfg['sarsa_max_episodes']:
        epsilon = max(cfg['epsilon_min'], 1.0 - ep / cfg['epsilon_decay_episodes'])

        # Phase 1: Collect episode
        t0 = time.time()
        obs, _ = env.reset()
        episode_obs = [np.array(obs)]
        episode_actions = []
        episode_rewards = []

        S = encode_single(obs, model, means, stds)
        A = pick_action(S, epsilon)

        while True:
            obs_next, R, terminated, truncated, _ = env.step(A)
            done = terminated or truncated
            episode_actions.append(A)
            episode_rewards.append(R)
            total_steps += 1

            if done:
                break

            episode_obs.append(np.array(obs_next))
            S = encode_single(obs_next, model, means, stds)
            A = pick_action(S, epsilon)

        time_collect += time.time() - t0

        if cfg['finetune'] or cfg.get('finetune_once'):
            recent_episodes.append(episode_obs)

        # Phase 2: Batch encode
        t0 = time.time()
        all_states = batch_encode(np.stack(episode_obs), model, means, stds)
        time_encode += time.time() - t0

        # Phase 3: SARSA(λ) with sparse traces
        t0 = time.time()
        trace_indices = set()
        e = np.zeros(hash_size)
        T = len(episode_actions)

        for t in range(T):
            active = get_tiles(all_states[t], episode_actions[t])
            q_sa = np.sum(w[active])

            if t == T - 1:
                td_error = episode_rewards[t] - q_sa
            else:
                active_next = get_tiles(all_states[t + 1], episode_actions[t + 1])
                q_next = np.sum(w[active_next])
                td_error = episode_rewards[t] + gamma * q_next - q_sa

            if trace_indices:
                trace_list = np.array(list(trace_indices))
                e[trace_list] *= gamma * lam
                dead = trace_list[np.abs(e[trace_list]) < 1e-4]
                e[dead] = 0.0
                trace_indices -= set(dead)

            for i in active:
                e[i] = 1.0
                trace_indices.add(i)

            if trace_indices:
                trace_list = np.array(list(trace_indices))
                w[trace_list] += alpha * td_error * e[trace_list]

        time_sarsa += time.time() - t0

        # Phase 4a: Fine-tune once (single big update)
        if (cfg.get('finetune_once') and not finetune_once_done
                and ep >= cfg['finetune_once_at']
                and len(recent_episodes) >= 10):
            t0 = time.time()
            n_obs = sum(len(eo) for eo in recent_episodes)
            print(f"\n  === Fine-tune ONCE at episode {ep} ===")
            print(f"  Using {len(recent_episodes)} recent episodes ({n_obs} obs), "
                  f"{cfg['finetune_once_steps']} gradient steps")

            ft_loss = finetune_model(model, recent_episodes, ft_opt,
                                     steps=cfg['finetune_once_steps'])
            means, stds = compute_norm_stats(env, model, n_episodes=20)

            if cfg['finetune_once_reset_weights']:
                w[:] = 0.0
                ep = 0  # reset epsilon schedule
                print(f"  Reset SARSA weights + epsilon schedule")

            time_finetune += time.time() - t0
            finetune_once_done = True
            finetune_count = 1
            print(f"  Fine-tune loss: {ft_loss:.5f} | Time: {time.time() - t0:.1f}s")
            print(f"  === Resuming SARSA ===\n")

        # Phase 4b: Periodic fine-tuning (original mode)
        elif (cfg['finetune'] and not cfg.get('finetune_once')
                and cfg['finetune_start'] <= ep <= cfg['finetune_end']
                and ep % cfg['finetune_every'] == 0):
            t0 = time.time()
            ft_loss = finetune_model(model, recent_episodes, ft_opt,
                                     steps=cfg['finetune_steps'])
            means, stds = compute_norm_stats(env, model, n_episodes=len(recent_episodes))
            time_finetune += time.time() - t0
            finetune_count += 1

        returns.append(sum(episode_rewards))
        timestep_checkpoints.append(total_steps)
        ep += 1

        if ep % 25 == 0:
            elapsed = time.time() - t_start
            avg = np.mean(returns[-25:])
            ft_str = ""
            if cfg.get('finetune_once'):
                if finetune_once_done:
                    ft_str = " | FT: done"
                else:
                    ft_str = f" | FT: at ep {cfg['finetune_once_at']}"
            elif cfg['finetune']:
                if cfg['finetune_start'] <= ep <= cfg['finetune_end']:
                    ft_str = f" | FT: ON (#{finetune_count})"
                elif ep > cfg['finetune_end']:
                    ft_str = f" | FT: done ({finetune_count})"
            print(f"Steps: {total_steps:>8d} | Ep {ep:4d} | "
                  f"Avg(25): {avg:7.1f} | Eps: {epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s{ft_str}")

    _save_and_plot(cfg, returns, timestep_checkpoints, total_steps, mode='pixels',
                   intrinsic_dim=intrinsic_dim)


def _save_and_plot(cfg, returns, timestep_checkpoints, total_steps,
                   mode='pixels', intrinsic_dim=None):
    """Shared save and plot logic for both modes."""
    window = 25
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    smoothed_steps = timestep_checkpoints[window - 1:]

    plt.figure(figsize=(10, 6))
    plt.plot(timestep_checkpoints, returns, alpha=0.3, label='Raw')
    plt.plot(smoothed_steps, smoothed, label=f'{window}-ep avg')
    plt.xlabel('Timesteps')
    plt.ylabel('Return')
    mode_label = 'Raw State' if mode == 'raw' else 'Learned Repr'
    plt.title(f"SARSA(λ) + Tile Coding ({mode_label}) — {cfg['env_name'].title()}")
    plt.legend()
    plt.savefig(env_path(cfg, f"sarsa_{mode}.png"))
    plt.close()

    results = {
        'env': cfg['env_name'],
        'mode': mode,
        'returns': [float(r) for r in returns],
        'timesteps': [int(t) for t in timestep_checkpoints],
        'final_avg_25': float(np.mean(returns[-25:])),
        'final_avg_100': float(np.mean(returns[-100:])) if len(returns) >= 100 else None,
        'total_steps': total_steps,
        'config': {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
    }
    if intrinsic_dim is not None:
        results['intrinsic_dim'] = intrinsic_dim

    save_path = env_path(cfg, f"sarsa_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ({mode}) ===")
    print(f"Final avg (last 25):  {np.mean(returns[-25:]):.1f}")
    if len(returns) >= 100:
        print(f"Final avg (last 100): {np.mean(returns[-100:]):.1f}")
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
