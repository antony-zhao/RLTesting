"""
CACLA (Continuous Actor-Critic Learning Automaton) with tile coding.
For continuous action environments.

Usage: python train_cacla.py --env pendulum --mode raw
       python train_cacla.py --env pendulum --mode pixels
       python train_cacla.py --env lunarlander --mode raw
"""
import argparse
import copy
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import get_config, make_env, add_env_arg, to_numpy, env_path
from rltesting.fvd_experiments.tiles import tiles
from rltesting.fvd_experiments.fvd_models import (
    Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
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
    parser = argparse.ArgumentParser(description='CACLA with tile coding')
    add_env_arg(parser)
    parser.add_argument('--mode', type=str, default=None, choices=['raw', 'pixels'])
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--noise-std', type=float, default=None,
                        help='Exploration noise std (default: 0.3 * action_range)')
    parser.add_argument('--noise-decay', type=float, default=0.999,
                        help='Noise decay per episode')
    parser.add_argument('--noise-min', type=float, default=0.05,
                        help='Minimum noise std (as fraction of action range)')
    args = parser.parse_args()

    cfg = get_config(args.env)

    if args.mode is not None:
        mode = args.mode
    elif cfg.get('env_type') in ('classic', 'continuous'):
        mode = 'raw'
    else:
        mode = 'pixels'

    max_steps = args.max_steps or cfg.get('cacla_max_timesteps', cfg['sarsa_max_timesteps'])
    gamma = cfg['gamma']
    lam = cfg['lam']

    np.random.seed(42)
    torch.manual_seed(0)

    # Setup environment and state encoding
    if mode == 'pixels':
        env_mode = 'pixels' if cfg.get('env_type') in ('classic', 'continuous') else 'default'
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
        state_dim = env.observation_space.shape[0]

        def get_state(obs):
            return np.array(obs, dtype=np.float64)

    # Action space info
    action_low = env.action_space.low
    action_high = env.action_space.high
    action_dim = env.action_space.shape[0]
    action_range = action_high - action_low

    # Tile coding setup
    num_tilings = cfg.get('cacla_num_tilings', cfg['num_tilings'])
    num_tiles_per_dim = cfg.get('cacla_num_tiles_per_dim', cfg['num_tiles_per_dim'])
    hash_size = cfg.get('cacla_hash_size', cfg['hash_size'])
    alpha_critic = cfg.get('cacla_alpha_critic', cfg['alpha'])
    alpha_actor = cfg.get('cacla_alpha_actor', alpha_critic * 0.1)

    if mode == 'raw':
        obs_low = np.clip(env.observation_space.low, -10.0, None)
        obs_high = np.clip(env.observation_space.high, None, 10.0)
        obs_range = obs_high - obs_low
        obs_range[obs_range == 0] = 1.0
        scale = np.full(state_dim, num_tiles_per_dim) / obs_range
        offset = obs_low
    else:
        # Compute actual bounds from encoded data
        print("Computing tile bounds from encoded data...")
        sample_z = []
        for _ in range(20):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                sample_z.append(get_state(obs))
                done = term or trunc
        sample_z = np.array(sample_z)
        z_min = sample_z.min(axis=0)
        z_max = sample_z.max(axis=0)
        # Add small margin
        margin = (z_max - z_min) * 0.05
        z_min -= margin
        z_max += margin
        z_range = z_max - z_min
        z_range[z_range == 0] = 1.0
        offset = z_min
        scale = np.full(state_dim, num_tiles_per_dim) / z_range
        print(f"  Tile bounds: {z_min} to {z_max}")
        print(f"  Scale: {scale}")

    def get_tiles(state):
        scaled = ((state - offset) * scale).tolist()
        return tiles(hash_size, num_tilings, scaled)

    # Critic: V(s) — tile coded
    w_critic = np.zeros(hash_size)

    def V(state):
        return np.sum(w_critic[get_tiles(state)])

    # Actor: mu(s) — linear from tile features to action
    # One weight vector per action dimension
    w_actor = np.zeros((action_dim, hash_size))

    def mu(state):
        t = get_tiles(state)
        action = np.array([np.sum(w_actor[d][t]) for d in range(action_dim)])
        return np.clip(action, action_low, action_high)

    # Exploration noise
    noise_std = args.noise_std if args.noise_std is not None else 0.3 * action_range
    noise_min = args.noise_min * action_range

    print(f"\n=== CACLA ({mode}) — {cfg['env_name']} ===")
    print(f"  state_dim={state_dim}, action_dim={action_dim}")
    print(f"  action_range=[{action_low}, {action_high}]")
    print(f"  tilings={num_tilings}, tiles_per_dim={num_tiles_per_dim}, hash={hash_size}")
    print(f"  alpha_critic={alpha_critic:.5f}, alpha_actor={alpha_actor:.5f}")
    print(f"  gamma={gamma}, lambda={lam}")
    print(f"  noise_std={noise_std}, decay={args.noise_decay}, min={noise_min}")
    print(f"  max_steps={max_steps}")

    # Training loop
    returns = []
    timestep_checkpoints = []
    total_steps = 0
    ep = 0
    t_start = time.time()
    current_noise = noise_std.copy() if isinstance(noise_std, np.ndarray) else np.full(action_dim, noise_std)

    while total_steps < max_steps:
        obs, _ = env.reset()
        state = get_state(obs)
        ep_reward = 0

        # Eligibility traces for critic
        trace_critic = np.zeros(hash_size)
        trace_indices = set()

        while True:
            # Actor + noise
            action = mu(state) + np.random.randn(action_dim) * current_noise
            action = np.clip(action, action_low, action_high)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1
            ep_reward += reward

            next_state = get_state(obs_next)
            v_s = V(state)
            if terminated:
                v_sp = 0.0
            else:
                v_sp = V(next_state)
            td_error = reward + gamma * v_sp - v_s

            # Update critic with eligibility traces
            active = get_tiles(state)

            if trace_indices:
                trace_list = np.array(list(trace_indices))
                trace_critic[trace_list] *= gamma * lam
                dead = trace_list[np.abs(trace_critic[trace_list]) < 1e-4]
                trace_critic[dead] = 0.0
                trace_indices -= set(dead)

            for i in active:
                trace_critic[i] = 1.0
                trace_indices.add(i)

            if trace_indices:
                trace_list = np.array(list(trace_indices))
                w_critic[trace_list] += alpha_critic * td_error * trace_critic[trace_list]

            # Update actor ONLY if td_error > 0
            if td_error > 0:
                for d in range(action_dim):
                    current_mu_d = np.sum(w_actor[d][active])
                    error = action[d] - current_mu_d
                    w_actor[d][active] += alpha_actor * error

            if done:
                break

            state = next_state

        # Decay noise
        current_noise = np.maximum(current_noise * args.noise_decay, noise_min)

        returns.append(ep_reward)
        timestep_checkpoints.append(total_steps)
        ep += 1

        if ep % 25 == 0:
            elapsed = time.time() - t_start
            avg = np.mean(returns[-25:])
            print(f"Steps: {total_steps:>8d} | Ep {ep:4d} | "
                  f"Avg(25): {avg:7.1f} | Noise: {current_noise[0]:.3f} | "
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
    plt.title(f"CACLA ({mode}) — {cfg['env_name'].title()}")
    plt.legend()
    plt.savefig(env_path(cfg, f"cacla_{mode}.png"))
    plt.close()

    # Save
    results = {
        'env': cfg['env_name'],
        'mode': mode,
        'method': 'cacla',
        'returns': [float(r) for r in returns],
        'timesteps': [int(t) for t in timestep_checkpoints],
        'final_avg_25': float(np.mean(returns[-25:])),
        'final_avg_100': float(np.mean(returns[-100:])) if len(returns) >= 100 else None,
        'total_steps': total_steps,
        'config': {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
    }
    if mode == 'pixels':
        results['intrinsic_dim'] = state_dim

    save_path = env_path(cfg, f"cacla_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Final avg (last 25):  {np.mean(returns[-25:]):.1f}")
    if len(returns) >= 100:
        print(f"Final avg (last 100): {np.mean(returns[-100:]):.1f}")
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
