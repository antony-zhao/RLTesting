"""
SARSA(λ) with Radial Basis Function (RBF) features.
Uses fixed RBF centers (sampled from data or grid) with Gaussian kernels
as features for linear SARSA(λ).

Reference: Sutton & Barto (2018) §9.5.5 — RBFs as a generalization of
coarse coding where each feature is a soft Gaussian activation rather
than a binary tile indicator.

Usage:
  python train_rbf.py --env acrobot --mode raw
  python train_rbf.py --env cartpole --mode raw --num-centers 200
  python train_rbf.py --env mountaincar --mode raw
  python train_rbf.py --env acrobot --mode pixels
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


# ---- Autoencoder loading (for pixel mode) ---- #

def load_autoencoder(cfg):
    from rltesting.fvd_experiments.fvd_models import (
        Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
    )
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
        from rltesting.fvd_experiments.fvd_models import IMPALAEncoder, IMPALADecoder
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


# ---- RBF Feature Network ---- #

class RBFFeatures:
    """
    Fixed RBF feature representation.
    
    Given state s, produces feature vector:
        phi(s)_i = exp(-||s - c_i||^2 / (2 * sigma_i^2))
    
    Centers are placed by sampling from random exploration data.
    Bandwidths are set per-center using k-nearest-neighbor heuristic,
    or a global bandwidth.
    """
    def __init__(self, centers, bandwidth='auto', k_neighbors=5):
        """
        Args:
            centers: (num_centers, state_dim) array of RBF centers
            bandwidth: 'auto' for per-center adaptive, or float for global sigma
            k_neighbors: number of neighbors for adaptive bandwidth
        """
        self.centers = np.array(centers, dtype=np.float64)  # (C, D)
        self.num_centers = len(centers)
        self.state_dim = centers.shape[1]

        if bandwidth == 'auto':
            # Per-center bandwidth: median distance to k nearest neighbors
            from scipy.spatial.distance import cdist
            dists = cdist(self.centers, self.centers)
            np.fill_diagonal(dists, np.inf)
            k = min(k_neighbors, self.num_centers - 1)
            knn_dists = np.sort(dists, axis=1)[:, :k]
            self.sigmas = np.median(knn_dists, axis=1)  # (C,)
            self.sigmas[self.sigmas < 1e-6] = 1.0  # avoid division by zero
            print(f"  Adaptive bandwidth: median={np.median(self.sigmas):.4f}, "
                  f"min={self.sigmas.min():.4f}, max={self.sigmas.max():.4f}")
        else:
            self.sigmas = np.full(self.num_centers, float(bandwidth))

        # Precompute 1/(2*sigma^2) for efficiency
        self.inv_2sigma2 = 1.0 / (2.0 * self.sigmas ** 2)  # (C,)

    def __call__(self, state):
        """Compute RBF features for a single state."""
        # state: (D,) -> diffs: (C, D)
        diffs = self.centers - state[np.newaxis, :]
        sq_dists = np.sum(diffs ** 2, axis=1)  # (C,)
        phi = np.exp(-sq_dists * self.inv_2sigma2)  # (C,)
        return phi

    @property
    def num_features(self):
        return self.num_centers


def sample_centers_from_env(env, get_state, num_centers, num_steps=50000):
    """Collect states from random exploration, then subsample centers."""
    print(f"  Collecting {num_steps} states for RBF centers...")
    states = []
    obs, _ = env.reset()
    for _ in range(num_steps):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        states.append(get_state(obs))
        if term or trunc:
            obs, _ = env.reset()

    states = np.array(states)
    # Subsample uniformly
    indices = np.random.choice(len(states), size=min(num_centers, len(states)),
                               replace=False)
    centers = states[indices]
    print(f"  Selected {len(centers)} centers from {len(states)} states")
    return centers


def main():
    parser = argparse.ArgumentParser(description='SARSA(λ) with RBF features')
    add_env_arg(parser)
    parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'pixels'])
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--num-centers', type=int, default=500,
                        help='Number of RBF centers')
    parser.add_argument('--bandwidth', type=str, default='auto',
                        help='"auto" for adaptive, or float for global sigma')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Learning rate (default: 0.1 / num_centers)')
    parser.add_argument('--lam', type=float, default=None,
                        help='Eligibility trace decay')
    parser.add_argument('--center-steps', type=int, default=50000,
                        help='Random steps for center sampling')
    args = parser.parse_args()

    cfg = get_config(args.env)
    mode = args.mode
    gamma = cfg['gamma']
    lam = args.lam if args.lam is not None else cfg['lam']
    max_steps = args.max_steps or cfg['sarsa_max_timesteps']
    num_actions = cfg.get('num_actions', None)

    np.random.seed(42)
    torch.manual_seed(0)

    # Setup environment and state encoding
    if mode == 'pixels':
        env = make_env(cfg, mode='pixels')
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

    if num_actions is None:
        num_actions = env.action_space.n

    # Sample RBF centers from environment
    bandwidth = args.bandwidth
    if bandwidth != 'auto':
        bandwidth = float(bandwidth)

    centers = sample_centers_from_env(env, get_state, args.num_centers,
                                      num_steps=args.center_steps)
    rbf = RBFFeatures(centers, bandwidth=bandwidth)
    num_features = rbf.num_features

    # Learning rate
    alpha = args.alpha if args.alpha is not None else 0.1 / num_features

    # Epsilon schedule
    eps_decay = cfg.get('epsilon_decay_episodes', 1000)
    eps_min = cfg.get('epsilon_min', 0.05)

    print(f"\n=== SARSA(λ) + RBF ({mode}) — {cfg['env_name']} ===")
    print(f"  state_dim={state_dim}, num_actions={num_actions}")
    print(f"  num_centers={num_features}, bandwidth={args.bandwidth}")
    print(f"  alpha={alpha:.6f}, lambda={lam}, gamma={gamma}")
    print(f"  max_steps={max_steps}")

    # Weight matrix: (num_actions, num_features)
    w = np.zeros((num_actions, num_features))

    def Q(phi, a):
        return np.dot(w[a], phi)

    def Q_all(phi):
        return w @ phi  # (num_actions,)

    def epsilon_greedy(phi, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(num_actions)
        return np.argmax(Q_all(phi))

    # Training loop
    returns = []
    timestep_checkpoints = []
    total_steps = 0
    ep = 0
    t_start = time.time()

    while total_steps < max_steps:
        obs, _ = env.reset()
        state = get_state(obs)
        phi = rbf(state)

        epsilon = max(eps_min, 1.0 - ep / eps_decay)
        action = epsilon_greedy(phi, epsilon)

        ep_reward = 0

        # Eligibility traces: (num_actions, num_features)
        # Use accumulating traces but only track active action
        e = np.zeros((num_actions, num_features))

        while True:
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1
            ep_reward += reward

            if done:
                td = reward - Q(phi, action)
                e[action] += phi  # accumulating trace
                w += alpha * td * e
                break

            next_state = get_state(obs_next)
            phi_next = rbf(next_state)
            next_action = epsilon_greedy(phi_next, epsilon)

            td = reward + gamma * Q(phi_next, next_action) - Q(phi, action)

            # Accumulating traces for SARSA
            e *= gamma * lam
            e[action] += phi

            w += alpha * td * e

            phi = phi_next
            action = next_action

        returns.append(ep_reward)
        timestep_checkpoints.append(total_steps)
        ep += 1

        if ep % 25 == 0:
            elapsed = time.time() - t_start
            avg = np.mean(returns[-25:])
            print(f"Steps: {total_steps:>8d} | Ep {ep:4d} | "
                  f"Avg(25): {avg:7.1f} | Eps: {epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")

    # Final stats
    final_avg_25 = float(np.mean(returns[-25:]))
    final_avg_100 = float(np.mean(returns[-100:])) if len(returns) >= 100 else None

    print(f"\n=== Results ({mode}) ===")
    print(f"Final avg (last 25):  {final_avg_25:.1f}")
    if final_avg_100 is not None:
        print(f"Final avg (last 100): {final_avg_100:.1f}")

    # Plot
    window = 25
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    smoothed_steps = timestep_checkpoints[window - 1:]

    plt.figure(figsize=(10, 6))
    plt.plot(timestep_checkpoints, returns, alpha=0.3, label='Raw')
    plt.plot(smoothed_steps, smoothed, label=f'{window}-ep avg')
    plt.xlabel('Timesteps')
    plt.ylabel('Return')
    plt.title(f"SARSA(λ)+RBF ({mode}) — {cfg['env_name'].title()}")
    plt.legend()
    plt.savefig(env_path(cfg, f"rbf_{mode}.png"))
    plt.close()

    # Save
    results = {
        'env': cfg['env_name'],
        'mode': mode,
        'method': 'rbf_sarsa',
        'returns': [float(r) for r in returns],
        'timesteps': [int(t) for t in timestep_checkpoints],
        'final_avg_25': final_avg_25,
        'final_avg_100': final_avg_100,
        'total_steps': total_steps,
        'num_centers': num_features,
        'bandwidth': str(args.bandwidth),
        'alpha': alpha,
        'config': {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
    }
    if mode == 'pixels':
        results['intrinsic_dim'] = state_dim

    save_path = env_path(cfg, f"rbf_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
