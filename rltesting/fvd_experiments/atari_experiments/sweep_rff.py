"""
RFF bandwidth/alpha/num_features sweep for SARSA(λ).
Supports parallel execution via --worker and --num-workers.

Usage:
  # Single process (all configs):
  python sweep_rff.py --env boxing

  # Parallel (4 workers in separate terminals):
  python sweep_rff.py --env boxing --num-workers 4 --worker 0
  python sweep_rff.py --env boxing --num-workers 4 --worker 1
  python sweep_rff.py --env boxing --num-workers 4 --worker 2
  python sweep_rff.py --env boxing --num-workers 4 --worker 3

  # Custom sweep ranges:
  python sweep_rff.py --env pong --bandwidths 1.0 2.0 4.0 --alphas 0.005 0.01 0.02
"""
import argparse
import copy
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from config import get_config, make_env, add_env_arg, to_numpy, env_path
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
    """Compute mean/std of intrinsic representations."""
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


def batch_encode(obs_array, model, means, stds, batch_size=128):
    with torch.no_grad():
        results = []
        for i in range(0, len(obs_array), batch_size):
            batch = obs_array[i:i + batch_size]
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            raw = to_numpy(model.double_encode(t))
            results.append(raw)
        return (np.concatenate(results, axis=0) - means) / stds


def run_single_config(env, model, means, stds, intrinsic_dim, num_actions, cfg,
                       bandwidth, alpha, num_features, num_episodes, eval_last):
    """Run SARSA(λ) + RFF with a single hyperparameter config."""
    lam = cfg['lam']
    gamma = cfg['gamma']
    total_features = num_features * num_actions

    # Initialize RFF
    np.random.seed(42)
    W_rff = np.random.randn(num_features, intrinsic_dim) / bandwidth
    b_rff = np.random.uniform(0, 2 * np.pi, num_features)
    np.random.seed(None)

    w = np.zeros(total_features)

    def rff_z(state):
        return np.cos(W_rff @ state + b_rff) * np.sqrt(2.0 / num_features)

    def Q_all(state):
        z = rff_z(state)
        qs = np.zeros(num_actions)
        for a in range(num_actions):
            s = a * num_features
            qs[a] = w[s:s + num_features] @ z
        return qs

    def pick_action(state, eps):
        if np.random.random() < eps:
            return np.random.randint(num_actions)
        return np.argmax(Q_all(state))

    returns = []
    total_steps = 0
    t_start = time.time()

    for ep in range(num_episodes):
        epsilon = max(0.05, 1.0 - ep / (0.5 * num_episodes))

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

        # Batch encode + SARSA(λ)
        all_states = batch_encode(np.stack(episode_obs), model, means, stds)
        trace = np.zeros(total_features)
        T = len(episode_actions)

        for t in range(T):
            z = rff_z(all_states[t])
            s = episode_actions[t] * num_features
            q_sa = w[s:s + num_features] @ z

            if t == T - 1:
                td_error = episode_rewards[t] - q_sa
            else:
                z_next = rff_z(all_states[t + 1])
                s_next = episode_actions[t + 1] * num_features
                q_next = w[s_next:s_next + num_features] @ z_next
                td_error = episode_rewards[t] + gamma * q_next - q_sa

            trace *= gamma * lam
            dot = np.dot(trace[s:s + num_features], z)
            trace[s:s + num_features] += z * (1 - alpha * gamma * lam * dot)
            nz = np.nonzero(trace)[0]
            w[nz] += alpha * td_error * trace[nz]
            trace[np.abs(trace) < 1e-4] = 0.0

        returns.append(sum(episode_rewards))

    elapsed = time.time() - t_start
    avg_last = np.mean(returns[-eval_last:])
    avg_all = np.mean(returns)

    return {
        'bandwidth': bandwidth,
        'alpha': alpha,
        'num_features': num_features,
        'avg_last': avg_last,
        'avg_all': avg_all,
        'returns': returns,
        'time': elapsed,
        'total_steps': total_steps,
    }


def _merge_sweep_results(cfg):
    """Merge all worker sweep JSONs into a single results file."""
    import glob
    sweep_dir = env_path(cfg, "")
    worker_files = sorted(glob.glob(os.path.join(sweep_dir, "rff_sweep_w*_*.json")))
    if not worker_files:
        print("No worker files found to merge.")
        return

    merged = {}
    for f in worker_files:
        with open(f) as fh:
            data = json.load(fh)
            merged.update(data)
        print(f"  Loaded {len(data)} configs from {os.path.basename(f)}")

    # Print sorted summary
    sorted_results = sorted(merged.items(), key=lambda x: x[1]['avg_last'], reverse=True)
    print(f"\n{'='*80}")
    print(f"MERGED RESULTS ({len(merged)} configs, sorted by score)")
    print(f"{'='*80}")
    for key, res in sorted_results[:10]:
        print(f"  {key:<30s} | Last avg: {res['avg_last']:7.1f} | Overall: {res['avg_all']:7.1f}")
    if len(sorted_results) > 10:
        print(f"  ... ({len(sorted_results)-10} more)")

    best_key = sorted_results[0][0]
    print(f"\nBest: {best_key} -> {merged[best_key]['avg_last']:.1f}")

    save_path = env_path(cfg, f"rff_sweep_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged results to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='RFF hyperparameter sweep')
    add_env_arg(parser)
    parser.add_argument('--bandwidths', type=float, nargs='+',
                        default=[1.0, 2.0, 4.0, 8.0, 12.0, 16.0],
                        help='Bandwidth values to sweep')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.001, 0.005, 0.01, 0.02, 0.05],
                        help='Alpha values to sweep')
    parser.add_argument('--num-features', type=int, nargs='+',
                        default=[2048, 4096],
                        help='Number of RFF features to sweep')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Episodes per config')
    parser.add_argument('--eval-last', type=int, default=200,
                        help='Episodes to average for final score')
    parser.add_argument('--worker', type=int, default=None,
                        help='Worker index (0-based) for parallel execution')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Total number of parallel workers')
    parser.add_argument('--parallel', type=int, default=None,
                        help='Launch N workers in parallel from this process')
    args = parser.parse_args()

    # If --parallel is set, spawn N subprocesses and wait
    if args.parallel is not None:
        import subprocess, sys
        n = args.parallel
        base_cmd = [sys.executable, __file__,
                    '--env', args.env,
                    '--episodes', str(args.episodes),
                    '--eval-last', str(args.eval_last),
                    '--num-workers', str(n),
                    '--bandwidths'] + [str(b) for b in args.bandwidths] + \
                   ['--alphas'] + [str(a) for a in args.alphas] + \
                   ['--num-features'] + [str(f) for f in args.num_features]

        procs = []
        for i in range(n):
            cmd = base_cmd + ['--worker', str(i)]
            print(f"Launching worker {i}/{n}: {' '.join(cmd[-4:])}")
            p = subprocess.Popen(cmd)
            procs.append(p)

        print(f"\n{n} workers launched. Waiting for completion...")
        for i, p in enumerate(procs):
            p.wait()
            print(f"Worker {i} finished (exit code {p.returncode})")

        print("\nAll workers done. Merging results...")
        _merge_sweep_results(get_config(args.env))
        return

    cfg = get_config(args.env)

    # Build config list
    configs = [(bw, al, nf)
               for bw in args.bandwidths
               for al in args.alphas
               for nf in args.num_features]

    # Split for parallel workers
    if args.worker is not None:
        chunk_size = (len(configs) + args.num_workers - 1) // args.num_workers
        start = args.worker * chunk_size
        end = min(start + chunk_size, len(configs))
        configs = configs[start:end]
        print(f"Worker {args.worker}/{args.num_workers}: running configs {start}-{end-1}")

    total_configs = len(configs)
    print(f"\nRFF Sweep: {cfg['env_name']} | {total_configs} configs, {args.episodes} eps each")
    print(f"Bandwidths: {args.bandwidths}")
    print(f"Alphas: {args.alphas}")
    print(f"Num features: {args.num_features}")
    print("=" * 80)

    # Setup
    env_mode = 'pixels' if cfg.get('env_type') == 'classic' else 'default'
    env = make_env(cfg, mode=env_mode)
    num_actions = env.action_space.n
    model, intrinsic_dim = load_autoencoder(cfg)
    model = copy.deepcopy(model)
    model.eval()
    means, stds = compute_norm_stats(env, model)

    # Run sweep
    results = {}
    for i, (bw, al, nf) in enumerate(configs):
        res = run_single_config(env, model, means, stds, intrinsic_dim, num_actions, cfg,
                                 bandwidth=bw, alpha=al, num_features=nf,
                                 num_episodes=args.episodes, eval_last=args.eval_last)
        key = f"bw={bw}_al={al}_nf={nf}"
        results[key] = res
        print(f"[{i+1:3d}/{total_configs}] BW={bw:5.1f} | α={al:.3f} | D={nf:5d} | "
              f"Last {args.eval_last} avg: {res['avg_last']:7.1f} | Time: {res['time']:.0f}s")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS (sorted by score)")
    print("=" * 80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_last'], reverse=True)
    for key, res in sorted_results:
        print(f"  {key:<30s} | Last {args.eval_last}: {res['avg_last']:7.1f} | "
              f"Overall: {res['avg_all']:7.1f} | Steps: {res['total_steps']:>8d}")

    best_key = sorted_results[0][0]
    best = results[best_key]
    print(f"\nBest: {best_key} -> {best['avg_last']:.1f}")

    # Save
    worker_suffix = f"_w{args.worker}" if args.worker is not None else ""
    save_path = env_path(cfg, f"rff_sweep{worker_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
