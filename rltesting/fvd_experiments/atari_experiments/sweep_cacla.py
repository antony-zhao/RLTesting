"""
CACLA hyperparameter sweep for continuous control on learned representations.

Usage:
  python sweep_cacla.py --env pendulum --parallel 4
  python sweep_cacla.py --env pendulum --parallel 4 --alpha-actors 0.005 0.01 0.02 --alpha-critics 0.05 0.1 0.2
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
from rltesting.fvd_experiments.tiles import tiles
from rltesting.fvd_experiments.fvd_models import (
    Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
)


def load_autoencoder(cfg):
    path = env_path(cfg, "autoencoder.pt")
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


def run_single_config(env, model, means, stds, intrinsic_dim, cfg,
                       alpha_critic, alpha_actor, num_tilings, num_tiles_per_dim,
                       noise_decay, noise_min_frac, max_steps):
    """Run CACLA with a single hyperparameter config."""
    gamma = cfg['gamma']
    lam = cfg['lam']
    hash_size = cfg.get('hash_size', 2**16)

    action_low = env.action_space.low
    action_high = env.action_space.high
    action_dim = env.action_space.shape[0]
    action_range = action_high - action_low
    noise_std = 0.3 * action_range
    noise_min = noise_min_frac * action_range

    # Tile coding
    offset = np.full(intrinsic_dim, -3.0)
    scale = np.full(intrinsic_dim, num_tiles_per_dim / 6.0)

    def get_tiles(state):
        scaled = ((state - offset) * scale).tolist()
        return tiles(hash_size, num_tilings, scaled)

    # Critic and Actor
    w_critic = np.zeros(hash_size)
    w_actor = np.zeros((action_dim, hash_size))

    def V(state):
        return np.sum(w_critic[get_tiles(state)])

    def mu(state):
        t = get_tiles(state)
        action = np.array([np.sum(w_actor[d][t]) for d in range(action_dim)])
        return np.clip(action, action_low, action_high)

    returns = []
    total_steps = 0
    current_noise = noise_std.copy() if isinstance(noise_std, np.ndarray) else np.full(action_dim, noise_std)
    t_start = time.time()

    while total_steps < max_steps:
        obs, _ = env.reset()
        state = encode_single(obs, model, means, stds)
        ep_reward = 0
        trace_critic = np.zeros(hash_size)
        trace_indices = set()

        while True:
            action = mu(state) + np.random.randn(action_dim) * current_noise
            action = np.clip(action, action_low, action_high)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1
            ep_reward += reward

            next_state = encode_single(obs_next, model, means, stds)
            v_s = V(state)
            if terminated:
                v_sp = 0.0
            else:
                v_sp = V(next_state)
            td_error = reward + gamma * v_sp - v_s

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

            if td_error > 0:
                for d in range(action_dim):
                    current_mu_d = np.sum(w_actor[d][active])
                    error = action[d] - current_mu_d
                    w_actor[d][active] += alpha_actor * error

            if done:
                break
            state = next_state

        current_noise = np.maximum(current_noise * noise_decay, noise_min)
        returns.append(ep_reward)

    elapsed = time.time() - t_start
    eval_last = min(200, len(returns))
    avg_last = np.mean(returns[-eval_last:])

    return {
        'alpha_critic': alpha_critic,
        'alpha_actor': alpha_actor,
        'num_tilings': num_tilings,
        'num_tiles_per_dim': num_tiles_per_dim,
        'noise_decay': noise_decay,
        'noise_min_frac': noise_min_frac,
        'avg_last': avg_last,
        'avg_all': np.mean(returns),
        'peak_100': float(np.max(np.convolve(returns, np.ones(100)/100, mode='valid'))) if len(returns) >= 100 else float(np.mean(returns)),
        'returns': returns,
        'time': elapsed,
        'total_steps': total_steps,
    }


def _merge_sweep_results(cfg):
    import glob
    sweep_dir = env_path(cfg, "")
    worker_files = sorted(glob.glob(os.path.join(sweep_dir, "cacla_sweep_w*_*.json")))
    if not worker_files:
        print("No worker files found to merge.")
        return

    merged = {}
    for f in worker_files:
        with open(f) as fh:
            data = json.load(fh)
            merged.update(data)
        print(f"  Loaded {len(data)} configs from {os.path.basename(f)}")

    sorted_results = sorted(merged.items(), key=lambda x: x[1]['avg_last'], reverse=True)
    print(f"\n{'='*80}")
    print(f"MERGED RESULTS ({len(merged)} configs, sorted by score)")
    print(f"{'='*80}")
    for key, res in sorted_results[:15]:
        print(f"  {key:<45s} | Last: {res['avg_last']:7.1f} | Peak100: {res['peak_100']:7.1f}")
    if len(sorted_results) > 15:
        print(f"  ... ({len(sorted_results)-15} more)")

    best_key = sorted_results[0][0]
    print(f"\nBest: {best_key} -> {merged[best_key]['avg_last']:.1f}")

    save_path = env_path(cfg, f"cacla_sweep_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged results to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='CACLA hyperparameter sweep')
    add_env_arg(parser)
    parser.add_argument('--alpha-actors', type=float, nargs='+',
                        default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--alpha-critics', type=float, nargs='+',
                        default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--tilings', type=int, nargs='+',
                        default=[16])
    parser.add_argument('--tiles-per-dim', type=int, nargs='+',
                        default=[4])
    parser.add_argument('--noise-decays', type=float, nargs='+',
                        default=[0.999, 0.9995, 0.9998])
    parser.add_argument('--noise-mins', type=float, nargs='+',
                        default=[0.1])
    parser.add_argument('--max-steps', type=int, default=300000,
                        help='Steps per config (shorter for sweep)')
    parser.add_argument('--worker', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.env)

    if args.parallel is not None:
        import subprocess, sys
        n = args.parallel
        base_cmd = [sys.executable, __file__,
                    '--env', args.env,
                    '--max-steps', str(args.max_steps),
                    '--num-workers', str(n),
                    '--alpha-actors'] + [str(a) for a in args.alpha_actors] + \
                   ['--alpha-critics'] + [str(a) for a in args.alpha_critics] + \
                   ['--tilings'] + [str(t) for t in args.tilings] + \
                   ['--tiles-per-dim'] + [str(t) for t in args.tiles_per_dim] + \
                   ['--noise-decays'] + [str(d) for d in args.noise_decays] + \
                   ['--noise-mins'] + [str(m) for m in args.noise_mins]

        procs = []
        for i in range(n):
            cmd = base_cmd + ['--worker', str(i)]
            print(f"Launching worker {i}/{n}")
            p = subprocess.Popen(cmd)
            procs.append(p)

        print(f"\n{n} workers launched. Waiting...")
        for i, p in enumerate(procs):
            p.wait()
            print(f"Worker {i} finished (exit code {p.returncode})")

        print("\nMerging results...")
        _merge_sweep_results(cfg)
        return

    # Build config list
    configs = [(ac, aa, nt, ntpd, nd, nm)
               for ac in args.alpha_critics
               for aa in args.alpha_actors
               for nt in args.tilings
               for ntpd in args.tiles_per_dim
               for nd in args.noise_decays
               for nm in args.noise_mins]

    if args.worker is not None:
        chunk_size = (len(configs) + args.num_workers - 1) // args.num_workers
        start = args.worker * chunk_size
        end = min(start + chunk_size, len(configs))
        configs = configs[start:end]
        print(f"Worker {args.worker}/{args.num_workers}: configs {start}-{end-1}")

    total = len(configs)
    print(f"\nCACLA Sweep: {cfg['env_name']} | {total} configs, {args.max_steps} steps each")

    # Setup
    env = make_env(cfg, mode='pixels')
    ae_model, intrinsic_dim = load_autoencoder(cfg)
    ae_model = copy.deepcopy(ae_model)
    ae_model.eval()
    means, stds = compute_norm_stats(env, ae_model)

    print(f"Intrinsic dim: {intrinsic_dim}")
    print(f"Norm stats - means: {means}, stds: {stds}")
    print("=" * 80)

    results = {}
    for i, (ac, aa, nt, ntpd, nd, nm) in enumerate(configs):
        np.random.seed(42)

        res = run_single_config(env, ae_model, means, stds, intrinsic_dim, cfg,
                                 alpha_critic=ac, alpha_actor=aa,
                                 num_tilings=nt, num_tiles_per_dim=ntpd,
                                 noise_decay=nd, noise_min_frac=nm,
                                 max_steps=args.max_steps)

        key = f"ac={ac}_aa={aa}_nt={nt}_ntpd={ntpd}_nd={nd}_nm={nm}"
        results[key] = res
        print(f"[{i+1:3d}/{total}] {key} | Last: {res['avg_last']:7.1f} | Peak100: {res['peak_100']:7.1f} | {res['time']:.0f}s")

    # Summary
    print("\n" + "=" * 80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_last'], reverse=True)
    for key, res in sorted_results[:10]:
        print(f"  {key:<45s} | Last: {res['avg_last']:7.1f} | Peak100: {res['peak_100']:7.1f}")

    worker_suffix = f"_w{args.worker}" if args.worker is not None else ""
    save_path = env_path(cfg, f"cacla_sweep{worker_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
