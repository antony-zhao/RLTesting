"""
Plot all experimental results for comparison.
Reads JSON files from per-env directories and uploaded results.

Usage: python plot_results.py [--dir results_dir]
"""
import json
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def load_results(search_dirs=None):
    """Load all JSON result files from directories."""
    if search_dirs is None:
        search_dirs = ['.']

    results = defaultdict(list)
    for d in search_dirs:
        for path in glob.glob(os.path.join(d, '**', '*.json'), recursive=True):
            try:
                with open(path) as f:
                    data = json.load(f)
                # Determine type
                fname = os.path.basename(path)
                env = data.get('env', '?')

                if 'sarsa' in fname:
                    mode = data.get('mode', 'pixels')
                    label = f'SARSA {mode}'
                elif 'tiny_dqn' in fname:
                    mode = data.get('mode', 'pixels')
                    label = f'Tiny DQN {mode}'
                elif 'ppo' in fname:
                    mode = data.get('mode', 'pixels')
                    label = f'PPO {mode}'
                else:
                    continue

                rets = data.get('returns', data.get('training_returns', []))
                steps = data.get('timesteps', data.get('training_timesteps', []))

                if len(rets) < 10:
                    continue

                results[env].append({
                    'label': label,
                    'returns': rets,
                    'timesteps': steps,
                    'path': path,
                    'data': data,
                })
            except (json.JSONDecodeError, KeyError):
                continue

    return results


def smooth(returns, window=50):
    """Smoothed returns."""
    if len(returns) < window:
        return returns
    return np.convolve(returns, np.ones(window) / window, mode='valid')


def _is_atari(env_name):
    """Check if an environment is Atari."""
    atari = {'boxing', 'bowling', 'pong', 'assault', 'spaceinvaders', 'breakout', 'freeway'}
    return env_name.lower() in atari


def plot_learning_curves(results, save_path='all_learning_curves.png'):
    """Plot learning curves, separate figures for Atari and Classic Control."""
    atari_envs = sorted([e for e in results if _is_atari(e)])
    classic_envs = sorted([e for e in results if not _is_atari(e)])

    groups = []
    if atari_envs:
        groups.append(('Atari', atari_envs, 'atari_learning_curves.png'))
    if classic_envs:
        groups.append(('Classic Control', classic_envs, 'classic_learning_curves.png'))

    colors = {
        'SARSA raw': '#1f77b4',
        'SARSA pixels': '#ff7f0e',
        'PPO ram': '#2ca02c',
        'PPO pixels': '#d62728',
        'Tiny DQN raw': '#8c564b',
        'Tiny DQN pixels': '#e377c2',
    }

    for group_name, envs, fname in groups:
        n = len(envs)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        fig.suptitle(group_name, fontsize=16, fontweight='bold', y=1.02)

        for idx, env in enumerate(envs):
            ax = axes[idx // cols][idx % cols]
            runs = results[env]

            # Deduplicate: keep best run per label
            best_runs = {}
            for run in runs:
                label = run['label']
                rets = run['returns']
                score = np.mean(rets[-100:]) if len(rets) >= 100 else np.mean(rets[-25:])
                if label not in best_runs or score > best_runs[label]['score']:
                    best_runs[label] = {**run, 'score': score}

            for label, run in sorted(best_runs.items()):
                rets = run['returns']
                steps = run['timesteps']
                color = colors.get(label, '#7f7f7f')

                window = 50 if len(rets) > 200 else 25
                sm = smooth(rets, window)
                sm_steps = steps[window - 1:] if len(steps) >= window else steps

                ax.plot(steps, rets, alpha=0.12, color=color)
                ax.plot(sm_steps[:len(sm)], sm, label=label, color=color, linewidth=2)

            ax.set_title(env.title(), fontsize=14, fontweight='bold')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Return')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {group_name} learning curves to {fname}")


def plot_bar_comparison(results, save_path='bar_comparison.png'):
    """Separate bar charts for Atari and Classic Control."""
    atari_envs = sorted([e for e in results if _is_atari(e)])
    classic_envs = sorted([e for e in results if not _is_atari(e)])

    groups = []
    if atari_envs:
        groups.append(('Atari', atari_envs))
    if classic_envs:
        groups.append(('Classic Control', classic_envs))

    if not groups:
        return

    colors = {
        'SARSA raw': '#1f77b4',
        'SARSA pixels': '#ff7f0e',
        'PPO ram': '#2ca02c',
        'PPO pixels': '#d62728',
        'Tiny DQN raw': '#8c564b',
        'Tiny DQN pixels': '#e377c2',
    }

    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5), squeeze=False)

    for g_idx, (group_name, envs) in enumerate(groups):
        ax = axes[0][g_idx]

        method_scores = defaultdict(dict)
        for env in envs:
            for run in results[env]:
                label = run['label']
                rets = run['returns']
                data = run['data']
                if 'eval_mean' in data:
                    score = data['eval_mean']
                elif len(rets) >= 100:
                    score = np.mean(rets[-100:])
                else:
                    score = np.mean(rets[-25:])
                if env not in method_scores[label] or score > method_scores[label][env]:
                    method_scores[label][env] = score

        methods = sorted(method_scores.keys())
        x = np.arange(len(envs))
        width = 0.8 / max(len(methods), 1)

        for i, method in enumerate(methods):
            scores = [method_scores[method].get(env, 0) for env in envs]
            offset = (i - len(methods) / 2 + 0.5) * width
            color = colors.get(method, '#7f7f7f')
            bars = ax.bar(x + offset, scores, width * 0.9, label=method, color=color)
            for bar, score in zip(bars, scores):
                if score != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{score:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([e.title() for e in envs], fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(group_name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar comparison to {save_path}")


def print_summary_table(results):
    """Print a text summary table."""
    envs = sorted(results.keys())
    if not envs:
        print("No results found!")
        return

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Env':<15s} | {'Method':<15s} | {'Final Score':>12s} | {'Peak 100-ep':>12s} | "
          f"{'Steps':>10s} | {'Episodes':>8s}")
    print("-" * 100)

    for env in envs:
        runs = sorted(results[env], key=lambda r: r['label'])
        for run in runs:
            label = run['label']
            rets = run['returns']
            steps = run['timesteps']
            data = run['data']

            if 'eval_mean' in data:
                final = f"{data['eval_mean']:.1f}±{data.get('eval_std', 0):.1f}"
            elif len(rets) >= 100:
                final = f"{np.mean(rets[-100:]):.1f}"
            else:
                final = f"{np.mean(rets[-25:]):.1f}"

            if len(rets) >= 100:
                rolling = np.convolve(rets, np.ones(100) / 100, mode='valid')
                peak = f"{np.max(rolling):.1f}"
            else:
                peak = "—"

            total_steps = data.get('total_steps', data.get('total_timesteps', 0))
            n_eps = len(rets)

            print(f"{env:<15s} | {label:<15s} | {final:>12s} | {peak:>12s} | "
                  f"{total_steps / 1e6:>9.1f}M | {n_eps:>8d}")
        print("-" * 100)

    print()


def plot_atari_vs_baselines(results, save_path='atari_comparison.png'):
    """Plot Atari results with published baselines."""
    baselines = {
        'boxing':  {'Random': 0.1, 'Human': 12.1, 'DQN (200M)': 88},
        'pong':    {'Random': -20.7, 'Human': 14.6},
        'bowling': {'Random': 23.1, 'Human': 160.7, 'DQN (200M)': 42},
        'assault': {'Random': 222, 'Human': 742, 'DQN (200M)': 3359},
        'spaceinvaders': {'Random': 148, 'Human': 1652, 'DQN (200M)': 1976},
    }

    atari_envs = [e for e in sorted(results.keys()) if e in baselines]
    if not atari_envs:
        return

    n = len(atari_envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for idx, env in enumerate(atari_envs):
        ax = axes[0][idx]
        bl = baselines.get(env, {})

        # Collect scores
        scores = {}
        for name, val in bl.items():
            scores[name] = val

        for run in results[env]:
            label = run['label']
            rets = run['returns']
            data = run['data']
            if 'eval_mean' in data:
                scores[label] = data['eval_mean']
            elif len(rets) >= 100:
                scores[label] = np.mean(rets[-100:])
            else:
                scores[label] = np.mean(rets[-25:])

        # Plot bars
        names = list(scores.keys())
        vals = list(scores.values())
        bar_colors = []
        for name in names:
            if 'SARSA' in name:
                bar_colors.append('#ff7f0e')
            elif 'Tiny DQN' in name:
                bar_colors.append('#e377c2')
            elif 'PPO' in name:
                bar_colors.append('#d62728')
            elif name == 'Human':
                bar_colors.append('#2ca02c')
            elif 'DQN' in name:
                bar_colors.append('#9467bd')
            else:
                bar_colors.append('#bdbdbd')

        bars = ax.barh(range(len(names)), vals, color=bar_colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(env.title(), fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Atari comparison to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot all results')
    parser.add_argument('--dir', type=str, nargs='+', default=['.'],
                        help='Directories to search for results')
    args = parser.parse_args()

    results = load_results(args.dir)

    if not results:
        print("No result files found! Make sure you're in the directory with env folders.")
        print("Searched:", args.dir)
        return

    print(f"Found results for: {', '.join(sorted(results.keys()))}")
    print(f"Total runs: {sum(len(v) for v in results.values())}")

    print_summary_table(results)
    plot_learning_curves(results)
    plot_bar_comparison(results)
    plot_atari_vs_baselines(results)

    print("\nAll plots saved.")


if __name__ == '__main__':
    main()