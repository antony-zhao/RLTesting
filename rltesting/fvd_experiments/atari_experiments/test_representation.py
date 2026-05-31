"""
Test learned representation quality by comparing intrinsic dims to true state.
Uses a SINGLE pixel env and extracts raw state from the unwrapped env body.

Usage:
  python test_representation.py --env pendulum
  python test_representation.py --env lunarlander
  python test_representation.py --env cartpole
  python test_representation.py --env acrobot
  python test_representation.py --env mountaincar
"""
import argparse
import numpy as np
import torch
from numpy import corrcoef

from config import get_config, make_env, add_env_arg, to_numpy, env_path
from train_cacla import load_autoencoder, compute_norm_stats, encode_single


# ---- Raw state extraction from pixel envs ---- #

def get_raw_state_pendulum(env):
    theta, thetadot = env.unwrapped.state
    return np.array([np.cos(theta), np.sin(theta), thetadot])

def get_raw_state_lunarlander(env):
    ll = env.unwrapped
    pos = ll.lander.position
    vel = ll.lander.linearVelocity
    return np.array([
        pos.x, pos.y,
        vel.x, vel.y,
        ll.lander.angle, ll.lander.angularVelocity,
        1.0 if ll.legs[0].ground_contact else 0.0,
        1.0 if ll.legs[1].ground_contact else 0.0,
    ])

def get_raw_state_cartpole(env):
    return np.array(env.unwrapped.state)

def get_raw_state_acrobot(env):
    s = env.unwrapped.state
    return np.array([
        np.cos(s[0]), np.sin(s[0]),
        np.cos(s[1]), np.sin(s[1]),
        s[2], s[3],
    ])

def get_raw_state_mountaincar(env):
    return np.array(env.unwrapped.state)


STATE_EXTRACTORS = {
    'pendulum':    (get_raw_state_pendulum,    ['cos_theta', 'sin_theta', 'velocity']),
    'lunarlander': (get_raw_state_lunarlander, ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'leg_L', 'leg_R']),
    'cartpole':    (get_raw_state_cartpole,    ['x', 'x_dot', 'theta', 'theta_dot']),
    'acrobot':     (get_raw_state_acrobot,     ['cos_t1', 'sin_t1', 'cos_t2', 'sin_t2', 'dtheta1', 'dtheta2']),
    'mountaincar': (get_raw_state_mountaincar, ['position', 'velocity']),
}


def collect_paired_data(env, model, means, stds, get_raw_state, n_steps=2000):
    """Collect (raw_state, intrinsic_state) pairs from a single pixel env."""
    raw_states, intrinsic_states = [], []
    obs, _ = env.reset()

    for _ in range(n_steps):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)

        try:
            raw = get_raw_state(env)
        except Exception as e:
            # Skip steps where state extraction fails (e.g. after reset)
            if term or trunc:
                obs, _ = env.reset()
            continue

        z = encode_single(obs, model, means, stds)
        raw_states.append(raw)
        intrinsic_states.append(z)

        if term or trunc:
            obs, _ = env.reset()

    return np.array(raw_states), np.array(intrinsic_states)


def run_correlation_test(raw_states, intrinsic_states, state_names, env_name):
    """Run linear correlation and nonlinear R² tests."""
    n_raw = raw_states.shape[1]
    n_intrinsic = intrinsic_states.shape[1]

    print(f"\n{'='*70}")
    print(f"Representation Quality: {env_name}")
    print(f"  Samples: {len(raw_states)}")
    print(f"  Raw state dims: {n_raw}, Intrinsic dims: {n_intrinsic}")
    print(f"{'='*70}")

    # Linear correlations
    print(f"\n--- Linear Correlations (|r| > 0.3) ---")
    found_any = False
    for i, name in enumerate(state_names):
        for j in range(n_intrinsic):
            c = corrcoef(raw_states[:, i], intrinsic_states[:, j])[0, 1]
            if abs(c) > 0.3:
                print(f"  {name:>12s} vs intrinsic[{j}]: r={c:+.3f}")
                found_any = True
    if not found_any:
        print("  None found! (all |r| < 0.3)")

    # Best linear correlation per state variable
    print(f"\n--- Best Linear |r| per State Variable ---")
    for i, name in enumerate(state_names):
        best_r = 0
        best_j = -1
        for j in range(n_intrinsic):
            c = corrcoef(raw_states[:, i], intrinsic_states[:, j])[0, 1]
            if abs(c) > abs(best_r):
                best_r = c
                best_j = j
        quality = "GOOD" if abs(best_r) > 0.7 else "OK" if abs(best_r) > 0.4 else "WEAK" if abs(best_r) > 0.2 else "NONE"
        print(f"  {name:>12s}: |r|={abs(best_r):.3f} (dim {best_j}) [{quality}]")

    # Nonlinear R² via MLP
    print(f"\n--- Nonlinear R² (MLP Regressor) ---")
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')

    for i, name in enumerate(state_names):
        scores = cross_val_score(
            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=42),
            intrinsic_states, raw_states[:, i], cv=5, scoring='r2')
        r2 = scores.mean()
        quality = "GOOD" if r2 > 0.8 else "OK" if r2 > 0.5 else "WEAK" if r2 > 0.0 else "FAIL"
        print(f"  {name:>12s}: R²={r2:.3f} [{quality}]")

    # Intrinsic dim statistics
    print(f"\n--- Intrinsic Dim Statistics ---")
    print(f"  Mins:  {intrinsic_states.min(axis=0)}")
    print(f"  Maxs:  {intrinsic_states.max(axis=0)}")
    print(f"  Stds:  {intrinsic_states.std(axis=0)}")
    print(f"  Means: {intrinsic_states.mean(axis=0)}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Test representation quality')
    add_env_arg(parser)
    parser.add_argument('--n-steps', type=int, default=2000,
                        help='Number of steps to collect')
    args = parser.parse_args()

    env_name = args.env
    cfg = get_config(env_name)

    if env_name not in STATE_EXTRACTORS:
        print(f"No state extractor for {env_name}. Available: {list(STATE_EXTRACTORS.keys())}")
        return

    get_raw_state, state_names = STATE_EXTRACTORS[env_name]

    # Load autoencoder
    pixel_env = make_env(cfg, mode='pixels')
    model, idim = load_autoencoder(cfg)
    means, stds = compute_norm_stats(pixel_env, model)

    # Collect paired data from single env
    raw_states, intrinsic_states = collect_paired_data(
        pixel_env, model, means, stds, get_raw_state, n_steps=args.n_steps)

    # Run tests
    run_correlation_test(raw_states, intrinsic_states, state_names, env_name)


if __name__ == '__main__':
    main()