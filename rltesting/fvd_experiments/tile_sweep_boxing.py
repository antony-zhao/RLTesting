"""
Hashed tile coding hyperparameter sweep on Boxing.
Run in tmux: python tile_sweep_boxing.py
Estimated runtime: 8-10 hours
"""
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation
import numpy as np
import torch
import os
import time
import json
from datetime import datetime
import ale_py

from wrapper import BasicEnvironmentRGB
from fvd_models import Encoder, Decoder, AEVAE
from tiles import tiles

to_numpy = lambda x: x.cpu().detach().numpy()

# ---- Config ---- #
framestack = 2
obs_shape = 64
latent_dim = 512
aevae_path = 'boxing_aevae.pt'
lam = 0.9
gamma = 0.99
num_episodes = 500
eval_last = 200

# Sweep parameters
tiles_per_dim_options = [2, 3, 4, 5]
alpha_multipliers = [0.5, 1.0, 2.0, 4.0]  # multiplied by 0.1/num_tilings
hash_sizes = [2**21]  # 512K, 1M, 2M

# ---- Setup environment ---- #
def make_env():
    gym.register_envs(ale_py)
    e = gym.make('ALE/Boxing-v5', render_mode='rgb_array')
    e = BasicEnvironmentRGB(e)
    e = ResizeObservation(e, (obs_shape, obs_shape))
    e = FrameStackObservation(e, stack_size=framestack)
    e = TransformObservation(e, lambda obs: np.transpose(obs, (1, 2, 0, 3)),
        observation_space=gym.spaces.Box(0, 255, (framestack, 3, obs_shape, obs_shape), dtype=np.uint8))
    e = TransformObservation(e, lambda obs: np.reshape(obs, (obs_shape, obs_shape, 3 * framestack)),
        observation_space=gym.spaces.Box(0, 255, (obs_shape, obs_shape, 3 * framestack), dtype=np.uint8))
    return e

env = make_env()
num_actions = env.action_space.n
print(f"Actions: {num_actions}")

# ---- Load AEVAE ---- #
print("Loading AEVAE...")
ckpt = torch.load(aevae_path, weights_only=False)
intrinsic_dim = ckpt['intrinsic_dim']
latent_dim = ckpt['latent_dim']
encoder = Encoder(framestack, latent_dim, obs_shape).cuda()
decoder = Decoder(encoder.conv_dim, framestack, latent_dim).cuda()
aevae = AEVAE(encoder, decoder, latent_dim, intrinsic_dim).cuda()
aevae.load_state_dict(ckpt['weights'])
aevae.eval()
print(f"Loaded. Intrinsic dim: {intrinsic_dim}, Latent dim: {latent_dim}")

# ---- Compute normalization stats ---- #
print("Computing normalization stats...")
all_intrinsic = []
for _ in range(20):
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
        raw = to_numpy(aevae.double_encode_deterministic(t))
        all_intrinsic.append(raw)

all_intrinsic = np.concatenate(all_intrinsic, axis=0)
means = all_intrinsic.mean(axis=0)
stds = all_intrinsic.std(axis=0)
stds[stds == 0] = 1.0
print(f"Stats from {len(all_intrinsic)} observations")

# ---- Helpers ---- #
def encode_obs(obs):
    with torch.no_grad():
        t = (torch.tensor(np.array(obs)).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
        raw = to_numpy(aevae.double_encode_deterministic(t)).squeeze()
        return (raw - means) / stds

def batch_encode(obs_array):
    with torch.no_grad():
        results = []
        for i in range(0, len(obs_array), 128):
            batch = obs_array[i:i+128]
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            raw = to_numpy(aevae.double_encode_deterministic(t))
            results.append(raw)
        return (np.concatenate(results, axis=0) - means) / stds

# ---- Sweep ---- #
results = {}
configs = [(tpd, am, hs) for tpd in tiles_per_dim_options 
           for am in alpha_multipliers for hs in hash_sizes]
total_configs = len(configs)
config_num = 0

print(f"\nStarting sweep: {total_configs} configs, {num_episodes} episodes each")
print(f"Tiles per dim: {tiles_per_dim_options}")
print(f"Alpha multipliers: {alpha_multipliers}")
print(f"Hash sizes: {hash_sizes}")
print("=" * 90)

for num_tiles_per_dim, alpha_mult, hash_size in configs:
    config_num += 1
    num_tilings = max(16, 4 * intrinsic_dim)  # at least 4x dims, power of 2-ish
    # Round up to next power of 2
    num_tilings = 1 << (num_tilings - 1).bit_length()
    
    alpha = alpha_mult * 0.1 / num_tilings
    
    mins = np.full(intrinsic_dim, -3.0)
    scale = np.full(intrinsic_dim, num_tiles_per_dim / 6.0)

    w = np.zeros(hash_size)

    def get_tiles(state, action):
        scaled = ((state - mins) * scale).tolist()
        return tiles(hash_size, num_tilings, scaled, ints=[action])

    def Q(state, action):
        return np.sum(w[get_tiles(state, action)])

    def Q_all(state):
        return np.array([Q(state, a) for a in range(num_actions)])

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

        S = encode_obs(obs)
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
            S = encode_obs(obs_next)
            A = pick_action(S, epsilon)

        # Batch encode + SARSA(λ)
        all_states = batch_encode(np.stack(episode_obs))
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

        returns.append(sum(episode_rewards))

    elapsed = time.time() - t_start
    avg_last = np.mean(returns[-eval_last:])
    avg_all = np.mean(returns)
    
    config_key = f"tpd={num_tiles_per_dim}_am={alpha_mult}_hs={hash_size}"
    results[config_key] = {
        'num_tiles_per_dim': num_tiles_per_dim,
        'alpha_multiplier': alpha_mult,
        'alpha': alpha,
        'hash_size': hash_size,
        'num_tilings': num_tilings,
        'avg_last': avg_last,
        'avg_all': avg_all,
        'returns': returns,
        'time': elapsed,
        'total_steps': total_steps
    }

    print(f"[{config_num:2d}/{total_configs}] TPD={num_tiles_per_dim} | "
          f"α×={alpha_mult:.1f} (α={alpha:.5f}) | HS={hash_size:>7d} | "
          f"Last {eval_last} avg: {avg_last:7.1f} | Steps: {total_steps:>8d} | "
          f"Time: {elapsed:.0f}s")

# ---- Summary ---- #
print("\n" + "=" * 90)
print("RESULTS SUMMARY (sorted by last avg)")
print("=" * 90)
print(f"{'TPD':>4s} | {'α×':>4s} | {'α':>8s} | {'HS':>8s} | {'NT':>4s} | "
      f"{'Last ' + str(eval_last):>10s} | {'Overall':>10s} | {'Steps':>10s} | {'Time':>6s}")
print("-" * 85)

sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_last'], reverse=True)
for key, res in sorted_results:
    print(f"{res['num_tiles_per_dim']:4d} | {res['alpha_multiplier']:4.1f} | "
          f"{res['alpha']:8.5f} | {res['hash_size']:8d} | {res['num_tilings']:4d} | "
          f"{res['avg_last']:10.1f} | {res['avg_all']:10.1f} | "
          f"{res['total_steps']:10d} | {res['time']:5.0f}s")

best_key = sorted_results[0][0]
best = results[best_key]
print(f"\nBest: tiles_per_dim={best['num_tiles_per_dim']}, "
      f"alpha_mult={best['alpha_multiplier']}, hash_size={best['hash_size']}, "
      f"return={best['avg_last']:.1f}")

# ---- Save ---- #
save_path = f"tile_sweep_boxing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {save_path}")