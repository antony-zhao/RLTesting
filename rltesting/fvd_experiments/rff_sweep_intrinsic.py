"""
RFF bandwidth/alpha sweep on learned intrinsic representation.
Run in tmux: python rff_sweep_intrinsic.py
"""
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation
import numpy as np
import torch
import os
import time
import json
from datetime import datetime

# ---- Import your local modules ---- #
from wrapper import BasicEnvironmentRGB
from fvd_models import Encoder, Decoder, AEVAE

to_numpy = lambda x: x.cpu().detach().numpy()

# ---- Config ---- #
framestack = 2
obs_shape = 64
aevae_path = 'aevae_original.pt'
num_features = 2048
lam = 0.9
gamma = 1.0
num_episodes = 1000
eval_last = 200

bandwidths = [1.0, 2.0, 3.0, 6.0, 10.0, 12.0, 16.0]
alphas = [0.005, 0.01, 0.02]

# ---- Setup environment ---- #
def make_env():
    e = gym.make('Acrobot-v1', render_mode='rgb_array')
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
    ep_obs = [obs]
    while not done:
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        ep_obs.append(obs)
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
print(f"Stats computed from {len(all_intrinsic)} observations")

# ---- Helper functions ---- #
def encode_obs(obs):
    with torch.no_grad():
        t = (torch.tensor(obs).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
        raw = to_numpy(aevae.double_encode_deterministic(t)).squeeze()
        return (raw - means) / stds

# ---- Sweep ---- #
results = {}
total_configs = len(bandwidths) * len(alphas)
config_num = 0

print(f"\nStarting sweep: {total_configs} configs, {num_episodes} episodes each")
print(f"Bandwidths: {bandwidths}")
print(f"Alphas: {alphas}")
print("=" * 80)

for bandwidth in bandwidths:
    for alpha in alphas:
        config_num += 1
        total_features = num_features * num_actions

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
        t_start = time.time()

        for ep in range(num_episodes):
            epsilon = max(0.01, 1.0 - ep / 400)
            obs, _ = env.reset()
            S = encode_obs(obs)
            A = pick_action(S, epsilon)
            total_reward = 0
            trace = np.zeros(total_features)

            while True:
                obs_next, R, terminated, truncated, _ = env.step(A)
                done = terminated or truncated
                total_reward += R

                z = rff_z(S)
                s = A * num_features
                q_sa = w[s:s + num_features] @ z

                if done:
                    td_error = R - q_sa
                else:
                    S_next = encode_obs(obs_next)
                    A_next = pick_action(S_next, epsilon)
                    z_next = rff_z(S_next)
                    s_next = A_next * num_features
                    q_next = w[s_next:s_next + num_features] @ z_next
                    td_error = R + gamma * q_next - q_sa

                trace *= gamma * lam
                trace[s:s + num_features] = z
                nz = np.nonzero(trace)[0]
                w[nz] += alpha * td_error * trace[nz]
                trace[np.abs(trace) < 1e-4] = 0.0

                if done:
                    break
                S, A = S_next, A_next

            returns.append(total_reward)

        elapsed = time.time() - t_start
        avg_last = np.mean(returns[-eval_last:])
        avg_all = np.mean(returns)
        results[(bandwidth, alpha)] = {
            'avg_last': avg_last,
            'avg_all': avg_all,
            'returns': returns,
            'time': elapsed
        }

        print(f"[{config_num:2d}/{total_configs}] BW={bandwidth:5.1f} | "
              f"α={alpha:.3f} | Last {eval_last} avg: {avg_last:7.1f} | "
              f"Overall avg: {avg_all:7.1f} | Time: {elapsed:.0f}s")

# ---- Summary ---- #
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"{'BW':>6s} | {'Alpha':>6s} | {'Last ' + str(eval_last):>10s} | {'Overall':>10s} | {'Time':>6s}")
print("-" * 50)

for (bw, al), res in sorted(results.items()):
    print(f"{bw:6.1f} | {al:6.3f} | {res['avg_last']:10.1f} | {res['avg_all']:10.1f} | {res['time']:5.0f}s")

best = max(results, key=lambda k: results[k]['avg_last'])
print(f"\nBest: bandwidth={best[0]}, alpha={best[1]}, "
      f"return={results[best]['avg_last']:.1f}")

# ---- Save results ---- #
save_results = {
    str(k): {'avg_last': v['avg_last'], 'avg_all': v['avg_all'],
             'returns': v['returns'], 'time': v['time']}
    for k, v in results.items()
}
save_path = f"rff_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(save_path, 'w') as f:
    json.dump(save_results, f, indent=2)
print(f"\nResults saved to {save_path}")