from wrapper import BasicEnvironmentRGB, DomainRandomization, RandomColorReplace
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation
from matplotlib import pyplot as plt
from rltesting.torch_rl.utils import random_sample_single_env
from rltesting.torch_rl.buffers import ReplayBuffer
from rltesting.torch_rl.models import MLP
from fvd_models import Encoder, Decoder, AEVAE
import numpy as np
import torch
import skdim
import ale_py
import os
from collections import deque

np.random.seed(42)
torch.manual_seed(0)

to_numpy = lambda x: x.cpu().detach().numpy()
framestack = 2
latent_dim = None
obs_shape = 64
data_steps = 200000
num_epochs = 50
steps_per_epoch = 1000
gym.register_envs(ale_py)
env = gym.make('ALE/Boxing-v5', render_mode="rgb_array", repeat_action_probability=0.0)
env = BasicEnvironmentRGB(env)
env = ResizeObservation(env, (obs_shape, obs_shape))
env = FrameStackObservation(env, stack_size=framestack)
env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0, 3)), observation_space=gym.spaces.Box(0, 255, (framestack, 3, obs_shape, obs_shape), dtype=np.uint8))
env = TransformObservation(env, lambda obs: np.reshape(obs, (obs_shape, obs_shape, 3 * framestack)), observation_space=gym.spaces.Box(0, 255, (obs_shape, obs_shape, 3 * framestack), dtype=np.uint8))
buffer_shapes = [(obs_shape, obs_shape, 3 * framestack), (), (), ()]
dtypes = [np.uint8, np.float32, np.float32, np.float32]
buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=data_steps)

filepath = "atari_buffer.npz"
if os.path.isfile(filepath):
    buffer.load(filepath)
else:
    samples = random_sample_single_env(env, num_steps=data_steps)
    for i in range(data_steps):
        buffer.add_sample([samples[j][i] for j in range(len(samples))].copy())
    buffer.save(filepath=filepath)

aevae_path = 'boxing_aevae.pt'
if os.path.isfile(aevae_path):
    print("Loading saved AEVAE...")
    ckpt = torch.load(aevae_path, weights_only=False)
    intrinsic_dim = ckpt['intrinsic_dim']
    latent_dim = ckpt['latent_dim']
    encoder = Encoder(framestack, latent_dim, obs_shape).cuda()
    decoder = Decoder(encoder.conv_dim, framestack, latent_dim).cuda()
    double_autoencoder = AEVAE(encoder, decoder, latent_dim, intrinsic_dim, hidden_dim=256).cuda()
    double_autoencoder.load_state_dict(ckpt['weights'])
    print(f"Loaded. Intrinsic dim: {intrinsic_dim}, Latent dim: {latent_dim}")

for name, module in [
    ("Stage 1 Encoder (conv + linear)", double_autoencoder.encoder),
    ("Stage 1 Decoder (linear + conv)", double_autoencoder.decoder),
    ("Intrinsic Encoder (MLP)",         double_autoencoder.intrinsic_encoder),
    ("Intrinsic mu head",               double_autoencoder.intrinsic_mu),
    ("Intrinsic logvar head",           double_autoencoder.intrinsic_logvar),
    ("Intrinsic Decoder (MLP)",         double_autoencoder.intrinsic_decoder),
]:
    n = sum(p.numel() for p in module.parameters())
    print(f"{name:<40s}  {n:>10,}")

total = sum(p.numel() for p in double_autoencoder.parameters())
print(f"{'TOTAL':<40s}  {total:>10,}")

# ---- SARSA(λ) + Hashed Tile Coding on Boxing ---- #
import copy
import time
from tiles import tiles

aevae = copy.deepcopy(double_autoencoder)
aevae.eval()

num_actions = env.action_space.n
num_tiles_per_dim = 2
num_tilings = 64
hash_size = 2 ** 21

w = np.zeros(hash_size)
alpha = 0.1 / num_tilings
lam = 0.9
gamma = 0.99
max_timesteps = 5_000_000
num_episodes = 10000
encode_batch_size = 128

# ---- Fine-tuning config ---- #
finetune_start = num_episodes#1000
finetune_end = num_episodes#2000
finetune_every = 5
finetune_steps = 50
finetune_lr = 1e-5
penalty_coef = 0
ft_opt = torch.optim.Adam(aevae.parameters(), finetune_lr)

# Rolling buffer of recent episode observations for fine-tuning
recent_episodes = deque(maxlen=30)

# Normalization stats
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
mins = np.full(intrinsic_dim, -3.0)
scale = np.full(intrinsic_dim, num_tiles_per_dim / 6.0)

print(f"Intrinsic dim: {intrinsic_dim}, Num tilings: {num_tilings}, Hash size: {hash_size}")


def encode_obs(obs):
    with torch.no_grad():
        t = (torch.tensor(np.array(obs)).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
        raw = to_numpy(aevae.double_encode_deterministic(t)).squeeze()
        return (raw - means) / stds


def batch_encode(obs_array):
    with torch.no_grad():
        results = []
        for i in range(0, len(obs_array), encode_batch_size):
            batch = obs_array[i:i + encode_batch_size]
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            raw = to_numpy(aevae.double_encode_deterministic(t))
            results.append(raw)
        return (np.concatenate(results, axis=0) - means) / stds


def recompute_stats():
    """Recompute normalization stats from recent on-policy episodes."""
    global means, stds
    all_z = []
    for ep_obs in recent_episodes:
        with torch.no_grad():
            batch = np.stack(ep_obs)
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            raw = to_numpy(aevae.double_encode_deterministic(t))
            all_z.append(raw)
    all_z = np.concatenate(all_z, axis=0)
    means = all_z.mean(axis=0)
    stds = all_z.std(axis=0)
    stds[stds == 0] = 1.0
    return len(all_z)


def finetune_aevae():
    """Fine-tune AEVAE on recent on-policy data."""
    # Build a flat observation pool from recent episodes
    all_obs = []
    for ep_obs in recent_episodes:
        all_obs.extend(ep_obs)
    all_obs = np.stack(all_obs)
    n = len(all_obs)

    aevae.train()
    total_loss = 0.0
    for step in range(finetune_steps):
        idx = np.random.choice(n, size=min(256, n), replace=False)
        batch = (torch.tensor(all_obs[idx]).float().transpose(-3, -1) / 255).cuda()
        loss = aevae.reconstruction_loss(batch, penalty_coef)
        loss.backward()
        ft_opt.step()
        ft_opt.zero_grad()
        total_loss += loss.item()
    aevae.eval()

    return total_loss / finetune_steps


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


# ---- Training loop ---- #
time_collect = 0
time_encode = 0
time_sarsa = 0
time_finetune = 0

returns = []
timestep_checkpoints = []
total_steps = 0
ep = 0
t_start = time.time()
finetune_count = 0

while total_steps < max_timesteps and ep < num_episodes:
    epsilon = max(0.05, 1.0 - ep / 1000)

    # Phase 1: Collect
    t0 = time.time()
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

    time_collect += time.time() - t0

    # Store episode observations for fine-tuning
    recent_episodes.append(episode_obs)

    # Phase 2: Batch encode
    t0 = time.time()
    all_states = batch_encode(np.stack(episode_obs))
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

    time_sarsa += time.time() - t0

    # Phase 4: Delayed fine-tuning
    if finetune_start <= ep <= finetune_end and ep % finetune_every == 0:
        t0 = time.time()
        ft_loss = finetune_aevae()
        n_stats = recompute_stats()
        time_finetune += time.time() - t0
        finetune_count += 1

        if finetune_count % 10 == 0:
            print(f"  [Fine-tune #{finetune_count}] Loss: {ft_loss:.5f} | "
                  f"Stats from {n_stats} obs | FT time: {time_finetune:.1f}s")

    returns.append(sum(episode_rewards))
    timestep_checkpoints.append(total_steps)
    ep += 1

    if ep % 25 == 0:
        elapsed = time.time() - t_start
        avg = np.mean(returns[-25:])
        ft_status = ""
        if finetune_start <= ep <= finetune_end:
            ft_status = f" | FT: ON (#{finetune_count})"
        elif ep > finetune_end:
            ft_status = f" | FT: done ({finetune_count} total)"
        print(f"Steps: {total_steps:>8d} | Episode {ep:4d} | "
              f"Avg Return (last 25): {avg:7.1f} | Epsilon: {epsilon:.3f} | "
              f"Time: {elapsed:.0f}s{ft_status}")
        print(f"  Collect: {time_collect:.1f}s | Encode: {time_encode:.1f}s | "
              f"SARSA: {time_sarsa:.1f}s | Finetune: {time_finetune:.1f}s")

# ---- Plot ---- #
window = 25
smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
smoothed_steps = [timestep_checkpoints[i] for i in range(window - 1, len(timestep_checkpoints))]
plt.figure(figsize=(10, 6))
plt.plot(timestep_checkpoints, returns, alpha=0.3, label='Raw')
plt.plot(smoothed_steps, smoothed, label=f'{window}-ep avg')

# Mark fine-tuning window
ft_start_step = timestep_checkpoints[min(finetune_start, len(timestep_checkpoints) - 1)]
ft_end_step = timestep_checkpoints[min(finetune_end, len(timestep_checkpoints) - 1)]
plt.axvspan(ft_start_step, ft_end_step, alpha=0.1, color='green', label='Fine-tune window')

plt.xlabel('Timesteps')
plt.ylabel('Return')
plt.title('SARSA(λ) + Hashed Tile Coding + Fine-tuning - Boxing')
plt.legend()
plt.savefig('sarsa_boxing_finetune.png')
plt.close()

total = time_collect + time_encode + time_sarsa + time_finetune
print(f"\n=== Timing ===")
print(f"Collect:  {time_collect:7.1f}s ({100 * time_collect / total:.0f}%)")
print(f"Encode:   {time_encode:7.1f}s ({100 * time_encode / total:.0f}%)")
print(f"SARSA:    {time_sarsa:7.1f}s ({100 * time_sarsa / total:.0f}%)")
print(f"Finetune: {time_finetune:7.1f}s ({100 * time_finetune / total:.0f}%)")
print(f"Fine-tune rounds: {finetune_count}")