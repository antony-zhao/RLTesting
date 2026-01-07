from wrapper import BasicEnvironmentRGB, DomainRandomization
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation
from matplotlib import pyplot as plt
from rltesting.torch_rl.utils import random_sample_single_env
from rltesting.torch_rl.buffers import ReplayBuffer
from rltesting.torch_rl.models import MLP, TargetNetwork
from rltesting.utils.logger import Logger
from fundamental_variable_discovery import Encoder, Decoder, DoubleAutoEncoder
import torch.nn.functional as F
import numpy as np
import torch
import skdim
import time
import ale_py

gym.register_envs(ale_py)

to_numpy = lambda x: x.cpu().detach().numpy()
framestack = 2
latent_dim = 2048
obs_shape = 64
data_steps = 50000
train_steps = 8000
l1_coeff = 0
env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
env = ResizeObservation(env, (64, 64))
env = FrameStackObservation(env, stack_size=framestack)
env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0, 3)), observation_space=gym.spaces.Box(0, 255, (framestack, 3, obs_shape, obs_shape), dtype=np.uint8))
env = TransformObservation(env, lambda obs: np.reshape(obs, (obs_shape, obs_shape, 3 * framestack)), observation_space=gym.spaces.Box(0, 255, (obs_shape, obs_shape, 3 * framestack), dtype=np.uint8))
buffer_shapes = [(obs_shape, obs_shape, 3 * framestack), (), (), ()]
dtypes = [np.uint8, np.float32, np.float32, np.float32]
buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=data_steps)

samples = random_sample_single_env(env, num_steps=data_steps)
for i in range(data_steps):
    buffer.add_sample([samples[j][i] for j in range(len(samples))].copy())
    
encoder = Encoder(framestack, latent_dim, obs_shape, filter_base=16).cuda()
decoder = Decoder(encoder.conv_dim, framestack, latent_dim, filter_base=16).cuda()
opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 3e-4)
losses = []
for i in range(train_steps):
    if i % 1000 == 0:
        print(i)
    samples = buffer.sample(256)
    obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).cuda()
    latent = encoder(obs)
    reconstruction = decoder(latent)
    loss = torch.nn.functional.mse_loss(reconstruction, obs)
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.detach().cpu().numpy())

latent_ID = []
image_ID = []

for _ in range(20):
    samples = buffer.sample(2048)
    obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).cuda()
    latent = encoder(obs)
    unique_latents = np.unique(latent.detach().cpu().numpy(), axis=0)
    unique_obs = np.unique(obs.detach().cpu().numpy(), axis=0).reshape(-1, 3 * framestack*obs_shape*obs_shape)

    latent_ID.append(skdim.id.MLE().fit(unique_latents).dimension_)
    image_ID.append(skdim.id.MLE().fit(unique_obs).dimension_)

print(f"Latent ID: {np.mean(latent_ID)} +- {np.std(latent_ID)}")
print(f"Image ID: {np.mean(image_ID)} +- {np.std(image_ID)}")

intrinsic_dim = round(2 * np.mean(latent_ID))
double_autoencoder = DoubleAutoEncoder(encoder, decoder, latent_dim, intrinsic_dim).cuda()
int_opt = torch.optim.AdamW(double_autoencoder.parameters(), 3e-4)

losses = []
for i in range(train_steps):
    if i % 1000 == 0:
        print(i)
    samples = buffer.sample(256)
    obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).cuda()
    intrinsic = double_autoencoder.double_encode(obs)
    reconstruction = double_autoencoder.double_decode(intrinsic)
    loss = torch.nn.functional.mse_loss(reconstruction, obs)
    loss.backward()
    int_opt.step()
    int_opt.zero_grad()
    opt.step()
    opt.zero_grad()
    losses.append(loss.detach().cpu().numpy())

del buffer

num_actions = 6
q_network = MLP(intrinsic_dim, num_actions, skip_connections=False).cuda()
q_opt = torch.optim.Adam(q_network.parameters(), 3e-4)
dtypes = [np.uint8, np.float32, np.float32, np.uint8, np.float32]
buffer_shapes = [(obs_shape, obs_shape, 3 * framestack), (), (), (obs_shape, obs_shape, 3 * framestack), ()]
dqn_buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=1_000_000)
q_target = TargetNetwork(q_network, tau=1e-4).cuda()
ae_target = TargetNetwork(double_autoencoder, tau=3e-5).cuda()
eps = 1
eps_decay = 0.99
eps_min = 0.1
num_episodes = 10000
train_after = 10000
gamma = 0.99

def dqn_loss(q_network, q_target, obs, action, reward, next_obs, done, q_opt):
    q_target.update()
    q_target = reward + gamma * q_target(next_obs).detach().gather(1, torch.argmax(q_network(next_obs).detach(), 1, keepdim=True)).squeeze() * (1 - done)
    q_values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze()
    q_loss = F.mse_loss(q_target, q_values)
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()
    return q_loss

def model_losses(q_network, q_target, double_autoencoder, ae_target, buffer, q_opt, double_ae_opt, batch_size=256):
    ae_target.update()
    obs, action, reward, next_obs, done = buffer.sample(batch_size)
    obs = (torch.as_tensor(obs).float().transpose(-3, -1) / 255).cuda()
    action = torch.as_tensor(action).long().cuda()
    reward = torch.as_tensor(reward).float().cuda()
    next_obs = (torch.as_tensor(next_obs).float().transpose(-3, -1) / 255).cuda()
    done = torch.as_tensor(done).float().cuda()
    intrinsic = ae_target.net.double_encode(obs)
    next_intrinsic = ae_target.net.double_encode(next_obs).detach()
    q_loss = dqn_loss(q_network, q_target, intrinsic.detach(), action, reward, next_intrinsic, done, q_opt)
    
    reconstruction_loss = double_autoencoder.reconstruction_loss(obs)
    double_ae_opt.zero_grad()
    reconstruction_loss.backward()
    double_ae_opt.step()
    
    return to_numpy(q_loss.cpu()), to_numpy(reconstruction_loss.cpu())

ep_rews = []
q_losses = []
ae_losses = []
smoothed_rewards = []
step = 0
ep_rew = 0
ep_rew_ema = None
smoothing = 0.9
logger = Logger('logs/pong_neural_variables')
for episode in range(1, num_episodes + 1):
    if episode % 50 == 0:
        print(episode, ep_rew_ema)
    obs, _ = env.reset()
    done = False
    ep_rew = 0
    q_loss_ = []
    ae_loss_ = []
    while not done:
        step += 1
        if step > train_after:
            obs_tensor = (torch.tensor(obs).float().transpose(-3, -1) / 255).cuda()
            if np.random.rand() < eps:
                action = np.random.randint(num_actions)
            else:
                intrinsic = double_autoencoder.double_encode(obs_tensor.unsqueeze(0))
                action = to_numpy(q_network(intrinsic).argmax())
            q_loss, ae_loss = model_losses(q_network, q_target, double_autoencoder, ae_target, dqn_buffer, q_opt, int_opt)
            q_loss_.append(q_loss)
            ae_loss_.append(ae_loss)
        else:
            action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        dqn_buffer.add_sample([obs, action, reward, next_obs, done])
        ep_rew += reward
        obs = next_obs
    logger.add_scalar('ep_rew', ep_rew)
    logger.add_scalar('q_loss', np.mean(q_loss_))
    logger.add_scalar('autoencoder_loss', np.mean(ae_loss_))
    logger.write(episode)
    ep_rews.append(ep_rew)
    q_losses.append(np.mean(q_loss_))
    ae_losses.append(np.mean(ae_loss_))
    if ep_rew_ema is None:
        ep_rew_ema = ep_rew
    else:
        ep_rew_ema = smoothing * ep_rew_ema + (1 - smoothing) * ep_rew
    smoothed_rewards.append(ep_rew_ema)
    eps = max(eps_min, eps * eps_decay)
    
plt.plot(smoothed_rewards)
plt.savefig("smoothed_ep_rew.png")
np.save("smoothed_rewards.npy", np.asarray(smoothed_rewards))
plt.plot(ep_rews)
plt.savefig("ep_rew.png")
plt.plot(q_losses)
plt.savefig("q_losses.png")
plt.plot(ae_losses)
plt.savefig("ae_losses.png")
