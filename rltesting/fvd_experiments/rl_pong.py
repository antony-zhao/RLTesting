from fvd_pretrain import load_or_create_model, pretrain, make_env, to_numpy
from matplotlib import pyplot as plt
from rltesting.torch_rl.utils import load_config, simple_process_config
from rltesting.torch_rl.buffers import ReplayBuffer
from rltesting.torch_rl.models import MLP, TargetNetwork
from rltesting.utils.logger import Logger
from fvd_models import Encoder, Decoder
import gymnasium as gym
import torch.nn.functional as F
import numpy as np
import torch

config_path = "rltesting/fvd_experiments/configs/atari/default.yaml"
config = load_config(config_path)
config = simple_process_config(config)

num_channels = 3 if not config.grayscale else 1

encoder = Encoder(config.framestack, config.latent_dim, config.image_size, num_channels).to(config.device)
decoder = Decoder(encoder.conv_dim, config.framestack, config.latent_dim, num_channels).to(config.device)
double_autoencoder = load_or_create_model(config, encoder, decoder)
    
num_envs = 16
train_every = 4
env = gym.vector.SyncVectorEnv([
    lambda: make_env(config.env_type, config.env_id, config.framestack, config.image_size, config.grayscale) for _ in range(num_envs)
])

num_actions = env.single_action_space.n
q_network = MLP(double_autoencoder.intrinsic_dim, num_actions, skip_connections=False).to(config.device)
int_opt = torch.optim.Adam(double_autoencoder.parameters(), 3e-4)
q_opt = torch.optim.Adam(q_network.parameters(), 3e-4)
dtypes = [np.uint8, np.float32, np.float32, np.uint8, np.float32]
buffer_shapes = [(config.image_size, config.image_size, num_channels * config.framestack), (), (), (config.image_size, config.image_size, num_channels * config.framestack), ()]
dqn_buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=1_000_000)
q_target = TargetNetwork(q_network, tau=1e-4).to(config.device)
ae_target = TargetNetwork(double_autoencoder, tau=3e-5).to(config.device)
eps = 1
eps_decay = 0.99
eps_min = 0.1
num_episodes = 10000
train_after = 10000
gamma = 0.99

num_iters = num_envs // train_every

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
    obs = (torch.as_tensor(obs).float().transpose(-3, -1) / 255).to(config.device)
    action = torch.as_tensor(action).long().to(config.device)
    reward = torch.as_tensor(reward).float().to(config.device)
    next_obs = (torch.as_tensor(next_obs).float().transpose(-3, -1) / 255).to(config.device)
    done = torch.as_tensor(done).float().to(config.device)
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
            obs_tensor = (torch.tensor(obs).float().transpose(-3, -1) / 255).to(config.device)
            if np.random.rand() < eps:
                action = np.random.randint(num_actions)
            else:
                intrinsic = double_autoencoder.double_encode(obs_tensor.unsqueeze(0))
                action = to_numpy(q_network(intrinsic).argmax())
            for _ in range(num_iters):
                q_loss, ae_loss = model_losses(q_network, q_target, double_autoencoder, ae_target, dqn_buffer, q_opt, int_opt)
                q_loss_.append(q_loss)
                ae_loss_.append(ae_loss)
        else:
            action = env.action_space.sample()
        next_obs, reward, term, trunc, info = env.step(action)
        for i in range(num_envs):
            dqn_buffer.add_sample([obs[i], action[i], reward[i], next_obs[i], term[i]])
        if term.any() or trunc.any():
            ep_rew = np.sum(info['episode']['r']) / np.sum(term + trunc)
            logger.add_scalar('ep_rew', ep_rew)
        obs = next_obs
    logger.add_scalar('q_loss', np.mean(q_loss_))
    logger.add_scalar('autoencoder_loss', np.mean(ae_loss_))
    logger.write(episode)
    # ep_rews.append(ep_rew)
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
