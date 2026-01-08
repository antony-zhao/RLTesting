from wrapper import BasicEnvironmentRGB, DomainRandomization
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation, GrayscaleObservation
from matplotlib import pyplot as plt
from rltesting.torch_rl.utils import random_sample_single_env
from rltesting.torch_rl.buffers import ReplayBuffer
from rltesting.torch_rl.models import MLP, TargetNetwork
from rltesting.torch_rl.utils import to_numpy
from fundamental_variable_discovery import Encoder, Decoder, DoubleAutoEncoder
import torch.nn.functional as F
import numpy as np
import pickle
import torch
import skdim

def pretrain(config):
    framestack = 2
    latent_dim = 512
    image_size = 64
    data_steps = 20000
    grayscale = True
    num_channels = 3 if not grayscale else 1
    buffer_size = 1_000_000
    train_steps = 1000
    vae = False
    env_type ="classic_control"
    env_id ="Pong-v5"
    exp_name = "Pong"
    if env_type == "classic_control":
        env = gym.make(env_id, render_mode="rgb_array")
        env = DomainRandomization(env, {"LINK_LENGTH_1": (0.5, 2.0), "LINK_LENGTH_2": (0.5, 2.0)})
        env = BasicEnvironmentRGB(env, (image_size, image_size))
    elif env_type == "atari":
        env = gym.make(f"ALE/{env_id}", render_mode="rgb_array")
        env = ResizeObservation(env, (image_size, image_size))
    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, stack_size=framestack)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0, 3)), 
                            observation_space=gym.spaces.Box(0, 255, (framestack, num_channels, image_size, image_size), dtype=np.uint8))
    env = TransformObservation(env, lambda obs: np.reshape(obs, (image_size, image_size, num_channels * framestack)), 
                            observation_space=gym.spaces.Box(0, 255, (image_size, image_size, num_channels * framestack), dtype=np.uint8))

    obs_shape = (image_size, image_size, num_channels * framestack)
    buffer_shapes = [obs_shape, (), (), obs_shape, ()]
    dtypes = [np.uint8, np.float32, np.float32, np.uint8, np.float32]
    buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=buffer_size)

    obs, _ = env.reset()
    done = None
    for _ in range(data_steps):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        buffer.add_sample(obs, action, reward, next_obs, done)
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
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
    
    torch.save(double_autoencoder, "models/{exp_name}_model.pt")