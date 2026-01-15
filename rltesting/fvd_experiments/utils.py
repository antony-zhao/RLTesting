from wrapper import BasicEnvironmentRGB, DomainRandomization
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation, GrayscaleObservation
from rltesting.torch_rl.utils import to_numpy
import numpy as np
import torch
from torch.nn import functional as F
import ale_py

def fill_buffer(env, buffer, steps):
    obs, _ = env.reset()
    done = None
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        buffer.add_sample([obs, action, reward, next_obs, done])
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

def make_env(env_type, env_id, framestack, image_size, grayscale):
    gym.register_envs(ale_py)

    num_channels = 3 if not grayscale else 1
    if env_type == "classic_control":
        env = gym.make(env_id, render_mode="rgb_array")
        env = DomainRandomization(env, {"LINK_LENGTH_1": (0.5, 2.0), "LINK_LENGTH_2": (0.5, 2.0)})
        env = BasicEnvironmentRGB(env, (image_size, image_size))
    elif env_type == "atari":
        env = gym.make(f"ALE/{env_id}-v5", render_mode="rgb_array")
        env = ResizeObservation(env, (image_size, image_size))
    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, stack_size=framestack)
    
    env = TransformObservation(
        env, 
        lambda obs: np.transpose(obs, (1, 2, 0, 3)), 
        observation_space=gym.spaces.Box(0, 255, (framestack, num_channels, image_size, image_size), dtype=np.uint8)
        )
    env = TransformObservation(
        env, 
        lambda obs: np.reshape(obs, (image_size, image_size, num_channels * framestack)), 
        observation_space=gym.spaces.Box(0, 255, (image_size, image_size, num_channels * framestack), dtype=np.uint8)
        )
    return env 

def dqn_loss(q_network, q_target, buffer, q_opt, batch_size=512, gamma=0.99):
    q_target.update()
    obs, action, reward, next_obs, done = buffer.sample(batch_size)
    obs = torch.tensor(obs).float().cuda()
    action = torch.tensor(action).long().cuda()
    reward = torch.tensor(reward).float().cuda()
    next_obs = torch.tensor(next_obs).float().cuda()
    done = torch.tensor(done).float().cuda()
    q_target = reward + gamma * q_target(next_obs).detach().gather(1, torch.argmax(q_network(next_obs).detach(), 1, keepdim=True)).squeeze() * (1 - done)
    q_values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze()
    q_loss = F.mse_loss(q_target, q_values)
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()
    return q_loss
