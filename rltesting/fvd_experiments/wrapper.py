import cv2
import numpy as np
from gymnasium import Wrapper
import gymnasium as gym

class BasicEnvironmentRGB(Wrapper):
    def __init__(self, env, obs_shape=(64, 64)):
        super().__init__(env)
        # make sure the render_mode is rgb_array
        self.env = env
        self.obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(0, 255, (obs_shape) + (3,))
        
    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        next_obs = self.env.render()
        next_obs = self.reshape_obs(next_obs)
        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        _, info = self.env.reset()
        obs = self.reshape_obs(self.env.render())
        return obs, info
    
    def close(self):
        return self.env.close()
    
    def reshape_obs(self, obs):
        return cv2.resize(obs, dsize=self.obs_shape)
    