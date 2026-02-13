import cv2
import numpy as np
from gymnasium import Wrapper
import gymnasium as gym
import random

class BasicEnvironmentRGB(Wrapper):
    def __init__(self, env, obs_shape=None):
        super().__init__(env)
        # make sure the render_mode is rgb_array
        self.env = env
        self.obs_shape = obs_shape
        if obs_shape is not None:
            self.observation_space = gym.spaces.Box(0, 255, (obs_shape) + (3,), dtype=np.uint8)
        else:
            self.env.reset()
            frame = env.render()
            self.observation_space = gym.spaces.Box(0, 255, frame.shape, dtype=np.uint8)
        
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
        if self.obs_shape is None:
            return obs
        return cv2.resize(obs, dsize=self.obs_shape, interpolation=cv2.INTER_AREA)

class DomainRandomization(Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.env = env
        # self.env.unwrapped.SCREEN_DIM = 800
        self.args = args
    
    def domain_randomization(self):
        unwrapped = self.env.unwrapped
        for k, v in self.args.items():
            setattr(unwrapped, k, v())
    
    def reset(self, **kwargs):
        self.domain_randomization()
        return self.env.reset(**kwargs)


class RandomColorReplace(gym.Wrapper):
    def __init__(self, env, threshold=30):
        super().__init__(env)
        self.colors = [
            (255, 0, 127),   # Electric Rose (High contrast to black/white)
            (0, 255, 127),   # Spring Green
            (127, 0, 255),   # Vivid Violet
            (255, 127, 0),   # Bright Orange
            (0, 127, 255),   # Azure Blue
            (255, 255, 0),   # Bright Yellow
            (0, 255, 255),   # Cyan
            (255, 20, 147),  # Deep Pink
            (50, 205, 50),   # Lime Green
            (138, 43, 226)   # Blue Violet
        ]
        self.threshold = threshold
        self.target_color = None
        self.replacement_color = None

    def _apply_fuzzy_swap(self, frame):
        if self.target_color is None:
            return frame
        
        # Calculate Euclidean distance for every pixel
        # (H, W, 3) -> distance (H, W)
        dist = np.linalg.norm(frame.astype(np.float32) - self.target_color, axis=-1)
        
        # Create a mask for pixels within the threshold
        mask = dist < self.threshold
        
        new_frame = frame.copy()
        new_frame[mask] = self.replacement_color
        return new_frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Get a sample frame to pick the target
        frame = self.env.render()
        
        # In Acrobot, the 'main' colors are usually extremes. 
        # Let's pick a random pixel's color as a target, 
        # or just target the Black arms (0,0,0) specifically.
        unique_colors = np.unique(frame.reshape(-1, 3), axis=0)
        self.target_color = unique_colors[np.random.choice(len(unique_colors))]
        
        self.replacement_color = np.array(random.choice(self.colors), dtype=np.uint8)
        obs = self._apply_fuzzy_swap(obs)
        
        return obs, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        return self._apply_fuzzy_swap(frame)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Note: We don't modify 'obs' here because Acrobot obs are vectors, 
        # but we return the modified info if you're using it for logging.
        obs = self._apply_fuzzy_swap(obs)
        return obs, reward, terminated, truncated, info
        