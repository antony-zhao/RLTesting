import numpy as np
import torch
import yaml
from types import SimpleNamespace

to_numpy = lambda x: x.detach().cpu().numpy()

class EMA:
    pass

def random_sample_single_env(env, num_steps=1000):
    observations = []
    actions = []
    rewards = []
    dones = []
    obs, _ = env.reset()
    done = None
    for _ in range(num_steps):
        observations.append(obs)
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        if done:
            obs, _ = env.reset()
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
    
    return np.stack(observations), np.stack(actions), np.stack(rewards), np.stack(dones)

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg

def flatten(d):
    flat = {}
    
    def thunk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                thunk(v)
            else:
                flat[k] = v
    thunk(d)
    
    return flat