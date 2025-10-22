import numpy as np

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
    
    return np.concat(observations), actions, rewards, dones