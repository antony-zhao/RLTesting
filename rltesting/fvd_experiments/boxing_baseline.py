"""
PPO baseline on Boxing using Stable Baselines 3.
Run: python ppo_boxing_baseline.py
"""
import gymnasium as gym
import numpy as np
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import json
from datetime import datetime

gym.register_envs(ale_py)

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_returns = []
        self.episode_timesteps = []
    
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_returns.append(info['episode']['r'])
                self.episode_timesteps.append(self.num_timesteps)
        return True

def make_env():
    def _init():
        env = gym.make('ALE/Boxing-v5', render_mode='rgb_array')
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    # 8 parallel envs
    n_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    total_timesteps = 5_000_000

    model = PPO('CnnPolicy', env, verbose=1,
                n_steps=128,
                batch_size=256,
                n_epochs=4,
                learning_rate=2.5e-4,
                gamma=0.99,
                gae_lambda=0.97,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5)

    logger = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # Evaluate
    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    eval_returns = []
    obs = eval_env.reset()
    for _ in range(100):
        obs = eval_env.reset()
        total = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = eval_env.step(action)
            total += r[0]
            if done[0]:
                break
        eval_returns.append(total)

    print(f"\nPPO Boxing | Mean: {np.mean(eval_returns):.1f} | Std: {np.std(eval_returns):.1f}")
    print(f"Training episodes logged: {len(logger.episode_returns)}")
    if logger.episode_returns:
        print(f"Last 100 training avg: {np.mean(logger.episode_returns[-100:]):.1f}")

    # Save results
    results = {
        'eval_mean': float(np.mean(eval_returns)),
        'eval_std': float(np.std(eval_returns)),
        'eval_returns': [float(r) for r in eval_returns],
        'training_returns': [float(r) for r in logger.episode_returns],
        'training_timesteps': [int(t) for t in logger.episode_timesteps],
        'total_timesteps': total_timesteps
    }

    save_path = f"ppo_boxing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")