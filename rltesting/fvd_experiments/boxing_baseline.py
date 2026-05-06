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
from stable_baselines3.common.utils import get_linear_fn
import json
from datetime import datetime
from matplotlib import pyplot as plt

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
        env = gym.make('ALE/Boxing-v5', render_mode='rgb_array', repeat_action_probability=0.0)
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

    total_timesteps = 10_000_000

    policy_kwargs = dict(
        share_features_extractor=False
    )

    model = PPO('CnnPolicy', env, verbose=1,
                n_steps=128,
                batch_size=256,
                n_epochs=3,                    # was 4
                learning_rate=lambda f: f * 2.5e-4,  # linear anneal
                gamma=0.99,
                gae_lambda=0.95,               # was 0.97
                clip_range=lambda f: f * 0.1,  # linear anneal from 0.1
                ent_coef=0.01,
                vf_coef=1.0,                   # was 0.5
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs
                )

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
    
    # Plot training curve
    window = 100
    if len(logger.episode_returns) > window:
        smoothed = np.convolve(logger.episode_returns, np.ones(window)/window, mode='valid')
        smoothed_steps = logger.episode_timesteps[window-1:]
        plt.figure(figsize=(10, 6))
        plt.plot(logger.episode_timesteps, logger.episode_returns, alpha=0.3, label='Raw')
        plt.plot(smoothed_steps, smoothed, label=f'{window}-ep avg')
        plt.xlabel('Timesteps'); plt.ylabel('Return')
        plt.title('PPO (SB3) - Boxing @ 5M steps')
        plt.legend()
        plt.savefig('ppo_boxing.png')
        plt.close()
        print("Saved plot to ppo_boxing.png")
    
    save_path = f"ppo_boxing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")