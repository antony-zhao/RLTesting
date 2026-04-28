"""
PPO baselines on Atari using Stable Baselines 3.
Supports pixel observations (CnnPolicy) and RAM (MlpPolicy).

Usage: python train_ppo.py --env boxing --mode pixels
       python train_ppo.py --env boxing --mode ram
"""
import argparse
import json
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import ale_py
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from config import get_config, add_env_arg, ENVS

gym.register_envs(ale_py)


class RewardLogger(BaseCallback):
    """Log per-episode returns and timesteps during training."""
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


def make_pixel_env(env_id):
    """Create a single pixel-observation Atari environment."""
    def _init():
        env = gym.make(env_id, render_mode='rgb_array')
        env = Monitor(env)
        env = AtariWrapper(env)
        return env
    return _init


def make_ram_env(env_id):
    """Create a single RAM-observation Atari environment."""
    def _init():
        env = gym.make(env_id, obs_type='ram', render_mode='rgb_array')
        env = Monitor(env)
        return env
    return _init


def train_pixels(cfg):
    """Train PPO with CnnPolicy and separate actor/critic conv encoders."""
    print(f"\n=== PPO (Pixels) — {cfg['env_name'].title()} ===")

    n_envs = cfg['ppo_n_envs']
    env = SubprocVecEnv([make_pixel_env(cfg['env_id']) for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    model = PPO(
        'CnnPolicy', env,
        n_steps=cfg['ppo_n_steps'],
        batch_size=cfg['ppo_batch_size'],
        n_epochs=cfg['ppo_n_epochs'],
        learning_rate=lambda f: f * cfg['ppo_lr'],
        gamma=cfg['gamma'],
        gae_lambda=cfg['ppo_gae_lambda'],
        clip_range=lambda f: f * cfg['ppo_clip'],
        ent_coef=cfg['ppo_ent_coef'],
        vf_coef=cfg['ppo_vf_coef'],
        max_grad_norm=cfg['ppo_max_grad_norm'],
        policy_kwargs=dict(share_features_extractor=False),
        verbose=1,
    )

    logger = RewardLogger()
    model.learn(total_timesteps=cfg['ppo_timesteps'], callback=logger)
    env.close()
    return model, logger, 'pixels'


def train_ram(cfg):
    """Train PPO with MlpPolicy on RAM observations."""
    print(f"\n=== PPO (RAM) — {cfg['env_name'].title()} ===")

    n_envs = cfg['ppo_n_envs']
    env = SubprocVecEnv([make_ram_env(cfg['env_id']) for _ in range(n_envs)])

    model = PPO(
        'MlpPolicy', env,
        n_steps=cfg['ppo_n_steps'],
        batch_size=cfg['ppo_batch_size'],
        n_epochs=cfg['ppo_n_epochs'],
        learning_rate=lambda f: f * cfg['ppo_lr'],
        gamma=cfg['gamma'],
        gae_lambda=cfg['ppo_gae_lambda'],
        clip_range=lambda f: f * cfg['ppo_clip'],
        ent_coef=cfg['ppo_ent_coef'],
        vf_coef=cfg['ppo_vf_coef'],
        max_grad_norm=cfg['ppo_max_grad_norm'],
        verbose=1,
    )

    logger = RewardLogger()
    model.learn(total_timesteps=cfg['ppo_timesteps'], callback=logger)
    env.close()
    return model, logger, 'ram'


def evaluate(model, cfg, mode, n_episodes=100):
    """Evaluate a trained model."""
    print(f"\nEvaluating for {n_episodes} episodes...")

    if mode == 'pixels':
        eval_env = SubprocVecEnv([make_pixel_env(cfg['env_id'])])
        eval_env = VecFrameStack(eval_env, n_stack=4)
        eval_env = VecTransposeImage(eval_env)
    else:
        eval_env = SubprocVecEnv([make_ram_env(cfg['env_id'])])

    eval_returns = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        total = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = eval_env.step(action)
            total += r[0]
            if done[0]:
                break
        eval_returns.append(total)

    eval_env.close()
    print(f"  Mean: {np.mean(eval_returns):.1f} | Std: {np.std(eval_returns):.1f}")
    return eval_returns


def save_results(logger, eval_returns, cfg, mode):
    """Save training curves, evaluation results, and plot."""
    # Plot
    window = 100
    if len(logger.episode_returns) > window:
        smoothed = np.convolve(logger.episode_returns,
                               np.ones(window) / window, mode='valid')
        smoothed_steps = logger.episode_timesteps[window - 1:]
        plt.figure(figsize=(10, 6))
        plt.plot(logger.episode_timesteps, logger.episode_returns, alpha=0.3, label='Raw')
        plt.plot(smoothed_steps, smoothed, label=f'{window}-ep avg')
        plt.xlabel('Timesteps')
        plt.ylabel('Return')
        plt.title(f"PPO ({mode}) — {cfg['env_name'].title()}")
        plt.legend()
        plt.savefig(f"ppo_{mode}_{cfg['env_name']}.png")
        plt.close()

    # JSON
    results = {
        'env': cfg['env_name'],
        'mode': mode,
        'eval_mean': float(np.mean(eval_returns)),
        'eval_std': float(np.std(eval_returns)),
        'eval_returns': [float(r) for r in eval_returns],
        'training_returns': [float(r) for r in logger.episode_returns],
        'training_timesteps': [int(t) for t in logger.episode_timesteps],
        'total_timesteps': cfg['ppo_timesteps'],
        'last_100_avg': float(np.mean(logger.episode_returns[-100:])),
    }
    save_path = f"ppo_{mode}_{cfg['env_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPO baseline for Atari')
    add_env_arg(parser)
    parser.add_argument('--mode', type=str, required=True, choices=['pixels', 'ram'],
                        help='Observation type')
    parser.add_argument('--timesteps', type=int, default=None, help='Override total timesteps')
    args = parser.parse_args()

    cfg = get_config(args.env)
    if args.timesteps:
        cfg['ppo_timesteps'] = args.timesteps

    if args.mode == 'pixels':
        model, logger, mode = train_pixels(cfg)
    else:
        model, logger, mode = train_ram(cfg)

    eval_returns = evaluate(model, cfg, mode)
    save_results(logger, eval_returns, cfg, mode)
    print("\nDone.")


if __name__ == '__main__':
    main()
