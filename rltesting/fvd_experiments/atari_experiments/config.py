"""
Shared configuration for Atari experiments.
Usage: import config; cfg = config.get_config('boxing')
"""
import argparse
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation
import numpy as np
import ale_py

gym.register_envs(ale_py)

# ---- Environment registry ---- #
ENVS = {
    'boxing':  {'env_id': 'ALE/Boxing-v5',  'num_actions': 18, 'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 16},
    'bowling': {'env_id': 'ALE/Bowling-v5', 'num_actions': 6, 'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 10},
    'pong':    {'env_id': 'ALE/Pong-v5',    'num_actions': 6, 'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 16},
}

# ---- Shared defaults ---- #
DEFAULTS = {
    # Environment
    'framestack': 4,
    'frameskip': 4,             # action repeat (standard Atari is 4)
    'image_channels': 1,       # 3 for RGB, 1 for grayscale
    'obs_shape': 64,
    'filter_base': 16,
    'gamma': 0.99,

    # Data collection
    'data_steps': 200_000,
    'batch_size': 256,

    # Autoencoder (stage 1 + 2)
    'latent_dim': None,         # None = no linear projection, use raw conv features
    'hidden_dim': 256,          # MLP hidden dim for intrinsic encoder/decoder
    'encoder_type': 'conv',     # 'conv' (original) or 'impala' (residual blocks)
    'id_multiplier': 1.5,      # intrinsic_dim = round(MLE * multiplier)
    'intrinsic_dim_override': None,  # set manually to bypass MLE (e.g. 16)
    'stage1_epochs': 50,
    'stage1_steps_per_epoch': 1500,
    'stage2_epochs': 80,
    'stage2_steps_per_epoch': 1500,
    'ae_lr': 3e-4,
    'ae_scheduler_gamma': 0.97,

    # SARSA(λ) + tile coding
    'num_tiles_per_dim': 2,
    'num_tilings': 64,
    'hash_size': 2 ** 22,
    'alpha_multiplier': 0.2,   # alpha = alpha_multiplier / num_tilings
    'lam': 0.9,
    'sarsa_max_timesteps': 10_000_000,
    'sarsa_max_episodes': 10_000,
    'epsilon_decay_episodes': 1000,
    'epsilon_min': 0.05,

    # Fine-tuning (off by default)
    'finetune': False,
    'finetune_start': 800,
    'finetune_end': 1500,
    'finetune_every': 5,
    'finetune_steps': 50,
    'finetune_lr': 1e-5,
    
    # Fine-tune once (off by default)
    'finetune_once': False,
    'finetune_once_at': 1500,           # episode to fine-tune at
    'finetune_once_steps': 5000,        # many gradient steps in one shot
    'finetune_once_buffer_size': 500,    # recent episodes to collect for fine-tuning
    'finetune_once_reset_weights': True, # reset SARSA weights after (tiles shifted)

    # PPO
    'ppo_timesteps': 10_000_000,
    'ppo_n_envs': 8,
    'ppo_n_steps': 128,
    'ppo_batch_size': 256,
    'ppo_n_epochs': 4,
    'ppo_lr': 2.5e-4,         # linearly annealed to 0
    'ppo_clip': 0.1,          # linearly annealed to 0
    'ppo_gae_lambda': 0.95,
    'ppo_ent_coef': 0.01,
    'ppo_vf_coef': 0.5,
    'ppo_max_grad_norm': 0.5,
}


def get_config(env_name):
    """Return a config dict for the given environment."""
    env_name = env_name.lower()
    if env_name not in ENVS:
        raise ValueError(f"Unknown environment: {env_name}. Choose from {list(ENVS.keys())}")
    cfg = {**DEFAULTS, **ENVS[env_name], 'env_name': env_name}
    cfg['alpha'] = cfg['alpha_multiplier'] / cfg['num_tilings']
    return cfg


def make_env(cfg, render_mode='rgb_array'):
    """Create a wrapped Atari environment."""
    img_channels = cfg.get('image_channels', 3)
    fs = cfg['framestack']
    obs = cfg['obs_shape']
    obs_type = "grayscale" if img_channels == 1 else "rgb"
    env = gym.make(cfg['env_id'], render_mode=render_mode, frameskip=cfg['frameskip'], obs_type=obs_type)
    env = ResizeObservation(env, (obs, obs))
    if img_channels == 1:
        # ALE grayscale returns (H, W) — add channel dim to get (H, W, 1)
        env = TransformObservation(
            env, lambda o: o[..., None],
            observation_space=gym.spaces.Box(0, 255, (obs, obs, 1), dtype=np.uint8))
    env = FrameStackObservation(env, stack_size=fs)
    env = TransformObservation(
        env, lambda o: np.transpose(o, (1, 2, 0, 3)),
        observation_space=gym.spaces.Box(
            0, 255, (fs, img_channels, obs, obs), dtype=np.uint8))
    env = TransformObservation(
        env, lambda o: np.reshape(o, (obs, obs, img_channels * fs)),
        observation_space=gym.spaces.Box(
            0, 255, (obs, obs, img_channels * fs), dtype=np.uint8))
    return env


def add_env_arg(parser):
    """Add --env argument to an argparse parser."""
    parser.add_argument('--env', type=str, required=True, choices=list(ENVS.keys()),
                        help='Environment name')
    return parser


def to_numpy(x):
    return x.cpu().detach().numpy()