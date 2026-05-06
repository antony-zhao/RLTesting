"""
Shared configuration for Atari and Classic Control experiments.
Usage: import config; cfg = config.get_config('boxing')
       import config; cfg = config.get_config('cartpole')
"""
import argparse
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation
import numpy as np
import ale_py

gym.register_envs(ale_py)

# ---- Environment registry ---- #
ENVS = {
    # Atari
    'boxing':       {'env_id': 'ALE/Boxing-v5',    'num_actions': 18, 'env_type': 'atari',
                     'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 16,
                     'weighted_loss': True},
    'bowling':      {'env_id': 'ALE/Bowling-v5',   'num_actions': 6,  'env_type': 'atari',
                     'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 10,
                     'weighted_loss': True},
    'pong':         {'env_id': 'ALE/Pong-v5',      'num_actions': 6,  'env_type': 'atari',
                     'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 16,
                     'weighted_loss': True},
    'spaceinvaders': {'env_id': 'ALE/SpaceInvaders-v5', 'num_actions': 6, 'env_type': 'atari',
                  'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 16,
                  'weighted_loss': True},
    'assault':      {'env_id': 'ALE/Assault-v5',   'num_actions': 7,  'env_type': 'atari',
                 'stage1_epochs': 80, 'stage2_epochs': 80, 'intrinsic_dim_override': 12,
                 'weighted_loss': True},

    # Classic control
    'cartpole':     {'env_id': 'CartPole-v1',      'num_actions': 2,  'env_type': 'classic',
                     'state_dim': 4,  'gamma': 0.99, 'framestack': 2,
                     'latent_dim': 256, 
                     'num_tilings': 16, 'num_tiles_per_dim': 4,
                     'hash_size': 2 ** 16,
                     'alpha_multiplier': 0.1,
                     'sarsa_max_timesteps': 1_000_000, 'sarsa_max_episodes': 100000,
                     'epsilon_decay_episodes': 500,
                     'ppo_timesteps': 1_000_000},
    'acrobot':      {'env_id': 'Acrobot-v1',       'num_actions': 3,  'env_type': 'classic',
                     'state_dim': 6,  'gamma': 1.0, 'framestack': 2,
                     'latent_dim': 256, 
                     'num_tilings': 32, 'num_tiles_per_dim': 4,
                     'hash_size': 2 ** 18,
                     'alpha_multiplier': 0.1,
                     'sarsa_max_timesteps': 1_000_000, 'sarsa_max_episodes': 100000,
                     'epsilon_decay_episodes': 500,
                     'ppo_timesteps': 1_000_000},
    'mountaincar':  {'env_id': 'MountainCar-v0',   'num_actions': 3,  'env_type': 'classic',
                 'state_dim': 2,  'gamma': 1.0, 'framestack': 2,
                 'latent_dim': 256, 
                 'num_tilings': 8, 'num_tiles_per_dim': 8,
                 'hash_size': 2 ** 16,
                 'alpha_multiplier': 0.3,
                 'lam': 0.97, 'data_steps': 200_000,
                 'sarsa_max_timesteps': 1_000_000, 'sarsa_max_episodes': 100000,
                 'epsilon_decay_episodes': 2000,
                 'ppo_timesteps': 1_000_000},
}

# ---- Shared defaults ---- #
DEFAULTS = {
    # Environment
    'env_type': 'atari',
    'framestack': 4,
    'image_channels': 1,       # 3 for RGB, 1 for grayscale
    'obs_shape': 64,
    'filter_base': 16,
    'gamma': 0.99,

    # Data collection
    'data_steps': 200_000,
    'batch_size': 256,

    # Autoencoder (stage 1 + 2)
    'latent_dim': None,         # None = no linear projection, use raw conv features
    'hidden_dim': 128,          # MLP hidden dim for intrinsic encoder/decoder
    'encoder_type': 'conv',     # 'conv' (original) or 'impala' (residual blocks)
    'id_multiplier': 1.5,      # intrinsic_dim = round(MLE * multiplier)
    'intrinsic_dim_override': None,  # set manually to bypass MLE (e.g. 16)
    'stage1_epochs': 50,
    'stage1_steps_per_epoch': 1500,
    'stage2_epochs': 50,
    'stage2_steps_per_epoch': 1500,
    'ae_lr': 3e-4,
    'ae_scheduler_gamma': 0.97,
    'weighted_loss': False,

    # SARSA(λ) + tile coding
    'num_tiles_per_dim': 2,
    'num_tilings': 64,
    'hash_size': 2 ** 22,
    'alpha_multiplier': 0.2,   # alpha = alpha_multiplier / num_tilings
    'lam': 0.9,
    'sarsa_max_timesteps': 5_000_000,
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
    'finetune_once_at': 1000,           # episode to fine-tune at
    'finetune_once_steps': 5000,        # many gradient steps in one shot
    'finetune_once_buffer_size': 50,    # recent episodes to collect for fine-tuning
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


def make_env(cfg, render_mode='rgb_array', mode='default'):
    """Create a wrapped environment.

    Args:
        cfg: config dict
        render_mode: 'rgb_array' or 'human'
        mode: 'default' — raw state for classic, pixel obs for atari
              'pixels'  — pixel obs for both (classic control uses render())
              'raw'     — raw state (classic control only)
    """
    env_type = cfg.get('env_type', 'atari')

    if env_type == 'classic':
        if mode == 'pixels':
            return _make_classic_pixel_env(cfg, render_mode)
        else:
            # Raw state — no wrappers needed
            return gym.make(cfg['env_id'], render_mode=render_mode)
    else:
        return _make_atari_env(cfg, render_mode)


def _make_atari_env(cfg, render_mode):
    """Create a wrapped Atari environment with pixel observations."""
    img_channels = cfg.get('image_channels', 1)
    fs = cfg['framestack']
    obs = cfg['obs_shape']
    obs_type = "grayscale" if img_channels == 1 else "rgb"
    env = gym.make(cfg['env_id'], render_mode=render_mode, obs_type=obs_type)
    env = ResizeObservation(env, (obs, obs))
    if img_channels == 1:
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


def _make_classic_pixel_env(cfg, render_mode):
    """Create a classic control environment with pixel observations."""
    from rltesting.fvd_experiments.wrapper import BasicEnvironmentRGB
    obs = cfg['obs_shape']
    fs = cfg.get('framestack', 2)
    img_channels = cfg.get('image_channels', 1)

    env = gym.make(cfg['env_id'], render_mode='rgb_array')
    env = BasicEnvironmentRGB(env)
    env = ResizeObservation(env, (obs, obs))
    if img_channels == 1:
        env = TransformObservation(
            env, lambda o: np.dot(o[..., :3], [0.299, 0.587, 0.114])[..., None].astype(np.uint8),
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


def env_dir(cfg):
    """Return and create a per-environment output directory."""
    import os
    d = cfg['env_name']
    os.makedirs(d, exist_ok=True)
    return d


def env_path(cfg, filename):
    """Return a path inside the per-environment output directory."""
    import os
    return os.path.join(env_dir(cfg), filename)