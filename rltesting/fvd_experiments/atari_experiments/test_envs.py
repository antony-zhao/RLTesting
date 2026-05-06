"""
Quick smoke test for all environments and modes.
Run: python test_envs.py
"""
from config import get_config, make_env
import numpy as np

print("=== Testing Environment Creation ===\n")

# Test all envs in default mode
for env_name in ['cartpole', 'acrobot', 'mountaincar', 'boxing', 'pong', 'bowling']:
    cfg = get_config(env_name)
    env_type = cfg.get('env_type', 'atari')
    img_ch = cfg.get('image_channels', 3)

    # Default mode (raw for classic, pixel for atari)
    try:
        env = make_env(cfg)
        obs, _ = env.reset()
        obs2, r, t, tr, info = env.step(env.action_space.sample())
        obs_arr = np.array(obs)
        obs2_arr = np.array(obs2)
        print(f"  {env_name:12s} default  | obs: {obs_arr.shape} dtype={obs_arr.dtype} | "
              f"actions: {env.action_space.n} | r={r:.1f} | ch={img_ch}")
        assert obs_arr.shape == obs2_arr.shape, f"Shape mismatch: {obs_arr.shape} vs {obs2_arr.shape}"
        env.close()
    except Exception as e:
        print(f"  {env_name:12s} default  | FAILED: {e}")

    # Pixel mode for classic control
    if env_type == 'classic':
        try:
            env = make_env(cfg, mode='pixels')
            obs, _ = env.reset()
            obs2, r, t, tr, info = env.step(env.action_space.sample())
            obs_arr = np.array(obs)
            obs2_arr = np.array(obs2)
            fs = cfg.get('framestack', 2)
            expected_shape = (cfg['obs_shape'], cfg['obs_shape'], img_ch * fs)
            print(f"  {env_name:12s} pixels   | obs: {obs_arr.shape} dtype={obs_arr.dtype} | "
                  f"expected: {expected_shape}")
            assert obs_arr.shape == expected_shape, f"Shape mismatch: {obs_arr.shape} vs {expected_shape}"
            assert obs_arr.shape == obs2_arr.shape, f"Step shape mismatch"
            env.close()
        except Exception as e:
            print(f"  {env_name:12s} pixels   | FAILED: {e}")

    # Verify atari pixel shape
    if env_type == 'atari':
        fs = cfg.get('framestack', 2)
        expected_shape = (cfg['obs_shape'], cfg['obs_shape'], img_ch * fs)
        obs_arr = np.array(obs)
        try:
            assert obs_arr.shape == expected_shape, f"Shape mismatch: {obs_arr.shape} vs {expected_shape}"
            print(f"  {env_name:12s}          | shape OK: {expected_shape}")
        except AssertionError as e:
            print(f"  {env_name:12s}          | SHAPE ERROR: {e}")

print()

# Test transpose/encode compatibility (simulates what train_autoencoder does)
print("=== Testing Tensor Conversion (simulates encoder input) ===\n")
import torch

for env_name in ['cartpole', 'boxing']:
    cfg = get_config(env_name)
    env_type = cfg.get('env_type', 'atari')
    img_ch = cfg.get('image_channels', 3)
    fs = cfg.get('framestack', 2)

    if env_type == 'classic':
        env = make_env(cfg, mode='pixels')
    else:
        env = make_env(cfg)

    obs, _ = env.reset()
    obs_arr = np.array(obs)

    # Simulate what the training code does
    try:
        t = torch.tensor(obs_arr).unsqueeze(0).float().transpose(-3, -1) / 255
        print(f"  {env_name:12s} | np: {obs_arr.shape} -> torch: {t.shape} | "
              f"expected channels: {img_ch * fs}")
        assert t.shape[1] == img_ch * fs, f"Channel mismatch: {t.shape[1]} vs {img_ch * fs}"
    except Exception as e:
        print(f"  {env_name:12s} | FAILED: {e}")

    # Test batch
    try:
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        batch = np.stack([np.array(obs), np.array(obs2)])
        t = torch.tensor(batch).float().transpose(-3, -1) / 255
        print(f"  {env_name:12s} | batch: {batch.shape} -> torch: {t.shape}")
    except Exception as e:
        print(f"  {env_name:12s} | BATCH FAILED: {e}")

    env.close()

print("\nAll tests complete.")
