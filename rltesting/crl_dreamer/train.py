from models import DreamerCRL
from rltesting.torch_rl.contrastive_rl.minigrid_wrappers import make_minigrid
from rltesting.utils.logger import Logger
from rltesting.torch_rl.utils import to_numpy
import torch
from torch import nn
import numpy as np
import gymnasium as gym
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--env_id", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--obs_type", default="vector", choices=["image", "vector"])
    parser.add_argument("--num_envs", default=32, type=int)
    parser.add_argument("--num_eval_envs", default=16, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--num_steps", default=3_000_000, type=int)
    parser.add_argument("--initial_data", default=10_000, type=int)

    # Encoder (image)
    parser.add_argument("--num_channels", default=3, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--kernel_size", default=4, type=int)
    parser.add_argument("--filter_base", default=96, type=int)
    parser.add_argument("--num_convs", default=4, type=int)

    # Vector obs
    parser.add_argument("--obs_dim", default=None, type=int)

    # World model
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--hidden_state_size", default=256, type=int)
    parser.add_argument("--num_hiddens_world_model", default=0, type=int)
    parser.add_argument("--num_hiddens_actor_critic", default=2, type=int)
    parser.add_argument("--action_type", default="discrete", choices=["discrete", "continuous"])
    parser.add_argument("--action_dim", default=6, type=int)
    parser.add_argument("--num_categoricals", default=16, type=int)
    parser.add_argument("--num_codes", default=16, type=int)
    parser.add_argument("--latent_unimix", default=0.01, type=float)
    parser.add_argument("--use_block_linear", default=True, type=bool)
    parser.add_argument("--act", default="silu", choices=["silu", "gelu", "relu"])

    # World model losses
    parser.add_argument("--prediction_loss_coef", default=1.0, type=float)
    parser.add_argument("--dynamics_loss_coef", default=0.5, type=float)
    parser.add_argument("--representation_loss_coef", default=0.1, type=float)
    parser.add_argument("--free_nats", default=1.0, type=float)

    # Reward bins
    parser.add_argument("--bin_low", default=-20, type=int)
    parser.add_argument("--bin_high", default=20, type=int)
    parser.add_argument("--num_bins", default=255, type=int)

    # Training
    parser.add_argument("--sample_batch_size", default=64, type=int)
    parser.add_argument("--crl_batch_size", default=256, type=int)
    parser.add_argument("--sample_seq_len", default=64, type=int)
    parser.add_argument("--rollout_length", default=16, type=int)
    parser.add_argument("--train_steps_per_update", default=1, type=int)
    parser.add_argument("--crl_steps_per_train", default=16, type=int)

    # Dreamer actor-critic
    parser.add_argument("--critic_tau", default=0.02, type=float)
    parser.add_argument("--entropy_coef", default=3e-4, type=float)
    parser.add_argument("--actor_unimix", default=0.01, type=float)
    parser.add_argument("--return_range_tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.997, type=float)
    parser.add_argument("--lambda_", default=0.95, type=float)
    parser.add_argument("--percentiles", default=0.05, type=float)

    # Contrastive RL
    parser.add_argument("--repr_dim", default=64, type=int)
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--penalty", default=0.1, type=float)
    parser.add_argument("--use_alpha", default=True, type=bool)
    parser.add_argument("--target_entropy_scale", default=0.5, type=float)

    # Optimizers
    parser.add_argument("--wm_lr", default=8e-5, type=float)
    parser.add_argument("--actor_lr", default=4e-5, type=float)
    parser.add_argument("--critic_lr", default=4e-5, type=float)
    parser.add_argument("--contrastive_lr", default=3e-4, type=float)
    parser.add_argument("--alpha_lr", default=3e-4, type=float)

    # Logging
    parser.add_argument("--log_interval", default=None, type=int)
    parser.add_argument("--eval_interval", default=None, type=int)
    parser.add_argument("--log_dir", default="logs/dreamer_crl")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    config = parser.parse_args()

    # Derived config values
    config.latent_size = config.num_categoricals * config.num_codes
    config.state_size = config.latent_size + config.hidden_state_size
    config.contrastive_state_size = config.hidden_dim + config.hidden_state_size

    if config.log_interval is None:
        config.log_interval = config.num_envs * config.episode_length
    if config.eval_interval is None:
        config.eval_interval = 4 * config.log_interval

    if config.act == "silu":
        config.act = nn.SiLU
    elif config.act == "gelu":
        config.act = nn.GELU
    elif config.act == "relu":
        config.act = nn.ReLU
    
    return config


def evaluate(agent, eval_env, config, num_episodes=32, max_steps=500):
    (obs, goal), _ = eval_env.reset()
    n = eval_env.num_envs
    successes = []
    episodes_done = 0
    steps = 0

    while episodes_done < num_episodes and steps < max_steps:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=config.device)
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=config.device)
            action = agent.eval_action(obs_t, goal_t, reset=(steps == 0))

        (obs, goal), rewards, terminations, truncations, infos = eval_env.step(action)
        steps += 1

        dones = terminations | truncations
        if dones.any():
            for i in np.where(dones)[0]:
                successes.append(bool(terminations[i]))
            episodes_done += dones.sum()

            # Reset hidden for done envs
            done_t = torch.as_tensor(dones, dtype=torch.float32, device=config.device).unsqueeze(1)
            agent.eval_hidden = (
                (1 - done_t) * agent.eval_hidden +
                done_t * agent.world_model._get_hidden(n)
            ).detach()

    return np.mean(successes) if successes else 0.0

if __name__ == "__main__":
    config = parse_args()

    # Environment setup
    env = make_minigrid(env_id=config.env_id, num_envs=config.num_envs, max_steps=config.episode_length)
    eval_env = make_minigrid(env_id=config.env_id, num_envs=config.num_eval_envs, max_steps=config.episode_length)

    # Infer action space
    if isinstance(env.single_action_space, gym.spaces.Discrete):
        config.action_type = "discrete"
        config.action_dim = int(env.single_action_space.n)
    else:
        config.action_type = "continuous"
        config.action_dim = env.action_space.shape[1]

    # Infer obs dim for vector observations
    if config.obs_type == "vector":
        config.obs_dim = env.single_observation_space.shape[0] // 2

    agent = DreamerCRL(config, obs_to_goal=env.obs_to_goal)
    logger = Logger(config.log_dir)

    (obs, goal), _ = env.reset()
    total_steps = 0
    recent_successes = []

    print(f"Starting training: {config.num_steps} steps, {config.num_envs} envs")
    print(f"  State size: {config.state_size} (latent {config.latent_size} + hidden {config.hidden_state_size})")
    print(f"  Action: {config.action_type}, dim={config.action_dim}")
    print(f"  Device: {config.device}")

    while total_steps < config.num_steps:
        # --- Collect experience ---
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=config.device)
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=config.device)
            action, latent = agent.choose_action(obs_t, goal_t)
            
            action_for_env = action.cpu().numpy()

        (next_obs, next_goal), rewards, terminations, truncations, infos = env.step(action_for_env)
        dones = terminations | truncations

        # Store in buffer and update hidden state
        agent.process_sample(obs, latent, action, rewards, dones)

        # Track successes
        if dones.any():
            for i in np.where(dones)[0]:
                recent_successes.append(bool(terminations[i]))

        obs = next_obs
        goal = next_goal
        total_steps += config.num_envs

        # --- Train ---
        if total_steps > config.initial_data:
            for _ in range(config.train_steps_per_update):
                loss_wm, loss_critic, loss_actor, metrics = agent.train()

            logger.add_scalar('wm_loss', loss_wm)
            logger.add_scalar('critic_loss', loss_critic)
            logger.add_scalar('actor_loss', loss_actor)
            logger.add_metrics(metrics)

        # --- Log ---
        if total_steps % config.log_interval == 0:
            if recent_successes:
                success_rate = np.mean(recent_successes)
                print(f"Steps: {total_steps:>8d} | Success: {success_rate:.2f} ({len(recent_successes)} eps)")
                logger.add_scalar('success_rate', success_rate)
                recent_successes = []

        # --- Eval ---
        if total_steps % config.eval_interval == 0 and total_steps > config.initial_data:
            eval_success = evaluate(agent, eval_env, config)
            print(f"Steps: {total_steps:>8d} | Eval: {eval_success:.2f}")
            logger.add_scalar('eval_success_rate', eval_success)

        logger.write(total_steps)

    print("Training complete.")