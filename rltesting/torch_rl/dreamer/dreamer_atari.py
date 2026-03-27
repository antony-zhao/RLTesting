import gymnasium as gym
from rltesting.utils.logger import Logger
from rltesting.torch_rl.dreamer.dreamer import *
import ale_py
from ale_py.vector_env import AtariVectorEnv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="just a temporary argparse while I debug the world model, will be moved elsewhere later")
    # specified for Atari specifically by default and using the 200M size model for Dreamer
    parser.add_argument("--obs_type", default="image", choices=["image", "vector", "multi"]) # need to handle multi for some robotics environments/experiments
    encoder_parser = parser.add_argument_group("encoder")
    vector_parser = parser.add_argument_group("vector")
    world_model_parser = parser.add_argument_group("world_model")
    encoder_parser.add_argument("--num_channels", default=3, type=int)
    encoder_parser.add_argument("--image_size", default=64, type=int) # images should be square
    encoder_parser.add_argument("--kernel_size", default=4, type=int) # best to keep it 4 or 6
    encoder_parser.add_argument("--filter_base", default=96, type=int) # the base number of filters, which is doubled for each convolutional layer
    encoder_parser.add_argument("--num_convs", default=4, type=int) # total number of convolutions, after which the dimension is size / 2^num_convs, and the final number of filters is filter_base * 2^(num_convs-1)
    vector_parser.add_argument("--obs_dim", default=None, type=int) # specify for vector, or both for vector and image stuff for multi
    world_model_parser.add_argument("--hidden_dim", default=1024, type=int) # hidden dims of MLPs
    world_model_parser.add_argument("--hidden_state_size", default=8192, type=int) # hidden state of GRU/RSSM
    world_model_parser.add_argument("--num_hiddens_world_model", default=0, type=int) # determines the depth for MLPs in the world model
    world_model_parser.add_argument("--num_hiddens_actor_critic", default=2, type=int)
    world_model_parser.add_argument("--action_type", default="discrete", choices=["discrete", "continuous"])
    world_model_parser.add_argument("--num_categoricals", default=32, type=int) # the number of rows in the latent
    world_model_parser.add_argument("--num_codes", default=32, type=int) # the actual dim that's softmaxed over in the latent
    world_model_parser.add_argument("--latent_unimix", default=0.01, type=float)
    world_model_parser.add_argument("--use_block_linear", default=True, type=bool)
    world_model_parser.add_argument("--act", default="silu", choices=["silu", "gelu", "relu"])
    world_model_parser.add_argument("--prediction_loss_coef", default=1, type=float)
    world_model_parser.add_argument("--dynamics_loss_coef", default=0.5, type=float)
    world_model_parser.add_argument("--representation_loss_coef", default=0.1, type=float)
    world_model_parser.add_argument("--free_nats", default=1, type=float)
    world_model_parser.add_argument("--bin_low", default=-20, type=int)
    world_model_parser.add_argument("--bin_high", default=20, type=int)
    world_model_parser.add_argument("--num_bins", default=255, type=int)
    parser.add_argument("--sample_batch_size", default=16, type=int)
    parser.add_argument("--sample_seq_len", default=64, type=int)
    parser.add_argument("--rollout_length", default=16, type=int)
    parser.add_argument("--critic_tau", default=0.02, type=float)
    parser.add_argument("--critic_imagination_loss_coef", default=1, type=float)
    parser.add_argument("--entropy_coef", default=3e-4, type=float)
    parser.add_argument("--actor_unimix", default=0.01, type=float)
    parser.add_argument("--return_range_tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.997, type=float)
    parser.add_argument("--lambda_", default=0.95, type=float)
    parser.add_argument("--percentiles", default=0.05, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wm_lr", default=8e-5, type=float)
    parser.add_argument("--reinforce_lr", default=4e-5, type=float)
    parser.add_argument("--num_envs", default=16, type=int)
    parser.add_argument("--replay_ratio", default=32, type=int) # batch_size * seq_len / timesteps_in_env = replay ratio, so default of 64 * 16 / num_envs * timesteps = 32
    parser.add_argument("--total_steps", default=50_000_000, type=int)
    parser.add_argument("--train_wm_after", default=10_000, type=int)
    parser.add_argument("--train_reinforce_after", default=100_000, type=int)
    parser.add_argument("--env_id", default="Assault")
    config = parser.parse_args()

    if config.act == "silu":
        config.act = nn.SiLU
    elif config.act == "gelu":
        config.act = nn.GELU
    elif config.act == "relu":
        config.act = nn.ReLU
    if config.obs_type == "image":
        config.image_dim = (config.num_channels, config.image_size, config.image_size)
        size = config.image_size // 2 ** (config.num_convs)
        config.output_dim = (config.filter_base * 2 ** (config.num_convs - 1), size, size)
    config.latent_size = config.num_categoricals * config.num_codes
    config.state_size = config.hidden_state_size + config.latent_size
    config.train_every = config.sample_batch_size * config.sample_seq_len / (config.num_envs * config.replay_ratio)
    if config.train_every > 1:
        config.train_every = int(config.train_every)
        config.num_iters = 1
    else:
        config.num_iters = int(1 / config.train_every)
        config.train_every = 1
    config.action_dim = gym.make(f"ALE/{config.env_id}-v5").action_space.n
    return config


def make_env(config, num_envs):
    # ALE vector env handles preprocessing natively in C++:
    # frame skip, resize, noop reset, fire reset
    # grayscale=False for RGB, stack_num=1 since RSSM handles temporal info
    env = AtariVectorEnv(
        game=config.env_id.lower(),
        num_envs=num_envs,
        frameskip=4,
        grayscale=False,
        stack_num=1,
        img_height=config.image_size,
        img_width=config.image_size,
        noop_max=30,
        use_fire_reset=True,
        reward_clipping=False,
        max_num_frames_per_episode=108000,  # 27000 steps * frameskip 4, standard for atari
        repeat_action_probability=0.25,
    )
    env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    return env

def process_obs(obs):
    # ALE vec env with grayscale=False, stack_num=1 outputs (N, 1, H, W, 3)
    # Dreamer expects (N, C, H, W)
    if obs.ndim == 5:
        obs = obs.squeeze(1)  # (N, H, W, 3)
    return obs.transpose(0, 3, 1, 2)  # (N, 3, H, W)

def make_eval_env(config):
    gym.register_envs(ale_py)
    env = gym.make(
        f"ALE/{config.env_id}-v5",
        render_mode="rgb_array",
        frameskip=4,
        repeat_action_probability=0.25,
    )
    env = gym.wrappers.ResizeObservation(env, (config.image_size, config.image_size))
    env = gym.wrappers.TimeLimit(env, 27000)
    return env

def eval(dreamer, eval_env, config):
    obs, _ = eval_env.reset()
    obs = obs.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    is_first = True
    total_reward = 0
    frames = []
    obs_traj = []
    while True:
        with torch.no_grad():
            action = dreamer.eval_action(torch.tensor(obs).float().to(config.device), reset=is_first)
        is_first = False
        action = action.item() if np.ndim(action) > 0 else action
        next_obs, reward, terminated, truncated, info = eval_env.step(action)
        frames.append(eval_env.render())  # full resolution rgb_array
        next_obs = next_obs.transpose(2, 0, 1)[np.newaxis]
        obs_traj.append(next_obs)
        total_reward += reward
        if terminated or truncated:
            break
        obs = next_obs
    return total_reward, obs_traj, frames

def imagine_rollout(dreamer, eval_env, config):
    obs, _ = eval_env.reset()
    obs = obs.transpose(2, 0, 1)[np.newaxis]
    obs = torch.tensor(obs).float().to(config.device)
    with torch.no_grad():
        eval_hidden = dreamer.world_model._get_hidden(1)
        latent_prob = dreamer.world_model.encoder(obs, eval_hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, eval_hidden)
        states = dreamer.imagine_rollout(state, 128)[0]
        imagined_rollout = dreamer.world_model.decoder.from_state(states) + 0.5
    return imagined_rollout.unsqueeze(0)

if __name__ == "__main__":
    config = parse_args()
    num_envs = config.num_envs
    env = make_env(config, num_envs)
    eval_env = make_eval_env(config)
    dreamer = DreamerV3(config)
    logger = Logger(f"logs/dreamer-v3/{config.env_id}")

    obs, _ = env.reset()
    obs = process_obs(obs)
    timestep = 0
    num_iterations = config.total_steps // config.num_envs
    for i in range(1, num_iterations + 1):
        timestep += config.num_envs
        if i % 100 == 0:
            print(i)
        with torch.no_grad():
            action, latent = dreamer.choose_action(torch.tensor(obs).float().to(config.device))
        next_obs, reward, terminated, truncated, infos = env.step(to_numpy(action))
        next_obs = process_obs(next_obs)
        done = terminated | truncated
        dreamer.process_sample(obs, latent, action, reward, done)
        # RecordEpisodeStatistics: infos["episode"] has r/l/t arrays, infos["_episode"] is boolean mask
        if "_episode" in infos and np.any(infos["_episode"]):
            finished = infos["_episode"]
            episode_rewards = infos["episode"]["r"][finished]
            logger.add_scalar("rewards/train reward", np.mean(episode_rewards))
            logger.write(timestep)
        if i % 1000 == 0 or i == 1:
            eval_reward, obs_traj, frames = eval(dreamer, eval_env, config)
            logger.add_scalar("rewards/eval reward", eval_reward)
            logger.add_video("videos/eval observed", np.concatenate(obs_traj)[np.newaxis, :].astype(np.uint8), fps=120)
            logger.add_video("videos/eval video", np.asarray(frames)[np.newaxis, :].transpose(0, 1, 4, 2, 3), fps=120)
            imagined_rollout = imagine_rollout(dreamer, eval_env, config)
            logger.add_video("videos/imagined rollout", imagined_rollout, fps=120)
            logger.write(timestep)
        if timestep > config.train_wm_after and i % config.train_every == 0:
            for _ in range(config.num_iters):
                loss_dict = dreamer.train()
            logger.add_metrics(loss_dict)
            logger.write(timestep)
        obs = next_obs
 
    env.close()
    eval_env.close()
    