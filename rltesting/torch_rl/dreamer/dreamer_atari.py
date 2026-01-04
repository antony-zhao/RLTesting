import gymnasium as gym
from rltesting.utils.logger import Logger
from dreamer import *

import ale_py
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import NoopResetEnv, FireResetEnv
import argparse
import imageio
import os


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
    # parser.add_argument("--imagination_batch_size", default=1024, type=int)
    parser.add_argument("--rollout_length", default=16, type=int)
    parser.add_argument("--critic_tau", default=0.02, type=float)
    parser.add_argument("--critic_imagination_loss_coef", default=1, type=float)
    parser.add_argument("--critic_replay_loss_coef", default=0.3, type=float)
    parser.add_argument("--entropy_coef", default=3e-4, type=float)
    parser.add_argument("--return_range_tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.997, type=float)
    parser.add_argument("--lambda_", default=0.95, type=float)
    parser.add_argument("--percentiles", default=0.05, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", default=8e-5, type=float)
    parser.add_argument("--num_envs", default=32, type=int)
    parser.add_argument("--replay_ratio", default=32, type=int) # batch_size * seq_len / timesteps_in_env = replay ratio, so default of 64 * 16 / num_envs * timesteps = 16
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


def make_env(config):
    def thunk():
        gym.register_envs(ale_py)
        env = gym.make(f"ALE/{config.env_id}-v5", render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.TransformObservation(env, lambda x: x.transpose(2, 0, 1), observation_space=gym.spaces.Box(0, 255, config.image_dim))
        env = gym.wrappers.TimeLimit(env, 10000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def eval(dreamer, eval_env, config):
    done = False
    obs = eval_env.reset()
    total_reward = 0
    num_completed = 0
    is_first = True
    frames = []
    obs_traj = []
    while not done:
        action = dreamer.eval_action(torch.tensor(obs).float().to(config.device), reset=is_first)
        is_first = False
        obs, reward, done, infos = eval_env.step(action)
        frames.append(eval_env.render())
        obs_traj.append(obs)
        for info in infos:
            if 'episode' in info.keys():
                total_reward += info['episode']['r']
                num_completed += 1
    return total_reward, obs_traj, frames

def imagine_rollout(dreamer, config):
    obs = eval_env.reset()
    obs = torch.tensor(obs).float().to(config.device)
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
    env = make_vec_env(make_env(config), num_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    eval_env = make_vec_env(make_env(config), 1)
    dreamer = DreamerV3(config)
    logger = Logger(f"logs/dreamer-v3/{config.env_id}")

    losses_wm = []
    losses_actor = []
    losses_critic = []
    losses_dict = {}
    obs = env.reset()
    timestep = 0
    for i in range(5_000_000 // config.num_envs):
        timestep += config.num_envs
        if i % 100 == 0:
            print(i)
        action, latent = dreamer.choose_action(torch.tensor(obs).float().to(config.device))
        next_obs, reward, done, info = env.step(to_numpy(action))
        dreamer.process_sample(obs, latent, action, reward, done)
        if True in done:
            indices = np.where(done)[0]
            temp_rew = []
            for index in indices:
                temp_rew.append(info[index]["episode"]["r"])
            logger.add_scalar("rewards/train reward", np.mean(temp_rew))
            logger.write(timestep)
        if i % 1000 == 0:
            eval_reward, obs_traj, frames = eval(dreamer, eval_env, config)
            logger.add_scalar("rewards/eval reward", eval_reward)
            logger.add_video("videos/eval observed", np.concatenate(obs_traj)[np.newaxis, :].astype(np.uint8), fps=120)
            logger.add_video("videos/eval video", np.asarray(frames)[np.newaxis, :].transpose(0, 1, 4, 2, 3), fps=120)
            imagined_rollout = imagine_rollout(dreamer, config)
            logger.add_video("videos/imagined rollout", imagined_rollout, fps=120)
            logger.write(timestep)
        if i > 500 and i % config.train_every == 0:
            for _ in range(config.num_iters):
                loss_wm, loss_critic, loss_actor, loss_dict = dreamer.train()
            logger.add_metrics(loss_dict)
            logger.write(timestep)
        obs = next_obs