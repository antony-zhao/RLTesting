from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import  SubprocVecEnv
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import ale_py
from rltesting.torch_rl.ppo.rnd import RND, AtariFeatureModel, RunningMeanStd, AtariPolicyNetwork, AtariRNDValueNetwork, PPONetwork, train_rnd, collect_rollout, layer_init
import imageio
from rltesting.utils.torch_utils import to_numpy
from rltesting.utils.logger import Logger
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def main(args):
    gym.register_envs(ale_py)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def make_env(gym_id, seed, time_limit=4500, eval=False):
        def thunk():
            env = gym.make(gym_id, render_mode='rgb_array')
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=1)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            if not eval:
                env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
            if not eval:
                env = gym.wrappers.TimeLimit(env, time_limit)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_vec_env(make_env(f'ALE/{args.env}-v5', args.seed), args.num_envs, args.seed, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    eval_env = make_vec_env(make_env(f'ALE/{args.env}-v5', args.seed, eval=True), 1, args.seed)
    
    def eval(ppo_network, rnd, eval_env):
        done = False
        obs = eval_env.reset()
        total_reward = 0
        num_completed = 0
        frames = []
        while not done:
            obs = obs#(obs - rms.mean) / (np.sqrt(rms.var) + 1e-8)
            obs = torch.as_tensor(obs).to(device).float()
            intrinsic_reward, _ = rnd.compute_intrinsic(obs)
            # total_reward += np.sum(intrinsic_reward)
            action, _ = ppo_network.policy_network.policy_fn(obs, det=False)
            obs, reward, done, infos = eval_env.step(to_numpy(action))
            total_reward += reward
            frames.append(eval_env.render())
            for info in infos:
                if 'episode' in info.keys():
                    total_reward += info['episode']['r']
                    num_completed += 1
        return total_reward, frames

    def anneal_lr(optim, lr, update, num_updates):
        frac = 1.0 - (update) / num_updates
        new_lr = frac * lr
        optim.param_groups[0]["lr"] = new_lr

    with torch.device(device):
        feature_model = AtariFeatureModel(num_hidden=3)
        target_model = AtariFeatureModel(num_hidden=0)
        rnd_opt = optim.Adam(feature_model.parameters(), lr=args.lr)
        rnd = RND(feature_model, target_model)
        
        policy = AtariPolicyNetwork(env.action_space.n)
        value = AtariRNDValueNetwork()
        ppo_network = PPONetwork(policy, value)
        for module in ppo_network.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_init(module)
        for module in rnd.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_init(module)
        ppo_opt = optim.Adam(ppo_network.parameters(), lr=args.lr, eps=1e-5)
    
        obs_rms = RunningMeanStd() # TODO, only normalize per channel dimension, so need to slightly modify the RMS code somehow
        rnd_rms = RunningMeanStd()
        obs = env.reset()
        for _ in range(50):
            action = [env.action_space.sample() for _ in range(args.num_envs)]
            obs, _, _, _ = env.step(action)
            obs_rms.update(obs) #
        
        total_updates = args.timesteps // (args.num_envs * args.rollout_length)
        obs = obs
        logger = Logger(f'logs/{args.env}')
        for i in range(total_updates):
            rollout, obs = collect_rollout(env, ppo_network, rnd, args.rollout_length, obs, obs_rms)
            metrics = train_rnd(rollout, ppo_network, rnd, obs_rms, rnd_rms, ppo_opt, rnd_opt, args.num_minibatches, args.num_epochs, device, rnd_coef=1,
                                max_grad_norm=args.max_grad_norm, clip=args.clip, ent_coef=args.ent_coef, val_coef=args.val_coef, sample_prob=32 / args.num_envs)
            anneal_lr(ppo_opt, args.lr, i, total_updates)
            logger.add_metrics(metrics)
            train_rew = np.sum(rollout.rewards) / max(1, np.sum(rollout.dones))
            print(f'{i * args.num_envs * args.rollout_length}: {train_rew}')
            logger.add_scalar('train_reward', train_rew)
            if i % 50 == 0:
                eval_reward, frames = eval(ppo_network, rnd, eval_env)
                print(f'epoch {i}: {np.mean(eval_reward)}')
                logger.add_scalar('eval_reward', eval_reward)
                imageio.mimwrite(f'gifs/{args.env}_{i}.gif', frames[::4], loop=0, fps=20)
            logger.write(i * args.num_envs * args.rollout_length)
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='MontezumaRevenge') #MontezumaRevenge
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--val-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--timesteps', type=int, default=50_000_000)
    parser.add_argument('--log-every', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
    