from wrappers import make_fetch_push, make_fetch_reach, make_fetch_reach_discrete, make_point_maze
import gymnasium as gym
import numpy as np
import torch
from crl import CRLAgent
from rltesting.utils.logger import Logger
from rltesting.utils.torch_utils import to_numpy

# most of this is done with the help of AI, didn't want to have to rewrite the training loop for the nth time

num_envs = 32
num_eval_envs = 16
num_steps = 3_000_000
initial_data = 10_000
batch_size = 256
episode_length = 100
train_steps_per_update = 16
log_interval = num_envs * episode_length
eval_interval = 4 * log_interval
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# env = make_fetch_reach(num_envs)          # continuous, obs_dim=10, act_dim=4
# env = make_fetch_reach_discrete(num_envs) # discrete,  obs_dim=10, num_actions=6
# env = make_fetch_push(num_envs)             # continuous, obs_dim=25, act_dim=4
# env = make_fetch_reach_discrete(num_envs, delta=0.1, max_episode_steps=episode_length)
env = make_point_maze(num_envs, maze="Medium_Diverse_GR", max_episode_steps=episode_length)
# eval_env = make_fetch_push(num_eval_envs)
# eval_env = make_fetch_reach_discrete(num_eval_envs, max_episode_steps=episode_length) 
eval_env = make_point_maze(num_eval_envs, maze="Medium_Diverse_GR", max_episode_steps=episode_length)

obs_dim = (env.single_observation_space.shape[0] // 2,)

if isinstance(env.single_action_space, gym.spaces.Discrete):
    action_type = "discrete"
    action_dim = int(env.single_action_space.n)
else:
    action_type = "continuous"
    action_dim = env.action_space.shape[1]

agent = CRLAgent(obs_dim, (action_dim, ), num_envs, device, env.obs_to_goal, action_type=action_type)

def evaluate(agent, eval_env, num_episodes=32, max_steps=episode_length, threshold=0.05):
    (obs, goal), _ = eval_env.reset()
    n = eval_env.num_envs
    start_idx = eval_env.start_index
    end_idx = eval_env.end_index

    time_at_goal = np.zeros(n)
    successes = []
    ep_time_at_goal = []
    episodes_done = 0
    steps = 0

    while episodes_done < num_episodes and steps < max_steps:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=agent.device)
            if agent.action_type == "continuous":
                x = agent.policy.encoder(obs_t)
                g = agent.policy.goal_encoder(goal_t)
                mu = agent.policy.policy_mu(torch.cat([x, g], dim=-1))
                action = torch.tanh(mu).cpu().numpy()
            else:
                dist = agent.policy.policy_dist(obs_t, goal_t)
                action = dist.probs.argmax(dim=-1).cpu().numpy()

        (obs, goal), rewards, terminations, truncations, infos = eval_env.step(action)
        steps += 1

        achieved = obs[:, start_idx:end_idx]
        desired = goal[:, start_idx:end_idx]
        reached = np.linalg.norm(achieved - desired, axis=-1) < threshold
        time_at_goal += reached

        dones = terminations | truncations
        if dones.any():
            for i in np.where(dones)[0]:
                successes.append(time_at_goal[i] > 0)
                ep_time_at_goal.append(time_at_goal[i])
                time_at_goal[i] = 0
            episodes_done += dones.sum()

    success_rate = np.mean(successes) if successes else 0.0
    avg_time_at_goal = np.mean(ep_time_at_goal) if ep_time_at_goal else 0.0
    return success_rate, avg_time_at_goal

(obs, goal), _ = env.reset()
total_steps = 0
logger = Logger('logs/crl')

while total_steps < num_steps:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
        action_t = agent.act(obs_t, goal_t)
        actions = action_t.cpu().numpy()

    (next_obs, next_goal), rewards, terminations, truncations, infos = env.step(actions)
    dones = terminations | truncations

    agent.store_sample(obs, actions, rewards, goal, dones)

    obs = next_obs
    goal = next_goal
    total_steps += num_envs

    if total_steps > initial_data:
        for _ in range(train_steps_per_update):
            critic_loss, actor_loss, metrics = agent.train()
        logger.add_scalar('critic_loss', to_numpy(critic_loss))
        logger.add_scalar('actor_loss', to_numpy(actor_loss))
        logger.add_metrics(metrics)

    if total_steps % log_interval == 0:
        if 'success' in infos:
            success_rate = np.mean(infos['success'])
            print(f"Steps: {total_steps:>8d} | Success: {success_rate:.2f}")
            logger.add_scalar('success_rate', success_rate)

    if total_steps % eval_interval == 0 and total_steps > initial_data:
        eval_success, eval_steps = evaluate(agent, eval_env)
        print(f"Steps: {total_steps:>8d} | Eval: {eval_success:.2f} | Time at goal: {eval_steps:.1f}")
        logger.add_scalar('eval_success_rate', eval_success)
        logger.add_scalar('eval_time_at_goal', eval_steps)

    logger.write(total_steps)