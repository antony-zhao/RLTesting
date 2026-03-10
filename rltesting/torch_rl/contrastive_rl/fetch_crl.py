from wrappers import make_fetch_push, make_fetch_reach, make_fetch_reach_discrete
import gymnasium as gym
import numpy as np
import torch
from crl import CRLAgent
from rltesting.utils.logger import Logger
from rltesting.utils.torch_utils import to_numpy

num_envs = 32
num_steps = 3_000_000
initial_data = 10_000
batch_size = 256
train_steps_per_update = 64
log_interval = num_envs * 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Pick one ---
# env = make_fetch_reach(num_envs)          # continuous, obs_dim=10, act_dim=4
# env = make_fetch_reach_discrete(num_envs) # discrete,  obs_dim=10, num_actions=6
env = make_fetch_push(num_envs)             # continuous, obs_dim=25, act_dim=4

# Extract shapes and action type from the wrapped env
obs_dim = (env.observation_space.shape[0] // 2,)  # wrapper doubles for [state, goal]

if isinstance(env.action_space, gym.spaces.Discrete):
    action_type = "discrete"
    action_dim = (env.action_space.n,)  # one-hot size
else:
    action_type = "continuous"
    action_dim = env.action_space.shape[1]

agent = CRLAgent(obs_dim, (action_dim, ), num_envs, device, env.obs_to_goal, action_type=action_type)

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

    # Buffer stores [obs, action, reward, goal, done] — future states sampled internally
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
        if 'is_success' in infos:
            success_rate = np.mean(infos['is_success'])
            print(f"Steps: {total_steps:>8d} | Success: {success_rate:.2f}")
            logger.add_scalar('success_rate', success_rate)

    logger.write(total_steps)