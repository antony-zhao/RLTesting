import gymnasium_robotics
import gymnasium as gym
import numpy as np
import torch
from crl import CRLAgent
from rltesting.utils.logger import Logger
from rltesting.utils.torch_utils import to_numpy

def format_vector_observation(obs_dict):
    """Converts batched Gymnasium dict observations into flat [state, padded_goal] arrays."""
    s = obs_dict['observation']           # shape: (num_envs, obs_dim)
    desired_goal = obs_dict['desired_goal'] # shape: (num_envs, goal_dim)
    
    # Create a padded goal array of the same size as states
    g = np.zeros_like(s)
    # Inject the 3D coordinates (assuming goal is 3D)
    g[:, 0:desired_goal.shape[1]] = desired_goal 
    
    # Concatenate along the feature dimension
    return s, g

num_envs = 16
num_steps = 1_000_000
initial_data = 10_000
batch_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = CRLAgent((10, ), (4, ), num_envs, device)

gym.register_envs(gymnasium_robotics)
env = gym.make_vec('FetchReachDense-v4', num_envs=num_envs, vectorization_mode="sync")
    
# Reset returns a batched dictionary and batched info
obs_dict, _ = env.reset()
obs, goal = format_vector_observation(obs_dict)

total_steps = 0
logger = Logger('logs/crl')

while total_steps < num_steps:
    # 2. Select Batched Actions
    with torch.no_grad():
        # Convert batched numpy array to tensor: shape [num_envs, obs_dim]
        obs_tensor = torch.tensor(obs).float().to(agent.device)
        goal_tensor = torch.tensor(goal).float().to(agent.device)
        # Agent outputs batched actions: shape [num_envs, action_dim]
        action_tensor = agent.act(obs_tensor, goal_tensor)
        actions = action_tensor.cpu().numpy()

    next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
    
    dones = terminations | truncations
    agent.store_sample(obs, actions, rewards, goal, dones)
        
    # 3. Step all environments simultaneously
    # Returns batched arrays for everything
    next_obs, goal = format_vector_observation(next_obs_dict)
    
    
    obs = next_obs
    total_steps += num_envs  # We took 'num_envs' steps in reality
    
    # 4. Train the Agent
    # Check if buffer has enough samples (you might need to adjust based on your buffer's API)
    if total_steps > initial_data:
        for _ in range(16):
            critic_loss, actor_loss = agent.train()
        logger.add_scalar('critic_loss', to_numpy(critic_loss))
        logger.add_scalar('actor_loss', to_numpy(actor_loss))
    
    # Optional: Logging success rate from the vectorized infos
    if total_steps % (num_envs * 50) == 0:
        if 'is_success' in infos:
            successes = infos['is_success']
            print(f"Steps: {total_steps} | Recent Success Rate: {np.mean(successes):.2f}")
            logger.add_scalar("success_rate", np.mean(successes))
    logger.write(total_steps)