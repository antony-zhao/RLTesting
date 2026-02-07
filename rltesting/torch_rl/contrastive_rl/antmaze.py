import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('AntMaze_UMaze-v5', max_episode_steps=100)

print(env.reset())