"""
CRL observation wrappers for gymnasium-robotics Fetch environments.

Converts the GoalEnv dict observation {observation, desired_goal, achieved_goal}
into the flat [state, padded_goal] format expected by Contrastive RL, matching
the observation format from the Google Research reference implementation.

Includes a discrete action wrapper for validating discrete CRL on FetchReach.

Usage:
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)

    # Continuous (single env)
    env = gym.make('FetchReach-v4')
    env = CRLFetchWrapper(env, start_index=0, end_index=3)

    # Continuous (vectorized)
    env = make_fetch_reach(num_envs=32)

    # Discrete FetchReach
    env = make_fetch_reach_discrete(num_envs=32)
"""

import gymnasium as gym
from gymnasium.vector import VectorWrapper, SyncVectorEnv
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Action Wrappers
# ══════════════════════════════════════════════════════════════════════════════


class DiscreteFetchWrapper(gym.ActionWrapper):
    """Maps discrete actions to fixed end-effector displacements.

    6 actions: ±x, ±y, ±z. Gripper dimension is always 0 (open).

    Args:
        env: A gymnasium Fetch environment with continuous actions.
        delta: Magnitude of each discrete step.
    """

    ACTIONS = np.array([
        [ 1,  0,  0, 0],  # +x
        [-1,  0,  0, 0],  # -x
        [ 0,  1,  0, 0],  # +y
        [ 0, -1,  0, 0],  # -y
        [ 0,  0,  1, 0],  # +z
        [ 0,  0, -1, 0],  # -z
    ], dtype=np.float32)

    def __init__(self, env, delta=0.3):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(6)
        self._actions = self.ACTIONS * delta

    def action(self, act):
        return self._actions[act]


# ══════════════════════════════════════════════════════════════════════════════
#  Observation Wrappers (single env)
# ══════════════════════════════════════════════════════════════════════════════


class CRLFetchWrapper(gym.Wrapper):
    """Wraps a single Fetch GoalEnv into flat [obs, padded_goal] format for CRL.

    Args:
        env: A gymnasium Fetch environment (GoalEnv).
        start_index: Start index of the achieved goal within the observation vector.
        end_index: End index (exclusive) of the achieved goal within the observation vector.
        goal_indices: Which indices in the goal vector to fill with the desired_goal.
            If None, defaults to [start_index, ..., end_index - 1].
            For FetchPush, pass [0, 1, 2, 3, 4, 5] to fill both gripper and block target.
    """

    def __init__(self, env, start_index, end_index, goal_indices=None):
        super().__init__(env)
        obs_sample = env.observation_space['observation']
        self.obs_dim = obs_sample.shape[0]
        self.start_index = start_index
        self.end_index = end_index
        self.goal_dim = end_index - start_index

        if goal_indices is None:
            self.goal_indices = list(range(start_index, end_index))
        else:
            self.goal_indices = goal_indices

        flat_dim = self.obs_dim * 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        s = obs_dict['observation']
        g = np.zeros(self.obs_dim, dtype=np.float32)
        desired = obs_dict['desired_goal']
        for i, idx in enumerate(self.goal_indices):
            g[idx] = desired[i % self.goal_dim]
        return np.concatenate([s, g]).astype(np.float32)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs_dict), reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════════════
#  Observation Wrappers (vectorized)
# ══════════════════════════════════════════════════════════════════════════════


class CRLFetchVecWrapper(VectorWrapper):
    """Wraps a vectorized Fetch GoalEnv into (state, goal) format for CRL.

    Returns separate (state, goal) arrays for direct use in training loops,
    since CRL needs to sample/manipulate goals independently.

    Args:
        env: A vectorized gymnasium Fetch environment.
        start_index: Start index of the achieved goal within the observation vector.
        end_index: End index (exclusive) of the achieved goal within the observation vector.
        goal_indices: Which indices in the goal vector to fill with the desired_goal.
            If None, defaults to [start_index, ..., end_index - 1].
    """

    def __init__(self, env, start_index, end_index, goal_indices=None):
        super().__init__(env)
        obs_sample = env.observation_space['observation']
        self.obs_dim = obs_sample.shape[1]
        self.start_index = start_index
        self.end_index = end_index
        self.goal_dim = end_index - start_index

        if goal_indices is None:
            self.goal_indices = list(range(start_index, end_index))
        else:
            self.goal_indices = goal_indices

        flat_dim = self.obs_dim * 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        """Returns (state, goal) as separate arrays, both shape (num_envs, obs_dim)."""
        s = obs_dict['observation']
        g = np.zeros_like(s, dtype=np.float32)
        desired = obs_dict['desired_goal']
        for i, idx in enumerate(self.goal_indices):
            g[:, idx] = desired[:, i % self.goal_dim]
        return s.astype(np.float32), g

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        state, goal = self._flatten_obs(obs_dict)
        return (state, goal), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        state, goal = self._flatten_obs(obs_dict)
        return (state, goal), reward, terminated, truncated, info

    def obs_to_goal(self, obs):
        """Extract the goal-relevant coordinates from a state observation.

        Used in the critic loss to convert future states into goal representations.

        Args:
            obs: State observations, shape (batch, obs_dim) or (batch, obs_dim * 2).
                 If obs_dim * 2, assumes [state, goal] concatenation and uses only state.

        Returns:
            Goal-formatted observation, shape (batch, obs_dim) with only goal-relevant
            indices filled and the rest zeroed.
        """
        if obs.shape[-1] == self.obs_dim * 2:
            obs = obs[..., :self.obs_dim]
        achieved = obs[..., self.start_index:self.end_index]
        g = np.zeros_like(obs)
        for i, idx in enumerate(self.goal_indices):
            g[..., idx] = achieved[..., i % self.goal_dim]
        return g


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience Constructors
# ══════════════════════════════════════════════════════════════════════════════


def make_fetch_reach(num_envs=None, dense=True, **kwargs):
    """Create a CRL-wrapped FetchReach environment (continuous actions).

    FetchReach: obs(10), achieved_goal = obs[0:3] (gripper position).
    Goal vector: desired_goal placed at indices [0, 1, 2].
    """
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env_id = 'FetchReachDense-v4' if dense else 'FetchReach-v4'
    if num_envs is not None:
        def _make_env():
            return gym.make(env_id, **kwargs)
        vec_env = SyncVectorEnv([_make_env for _ in range(num_envs)])
        return CRLFetchVecWrapper(vec_env, start_index=0, end_index=3)
    else:
        env = gym.make(env_id, **kwargs)
        return CRLFetchWrapper(env, start_index=0, end_index=3)


def make_fetch_reach_discrete(num_envs=None, dense=True, delta=0.3, **kwargs):
    """Create a CRL-wrapped FetchReach with discrete actions (6 directions).

    Same observation format as make_fetch_reach. Action space is Discrete(6):
        0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z

    Args:
        num_envs: If set, create vectorized env via SyncVectorEnv.
        dense: Use dense reward variant.
        delta: Step size for each discrete action.
    """
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env_id = 'FetchReachDense-v4' if dense else 'FetchReach-v4'
    if num_envs is not None:
        def _make_env():
            env = gym.make(env_id, **kwargs)
            return DiscreteFetchWrapper(env, delta=delta)
        vec_env = SyncVectorEnv([_make_env for _ in range(num_envs)])
        return CRLFetchVecWrapper(vec_env, start_index=0, end_index=3)
    else:
        env = gym.make(env_id, **kwargs)
        env = DiscreteFetchWrapper(env, delta=delta)
        return CRLFetchWrapper(env, start_index=0, end_index=3)


def make_fetch_push(num_envs=None, dense=True, **kwargs):
    """Create a CRL-wrapped FetchPush environment.

    FetchPush: obs(25), achieved_goal = obs[3:6] (block position).
    Goal vector: desired_goal placed at indices [0,1,2] (gripper target)
                 AND [3,4,5] (block target), matching reference implementation.
    """
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env_id = 'FetchPushDense-v4' if dense else 'FetchPush-v4'
    goal_indices = [0, 1, 2, 3, 4, 5]
    if num_envs is not None:
        def _make_env():
            return gym.make(env_id, **kwargs)
        vec_env = SyncVectorEnv([_make_env for _ in range(num_envs)])
        return CRLFetchVecWrapper(vec_env, start_index=3, end_index=6, goal_indices=goal_indices)
    else:
        env = gym.make(env_id, **kwargs)
        return CRLFetchWrapper(env, start_index=3, end_index=6, goal_indices=goal_indices)