"""
Goal-conditioned MiniGrid wrapper for discrete CRL validation.

Observation: (x, y, dir_onehot[4], neighbor_walls[4]) = 10 dims
Goal:        (x, y, 0, 0, 0, 0, 0, 0, 0, 0)          = 10 dims
obs_to_goal zeros out everything except x, y.

Usage:
    env = make_minigrid(num_envs=16)
"""

import gymnasium as gym
from gymnasium.vector import VectorWrapper, SyncVectorEnv
import numpy as np
from minigrid.wrappers import FullyObsWrapper


OBS_DIM = 10  # x, y, dir_onehot(4), neighbors(4)


class MiniGridGoalEnv(gym.Env):
    """Goal-conditioned MiniGrid with augmented position observations.

    obs  = (x, y, dir_onehot[4], wall_right, wall_left, wall_down, wall_up)
    goal = (x, y, 0, 0, 0, 0, 0, 0, 0, 0)
    """

    def __init__(self, env_id="MiniGrid-FourRooms-v0", max_steps=None):
        super().__init__()
        kwargs = {}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps

        self._env = gym.make(env_id, **kwargs)
        self._env = FullyObsWrapper(self._env)

        self.num_actions = 6
        self.action_space = gym.spaces.Discrete(self.num_actions)

        obs, _ = self._env.reset()
        grid = obs["image"]
        self._grid_max = float(max(grid.shape[0], grid.shape[1]))

        obs_space = gym.spaces.Box(
            low=0, high=max(self._grid_max, 1),
            shape=(OBS_DIM,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "observation": obs_space,
            "achieved_goal": obs_space,
            "desired_goal": obs_space,
        })

        self._empty_cells = []
        self._goal_pos = None

    def _find_empty_cells(self):
        grid = self._env.unwrapped.grid
        empty = []
        for x in range(grid.width):
            for y in range(grid.height):
                if grid.get(x, y) is None:
                    empty.append((x, y))
        return empty

    def _get_neighbors(self):
        grid = self._env.unwrapped.grid
        x, y = self._env.unwrapped.agent_pos
        return np.array([
            float(grid.get(x + 1, y) is not None),  # wall right
            float(grid.get(x - 1, y) is not None),  # wall left
            float(grid.get(x, y + 1) is not None),  # wall down
            float(grid.get(x, y - 1) is not None),  # wall up
        ], dtype=np.float32)

    def _make_obs(self):
        pos = self._env.unwrapped.agent_pos
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self._env.unwrapped.agent_dir] = 1.0
        neighbors = self._get_neighbors()
        return np.concatenate([
            np.array([pos[0], pos[1]], dtype=np.float32),
            dir_onehot,
            neighbors,
        ])

    def _make_goal(self):
        goal = np.zeros(OBS_DIM, dtype=np.float32)
        goal[0] = self._goal_pos[0]
        goal[1] = self._goal_pos[1]
        return goal

    def _get_obs(self):
        obs = self._make_obs()
        return {
            "observation": obs,
            "achieved_goal": obs,
            "desired_goal": self._make_goal(),
        }

    def _is_success(self):
        p = self._env.unwrapped.agent_pos
        return p[0] == self._goal_pos[0] and p[1] == self._goal_pos[1]

    def reset(self, *, seed=None, options=None):
        self._env.reset(seed=seed, options=options)
        self._empty_cells = self._find_empty_cells()

        agent = tuple(self._env.unwrapped.agent_pos)
        candidates = [c for c in self._empty_cells if c != agent]
        self._goal_pos = candidates[self.np_random.integers(len(candidates))]

        return self._get_obs(), {"is_success": False}

    def step(self, action):
        _, _, terminated, truncated, info = self._env.step(action)

        success = self._is_success()
        reward = 0.0 if success else -1.0
        info["is_success"] = success
        if success:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class CRLMiniGridVecWrapper(VectorWrapper):
    """Wraps vectorized MiniGridGoalEnvs for CRL.

    Returns (state, goal) tuples, both (num_envs, 10).
    obs_to_goal zeros out everything except x, y.
    """

    def __init__(self, env):
        super().__init__(env)
        flat_dim = OBS_DIM * 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.single_action_space = env.single_action_space

        # For eval distance checking — only compare x, y
        self.start_index = 0
        self.end_index = 2

    def _split(self, obs_dict):
        state = obs_dict["observation"].astype(np.float32)
        goal = obs_dict["desired_goal"].astype(np.float32)
        return state, goal

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        state, goal = self._split(obs_dict)
        return (state, goal), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        state, goal = self._split(obs_dict)
        return (state, goal), reward, terminated, truncated, info

    def obs_to_goal(self, obs):
        goal = np.zeros_like(obs)
        goal[..., :2] = obs[..., :2]
        return goal


class CRLMiniGridWrapper(gym.Wrapper):
    """Wraps a single MiniGridGoalEnv for CRL.

    Returns flat obs: (20,) = [obs(10), goal(10)].
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM * 2,), dtype=np.float32
        )
        self.start_index = 0
        self.end_index = 2

    def _flatten(self, obs_dict):
        obs = obs_dict["observation"].astype(np.float32)
        goal = obs_dict["desired_goal"].astype(np.float32)
        return np.concatenate([obs, goal])

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs_dict), reward, terminated, truncated, info

    def obs_to_goal(self, obs):
        goal = np.zeros_like(obs)
        goal[..., :2] = obs[..., :2]
        return goal


def make_minigrid(env_id="MiniGrid-FourRooms-v0", num_envs=None, max_steps=None):
    """Create a CRL-wrapped MiniGrid with augmented observations.

    Obs: (x, y, dir_onehot[4], neighbor_walls[4]) = 10 dims
    Goal: (x, y, 0...) = 10 dims, only position filled

    Returns:
        Single: CRLMiniGridWrapper  — obs is (20,) = [obs, goal]
        Vector: CRLMiniGridVecWrapper — returns (state, goal) tuple, both (N, 10)
    """
    if num_envs is not None:
        def _make_env():
            return MiniGridGoalEnv(env_id=env_id, max_steps=max_steps)
        vec_env = SyncVectorEnv([_make_env for _ in range(num_envs)])
        return CRLMiniGridVecWrapper(vec_env)
    else:
        env = MiniGridGoalEnv(env_id=env_id, max_steps=max_steps)
        return CRLMiniGridWrapper(env)


if __name__ == "__main__":
    print("Single env:")
    env = make_minigrid()
    obs, info = env.reset(seed=42)
    print(f"  obs: {obs}")
    print(f"  shape: {obs.shape}  (should be (20,))")
    for step in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            print(f"  Done at step {step+1}, success={info['is_success']}")
            break
    env.close()

    print("\nVectorized (4 envs):")
    env = make_minigrid(num_envs=4)
    (state, goal), info = env.reset(seed=42)
    print(f"  state shape: {state.shape}  (should be (4, 10))")
    print(f"  goal shape:  {goal.shape}   (should be (4, 10))")
    print(f"  state[0]: {state[0]}")
    print(f"  goal[0]:  {goal[0]}")
    print(f"  obs_to_goal(state)[0]: {env.obs_to_goal(state)[0]}")
    print(f"  single_obs_space: {env.single_observation_space}")
    print(f"  single_action_space: {env.single_action_space}")
    env.close()