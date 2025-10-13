import torch
import numpy as np
from gymnasium import spaces

to_numpy = lambda x: x.detach().cpu().numpy()

def get_action_dim(action_space: spaces.Space) -> int:
    # Both this and get obs shape are from stable baselines, don't need to adapt too many environments so just using relevant sections
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1

def get_obs_shape(
    observation_space: spaces.Space,
):
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)