import jax
import jax.numpy as jnp
from abc import ABC
from typing import Callable

class ObservationWrapper(ABC):
    # Two ways to handle these, pass it into the ppo network and process every time it's called (slower)
    # Or run it on the step function outputs, is probably fine for PPO but could lead to storage issues 
    # for things like off-policy learning.
    nested_wrapper: Callable
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        
    def __call__(self, observation):
        return observation
    
class RewardWrapper(ABC):
    nested_wrapper: Callable
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        
    def __call__(self, observation):
        return observation
    
class FlattenObservation(ObservationWrapper):
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        f = lambda x: jax.lax.collapse(x, 1)
        self.f = jax.jit(f)
        
    def __call__(self, observation):
        # Assume that the first dimension is the batch dimension (should generally be correct)
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        return self.f(observation)
    
class TransposeObservation(ObservationWrapper):
    def __init__(self, axes, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        f = lambda x: jnp.transpose(x, axes)
        self.f = jax.jit(jax.vmap(f))
        
    def __call__(self, observation):
        # Assume that the first dimension is the batch dimension (should generally be correct)
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        return self.f(observation)
    
class ToDtype(ObservationWrapper):
    def __init__(self, dtype, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        f = lambda x: jnp.astype(x, dtype)
        self.f = jax.jit(f)
        
    def __call__(self, observation):
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        return self.f(observation)
    
class _2048Reward(RewardWrapper):
    def __init__(self, nested_wrapper=None, eps=0.001):
        self.nested_wrapper = nested_wrapper
        f = lambda x: jnp.sign(x) * (jnp.sqrt(x + 1) - 1 + eps * x)
        self.f = jax.jit(f)
        
    def __call__(self, reward):
        if self.nested_wrapper is not None:
            reward = self.nested_wrapper(reward)
        return self.f(reward)