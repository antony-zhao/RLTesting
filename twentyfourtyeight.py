import jax
import jax.numpy as jnp
import equinox as eqx
import pgx
import optax
from typing import Callable
from abc import ABC

key = jax.random.key(0)

class Trajectory(eqx.Module):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array

class MLP(eqx.Module):
    input_layer: eqx.Module
    hiddens: list
    action: eqx.Module
    act: Callable
    skip_connections: bool
    
    def __init__(self, input_dim, output_dim, key, hidden_dim=256, num_hiddens=3, act=jax.nn.selu, hidden_dims=None):
        # Can specify hidden_dims if want more precise control, but generally keeping them the same should be "good enough," we also
        # assume that if there aren't specified hidden dims, that we can use skip connections
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
            self.skip_connections = False
        else:
            self.skip_connections = True
        key, subkey = jax.random.split(key)
        self.input_layer = eqx.nn.Linear(input_dim, hidden_dim, key=subkey)
        self.hiddens = []
        for i in range(num_hiddens):
            key, subkey = jax.random.split(key)
            if hidden_dims is None:
                self.hiddens.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=subkey))
            else:
                self.hiddens.append(eqx.nn.Linear(hidden_dim, hidden_dims[i + 1], key=subkey))
                hidden_dim = hidden_dims[i + 1]
        self.action = eqx.nn.Linear(hidden_dim, output_dim, key=key)
        self.act = act
    
    def __call__(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        action_logits = self.action(x)
        return action_logits

    def policy_fn(self, x, legal_action_mask, key, obs_wrapper=None):
        if obs_wrapper is not None:
            x = obs_wrapper(x)
        action_logits = jax.vmap(self)(x)
        action_logits = jnp.where(legal_action_mask, action_logits, -jnp.inf)
        actions = jax.random.categorical(key, action_logits) 
        action_dist = jax.nn.softmax(action_logits)
        action_probs = jnp.take_along_axis(action_dist, jnp.expand_dims(actions, 1), axis=1)
        return actions, action_probs

def collect_trajectory(step, state, current_obs, policy, key, num_timesteps, obs_wrapper=None):
    # slightly inspired by the brax code, but just trying to implement from memory to learn
    def f(carry, _):
        state, current_obs, key = carry
        random_key, key = jax.random.split(key)
        next_state, next_obs, trajectory = single_step(step, state, current_obs, policy, random_key, obs_wrapper)
        return (next_state, next_obs, key), trajectory
    
    (_, final_obs, _), trajectory = jax.lax.scan(f, (state, current_obs, key), (), num_timesteps) # note to self that scan returns ys in a stacked way.
    return final_obs, trajectory

def single_step(step, state, current_obs, policy, key, obs_wrapper=None):
    key, subkey = jax.random.split(key)
    batch_size = current_obs.shape[0]
    keys = jax.random.split(subkey, batch_size)
    action, action_probs = policy(current_obs, state.legal_action_mask, subkey, obs_wrapper)
    next_state = step(state, action, keys)  # pgx specifically, need to rewrite for other types of environments (especially non-jax ones)
    next_obs = next_state.observation
    rewards = next_state.rewards
    terminated = next_state.terminated
    truncated = next_state.truncated
    dones = jnp.bitwise_or(truncated, terminated)  # Probably a better way to handle this but this should suffice for now
    traj = Trajectory(current_obs, action, rewards, dones)
    return next_state, next_obs, traj

def generalized_advantage_estimate(value):
    pass

class ObservationWrapper(ABC):
    nested_wrapper: Callable
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        
    def __call__(self, observation):
        return observation
    
class FlattenObservation(ObservationWrapper):
    def __call__(self, observation):
        # Assume that the first dimension is the batch dimension (should generally be correct)
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        batch_shape = observation.shape[0]
        return observation.reshape(batch_shape, -1)
    
class ToInt(ObservationWrapper):
    def __call__(self, observation):
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        return jnp.astype(observation, int)
        

"""
TODO
Add GAE and loss functions/gradients
Note to self, there is still an issue that the shape is like (timesteps, batch_size, w, h, channels)
"""