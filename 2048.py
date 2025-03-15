import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random
import pgx
import optax

key = random.key(0)

class Trajectory(eqx.Module):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array

class MLP(eqx.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_hiddens=3, act=jax.nn.selu, hidden_dims=None):
        # Can specify hidden_dims if want more precise control, but generally keeping them the same should be "good enough"
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
        self.input_layer = eqx.nn.Linear(input_dim, hidden_dim)
        self.hiddens = []
        for i in range(num_hiddens):
            if hidden_dims is None:
                self.hiddens.append(eqx.nn.Linear(hidden_dim, hidden_dim))
            else:
                self.hiddens.append(eqx.nn.Linear(hidden_dim, hidden_dims[i + 1]))
                hidden_dim = hidden_dims[i + 1]
        self.action = eqx.nn.Linear(hidden_dim, output_dim)
        self.act = act
    
    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x)) + x
        x = self.act(self.lin3(x)) + x
        x = self.act(self.lin4(x)) + x
        action_logits = self.action(x)
        return action_logits
        
    
def policy_fn(model, x, key):
    action_logits = model(x)
    actions = jax.random.categorical(key, action_logits) 
    action_dist = jax.nn.softmax(action_logits)
    action_probs = jnp.choose(action_dist, actions)
    return actions, action_probs

def collect_trajectory(step, current_obs, policy, key, num_timesteps):
    # slightly inspired by the brax code, but just trying to implement from memory to learn
    def f(carry, _):
        current_obs, key = carry
        random_key, key = random.split(key)
        next_obs, trajectory = single_step(step, current_obs, policy, random_key)
        return (next_obs, key), trajectory
    
    (final_obs, _), trajectory = jax.lax.scan(f, (current_obs, key), (), num_timesteps) # note to self that scan returns ys in a stacked way.
    return final_obs, trajectory

def single_step(step, current_obs, policy, key):
    action = policy(current_obs)
    state = step(key, action)  # pgx specifically, need to rewrite for other types of environments (especially non-jax ones)
    next_obs = state.observation
    rewards = state.rewards
    terminated = state.terminated
    truncated = state.truncated
    dones = jnp.bitwise_or(truncated, terminated)  # Probably a better way to handle this but this should suffice for now
    traj = Trajectory(current_obs, action, rewards, dones)
    return next_obs, traj

def generalized_advantage_estimate(value):
    pass


"""
TODO
Handle action masking, add GAE and loss functions/gradients
"""