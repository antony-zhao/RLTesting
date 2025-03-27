import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Callable
from abc import ABC

combine_dims = lambda x: jax.lax.collapse(x, 0, 2)

class Trajectory(eqx.Module):
    observations: jax.Array
    actions: jax.Array
    action_probs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    mask: jax.Array

class ObservationWrapper(ABC):
    # Two ways to handle these, pass it into the ppo network and process every time it's called (slower)
    # Or run it on the step function outputs, is probably fine for PPO but could lead to storage issues 
    # for things like off-policy learning.
    nested_wrapper: Callable
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        
    def __call__(self, observation):
        return observation

class MLP(eqx.Module):
    input_layer: eqx.Module
    hiddens: list
    output_layer: eqx.Module
    act: Callable
    skip_connections: bool
    output_act: Callable
    forward: Callable
    
    def __init__(self, input_dim, output_dim, key, hidden_dim=256, num_hiddens=3, act=jax.nn.selu, hidden_dims=None, output_act=None):
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
        self.output_layer = eqx.nn.Linear(hidden_dim, output_dim, key=key)
        self.act = act
        self.output_act = output_act
        def f(x):
            x = self.act(self.input_layer(x))
            for i in range(len(self.hiddens)):
                if self.skip_connections:
                    x = self.act(self.hiddens[i](x)) + x
                else:
                    x = self.act(self.hiddens[i](x))
            logits = self.output_layer(x)
            if self.output_act is not None:
                return self.output_act(logits)
            return logits
        self.forward = eqx.filter_jit(f)
    
    def __call__(self, x):
        return self.forward(x)
    
class PPONetwork(eqx.Module):
    policy_network: MLP
    value_network: MLP
    obs_wrapper: ObservationWrapper
    
    def __init__(self, input_dim, num_actions, key, hidden_dim=256, 
                 num_hiddens=3, act=jax.nn.selu, hidden_dims=None, observation_wrapper=None):
        # Assuming we have the same general MLP structure for both the policy network and value network
        # Potentially TODO: Support better arg handling for different structures for both models
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.policy_network = MLP(input_dim, num_actions, subkey1, hidden_dim=hidden_dim, 
                                  num_hiddens=num_hiddens, act=act, hidden_dims=hidden_dims)
        self.value_network = MLP(input_dim, 1, subkey2, hidden_dim=hidden_dim, 
                                  num_hiddens=num_hiddens, act=act, hidden_dims=hidden_dims)
        self.obs_wrapper = observation_wrapper
    
    def policy_forward(self, x):
        if self.obs_wrapper is not None:
            x = self.obs_wrapper(x)
        action_logits = jax.vmap(self.policy_network)(x)
        return action_logits
    
    def value_forward(self, x):
        if self.obs_wrapper is not None:
            x = self.obs_wrapper(x)
        action_logits = jax.vmap(self.value_network)(x)
        return action_logits
    
def make_policy_fn(model):
    @jax.jit
    def policy_fn(x, legal_action_mask, key):
        action_logits = model(x)
        action_logits = jnp.where(legal_action_mask, action_logits, -jnp.inf)
        action_dist = jax.nn.softmax(action_logits)
        actions = jax.random.categorical(key, action_logits)
        action_probs = jnp.take_along_axis(action_dist, jnp.expand_dims(actions, 1), axis=1)
        return actions, action_probs, action_dist
    return policy_fn

def make_value_fn(model):
    @jax.jit
    def value_fn(x):
        values = model(x)
        return values
    return value_fn

def collect_trajectory(step, state, current_obs, policy_fn, key, num_timesteps, obs_wrapper=None):
    # slightly inspired by the brax code, but just trying to implement from memory to learn
    # @jax.jit
    def f(carry, _):
        state, current_obs, key = carry
        random_key, key = jax.random.split(key)
        next_state, next_obs, trajectory = single_step(step, state, current_obs, policy_fn, random_key, obs_wrapper=obs_wrapper)
        return (next_state, next_obs, key), trajectory
    
    (next_state, final_obs, _), trajectory = jax.lax.scan(
        f, (state, current_obs, key), (), num_timesteps
    ) # note to self that scan returns ys in a stacked way.
    # try to stack final obs into the trajectory, not sure if possible but worth a try
    return next_state, final_obs, trajectory

def single_step(step, state, current_obs, policy_fn, key, obs_wrapper=None):
    key, subkey = jax.random.split(key)
    batch_size = current_obs.shape[0]
    keys = jax.random.split(subkey, batch_size)
    if obs_wrapper is not None:
        current_obs = obs_wrapper(current_obs)
    key, subkey = jax.random.split(key)
    action, action_prob, _ = policy_fn(current_obs, state.legal_action_mask, subkey)
    next_state = step(state, action, keys)  # pgx specifically, need to rewrite for other types of environments (especially non-jax ones)
    next_obs = next_state.observation
    rewards = next_state.rewards
    terminated = next_state.terminated
    truncated = next_state.truncated
    dones = jnp.bitwise_or(truncated, terminated)  # Probably a better way to handle this but this should suffice for now
    traj = Trajectory(current_obs, action, action_prob, rewards, dones, state.legal_action_mask)
    return next_state, next_obs, traj

# @jax.jit
def generalized_advantage_estimate(values, rewards, dones, discount, _lambda):
    # Also heavily inspired by brax implementation
    # assume observations include s_t-s_t+l (so both initial and the final obs) 
    # delta_t = r_t + gamma * v_t+1 - v_t
    # A_t = delta_t + gamma * lambda * A_t+1
    values_t = values[:-1]
    values_t_p1 = values[1:]
    deltas = rewards + values_t_p1 * discount - values_t
    def f(carry, xs):
        delta, done = xs
        a_t_p1, _ = carry
        a_t = delta + a_t_p1 * _lambda * discount * (1 - done)
        return (a_t, _), (a_t)
    
    a_init = (jnp.zeros((values.shape[1], 1)), None)
    dones = jnp.expand_dims(dones, axis=-1)
    _, gae = jax.lax.scan(f, a_init, (deltas, dones), reverse=True)
    vs = gae + values_t
    vs_t_p1 = jnp.concatenate([vs[1:], jnp.expand_dims(values[-1], 0)])
    
    advantages = (1 - dones) * (rewards + discount * vs_t_p1 - vs)
    
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(vs)

def ppo_loss(trajectory, final_obs, ppo_network, value_fn, discount, _lambda, eps, value_loss_coeff, entropy_loss_coeff):
    # Value loss: L = (r_t + gamma * V(s_t+1) - V(s_t)) ^ 2
    # R_ratio = pi_theta(a_t|s_t) / pi_old(a_t | s_t)
    # Policy loss = min(R_ratio * advantage, clip(R_ratio, 1-eps, 1+eps) * advantage)
    # Entropy = - sum_i p(a_i | s_t) * log (p(a_i | s_t) + 1e-10) (or some other small term)
    observations, rewards, dones, actions, mask = trajectory.observations, trajectory.rewards, \
        trajectory.dones, trajectory.actions, trajectory.mask
    timesteps, batch_size = observations.shape[:2]
    all_observations = jnp.concatenate([observations, jnp.expand_dims(final_obs, 0)]) 
    old_action_probs = trajectory.action_probs
    # since observations have a shape of (timesteps, batch, (shape)), we have to modify it by rehsaping 
    # first two dims, either that or have to make a more annoying change with a double vmap.
    observations = combine_dims(observations)
    mask = combine_dims(mask)
    actions = combine_dims(actions)
    action_logits = ppo_network.policy_forward(observations)
    action_logits = jnp.where(mask, action_logits, -jnp.inf)
    action_dist = jax.nn.softmax(action_logits)
    new_action_probs = jnp.take_along_axis(action_dist, jnp.expand_dims(actions, 1), axis=1)
    new_action_probs = jnp.reshape(new_action_probs, (timesteps, batch_size, -1))
    
    all_observations = combine_dims(all_observations)
    all_values = value_fn(all_observations)
    all_values = jnp.reshape(all_values, (timesteps + 1, batch_size, -1))
    values = all_values[:-1]
    advantages, value_target = generalized_advantage_estimate(all_values, rewards, dones,
                                                              discount, _lambda)
    value_loss = jnp.mean((values - value_target) ** 2)
    
    policy_ratio = new_action_probs / old_action_probs
    policy_loss = -jnp.mean(jnp.minimum(policy_ratio * advantages, 
                                        jnp.clip(policy_ratio, 1-eps, 1+eps) * advantages))
    
    # TODO Entropy loss
    entropy_loss = jnp.mean(jnp.sum(action_dist * jnp.log(action_dist + 1e-10), axis=-1))
    total_loss = value_loss * value_loss_coeff + policy_loss + entropy_loss * entropy_loss_coeff
    metrics = {
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "policy_loss": policy_loss
    }
    return total_loss, metrics
    
class FlattenObservation(ObservationWrapper):
    @jax.jit
    def __call__(self, observation):
        # Assume that the first dimension is the batch dimension (should generally be correct)
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        return jax.lax.collapse(observation, 1)
    
class ToInt(ObservationWrapper):
    @jax.jit
    def __call__(self, observation):
        if self.nested_wrapper is not None:
            observation = self.nested_wrapper(observation)
        return jnp.astype(observation, int)
        

"""
TODO
Add loss functions/gradients
Note to self, there is still an issue that the shape is like (timesteps, batch_size, w, h, channels)
"""

def main(args):
    key = jax.random.key(args.seed)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--rollout_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)