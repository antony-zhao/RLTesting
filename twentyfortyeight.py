import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Callable
from abc import ABC
from functools import partial
import pgx
from pgx.experimental import auto_reset, act_randomly
from matplotlib import pyplot as plt

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
    
class RewardWrapper(ABC):
    nested_wrapper: Callable
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        
    def __call__(self, observation):
        return observation

class MLP(eqx.Module):
    input_layer: eqx.Module
    hiddens: list[eqx.Module]
    output_layer: eqx.Module
    act: Callable
    skip_connections: bool
    output_act: Callable
    
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
    
    @eqx.filter_jit
    def __call__(self, x):
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
    
class PPONetwork(eqx.Module):
    policy_network: MLP
    value_network: MLP
    obs_wrapper: ObservationWrapper
    
    def __init__(self, input_dim, num_actions, key, hidden_dim=256, 
                 num_hiddens=3, act=jax.nn.selu, hidden_dims=None, obs_wrapper=None):
        # Assuming we have the same general MLP structure for both the policy network and value network
        # Potentially TODO: Support better arg handling for different structures for both models
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.policy_network = MLP(input_dim, num_actions, subkey1, hidden_dim=hidden_dim, 
                                  num_hiddens=num_hiddens, act=act, hidden_dims=hidden_dims)
        self.value_network = MLP(input_dim, 1, subkey2, hidden_dim=hidden_dim, 
                                  num_hiddens=num_hiddens, act=act, hidden_dims=hidden_dims)
        self.obs_wrapper = obs_wrapper
    
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
    
    @eqx.filter_jit
    def policy_fn(self, x, legal_action_mask, key):
        action_logits = self.policy_forward(x)
        action_logits = jnp.where(legal_action_mask, action_logits, -jnp.inf)
        action_dist = jax.nn.softmax(action_logits)
        actions = jax.random.categorical(key, action_logits)
        action_probs = jnp.take_along_axis(action_dist, jnp.expand_dims(actions, 1), axis=1)
        return actions, action_probs, action_dist

@eqx.filter_jit
def collect_trajectory(ppo_network, step, state, current_obs, key, num_timesteps, obs_wrapper=None):
    # slightly inspired by the brax code, but just trying to implement from memory to learn
    def f(carry, _):
        state, current_obs, key = carry
        random_key, key = jax.random.split(key)
        next_state, next_obs, trajectory = single_step(step, state, current_obs, ppo_network.policy_fn, random_key, obs_wrapper=obs_wrapper)
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

@jax.jit
def generalized_advantage_estimate(values, rewards, dones, discount, lambda_):
    # Also heavily inspired by brax implementation
    # assume observations include s_t-s_t+l (so both initial and the final obs) 
    # delta_t = r_t + gamma * v_t+1 - v_t
    # A_t = delta_t + gamma * lambda * A_t+1
    rewards = jnp.sign(rewards) * (jnp.sqrt(rewards + 1) - 1 + 0.001 * rewards)
    values_t = values[:-1]
    values_t_p1 = values[1:]
    deltas = rewards + values_t_p1 * discount - values_t
    def f(carry, xs):
        delta, done = xs
        a_t_p1, _ = carry
        a_t = delta + a_t_p1 * lambda_ * discount * (1 - done)
        return (a_t, _), (a_t)
    
    a_init = (jnp.zeros((values.shape[1], 1)), None)
    dones = jnp.expand_dims(dones, axis=-1)
    _, gae = jax.lax.scan(f, a_init, (deltas, dones), reverse=True)
    value_target = gae + values_t
    vs_t_p1 = jnp.concatenate([value_target[1:], jnp.expand_dims(values[-1], 0)])
    
    advantages = (1 - dones) * (rewards + discount * vs_t_p1 - values_t)
    
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(value_target)

@eqx.filter_jit
def ppo_loss(ppo_network, trajectory, final_obs, discount, lambda_, eps, value_loss_coeff, entropy_loss_coeff):
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
    all_values = ppo_network.value_forward(all_observations)
    all_values = jnp.reshape(all_values, (timesteps + 1, batch_size, -1))
    values = all_values[:-1]
    advantages, value_target = generalized_advantage_estimate(all_values, rewards, dones,
                                                              discount, lambda_)
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-10)
    value_loss = jnp.mean((values - value_target) ** 2)
    
    policy_ratio = new_action_probs / old_action_probs
    policy_loss = -jnp.mean(jnp.minimum(policy_ratio * advantages, 
                                        jnp.clip(policy_ratio, 1-eps, 1+eps) * advantages))
    
    entropy_loss = jnp.mean(jnp.sum(action_dist * jnp.log(action_dist + 1e-10), axis=-1))
    total_loss = value_loss * value_loss_coeff + policy_loss + entropy_loss * entropy_loss_coeff
    metrics = {
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "policy_loss": policy_loss
    }
    return total_loss, metrics

def make_grad_step(model, loss_fn, optim, key, num_minibatches):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    def grad_step(ppo_network, trajectory, final_obs, discount, lambda_, eps, value_loss_coeff, entropy_loss_coeff):
        nonlocal opt_state, optim, key
        key, subkey = jax.random.split(key)
        model = ppo_network
        
        def create_minibatches(data):
            data = jax.random.permutation(subkey, data, axis=1)
            data = jnp.swapaxes(data, 0, 1)
            data = jnp.reshape(data, (num_minibatches, -1) + data.shape[1:])
            data = jnp.swapaxes(data, 1, 2)
            return data
            
        trajectory = jax.tree_util.tree_map(create_minibatches, trajectory)
        final_obs = jax.random.permutation(subkey, final_obs, axis=1)
        final_obs = jnp.reshape(final_obs, (num_minibatches, -1) + final_obs.shape[1:])
        
        # @eqx.filter_jit
        # def f(carry, data):
        #     model, opt_state = carry
        #     mini_traj, final = data
        #     loss_out, grads = eqx.filter_value_and_grad(loss_fn, allow_int=True, has_aux=True)(ppo_network, mini_traj, final, discount, lambda_, eps, value_loss_coeff, entropy_loss_coeff)
        #     updates, opt_state = optim.update(
        #         grads, opt_state, eqx.filter(model, eqx.is_array)
        #     )
        #     model = eqx.apply_updates(model, updates)
        #     return (model, opt_state), loss_out
        def f(i, model, opt_state):
            mini_traj = Trajectory(trajectory.observations[i], trajectory.actions[i], trajectory.action_probs[i], trajectory.rewards[i], trajectory.dones[i], trajectory.mask[i])
            final = final_obs[i]
            loss_out, grads = eqx.filter_value_and_grad(loss_fn, allow_int=True, has_aux=True)(ppo_network, mini_traj, final, discount, lambda_, eps, value_loss_coeff, entropy_loss_coeff)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return (model, opt_state), loss_out
        
        for i in range(num_minibatches):
            (model, opt_state), loss_out = f(i, model, opt_state)
        # (model, opt_state), loss_out = jax.lax.scan(f, (model, opt_state), (trajectory, final_obs))
        return loss_out, model
    return grad_step
    
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
    
class ToInt(ObservationWrapper):
    def __init__(self, nested_wrapper=None):
        self.nested_wrapper = nested_wrapper
        f = lambda x: jnp.astype(x, int)
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

def main(args):
    env = pgx.make("2048")
    
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    
    key = jax.random.key(args.seed)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, args.num_envs)
    
    state = init(keys)
    
    obs_wrapper = ToInt(FlattenObservation())
    ppo_network = PPONetwork(496, 4, key, obs_wrapper=obs_wrapper)
    optim = optax.adamw(args.lr)
    grad_step = make_grad_step(ppo_network, ppo_loss, optim)
    losses = []
    rewards = []
    for i in range(100000):
        state, _, traj = collect_trajectory(ppo_network, step, state, state.observation, key, args.rollout_length)
        for _ in range(1):
            (loss, metrics), ppo_network = grad_step(ppo_network, traj, state.observation, args.discount, args.lambda_, args.clip, 10, 0.01)
            losses.append(loss)
        rewards.append(jnp.mean(traj.rewards))
    plt.plot(losses)
    plt.show()
    plt.plot(rewards)
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--rollout_length', type=int, default=16)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    main(args)