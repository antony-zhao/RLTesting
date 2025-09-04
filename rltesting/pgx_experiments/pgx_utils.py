import jax
import jax.numpy as jnp
import equinox as eqx
from rltesting.jax_rl.utils import Rollout
from rltesting.jax_rl.ppo import compute_gae

@eqx.filter_jit
def collect_rollout(ppo_network, step, state, current_obs, key, num_timesteps, obs_wrapper=None):
    # slightly inspired by the brax code, but just trying to implement from memory to learn
    def f(carry, _):
        state, current_obs, key = carry
        random_key, key = jax.random.split(key)
        next_state, next_obs, rollout = single_step(step, state, current_obs, ppo_network.policy_fn, random_key, obs_wrapper=obs_wrapper)
        return (next_state, next_obs, key), rollout
    
    (next_state, final_obs, _), rollout = jax.lax.scan(
        f, (state, current_obs, key), (), num_timesteps
    ) # note to self that scan returns ys in a stacked way.
    # try to stack final obs into the rollout, not sure if possible but worth a try
    if obs_wrapper is not None:
        final_obs = obs_wrapper(final_obs)
    return next_state, final_obs, rollout

@eqx.filter_jit
def single_step(step, state, current_obs, policy_fn, key, obs_wrapper=None):
    key, subkey = jax.random.split(key)
    batch_size = current_obs.shape[0]
    keys = jax.random.split(subkey, batch_size)
    if obs_wrapper is not None:
        current_obs = obs_wrapper(current_obs)
    key, subkey = jax.random.split(key)
    action, action_prob = policy_fn(current_obs, state.legal_action_mask, subkey)
    next_state = step(state, action, keys)  # pgx specifically, need to rewrite for other types of environments (especially non-jax ones)
    next_obs = next_state.observation
    rewards = next_state.rewards
    terminated = next_state.terminated
    truncated = next_state.truncated
    dones = jnp.bitwise_or(truncated, terminated)  # Probably a better way to handle this but this should suffice for now
    roll = Rollout(current_obs, action, action_prob, rewards, dones, state.legal_action_mask)
    return next_state, next_obs, roll

def make_eval_step(eval_init, eval_step_fn, eval_batch_size, key, obs_wrapper=None):
    @eqx.filter_jit 
    def eval_step(ppo_network):
        nonlocal key
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, eval_batch_size)

        eval_state = eval_init(keys)
        rewards = 0
        timesteps = 0

        def cond_fn(carry):
            eval_state, _, _, _ = carry
            return jnp.logical_not((eval_state.terminated | eval_state.truncated).all())
        
        def body_fn(carry):
            eval_state, rewards, timesteps, key = carry
            key, subkey, subkey2 = jax.random.split(key, 3)
            
            if obs_wrapper is not None:
                action, _ = ppo_network.policy_fn(obs_wrapper(eval_state.observation), eval_state.legal_action_mask, subkey, deterministic=True)
            else:
                action, _ = ppo_network.policy_fn(eval_state.observation, eval_state.legal_action_mask, subkey, deterministic=True)
            
            keys = jax.random.split(subkey2, eval_batch_size)
            eval_state = eval_step_fn(eval_state, action, keys)
            rewards += jnp.mean(eval_state.rewards)
            timesteps += 1
            
            return eval_state, rewards, timesteps, key
        
        carry = (eval_state, 0, 0, key)
        
        eval_state, rewards, timesteps, _ = jax.lax.while_loop(cond_fn, body_fn, carry)
        
        return rewards, timesteps
    return eval_step

def make_grad_step(model, loss_fn, optim, key, num_minibatches, num_training_steps):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    @eqx.filter_jit
    def grad_step(ppo_network, rollout, final_obs, discount, lambda_, eps, value_loss_coeff, entropy_loss_coeff):
        nonlocal opt_state, optim, key
        key, subkey = jax.random.split(key)
        
        observations, rewards, dones, actions, action_log_probs, mask = rollout.observations, rollout.rewards, \
        rollout.dones, rollout.actions, rollout.action_log_probs, rollout.mask
        rewards = jnp.sign(rewards) * (jnp.sqrt(rewards + 1) - 1 + 0.001 * rewards)
        timesteps, batch_size = observations.shape[:2]
        all_observations = jnp.concatenate([observations, jnp.expand_dims(final_obs, 0)]) 
        all_values = jax.vmap(ppo_network.value_forward)(all_observations)
        all_values = jnp.reshape(all_values, (timesteps + 1, batch_size, -1))
        advantages, value_target = compute_gae(all_values, rewards, dones, discount, lambda_)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-10)
        
        def create_minibatches(data):
            data = jax.random.permutation(subkey, data, axis=1)
            data = jnp.swapaxes(data, 0, 1)
            data = jnp.reshape(data, (num_minibatches, -1) + data.shape[1:])
            data = jnp.swapaxes(data, 1, 2)
            return data
        
        obs_batches = create_minibatches(observations)
        action_batches = create_minibatches(actions)
        log_prob_batches = create_minibatches(action_log_probs)
        mask_batches = create_minibatches(mask)
        value_target_batches = create_minibatches(value_target)
        adv_batches = create_minibatches(advantages)
        
        arr, static = eqx.partition(ppo_network, eqx.is_array)
        
        def f(carry, data):
            arr, opt_state = carry
            ppo_network = eqx.combine(arr, static)
            obs_batch, action_batch, log_prob_batch, mask_batch, value_target_batch, adv_batch = data
            loss_out, grads = eqx.filter_value_and_grad(loss_fn, allow_int=True, has_aux=True)\
            (ppo_network, obs_batch, action_batch, log_prob_batch, mask_batch, value_target_batch, adv_batch, eps, value_loss_coeff, entropy_loss_coeff)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(ppo_network, eqx.is_array)
            )
            ppo_network = eqx.apply_updates(ppo_network, updates)
            arr, _ = eqx.partition(ppo_network, eqx.is_array)
            return (arr, opt_state), loss_out
        
        loss = []
        for _ in range(num_training_steps):
            (arr, opt_state), loss_out = jax.lax.scan(f, (arr, opt_state), (obs_batches, action_batches, log_prob_batches, mask_batches, value_target_batches, adv_batches))
            loss_out = jax.tree_util.tree_map(jnp.mean, loss_out)
            loss.append(loss_out)
        loss = jax.tree_util.tree_map(jnp.mean, loss)
        ppo_network = eqx.combine(arr, static)
        return loss_out, ppo_network
    return grad_step
