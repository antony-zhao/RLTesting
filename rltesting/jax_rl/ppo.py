import equinox as eqx
import jax
import jax.numpy as jnp
from rltesting.utils.jax_utils import combine_dims
from distreqx.distributions.categorical import Categorical

@jax.jit
def compute_gae(values, rewards, dones, discount, lambda_):
    # Also heavily inspired by brax implementation
    # assume observations include s_t-s_t+l (so both initial and the final obs) 
    # delta_t = r_t + gamma * v_t+1 - v_t
    # A_t = delta_t + gamma * lambda * A_t+1
    values_t = values[:-1]
    values_t_p1 = values[1:]
    dones = jnp.expand_dims(dones, axis=-1)
    deltas = rewards + values_t_p1 * discount * (1 - dones) - values_t
    def f(carry, xs):
        delta, done = xs
        a_t_p1, _ = carry
        a_t = delta + a_t_p1 * lambda_ * discount * (1 - done)
        return (a_t, _), (a_t)
    
    a_init = jnp.zeros((values.shape[1], 1))
    _, gae = jax.lax.scan(f, (a_init, None), (deltas, dones), reverse=True)
    value_target = gae + values_t
            
    return gae, value_target


@eqx.filter_jit
def ppo_loss(ppo_network, observations, actions, old_action_log_probs, mask, value_target, advantages, eps, value_loss_coeff, entropy_loss_coeff):
    # Value loss: L = (r_t + gamma * V(s_t+1) - V(s_t)) ^ 2
    # R_ratio = pi_theta(a_t|s_t) / pi_old(a_t | s_t)
    # Policy loss = min(R_ratio * advantage, clip(R_ratio, 1-eps, 1+eps) * advantage)
    # since observations have a shape of (timesteps, batch, (shape)), we have to modify it by rehsaping 
    # first two dims, either that or have to make a more annoying change with a double vmap.
    values = jax.vmap(ppo_network.value_forward)(observations)
    observations = combine_dims(observations)
    mask = combine_dims(mask)
    actions = combine_dims(actions)
    action_logits = ppo_network.policy_forward(observations)
    action_logits = jnp.where(mask, action_logits, -jnp.inf)
    actions_dist = Categorical(action_logits)
    new_action_log_probs = actions_dist.log_prob(actions)
    new_action_log_probs = jnp.reshape(new_action_log_probs, advantages.shape)
    old_action_log_probs = jnp.reshape(old_action_log_probs, advantages.shape)
    
    value_loss = jnp.mean((values - value_target) ** 2 / 2) 
    
    policy_ratio = jnp.exp(new_action_log_probs - old_action_log_probs)
    policy_loss = -jnp.mean(jnp.minimum(policy_ratio * advantages, 
                                        jnp.clip(policy_ratio, 1-eps, 1+eps) * advantages))
    
    entropy_loss = -actions_dist.entropy().mean()
    total_loss = value_loss * value_loss_coeff + policy_loss + entropy_loss * entropy_loss_coeff
    metrics = {
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "policy_loss": policy_loss
    }
    return total_loss, metrics
