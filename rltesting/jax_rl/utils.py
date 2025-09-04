import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable
from distreqx.distributions.categorical import Categorical
from rltesting.jax_rl.wrappers import ObservationWrapper

class Rollout(eqx.Module):
    observations: jax.Array
    actions: jax.Array
    action_log_probs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    mask: jax.Array

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
    
class ConvNetwork(eqx.Module):
    conv: eqx.Module
    mlp: eqx.Module
    
    @eqx.filter_jit
    def __call__(self, x):
        x = self.conv(x)
        return self.mlp(x)

class DiscretePPONetwork(eqx.Module):
    policy_network: MLP
    value_network: MLP
    obs_wrapper: ObservationWrapper
    
    def __init__(self, policy_network, value_network, obs_wrapper=None):
        self.policy_network = policy_network
        self.value_network = value_network
        self.obs_wrapper = obs_wrapper
    
    def policy_forward(self, x):
        if self.obs_wrapper is not None:
            x = self.obs_wrapper(x)
        action_logits = jax.vmap(self.policy_network)(x)
        return action_logits
    
    def value_forward(self, x):
        if self.obs_wrapper is not None:
            x = self.obs_wrapper(x)
        values = jax.vmap(self.value_network)(x)
        return values
    
    @eqx.filter_jit
    def policy_fn(self, x, legal_action_mask, key, deterministic=False):
        action_logits = self.policy_forward(x)
        action_logits = jnp.where(legal_action_mask, action_logits, -jnp.inf)
        if deterministic:
            return jnp.argmax(action_logits, -1), None
        else:
            actions_dist = Categorical(action_logits)
            actions = actions_dist.sample(key)
            action_log_probs = actions_dist.log_prob(actions)
        return actions.astype(jnp.int32), action_log_probs
