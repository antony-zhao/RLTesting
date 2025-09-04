import jax
import jax.numpy as jnp
import equinox as eqx

orthogonal_initializer = jax.nn.initializers.orthogonal()
combine_dims = lambda x: jax.lax.collapse(x, 0, 2)

is_linear = lambda x: isinstance(x, eqx.nn.Linear)
is_conv = lambda x: isinstance(x, eqx.nn.Conv)

def orthogonal_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    return orthogonal_initializer(key, weight.shape, jnp.float32)

def init_linear_weight(model, init_fn, key):
    # from https://docs.kidger.site/equinox/tricks/#custom-parameter-initialisation
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model

def init_convnet_weights(model, init_fn, key):
    cond = lambda x: is_conv(x) or is_linear(x)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=cond)
                            if cond(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model

if __name__ == '__main__':
    test_model = eqx.nn.MLP(10, 50, 20, 3, key=jax.random.key(0))
    init_linear_weight(test_model, orthogonal_init, jax.random.key(0))