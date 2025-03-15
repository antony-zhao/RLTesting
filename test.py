import jax
import pgx
from pgx.experimental import act_randomly

print(f"{pgx.__version__=}")

env = pgx.make("chess")

init = jax.jit(jax.vmap(env.init))  # vectorize and JIT-compile
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)

batch_size = 9

# prepare PRNGKeys
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)

state = init(keys)  # vectorized states
while not (state.terminated | state.truncated).all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state.legal_action_mask)
    # action = model(state.current_player, state.observation, state.legal_action_mask)
    state = step(state, action)  # state.reward (2,)
    
state