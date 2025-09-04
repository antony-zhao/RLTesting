import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Callable
import pgx
from pgx.experimental import auto_reset
from matplotlib import pyplot as plt
from rltesting.utils.logger import Logger
from rltesting.utils.jax_utils import init_convnet_weights, orthogonal_init
from rltesting.jax_rl.ppo import ppo_loss
from rltesting.jax_rl.wrappers import TransposeObservation, ToDtype
from rltesting.jax_rl.utils import MLP, ConvNetwork, DiscretePPONetwork
from rltesting.pgx_experiments.pgx_utils import collect_rollout, make_eval_step, make_grad_step
    
class Conv2048(eqx.Module):
    # essentially just a bunch of residual 2x2 blocks
    input_layer: eqx.Module
    output_layer: eqx.Module
    layers: list[eqx.Module]
    act: Callable
    output_dim: int
    
    def __init__(self, key, num_convs=5, act=jax.nn.selu, channels=128):
        self.act = act
        self.layers = []
        key, subkey = jax.random.split(key)
        self.input_layer = eqx.nn.Conv2d(31, channels, 3, 1, padding='same', key=subkey)
        for _ in range(num_convs):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Conv2d(channels, channels, 2, 1, padding='same', key=subkey))
        key, subkey = jax.random.split(key)
        self.output_layer = eqx.nn.Conv2d(channels, channels, 2, 2, padding='same', key=subkey)
        self.output_dim = channels * 4
        
    @eqx.filter_jit
    def __call__(self, x):
        x = x_skip = self.act(self.input_layer(x))
        for i in range(len(self.layers)):
            x = x_skip = self.act(self.layers[i](x)) + x_skip
        x = self.act(self.output_layer(x))
        return x.flatten()

def main(args):
    import numpy as np
    import time
    env_id = "2048"
    env = pgx.make(env_id)
    batch_size = args.num_envs
    eval_batch_size = args.num_eval_envs

    init = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    eval_init = jax.jit(jax.vmap(env.init))
    eval_step_fn = jax.jit(jax.vmap(env.step))

    key = jax.random.key(args.seed)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    
    timesteps = args.rollout_length
    minibatches = args.num_minibatches
    epochs = args.epochs
    eval_every = args.eval_every
    train_steps = args.train_steps

    obs_wrapper = ToDtype(float, TransposeObservation((2, 0, 1)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    conv1 = Conv2048(subkey1)
    policy_network = ConvNetwork(conv1, MLP(conv1.output_dim, 4, subkey2))
    key, subkey1, subkey2 = jax.random.split(key, 3)
    conv2 = Conv2048(subkey1)
    value_network = ConvNetwork(conv2, MLP(conv2.output_dim, 1, subkey2))
    ppo_network = DiscretePPONetwork(policy_network, value_network)

    key, subkey = jax.random.split(key)
    eval_step = make_eval_step(eval_init, eval_step_fn, eval_batch_size, subkey, obs_wrapper)
    state = init(keys)
    state, _, _ = collect_rollout(ppo_network, step_fn, state, state.observation, key, args.init_steps, obs_wrapper=obs_wrapper)
    key, subkey = jax.random.split(key)
    optim = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(args.lr))
    grad_step = make_grad_step(ppo_network, ppo_loss, optim, subkey, minibatches, train_steps)
    
    logger = Logger('logs')
    losses = []
    value_losses = []
    policy_losses = []
    entropy_losses = []
    train_rewards = []
    eval_rewards = []
    eval_timesteps = []
    start_time = time.time()
    for i in range(epochs + 1):
        if i % eval_every == 0:
            eval_reward, eval_length = eval_step(ppo_network)
            print(f"epoch {i}: reward: {eval_reward}, episode length: {eval_length}")
            eval_rewards.append(eval_reward)
            eval_timesteps.append(eval_length)
            logger.add_scalar("eval_reward", eval_reward)
            logger.add_scalar("eval_length", eval_length)
        state, final_obs, roll = collect_rollout(ppo_network, step_fn, state, state.observation, key, timesteps, obs_wrapper=obs_wrapper)
        (loss, metrics), ppo_network = grad_step(ppo_network, roll, final_obs, args.discount, args.lambda_, args.clip, args.val_coef, args.ent_coef)
        losses.append(np.array(loss))
        value_losses.append(np.array(metrics['value_loss']))
        policy_losses.append(np.array(metrics['policy_loss']))
        entropy_losses.append(np.array(metrics['entropy_loss']))
        train_rewards.append(np.array(jnp.sum(roll.rewards) / jnp.sum(roll.dones)))
        logger.add_scalar("loss", np.array(loss))
        logger.add_scalar("value_loss", np.array(metrics['value_loss']))
        logger.add_scalar("policy_loss", np.array(metrics['policy_loss']))
        logger.add_scalar("entropy_loss", np.array(metrics['entropy_loss']))
        logger.add_scalar("train_reward", np.array(jnp.sum(roll.rewards) / jnp.sum(roll.dones)))
        logger.write(i * batch_size * timesteps)
        logger.add_scalar("epochs over time", i)
        logger.write(time.time() - start_time)

    fig, axs = plt.subplots(2, 4)
    fig.set_figwidth(20)
    fig.set_figheight(10)
    axs[0, 0].plot(losses)
    axs[0, 0].set_title("Overall Loss")
    axs[0, 1].plot(entropy_losses)
    axs[0, 1].set_title("Entropy Loss")
    axs[0, 2].plot(policy_losses)
    axs[0, 2].set_title("Policy Loss")
    axs[0, 3].plot(value_losses)
    axs[0, 3].set_title("Value Loss")
    axs[1, 0].plot(train_rewards)
    axs[1, 0].set_title("Train Reward")
    eval_axis = np.arange(len(eval_rewards)) * eval_every
    axs[1, 1].plot(eval_axis, eval_rewards)
    axs[1, 1].set_title("Eval Reward")
    axs[1, 2].plot(eval_axis, eval_timesteps)
    axs[1, 2].set_title("Eval Episode Length")
    fig.savefig(f"{env_id}_plot.png")
    print(metrics)
    
    render_batch_size = 4

    render_step_fn = jax.jit(jax.vmap(env.step))

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, render_batch_size)

    states = []
    state = init(keys)
    states.append(state)
    entropy_losses = 0

    while not (state.terminated | state.truncated).all():
        key, subkey, subkey2 = jax.random.split(key, 3)
        action, _ = ppo_network.policy_fn(obs_wrapper(state.observation), state.legal_action_mask, subkey, True)
        keys = jax.random.split(subkey2, render_batch_size)
        state = render_step_fn(state, action, keys)
        states.append(state)
        entropy_losses += jnp.mean(state.rewards)
        
    print('saving animation')
    pgx.save_svg_animation(states, f"{env_id}.svg", frame_duration_seconds=0.05)
    eqx.tree_serialise_leaves('2048ppo.eqx', ppo_network)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--val_coef', type=float, default=0.5)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--rollout_length', type=int, default=32)
    parser.add_argument('--num_envs', type=int, default=8192)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--init_steps', type=int, default=100)
    parser.add_argument('--train_steps', type=int, default=4)
    args = parser.parse_args()
    main(args)
    