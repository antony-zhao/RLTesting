"""
Quick NN actor-critic test on stage 1 (conv) latent representation.
Tests whether the 256-dim conv features support control, bypassing the intrinsic bottleneck.

Usage:
  python test_nn_latent.py --env lunarlander
  python test_nn_latent.py --env pendulum
  python test_nn_latent.py --env lunarlander --use-intrinsic
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import get_config, make_env, add_env_arg, to_numpy, env_path
from rltesting.fvd_experiments.fvd_models import (
    Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
)


def load_and_setup(cfg, use_intrinsic=False):
    """Load autoencoder and return encoder function + state dim."""
    path = env_path(cfg, "autoencoder.pt")
    ckpt = torch.load(path, weights_only=False)
    latent_dim = ckpt['latent_dim']
    intrinsic_dim = ckpt['intrinsic_dim']
    hidden_dim = ckpt.get('hidden_dim', 128)
    filter_base = ckpt.get('filter_base', 16)
    encoder_type = ckpt.get('encoder_type', 'conv')
    img_ch = ckpt.get('image_channels', cfg.get('image_channels', 1))

    if encoder_type == 'impala':
        encoder = IMPALAEncoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                                image_channels=img_ch, filter_base=filter_base).cuda()
        decoder = IMPALADecoder(encoder.conv_dim, cfg['framestack'], latent_dim,
                                image_channels=img_ch, filter_base=filter_base).cuda()
    else:
        encoder = Encoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                          image_channels=img_ch, filter_base=filter_base).cuda()
        decoder = Decoder(encoder.conv_dim, cfg['framestack'], latent_dim,
                          image_channels=img_ch, filter_base=filter_base).cuda()

    model = DoubleAutoEncoder(encoder, decoder, latent_dim,
                              intrinsic_dim, hidden_dim=hidden_dim).cuda()
    model.load_state_dict(ckpt['weights'])
    model.eval()

    if use_intrinsic:
        state_dim = intrinsic_dim
        print(f"Using intrinsic (double_encode): {state_dim} dims")
    else:
        state_dim = latent_dim
        print(f"Using stage 1 latent (encoder): {state_dim} dims")

    # Compute norm stats
    env_tmp = make_env(cfg, mode='pixels')
    all_z = []
    for _ in range(20):
        obs, _ = env_tmp.reset()
        done = False
        ep_obs = [np.array(obs)]
        while not done:
            obs, _, term, trunc, _ = env_tmp.step(env_tmp.action_space.sample())
            ep_obs.append(np.array(obs))
            done = term or trunc
        with torch.no_grad():
            batch = np.stack(ep_obs)
            t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
            if use_intrinsic:
                z = to_numpy(model.double_encode(t))
            else:
                z = to_numpy(model.encoder(t))
            all_z.append(z)
    all_z = np.concatenate(all_z, axis=0)
    means = all_z.mean(axis=0)
    stds = all_z.std(axis=0)
    stds[stds == 0] = 1.0
    env_tmp.close()

    def encode(obs):
        with torch.no_grad():
            t = (torch.tensor(np.array(obs)).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
            if use_intrinsic:
                z = to_numpy(model.double_encode(t)).squeeze()
            else:
                z = to_numpy(model.encoder(t)).squeeze()
            return ((z - means) / stds).astype(np.float32)

    return encode, state_dim


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, hidden=128):
        super().__init__()
        self.action_high = action_high
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh())

    def forward(self, s):
        a_high = torch.tensor(self.action_high, dtype=torch.float32, device=s.device)
        return self.actor(s) * a_high, self.critic(s)


def main():
    parser = argparse.ArgumentParser(description='NN actor-critic on learned latent')
    add_env_arg(parser)
    parser.add_argument('--use-intrinsic', action='store_true',
                        help='Use intrinsic (bottleneck) dims instead of stage 1 latent')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--noise-init', type=float, default=0.5)
    parser.add_argument('--noise-decay', type=float, default=0.999)
    parser.add_argument('--noise-min', type=float, default=0.05)
    args = parser.parse_args()

    cfg = get_config(args.env)
    env = make_env(cfg, mode='pixels')

    import gymnasium
    is_discrete = isinstance(env.action_space, gymnasium.spaces.Discrete)

    if is_discrete:
        num_actions = env.action_space.n
        print(f"Discrete action space: {num_actions} actions")
    else:
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low

    encode, state_dim = load_and_setup(cfg, use_intrinsic=args.use_intrinsic)

    if is_discrete:
        # Simple DQN-style: Q-network
        q_net = nn.Sequential(
            nn.Linear(state_dim, args.hidden), nn.ReLU(),
            nn.Linear(args.hidden, args.hidden), nn.ReLU(),
            nn.Linear(args.hidden, num_actions)).cuda()
        opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
        epsilon = 1.0

        print(f"\n=== NN DQN on {'intrinsic' if args.use_intrinsic else 'stage1 latent'} — {cfg['env_name']} ===")
        print(f"  state_dim={state_dim}, num_actions={num_actions}")
        print(f"  hidden={args.hidden}, lr={args.lr}")

        returns = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            s = encode(obs)
            ep_r = 0

            while True:
                st = torch.tensor(s).unsqueeze(0).cuda()
                with torch.no_grad():
                    q_vals = q_net(st)

                # Epsilon-greedy
                if np.random.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = q_vals.argmax(dim=-1).item()

                obs2, r, term, trunc, _ = env.step(a)
                done = term or trunc
                s2 = encode(obs2)
                ep_r += r

                # TD update
                st2 = torch.tensor(s2).unsqueeze(0).cuda()
                with torch.no_grad():
                    q_next = q_net(st2).max(dim=-1).values
                    target = r + (0.0 if term else 0.99 * q_next.item())

                q_pred = q_net(st)[0, a]
                loss = F.mse_loss(q_pred, torch.tensor(target, dtype=torch.float32).cuda())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if done:
                    break
                s = s2

            epsilon = max(0.05, epsilon * 0.998)
            returns.append(ep_r)

            if (ep + 1) % 50 == 0:
                avg = np.mean(returns[-50:])
                print(f"Ep {ep+1:4d} | Avg(50): {avg:7.1f} | Last: {ep_r:7.1f} | Eps: {epsilon:.3f}")

    else:
        # Continuous: CACLA-style
        ac = ActorCritic(state_dim, action_dim, action_high, hidden=args.hidden).cuda()
        opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
        noise = args.noise_init

        print(f"\n=== NN CACLA on {'intrinsic' if args.use_intrinsic else 'stage1 latent'} — {cfg['env_name']} ===")
        print(f"  state_dim={state_dim}, action_dim={action_dim}")
        print(f"  action_range=[{action_low}, {action_high}]")
        print(f"  hidden={args.hidden}, lr={args.lr}")
        print(f"  noise={args.noise_init}, decay={args.noise_decay}, min={args.noise_min}")

        returns = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            s = encode(obs)
            ep_r = 0

            while True:
                st = torch.tensor(s).unsqueeze(0).cuda()
                with torch.no_grad():
                    a_mean, _ = ac(st)
                a = a_mean.cpu().numpy().squeeze()
                a = (a + np.random.randn(action_dim) * noise * action_high).clip(action_low, action_high)

                obs2, r, term, trunc, _ = env.step(a)
                done = term or trunc
                s2 = encode(obs2)
                ep_r += r

                # TD target
                st2 = torch.tensor(s2).unsqueeze(0).cuda()
                with torch.no_grad():
                    _, v2 = ac(st2)
                    target = torch.tensor(
                        [r + (0.0 if term else 0.99 * v2.item())],
                        dtype=torch.float32).cuda()

                a_pred, v = ac(st)
                critic_loss = F.mse_loss(v.squeeze(), target.squeeze())

                td = target.item() - v.item()
                if td > 0:
                    at = torch.tensor(a, dtype=torch.float32).unsqueeze(0).cuda()
                    actor_loss = F.mse_loss(a_pred, at)
                else:
                    actor_loss = torch.tensor(0.0).cuda()

                loss = critic_loss + actor_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                if done:
                    break
                s = s2

            noise = max(args.noise_min, noise * args.noise_decay)
            returns.append(ep_r)

            if (ep + 1) % 50 == 0:
                avg = np.mean(returns[-50:])
                print(f"Ep {ep+1:4d} | Avg(50): {avg:7.1f} | Last: {ep_r:7.1f} | Noise: {noise:.3f}")

    print(f"\nFinal avg (last 100): {np.mean(returns[-100:]):.1f}")
    env.close()


if __name__ == '__main__':
    main()