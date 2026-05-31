# Quick test script — save as test_nn_cacla.py and run
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, copy
from config import get_config, make_env, to_numpy, env_path
from train_cacla import load_autoencoder, compute_norm_stats, encode_single

cfg = get_config('pendulum')
env = make_env(cfg, mode='pixels')
model, idim = load_autoencoder(cfg)
means, stds = compute_norm_stats(env, model)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, action_dim), nn.Tanh())
    def forward(self, s):
        return self.actor(s) * 2.0, self.critic(s)  # scale to [-2, 2]

ac = ActorCritic(idim, 1).cuda()
opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
noise = 0.5

for ep in range(2000):
    obs, _ = env.reset()
    s = encode_single(obs, model, means, stds)
    ep_r = 0
    while True:
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            a_mean, _ = ac(st)
        a = (a_mean.cpu().numpy().squeeze() + np.random.randn() * noise).clip(-2, 2)

        obs2, r, term, trunc, _ = env.step(np.array([a]))
        done = term or trunc
        s2 = encode_single(obs2, model, means, stds)
        ep_r += r

        # TD update
        st2 = torch.tensor(s2, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            _, v2 = ac(st2)
            target = torch.tensor([r + (0.0 if term else 0.99 * v2.item())], dtype=torch.float32).cuda()

        a_pred, v = ac(st)
        critic_loss = F.mse_loss(v.squeeze(), target.squeeze())

        # Actor: policy gradient style
        td = target.item() - v.item()
        if td > 0:
            at = torch.tensor([[a]], dtype=torch.float32).cuda()
            actor_loss = F.mse_loss(a_pred, at)
        else:
            actor_loss = torch.tensor(0.0).cuda()

        loss = critic_loss + actor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if done: break
        s = s2

    noise = max(0.05, noise * 0.999)
    if (ep+1) % 50 == 0:
        print(f"Ep {ep+1:4d} | Avg: {ep_r:7.1f} | Noise: {noise:.3f}")