# test_nn_cacla_latent.py — actor-critic on stage 1 latent (pre-bottleneck)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, copy
from config import get_config, make_env, to_numpy, env_path
from rltesting.fvd_experiments.fvd_models import Encoder, Decoder, DoubleAutoEncoder

cfg = get_config('pendulum')
env = make_env(cfg, mode='pixels')

# Load autoencoder but use stage 1 encoder only
path = env_path(cfg, "autoencoder.pt")
ckpt = torch.load(path, weights_only=False)
latent_dim = ckpt['latent_dim']  # 256
encoder = Encoder(cfg['framestack'], latent_dim, cfg['obs_shape'],
                  image_channels=cfg.get('image_channels', 1),
                  filter_base=ckpt.get('filter_base', 16)).cuda()

# Load just encoder weights
model = DoubleAutoEncoder(encoder, 
    Decoder(encoder.conv_dim, cfg['framestack'], latent_dim,
            image_channels=cfg.get('image_channels', 1),
            filter_base=ckpt.get('filter_base', 16)).cuda(),
    latent_dim, ckpt['intrinsic_dim'], hidden_dim=ckpt.get('hidden_dim', 128)).cuda()
model.load_state_dict(ckpt['weights'])
model.eval()

# Norm stats for stage 1 latent
all_z = []
for _ in range(20):
    obs, _ = env.reset()
    done = False
    ep_obs = [np.array(obs)]
    while not done:
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        ep_obs.append(np.array(obs))
        done = term or trunc
    with torch.no_grad():
        batch = np.stack(ep_obs)
        t = (torch.tensor(batch).float().transpose(-3, -1) / 255).cuda()
        z = to_numpy(model.encoder(t))  # stage 1 latent, not double_encode
        all_z.append(z)
all_z = np.concatenate(all_z, axis=0)
means = all_z.mean(axis=0)
stds = all_z.std(axis=0)
stds[stds == 0] = 1.0

def encode(obs):
    with torch.no_grad():
        t = (torch.tensor(np.array(obs)).unsqueeze(0).float().transpose(-3, -1) / 255).cuda()
        z = to_numpy(model.encoder(t)).squeeze()
        return (z - means) / stds

state_dim = latent_dim  # 256
print(f"Using stage 1 latent: {state_dim} dims")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, action_dim), nn.Tanh())
    def forward(self, s):
        return self.actor(s) * 2.0, self.critic(s)

ac = ActorCritic(state_dim, 1, hidden=128).cuda()
opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
noise = 0.5

for ep in range(2000):
    obs, _ = env.reset()
    s = encode(obs)
    ep_r = 0
    while True:
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            a_mean, _ = ac(st)
        a = (a_mean.cpu().numpy().squeeze() + np.random.randn() * noise).clip(-2, 2)

        obs2, r, term, trunc, _ = env.step(np.array([a]))
        done = term or trunc
        s2 = encode(obs2)
        ep_r += r

        st2 = torch.tensor(s2, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            _, v2 = ac(st2)
            target = torch.tensor([r + (0.0 if term else 0.99 * v2.item())], dtype=torch.float32).cuda()

        a_pred, v = ac(st)
        critic_loss = F.mse_loss(v.squeeze(), target.squeeze())

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
        print(f"Ep {ep+1:4d} | Return: {ep_r:7.1f} | Noise: {noise:.3f}")