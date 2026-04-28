"""
Train a dual-stage autoencoder on Atari observations.
Stage 1: Conv encoder + decoder (image reconstruction)
Stage 2: DoubleAutoEncoder (intrinsic dim compression)
Optional: Forward prediction loss to encode dynamics-relevant features.

Usage: python train_autoencoder.py --env boxing
       python train_autoencoder.py --env boxing --forward-prediction
       python train_autoencoder.py --env boxing --forward-prediction --fp-coef 0.5 --pred-coef 0.1
       python train_autoencoder.py --env boxing --forward-prediction --direct-grads  # old style
       python train_autoencoder.py --env boxing --intrinsic-forward                  # forward pred in stage 2
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skdim
from matplotlib import pyplot as plt

from config import get_config, make_env, add_env_arg, to_numpy
from rltesting.fvd_experiments.fvd_models import (
    Encoder, Decoder, DoubleAutoEncoder, IMPALAEncoder, IMPALADecoder
)
from rltesting.torch_rl.utils import random_sample_single_env
from rltesting.torch_rl.buffers import ReplayBuffer


# ---- Forward prediction model ---- #

class ForwardPredictor(nn.Module):
    """Predicts z_{t+1} from (z_t, a_t). Action is one-hot concatenated."""
    def __init__(self, latent_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, action):
        """
        z: (B, latent_dim) — current latent
        action: (B,) — integer actions
        Returns: (B, latent_dim) — predicted next latent
        """
        a_onehot = F.one_hot(action.long(), self.num_actions).float()
        x = torch.cat([z, a_onehot], dim=-1)
        return self.net(x)


def weighted_reconstruction_loss(reconstruction, target, bg_weight=1.0, fg_weight=10.0):
    """Upweight non-background pixels in the MSE loss."""
    with torch.no_grad():
        B, C, H, W = target.shape
        bg = target.reshape(B, C, -1).median(dim=-1).values[:, :, None, None]
        diff = (target - bg).abs().sum(dim=1, keepdim=True)
        mask = (diff > 0.05).float()
        weights = bg_weight + mask * (fg_weight - bg_weight)
    loss = (reconstruction - target) ** 2 * weights
    return loss.mean()


# ---- Data collection ---- #

def collect_data(env, cfg):
    """Collect random-policy data into a replay buffer."""
    img_ch = cfg.get('image_channels', 3)
    buffer_shapes = [(cfg['obs_shape'], cfg['obs_shape'], img_ch * cfg['framestack']), (), (), ()]
    dtypes = [np.uint8, np.float32, np.float32, np.float32]
    buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=cfg['data_steps'])

    filepath = f"{cfg['env_name']}_buffer.npz"
    if os.path.isfile(filepath):
        print(f"Loading buffer from {filepath}")
        buffer.load(filepath)
    else:
        print(f"Collecting {cfg['data_steps']} steps of random data...")
        samples = random_sample_single_env(env, num_steps=cfg['data_steps'])
        for i in range(cfg['data_steps']):
            buffer.add_sample([samples[j][i] for j in range(len(samples))].copy())
        buffer.save(filepath=filepath)
        print(f"Saved buffer to {filepath}")
    return buffer


def prep_batch(buffer, batch_size):
    """Sample and preprocess a batch of observations."""
    samples = buffer.sample(batch_size)
    obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).cuda()
    return obs


def prep_transition_batch(buffer, batch_size):
    """Sample consecutive (obs_t, action_t, obs_{t+1}) pairs using buffer's seq_len."""
    samples = buffer.sample(batch_size, seq_len=2)
    obs_t = (torch.tensor(samples[0][:, 0]).float().transpose(-3, -1) / 255).cuda()
    actions = torch.tensor(samples[1][:, 0]).long().cuda()
    obs_tp1 = (torch.tensor(samples[0][:, 1]).float().transpose(-3, -1) / 255).cuda()
    return obs_t, actions, obs_tp1


# ---- Training stages ---- #

def train_stage1(encoder, decoder, buffer, cfg, use_weighted_loss=False):
    """Train the convolutional autoencoder."""
    print("\n=== Stage 1: Conv Autoencoder ===")
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), cfg['ae_lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg['ae_scheduler_gamma'])

    epochs = cfg['stage1_epochs']
    spe = cfg['stage1_steps_per_epoch']
    losses = []
    for epoch in range(1, epochs + 1):
        for _ in range(spe):
            obs = prep_batch(buffer, cfg['batch_size'])
            latent = encoder(obs)
            reconstruction = decoder(latent)
            if use_weighted_loss:
                loss = weighted_reconstruction_loss(reconstruction, obs)
            else:
                loss = F.mse_loss(reconstruction, obs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs} | "
                  f"Loss: {np.mean(losses[-spe:]):.5f}")

    plt.figure()
    plt.plot(losses)
    plt.title('Stage 1 Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.savefig(f"{cfg['env_name']}_stage1_loss.png")
    plt.close()
    return losses


def train_stage1_with_forward(encoder, decoder, forward_model, buffer, cfg,
                               fp_coef=0.5, pred_coef=0.1,
                               use_weighted_loss=False, split_gradients=True):
    """Train conv autoencoder with forward prediction auxiliary loss.

    Two modes controlled by split_gradients:

    split_gradients=True (DreamerV3-style, default):
      - Dynamics loss:        train forward model only (encoder detached)
      - Predictability loss:  train encoder only (forward model detached)
      Encoder is mostly driven by reconstruction with a gentle nudge
      toward predictability. Avoids collapsing the latent space.

    split_gradients=False (direct):
      - Forward loss flows directly through z_t into the encoder.
      Simpler but can degrade reconstruction quality.
    """
    mode = "split (DreamerV3-style)" if split_gradients else "direct"
    print(f"\n=== Stage 1: Conv AE + Forward Prediction ({mode}) ===")
    print(f"  fp_coef={fp_coef}, pred_coef={pred_coef}")

    all_params = (list(encoder.parameters()) + list(decoder.parameters())
                  + list(forward_model.parameters()))
    opt = torch.optim.Adam(all_params, cfg['ae_lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg['ae_scheduler_gamma'])

    recon_losses, dyn_losses, pred_losses = [], [], []
    for epoch in range(1, cfg['stage1_epochs'] + 1):
        for _ in range(cfg['stage1_steps_per_epoch']):
            # Sample transitions — use obs_t for reconstruction too
            obs_t, actions, obs_tp1 = prep_transition_batch(buffer, cfg['batch_size'])

            # Two encoder passes total (z_t and z_tp1)
            z_t = encoder(obs_t)
            z_tp1 = encoder(obs_tp1)

            # Reconstruction from z_t
            reconstruction = decoder(z_t)
            if use_weighted_loss:
                loss_recon = weighted_reconstruction_loss(reconstruction, obs_t)
            else:
                loss_recon = F.mse_loss(reconstruction, obs_t)

            # Forward prediction
            if split_gradients:
                # Dynamics: train forward model only
                z_tp1_pred = forward_model(z_t.detach(), actions)
                loss_dyn = F.mse_loss(z_tp1_pred, z_tp1.detach())

                # Predictability: train encoder only (through z_tp1)
                loss_pred = F.mse_loss(z_tp1, z_tp1_pred.detach())

                loss_fwd = fp_coef * loss_dyn + pred_coef * loss_pred
            else:
                with torch.no_grad():
                    z_tp1_target = z_tp1.detach()
                z_tp1_pred = forward_model(z_t, actions)
                loss_dyn = F.mse_loss(z_tp1_pred, z_tp1_target)
                loss_pred = torch.tensor(0.0)
                loss_fwd = fp_coef * loss_dyn

            loss = loss_recon + loss_fwd
            loss.backward()
            opt.step()
            opt.zero_grad()

            recon_losses.append(loss_recon.item())
            dyn_losses.append(loss_dyn.item())
            pred_losses.append(loss_pred.item() if split_gradients else 0.0)

        scheduler.step()
        if epoch % 10 == 0:
            spe = cfg['stage1_steps_per_epoch']
            log = (f"  Epoch {epoch}/{cfg['stage1_epochs']} | "
                   f"Recon: {np.mean(recon_losses[-spe:]):.5f} | "
                   f"Dynamics: {np.mean(dyn_losses[-spe:]):.5f}")
            if split_gradients:
                log += f" | Predict: {np.mean(pred_losses[-spe:]):.5f}"
            print(log)

    fig, axes = plt.subplots(1, 3 if split_gradients else 2, figsize=(16, 4))
    axes[0].plot(recon_losses); axes[0].set_title('Reconstruction Loss')
    axes[1].plot(dyn_losses); axes[1].set_title('Dynamics Loss')
    if split_gradients:
        axes[2].plot(pred_losses); axes[2].set_title('Predictability Loss')
    for ax in axes:
        ax.set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(f"{cfg['env_name']}_stage1_fwd_loss.png")
    plt.close()
    return recon_losses, dyn_losses


def estimate_intrinsic_dim(encoder, buffer, cfg):
    """Estimate intrinsic dimensionality via MLE."""
    print("\n=== Estimating Intrinsic Dimensionality ===")
    latent_ids = []
    for _ in range(20):
        obs = prep_batch(buffer, 2048)
        with torch.no_grad():
            latent = encoder(obs)
        unique = np.unique(to_numpy(latent), axis=0)
        latent_ids.append(skdim.id.MLE().fit(unique).dimension_)

    mean_id = np.mean(latent_ids)
    std_id = np.std(latent_ids)
    intrinsic_dim = round(mean_id * cfg['id_multiplier'])
    print(f"  MLE estimate: {mean_id:.1f} +/- {std_id:.1f}")
    print(f"  Using intrinsic_dim = round({mean_id:.1f} * {cfg['id_multiplier']}) = {intrinsic_dim}")
    return intrinsic_dim


def train_stage2(model, buffer, cfg, use_weighted_loss=False):
    """Train the full DoubleAutoEncoder (intrinsic compression)."""
    print("\n=== Stage 2: Double Autoencoder ===")
    opt = torch.optim.AdamW(model.parameters(), cfg['ae_lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg['ae_scheduler_gamma'])

    epochs = cfg['stage2_epochs']
    spe = cfg['stage2_steps_per_epoch']
    losses = []
    for epoch in range(1, epochs + 1):
        for _ in range(spe):
            obs = prep_batch(buffer, cfg['batch_size'])
            intrinsic = model.double_encode(obs)
            reconstruction = model.double_decode(intrinsic)
            if use_weighted_loss:
                loss = weighted_reconstruction_loss(reconstruction, obs)
            else:
                loss = F.mse_loss(reconstruction, obs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs} | "
                  f"Loss: {np.mean(losses[-spe:]):.5f}")

    plt.figure()
    plt.plot(losses)
    plt.title('Stage 2 Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.savefig(f"{cfg['env_name']}_stage2_loss.png")
    plt.close()
    return losses


def train_stage2_with_forward(model, forward_model, buffer, cfg,
                               fp_coef=0.5, pred_coef=0.1,
                               use_weighted_loss=False):
    """Train DoubleAutoEncoder with forward prediction in intrinsic space.

    DreamerV3-style split gradients:
      - Dynamics loss:       train forward model to predict next intrinsic (encoder detached)
      - Predictability loss: nudge encoder to produce predictable intrinsics (forward model detached)
      - Reconstruction loss: main training signal for the full autoencoder

    The forward model operates on the compact intrinsic space, making it
    fast and focused on control-relevant dynamics.
    """
    print(f"\n=== Stage 2: Double AE + Intrinsic Forward Prediction ===")
    print(f"  fp_coef={fp_coef}, pred_coef={pred_coef}")

    all_params = list(model.parameters()) + list(forward_model.parameters())
    opt = torch.optim.AdamW(all_params, cfg['ae_lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg['ae_scheduler_gamma'])

    recon_losses, dyn_losses, pred_losses = [], [], []
    for epoch in range(1, cfg['stage2_epochs'] + 1):
        for _ in range(cfg['stage2_steps_per_epoch']):
            # Sample transitions
            obs_t, actions, obs_tp1 = prep_transition_batch(buffer, cfg['batch_size'])

            # Encode both frames to intrinsic space
            z_t = model.double_encode(obs_t)
            z_tp1 = model.double_encode(obs_tp1)

            # Reconstruction loss (from z_t)
            reconstruction = model.double_decode(z_t)
            if use_weighted_loss:
                loss_recon = weighted_reconstruction_loss(reconstruction, obs_t)
            else:
                loss_recon = F.mse_loss(reconstruction, obs_t)

            # Dynamics loss: train forward model only
            z_tp1_pred = forward_model(z_t.detach(), actions)
            loss_dyn = F.mse_loss(z_tp1_pred, z_tp1.detach())

            # Predictability loss: nudge encoder only
            loss_pred = F.mse_loss(z_tp1, z_tp1_pred.detach())

            loss = loss_recon + fp_coef * loss_dyn + pred_coef * loss_pred
            loss.backward()
            opt.step()
            opt.zero_grad()

            recon_losses.append(loss_recon.item())
            dyn_losses.append(loss_dyn.item())
            pred_losses.append(loss_pred.item())

        scheduler.step()
        if epoch % 10 == 0:
            spe = cfg['stage2_steps_per_epoch']
            print(f"  Epoch {epoch}/{cfg['stage2_epochs']} | "
                  f"Recon: {np.mean(recon_losses[-spe:]):.5f} | "
                  f"Dynamics: {np.mean(dyn_losses[-spe:]):.5f} | "
                  f"Predict: {np.mean(pred_losses[-spe:]):.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(recon_losses); axes[0].set_title('Reconstruction Loss')
    axes[1].plot(dyn_losses); axes[1].set_title('Dynamics Loss')
    axes[2].plot(pred_losses); axes[2].set_title('Predictability Loss')
    for ax in axes:
        ax.set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(f"{cfg['env_name']}_stage2_fwd_loss.png")
    plt.close()
    return recon_losses, dyn_losses


def visualize_reconstructions(model, buffer, cfg, n=4, stage='stage2'):
    """Save side-by-side original vs reconstruction."""
    obs = prep_batch(buffer, n)
    with torch.no_grad():
        if stage == 'stage1':
            # Encoder → Decoder directly (no intrinsic bottleneck)
            recon = model.decoder(model.encoder(obs))
        else:
            intrinsic = model.double_encode(obs)
            recon = model.double_decode(intrinsic)

    fig, axs = plt.subplots(n, 2, figsize=(4, 2 * n))
    for i in range(n):
        orig = to_numpy(obs[i].transpose(-3, -1))[:, :, :3]
        rec = to_numpy(recon[i].transpose(-3, -1))[:, :, :3]
        axs[i][0].imshow(orig)
        axs[i][0].set_title('Original' if i == 0 else '')
        axs[i][0].axis('off')
        axs[i][1].imshow(rec)
        title = 'Stage 1 Recon' if stage == 'stage1' else 'Double AE Recon'
        axs[i][1].set_title(title if i == 0 else '')
        axs[i][1].axis('off')
    plt.tight_layout()
    fname = f"{cfg['env_name']}_{stage}_reconstructions.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {stage} reconstructions to {fname}")


def save_checkpoint(model, intrinsic_dim, cfg, forward_model=None):
    """Save the trained model."""
    path = f"{cfg['env_name']}_autoencoder.pt"
    data = {
        'weights': model.state_dict(),
        'intrinsic_dim': intrinsic_dim,
        'latent_dim': cfg['latent_dim'],
        'hidden_dim': cfg['hidden_dim'],
        'framestack': cfg['framestack'],
        'obs_shape': cfg['obs_shape'],
        'filter_base': cfg['filter_base'],
        'encoder_type': cfg.get('encoder_type', 'conv'),
        'image_channels': cfg.get('image_channels', 3),
    }
    if forward_model is not None:
        data['forward_model'] = forward_model.state_dict()
    torch.save(data, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train dual-stage autoencoder')
    add_env_arg(parser)
    parser.add_argument('--force', action='store_true', help='Retrain even if checkpoint exists')
    parser.add_argument('--forward-prediction', action='store_true',
                        help='Enable forward prediction auxiliary loss in stage 1')
    parser.add_argument('--intrinsic-forward', action='store_true',
                        help='Enable forward prediction in intrinsic space during stage 2')
    parser.add_argument('--fp-coef', type=float, default=0.5,
                        help='Dynamics loss coefficient (default: 0.5)')
    parser.add_argument('--pred-coef', type=float, default=0.1,
                        help='Predictability loss coefficient, only used with split grads (default: 0.1)')
    parser.add_argument('--direct-grads', action='store_true',
                        help='Use direct forward gradients instead of DreamerV3-style split')
    parser.add_argument('--weighted-loss', action='store_true',
                        help='Upweight foreground pixels in reconstruction loss')
    parser.add_argument('--impala', action='store_true',
                        help='Use IMPALA-style residual encoder/decoder')
    args = parser.parse_args()

    cfg = get_config(args.env)
    ckpt_path = f"{cfg['env_name']}_autoencoder.pt"

    if os.path.isfile(ckpt_path) and not args.force:
        print(f"Checkpoint {ckpt_path} already exists. Use --force to retrain.")
        return

    np.random.seed(42)
    torch.manual_seed(0)

    env = make_env(cfg)

    import time
    t0 = time.time()
    buffer = collect_data(env, cfg)
    t_data = time.time() - t0
    print(f"  Data collection: {t_data:.1f}s")

    # Stage 1
    use_impala = args.impala or cfg.get('encoder_type') == 'impala'
    img_ch = cfg.get('image_channels', 3)
    if use_impala:
        cfg['encoder_type'] = 'impala'
        print("Using IMPALA-style encoder/decoder")
        encoder = IMPALAEncoder(cfg['framestack'], cfg['latent_dim'], cfg['obs_shape'],
                                image_channels=img_ch,
                                filter_base=cfg['filter_base']).cuda()
        decoder = IMPALADecoder(encoder.conv_dim, cfg['framestack'], cfg['latent_dim'],
                                image_channels=img_ch,
                                filter_base=cfg['filter_base']).cuda()
    else:
        print("Using standard conv encoder/decoder")
        encoder = Encoder(cfg['framestack'], cfg['latent_dim'], cfg['obs_shape'],
                          image_channels=img_ch,
                          filter_base=cfg['filter_base']).cuda()
        decoder = Decoder(encoder.conv_dim, cfg['framestack'], cfg['latent_dim'],
                          image_channels=img_ch,
                          filter_base=cfg['filter_base']).cuda()

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Encoder: {enc_params:,} params | Decoder: {dec_params:,} params")
    print(f"  Conv dim: {encoder.conv_dim}")

    forward_model = None
    t0 = time.time()
    if args.forward_prediction:
        # Determine encoder output dim
        latent_dim_actual = (np.prod(encoder.conv_dim)
                             if cfg['latent_dim'] is None else cfg['latent_dim'])
        forward_model = ForwardPredictor(
            latent_dim_actual, cfg['num_actions'], hidden_dim=256).cuda()
        train_stage1_with_forward(encoder, decoder, forward_model, buffer, cfg,
                                   fp_coef=args.fp_coef,
                                   pred_coef=args.pred_coef,
                                   use_weighted_loss=args.weighted_loss,
                                   split_gradients=not args.direct_grads)
    else:
        train_stage1(encoder, decoder, buffer, cfg,
                     use_weighted_loss=args.weighted_loss)
    t_stage1 = time.time() - t0
    print(f"  Stage 1 training: {t_stage1:.1f}s")

    # Estimate intrinsic dim
    t0 = time.time()
    if cfg.get('intrinsic_dim_override'):
        intrinsic_dim = cfg['intrinsic_dim_override']
        print(f"\n=== Using manual intrinsic_dim = {intrinsic_dim} ===")
    else:
        intrinsic_dim = estimate_intrinsic_dim(encoder, buffer, cfg)
    t_id = time.time() - t0
    print(f"  ID estimation: {t_id:.1f}s")

    # Stage 2
    model = DoubleAutoEncoder(encoder, decoder, cfg['latent_dim'],
                              intrinsic_dim, hidden_dim=cfg['hidden_dim']).cuda()

    # Visualize stage 1 before stage 2 training changes anything
    visualize_reconstructions(model, buffer, cfg, stage='stage1')

    intrinsic_forward_model = None
    t0 = time.time()
    if args.intrinsic_forward:
        intrinsic_forward_model = ForwardPredictor(
            intrinsic_dim, cfg['num_actions'], hidden_dim=128).cuda()
        train_stage2_with_forward(model, intrinsic_forward_model, buffer, cfg,
                                   fp_coef=args.fp_coef,
                                   pred_coef=args.pred_coef,
                                   use_weighted_loss=args.weighted_loss)
    else:
        train_stage2(model, buffer, cfg, use_weighted_loss=args.weighted_loss)
    t_stage2 = time.time() - t0
    print(f"  Stage 2 training: {t_stage2:.1f}s")

    visualize_reconstructions(model, buffer, cfg, stage='stage2')

    # Save
    save_checkpoint(model, intrinsic_dim, cfg, forward_model=forward_model)
    if intrinsic_forward_model is not None:
        fwd_path = f"{cfg['env_name']}_intrinsic_forward.pt"
        torch.save(intrinsic_forward_model.state_dict(), fwd_path)
        print(f"Saved intrinsic forward model to {fwd_path}")

    print(f"\n=== Timing Summary ===")
    print(f"  Data:     {t_data:7.1f}s")
    print(f"  Stage 1:  {t_stage1:7.1f}s")
    print(f"  ID est:   {t_id:7.1f}s")
    print(f"  Stage 2:  {t_stage2:7.1f}s")
    print(f"  Total:    {t_data + t_stage1 + t_id + t_stage2:7.1f}s")
    print("\nDone.")


if __name__ == '__main__':
    main()