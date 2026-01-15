
from rltesting.torch_rl.buffers import ReplayBuffer
from fvd_models import DoubleAutoEncoder
from utils import *
import torch.nn.functional as F
import numpy as np
import torch
import skdim
import os

def pretrain(config, encoder, decoder, reconstruction_loss, penalty=None):
    framestack = config.framestack
    latent_dim = config.latent_dim
    image_size = config.image_size
    data_steps = config.data_steps
    grayscale = config.grayscale
    num_channels = 3 if not grayscale else 1
    train_steps = config.train_steps
    env_type = config.env_type
    env_id = config.env_id
    lr = config.learning_rate
    batch_size = config.batch_size
    scale_factor = config.scale_factor
    device = config.device

    env = make_env(env_type, env_id, framestack, image_size, grayscale)
    
    obs_shape = (image_size, image_size, num_channels * framestack)
    buffer_shapes = [obs_shape, (), (), obs_shape, ()]
    dtypes = [np.uint8, np.float32, np.float32, np.uint8, np.float32]
    buffer = ReplayBuffer(buffer_shapes, dtypes, buffer_size=data_steps)
    fill_buffer(env, buffer, data_steps)
    
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)
    losses = []
    for i in range(1, train_steps + 1):
        if i % 1000 == 0:
            print(i)
        samples = buffer.sample(batch_size)
        obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).to(device)
        latent = encoder(obs)
        reconstruction = decoder(latent)
        loss = reconstruction_loss(reconstruction, obs)
        if penalty is not None:
            loss += penalty(latent)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.detach().cpu().numpy())

    latent_ID = []
    image_ID = []

    for _ in range(config.id_num_batches):
        samples = buffer.sample(config.id_batch_size)
        obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).to(device)
        latent = encoder(obs)
        unique_latents = np.unique(latent.detach().cpu().numpy(), axis=0)
        unique_obs = np.unique(obs.detach().cpu().numpy(), axis=0).reshape(config.id_batch_size, -1)

        latent_ID.append(skdim.id.MLE().fit(unique_latents).dimension_)
        image_ID.append(skdim.id.MLE().fit(unique_obs).dimension_)

    print(f"Latent ID: {np.mean(latent_ID)} +- {np.std(latent_ID)}")
    print(f"Image ID: {np.mean(image_ID)} +- {np.std(image_ID)}")

    intrinsic_dim = round(scale_factor * np.mean(latent_ID))
    double_autoencoder = DoubleAutoEncoder(encoder, decoder, latent_dim, intrinsic_dim).to(device)
    int_opt = torch.optim.AdamW(double_autoencoder.parameters(), lr)

    losses = []
    for i in range(1, train_steps + 1):
        if i % 1000 == 0:
            print(i)
        samples = buffer.sample(batch_size)
        obs = (torch.tensor(samples[0]).float().transpose(-3, -1) / 255).to(device)
        intrinsic = double_autoencoder.double_encode(obs)
        reconstruction = double_autoencoder.double_decode(intrinsic)
        loss = reconstruction_loss(reconstruction, obs)
        loss.backward()
        int_opt.step()
        int_opt.zero_grad()
        opt.step()
        opt.zero_grad()
        losses.append(loss.detach().cpu().numpy())
    
    torch.save(double_autoencoder, config.model_path)
    return double_autoencoder

def load_or_create_model(config, encoder, decoder):
    if os.path.exists(config.model_path):
        model = torch.load(config.model_path, map_location=config.device, weights_only=False)
        print("Model Loaded")
        return model
    else:
        double_autoencoder = pretrain(config, encoder, decoder, F.mse_loss)
        print("Done Pretraining")
        return double_autoencoder
