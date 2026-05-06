import torch
from torch import nn
import torch.nn.functional as F
from rltesting.torch_rl.models import MLP, ChannelNorm, IMPALABlock, ResBlock
import numpy as np

def compute_pad(kernel_size, stride):
    return int(np.ceil((kernel_size - stride) / 2))

class EncoderConv(nn.Module):
    # built for 64x64 observations and downscales them to 4x4, can do other sizes but would need to be changed a bit
    def __init__(self, filter_base=8, num_convs=4, kernel_size=4, image_channels=3, input_size=64, act=nn.GELU):
        super().__init__()
        filters = [filter_base * 2 ** i for i in range(num_convs)]
        layers = []
        for i, filter in enumerate(filters):
            layers.append(nn.Conv2d(image_channels if i == 0 else filters[i - 1], filter, kernel_size, 
                                    stride=2, padding=compute_pad(kernel_size, 2), bias=i == len(filters) - 1))
            if i < len(filters) - 1:
                layers.append(ChannelNorm(filter))
            layers.append(act())
        self.layers = nn.Sequential(*layers)
        size = input_size // (2 ** num_convs)
        self.output_size = (filters[-1], size, size)
    
    def forward(self, x):
        return self.layers(x)
    
class DecoderConv(nn.Module):
    # does the reverse of the encoder conv, pass in reversed filters
    def __init__(self, filter_base=8, num_convs=4, kernel_size=4, image_channels=3, act=nn.GELU):
        super().__init__()
        filters = [filter_base * 2 ** i for i in reversed(range(num_convs))]
        layers = []
        for i, filter in enumerate(filters):
            layers.append(nn.ConvTranspose2d(filter, image_channels if i == len(filters) - 1 else filters[i + 1], kernel_size, 
                                             stride=2, padding=compute_pad(kernel_size, 2), bias=i == len(filters) - 1))
            if i < len(filters) - 1:
                layers.append(ChannelNorm(filters[i + 1]))
                layers.append(act())
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, framestack, latent_dim, image_dim, image_channels=3, filter_base=16):
        super().__init__()
        self.conv = EncoderConv(filter_base=filter_base, image_channels=image_channels * framestack, input_size=image_dim) # 256, 4x4
        self.conv_dim = self.conv.output_size
        if latent_dim is None:
            self.out = None
            self.latent_dim = self.conv_dim
        else:
            self.out = nn.Linear(np.prod(self.conv.output_size), latent_dim)
            self.latent_dim = latent_dim
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(-3)
        if self.out:
            x = self.out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, conv_dim, framestack, latent_dim, image_channels=3, filter_base=16):
        super().__init__()
        if latent_dim is None:
            self._in = None
        else:
            self._in = nn.Linear(latent_dim, np.prod(conv_dim))
        self.conv_dim = conv_dim
        self.conv = DecoderConv(filter_base=filter_base, image_channels=image_channels * framestack)
    
    def forward(self, x):
        if self._in:
            x = self._in(x)
        x = x.reshape(-1, *self.conv_dim)
        x = self.conv(x)
        return x

class DoubleAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, intrinsic_dim, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = np.prod(encoder.latent_dim) if latent_dim is None else latent_dim
        
        self.intrinsic_encoder = MLP(self.latent_dim, intrinsic_dim, hidden_dim, act=nn.GELU, skip_connections=True)
        self.intrinsic_decoder = MLP(intrinsic_dim, self.latent_dim, hidden_dim, act=nn.GELU, skip_connections=True)
        self.intrinsic_dim = intrinsic_dim
    
    def double_encode(self, image):
        latent = self.encoder(image)
        intrinsic = self.intrinsic_encoder(latent)
        return intrinsic
    
    def double_decode(self, intrinsic):
        reconstructed_latent = self.intrinsic_decoder(intrinsic)
        reconstructed = self.decoder(reconstructed_latent)
        return reconstructed
    
    def compute_latent(self, image):
        return self.encoder(image)
    
    def compute_intrinsic(self, latent):
        return self.intrinsic_encoder(latent)
    
    def reconstruct_latent(self, intrinsic):
        return self.intrinsic_decoder(intrinsic)
    
    def reconstruct_image(self, latent):
        return self.decoder(latent)
    
    def reconstruction_loss(self, image):
        latent = self.compute_latent(image)
        intrinsic = self.compute_intrinsic(latent)
        latent_reconstruction = self.reconstruct_latent(intrinsic)
        reconstruction = self.reconstruct_image(latent)
        return F.mse_loss(image, reconstruction) #+ F.mse_loss(latent.detach(), latent_reconstruction)

class AEVAE(DoubleAutoEncoder):
    def __init__(self, encoder, decoder, latent_dim, intrinsic_dim, hidden_dim=128, penalty_coef=1e-4):
        super().__init__(encoder, decoder, latent_dim, intrinsic_dim, hidden_dim)
        # treat intrinsic_encoder as producing the mean
        self.penalty_coef = penalty_coef
        self.intrinsic_encoder = MLP(self.latent_dim, hidden_dim, hidden_dim, num_hiddens=2, act=nn.GELU, skip_connections=False)
        self.intrinsic_decoder = MLP(intrinsic_dim, self.latent_dim, hidden_dim, num_hiddens=0, act=nn.GELU, skip_connections=False)
        self.intrinsic_mu = nn.Linear(hidden_dim, intrinsic_dim)
        self.intrinsic_logvar = nn.Linear(hidden_dim, intrinsic_dim)
    
    def double_encode(self, image):
        latent = self.encoder(image)
        intrinsic = self.compute_intrinsic(latent)
        return intrinsic

    def double_encode_deterministic(self, image):
        latent = self.encoder(image)
        temp = self.intrinsic_encoder(latent)
        intrinsic_mu = self.intrinsic_mu(temp)
        return intrinsic_mu
    
    def double_decode(self, intrinsic):
        reconstructed_latent = self.intrinsic_decoder(intrinsic)
        reconstructed = self.decoder(reconstructed_latent)
        return reconstructed
    
    def compute_intrinsic(self, latent):
        temp = self.intrinsic_encoder(latent)
        intrinsic_mu = self.intrinsic_mu(temp)
        intrinsic_logvar = self.intrinsic_logvar(temp)
        intrinsic = reparameterize(intrinsic_mu, intrinsic_logvar)
        return intrinsic
    
    def reconstruction_loss(self, image, penalty_coef=None):
        latent = self.encoder(image)
        temp = self.intrinsic_encoder(latent)
        intrinsic_mu = self.intrinsic_mu(temp)
        intrinsic_logvar = self.intrinsic_logvar(temp)
        intrinsic = reparameterize(intrinsic_mu, intrinsic_logvar)
        reconstruction = self.double_decode(intrinsic)
        kl_penalty = vae_kl(intrinsic_mu, intrinsic_logvar).mean() * (self.penalty_coef if penalty_coef is None else penalty_coef)
        reconstruction_losses = F.mse_loss(image, reconstruction)
        return reconstruction_losses + kl_penalty

class HybridAEVAE(AEVAE):
    def __init__(self, encoder, decoder, latent_dim, intrinsic_dim, num_codes=20, hidden_dim=256, penalty_coef=1e-5):
        super().__init__(encoder, decoder, latent_dim, intrinsic_dim * 2, hidden_dim, penalty_coef)
        self.codes = nn.Embedding(num_codes, intrinsic_dim)
        self.dim = intrinsic_dim
    
    def discretize(self, latent):
        distances = torch.sum((latent.unsqueeze(1) - self.codes.weight) ** 2, dim=-1)
        codebook_ind = torch.argmin(distances, dim=-1)
        
        latent_q = self.codes(codebook_ind)
        latent_q_straight_through = latent + (latent_q - latent).detach()
        return latent_q_straight_through, latent_q
    
    def double_encode(self, image):
        latent = self.encoder(image)
        intrinsic_mu = self.intrinsic_encoder(latent)
        intrinsic_logvar = self.intrinsic_logvar(latent)
        intrinsic = reparameterize(intrinsic_mu, intrinsic_logvar)
        disc_intrinsic = intrinsic[:, self.dim:]
        disc_intrinsic, _ = self.discretize(disc_intrinsic)
        intrinsic[:, self.dim:] = disc_intrinsic
        return intrinsic
    
    def double_decode(self, intrinsic):
        reconstructed_latent = self.intrinsic_decoder(intrinsic)
        reconstructed = self.decoder(reconstructed_latent)
        return reconstructed
    
    def compute_intrinsic(self, latent):
        intrinsic_mu = self.intrinsic_encoder(latent)
        intrinsic_logvar = self.intrinsic_logvar(latent)
        intrinsic = reparameterize(intrinsic_mu, intrinsic_logvar)
        disc_intrinsic = intrinsic[:, self.dim:]
        disc_intrinsic, _ = self.discretize(disc_intrinsic)
        intrinsic[:, self.dim:] = disc_intrinsic
        return intrinsic
    
    def reconstruction_loss(self, image, eps=1e-6):
        latent = self.encoder(image)
        intrinsic_mu = self.intrinsic_encoder(latent)
        intrinsic_logvar = self.intrinsic_logvar(latent)
        intrinsic = reparameterize(intrinsic_mu, intrinsic_logvar)
        disc_intrinsic = intrinsic[:, self.dim:]
        disc_intrinsic_straight_through, codebook = self.discretize(disc_intrinsic)
        intrinsic[:, self.dim:] = disc_intrinsic_straight_through
        reconstruction = self.double_decode(intrinsic)
        codebook_loss = F.mse_loss(codebook, disc_intrinsic.detach())
        commitment_loss = F.mse_loss(codebook.detach(), disc_intrinsic)
        dist_loss = distance_correlation_loss(intrinsic[:, :self.dim], intrinsic[:, self.dim:])
        return F.mse_loss(image, reconstruction) + vae_kl(intrinsic_mu, intrinsic_logvar).sum(-1).mean() * self.penalty_coef + codebook_loss + commitment_loss * 0.1 #+ dist_loss * 0.1 # TODO add info gan terms and beta-VAE

def vae_kl(mu, logvar):
    return 0.5 * (-logvar - 1 + torch.exp(logvar) + mu ** 2)

def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    z = mu + eps * torch.exp(logvar * 0.5)
    return z

def distance_correlation_loss(z, c):
    n = z.size(0)
    
    def get_distance_matrix(x):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        dist_sq = x_norm + x_norm.t() - 2 * (x @ x.t())
        return torch.clamp(dist_sq, min=0)

    def u_center(mat):
        row_mean = mat.mean(dim=1, keepdim=True)
        col_mean = mat.mean(dim=0, keepdim=True)
        grand_mean = mat.mean()
        centered = mat - row_mean - col_mean + grand_mean
        centered.fill_diagonal_(0)
        return centered

    A = u_center(get_distance_matrix(z))
    B = u_center(get_distance_matrix(c))

    dcov2 = torch.sum(A * B) / (n * (n - 3))
    return dcov2

class IMPALADecoderBlock(nn.Module):
    """Reverse of IMPALABlock: ResBlocks → Upsample → Conv."""
    def __init__(self, in_channels, out_channels, act=nn.GELU):
        super().__init__()
        self.res1 = ResBlock(in_channels, in_channels, 3, act)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
 
    def forward(self, x):
        x = self.res1(x)
        x = self.upsample(x)
        x = self.conv(x)
        return x
 
 
class IMPALAEncoder(nn.Module):
    """IMPALA-style encoder. Drop-in replacement for Encoder.
 
    Uses IMPALABlock (Conv → MaxPool → ResBlock → ResBlock) at each level.
    MaxPool downsamples more gently than strided conv, preserving small features.
    Residual connections improve gradient flow through the network.
 
    Works with any even input size (64, 80, 96, 128, etc).
    """
    def __init__(self, framestack, latent_dim, image_dim, image_channels=3,
                 filter_base=16, num_blocks=4, act=nn.GELU):
        super().__init__()
        in_channels = image_channels * framestack
        channels = [in_channels] + [filter_base * 2 ** i for i in range(num_blocks)]
 
        self.blocks = nn.Sequential(
            *[IMPALABlock(channels[i], channels[i + 1], act, num_blocks=1) for i in range(num_blocks)]
        )
        self.final_act = act()
 
        # Compute output spatial size after num_blocks MaxPool(3, stride=2, pad=1)
        size = image_dim
        for _ in range(num_blocks):
            size = (size - 1) // 2 + 1  # MaxPool2d(3, stride=2, padding=1) formula
        self.conv_dim = (channels[-1], size, size)
 
        if latent_dim is None:
            self.out = None
            self.latent_dim = self.conv_dim
        else:
            self.out = nn.Linear(int(np.prod(self.conv_dim)), latent_dim)
            self.latent_dim = latent_dim
 
    def forward(self, x):
        x = self.blocks(x)
        x = self.final_act(x)
        x = x.flatten(-3)
        if self.out:
            x = self.out(x)
        return x
 
 
class IMPALADecoder(nn.Module):
    """IMPALA-style decoder (reverse of IMPALAEncoder). Drop-in replacement for Decoder.
 
    Uses IMPALADecoderBlock (ResBlock → ResBlock → Upsample → Conv) at each level.
    Final output is passed through sigmoid to [0, 1].
 
    Works with any even input size (64, 80, 96, 128, etc).
    """
    def __init__(self, conv_dim, framestack, latent_dim, image_channels=3,
                 filter_base=16, num_blocks=4, act=nn.GELU):
        super().__init__()
        if latent_dim is None:
            self._in = None
        else:
            self._in = nn.Linear(latent_dim, int(np.prod(conv_dim)))
        self.conv_dim = conv_dim
 
        # Channels: high → low → output image channels
        channels = [filter_base * 2 ** i for i in reversed(range(num_blocks))]
        out_channels = channels[1:] + [image_channels * framestack]
 
        self.blocks = nn.Sequential(
            *[IMPALADecoderBlock(channels[i], out_channels[i], act) for i in range(num_blocks)]
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        if self._in:
            x = self._in(x)
        x = x.reshape(-1, *self.conv_dim)
        x = self.blocks(x)
        x = self.sigmoid(x)
        return x