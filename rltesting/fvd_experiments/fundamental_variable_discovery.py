import torch
from torch import nn
import torch.nn.functional as F
from rltesting.torch_rl.models import MLP
import numpy as np
import skdim
import scipy

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
            layers.append(nn.BatchNorm2d(filter))
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
                layers.append(nn.BatchNorm2d(filters[i + 1]))
                layers.append(act())
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, framestack, latent_dim, image_dim, filter_base=8):
        super().__init__()
        self.conv = EncoderConv(filter_base=filter_base, image_channels=3 * framestack, input_size=image_dim) # 256, 4x4
        self.conv_dim = self.conv.output_size
        self.out = nn.Linear(np.prod(self.conv.output_size), latent_dim) #MLP(np.prod(self.conv.output_size), latent_dim, 1024, skip_connections=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(-3)
        x = self.out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, conv_dim, framestack, latent_dim, filter_base=8):
        super().__init__()
        self._in = nn.Linear(latent_dim, np.prod(conv_dim))
        self.conv_dim = conv_dim
        self.conv = DecoderConv(filter_base=filter_base, image_channels=3 * framestack)
    
    def forward(self, x):
        x = self._in(x)
        x = x.reshape(-1, *self.conv_dim)
        x = self.conv(x)
        return x

class DoubleAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, intrinsic_dim, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.intrinsic_encoder = MLP(latent_dim, intrinsic_dim, hidden_dim, act=nn.GELU, skip_connections=False)
        self.intrinsic_decoder = MLP(intrinsic_dim, latent_dim, hidden_dim, act=nn.GELU, skip_connections=False)
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
        return self.encoder(image).detach()
    
    def compute_intrinsic(self, latent):
        return self.intrinsic_encoder(latent)
    
    def reconstruct_latent(self, intrinsic):
        return self.intrinsic_decoder(intrinsic)
    
    def reconstruct_image(self, latent):
        return self.decoder(latent)
    
    def reconstruction_loss(self, image):
        int_ = self.double_encode(image)
        reconstruction = self.double_decode(int_)
        return F.mse_loss(image, reconstruction)


# class LevinaBickelAlgorithm:
#     def __init__(self, samples):
        