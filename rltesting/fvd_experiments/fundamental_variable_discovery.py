import torch
from torch import nn
import torch.nn.functional as F
from rltesting.torch_rl.models import DreamerDecoderConv, DreamerEncoderConv
import numpy as np
import skdim

class Encoder(nn.Module):
    def __init__(self, framestack, latent_size, image_dim):
        super().__init__()
        self.conv = DreamerEncoderConv(image_channels=3 * framestack, input_size=image_dim)
        self.conv_dim = self.conv.output_size
        self.out = nn.Linear(np.prod(self.conv.output_size), latent_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(-3)
        x = self.out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, conv_dim, framestack, latent_size):
        super().__init__()
        self._in = nn.Linear(latent_size, np.prod(conv_dim))
        self.conv_dim = conv_dim
        self.conv = DreamerDecoderConv(image_channels=3 * framestack)
    
    def forward(self, x):
        x = self._in(x)
        x = x.reshape(-1, *self.conv_dim)
        x = self.conv(x)
        return F.sigmoid(x)

class IntrinsicEncoder(nn.Module):
    pass

class IntrinsicDecoder(nn.Module):
    pass

# class LevinaBickelAlgorithm:
#     def __init__(self, samples):
        