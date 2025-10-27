import torch
from torch import nn
import torch.nn.functional as F
from rltesting.torch_rl.models import DreamerDecoderConv, DreamerEncoderConv
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = DreamerEncoderConv()
        self.conv_dim = self.conv.output_size
        self.out = nn.Linear(np.prod(self.conv.output_size), 128)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(-3)
        x = self.out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, conv_dim):
        super().__init__()
        self._in = nn.Linear(128, np.prod(conv_dim))
        self.conv_dim = conv_dim
        self.conv = DreamerDecoderConv()
    
    def forward(self, x):
        x = self._in(x)
        x = x.reshape(-1, *self.conv_dim)
        x = self.conv(x)
        return F.sigmoid(x)

class IntrinsicEncoder(nn.Module):
    pass

class IntrinsicDecoder(nn.Module):
    pass

class LevinaBickelAlgorithm:
    pass