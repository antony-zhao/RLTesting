from torch import nn
import torch
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_hiddens=2, act=nn.SiLU, hidden_dims=None):
        super(MLP, self).__init__()
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
            self.skip_connections = False
        else:
            self.skip_connections = True
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hiddens = []
        for i in range(num_hiddens):
            if hidden_dims is None:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
                hidden_dim = hidden_dims[i + 1]
        self.hiddens = nn.ModuleList(self.hiddens)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = act()
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        logits = self.output_layer(x)
        return logits

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=nn.SiLU):
        super(ResBlock, self).__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        if stride > 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride)
        else:
            self.skip = None
        self.act = act()
    
    def forward(self, x):
        x_skip = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        if self.skip is not None:
            x_skip = self.skip(x_skip)
        return x + x_skip

class AtariConv(nn.Module):
    # assumes the 84x84 grayscale and 4 frame stack
    def __init__(self, act=nn.SiLU, flatten_out=False, input_channels=4):
        super(AtariConv, self).__init__()
        # self.convs = nn.Sequential(
        #     ResBlock(in_channels=input_channels, out_channels=32, kernel_size=7, stride=3), # (4, 84, 84) -> (32, 28, 28)
        #     act(),
        #     ResBlock(in_channels=32, out_channels=64, kernel_size=5, stride=2), # (32, 28, 28) -> (64, 14, 14)
        #     act(),
        #     ResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2), # (64, 14, 14) -> (128, 7, 7)
        #     act(),
        #     ResBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2) # (128, 7, 7) -> (256, 4, 4) or 4096
        # )
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            act(),
            nn.Conv2d(32, 64, 4, stride=2),
            act(),
            nn.Conv2d(64, 64, 3, stride=1),
            act(),
        )
        self.output_dim = self.compute_output_dim(input_channels)
        self.flatten_out = flatten_out
    
    def compute_output_dim(self, input_channels):
        x = torch.zeros(1, input_channels, 84, 84)
        x = self.convs(x)
        return x.view(-1).shape[0]
        
    def forward(self, x):
        x = self.convs(x)
        if self.flatten_out:
            return x.view(-1, self.output_dim)
        else:
            return x
    