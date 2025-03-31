import torch
from torch import nn
import torch.functional as F
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from torch.distributions import Categorical
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, key, hidden_dim=256, num_hiddens=3, act=nn.SELU, hidden_dims=None, output_act=None):
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
        self.output_layer = nn.Linear(hidden_dim, output_dim, key=key)
        self.act = act
        self.output_act = output_act
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        logits = self.output_layer(x)
        if self.output_act is not None:
            return self.output_act(logits)
        return logits


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=nn.SELU):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding='same')
        if kernel_size > 1:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding='same')
        else:
            self.skip_conv = None
        self.act = act
    
    def forward(self, x):
        x_skip = x.copy()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        if self.skip_conv is not None:
            x_skip = self.skip_conv(x_skip)
        return x + x_skip
        
        
class AtariConv(nn.Module):
    # assumes the 84x84 grayscale and 4 frame stack
    def __init__(self, act=nn.SELU):
        super(AtariConv, self).__init__()
        self.convs = nn.ModuleList([
            ResBlock(in_channels=4, out_channels=32, kernel_size=7, stride=3, padding='same'), # (4, 84, 84) -> (32, 28, 28)
            act,
            ResBlock(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding='same'), # (32, 28, 28) -> (64, 14, 14)
            act,
            ResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding='same'), # (64, 14, 14) -> (128, 7, 7)
            act, 
            ResBlock(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding='same') # (128, 7, 7) -> (256, 3, 3) or 2304
        ])
    def forward(self, x):
        x = self.convs(x)
        return x
    

class ICM:
    pass

def main(args):
    env = make_atari_env(args.env + '-v5', args.num_envs, args.seed)
    torch.random.seed(args.seed)
    np.random.seed(args.seed)
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--env', type=str, default='MontezumaRevenge')
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--rollout_length', type=int, default=16)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    main(args)
    