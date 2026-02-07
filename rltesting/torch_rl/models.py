from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import copy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def compute_pad(kernel_size, stride):
    return int(np.ceil((kernel_size - stride) / 2))

class MLP(nn.Module):
    # if hidden dims is specified then doesn't use skip connections
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_hiddens=2, act=nn.GELU, hidden_dims=None, final_act=None, skip_connections=None):
        super().__init__()
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
            self.skip_connections = False
        else:
            self.skip_connections = True
        if skip_connections is not None:
            self.skip_connections = skip_connections
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
        self.final_act = final_act
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        logits = self.output_layer(x)
        if self.final_act is not None:
            return self.final_act(logits)
        return logits

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=nn.GELU):
        super().__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = act()
    
    def forward(self, x):
        x_skip = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + x_skip

class NatureCNN(nn.Module):
    # from the original Nature DQN paper
    # assumes the 84x84 grayscale and 4 frame stack
    def __init__(self, act=nn.GELU, flatten_out=False, input_channels=4):
        super().__init__()
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

class IMPALABlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResBlock(out_channels, out_channels, 3, act)
        self.res2 = ResBlock(out_channels, out_channels, 3, act)
        self.act = act()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class IMPALACNN(nn.Module):
    def __init__(self, image_size, num_blocks, image_channels=3, channel_base=16, act=nn.GELU):
        channels = [image_channels] + [channel_base * 2 ** i for i in range(num_blocks)]
        self.image_size = image_size
        self.image_channels = image_channels
        super().__init__()
        self.layers = nn.Sequential(*[IMPALABlock(channels[i], channels[i + 1], act) for i in range(num_blocks)])
        self.output_dim = self.compute_output_dim()
    
    def compute_output_dim(self):
        x = torch.zeros(1, self.image_channels, self.image_size, self.image_size)
        x = self.layers(x)
        return x.view(-1).shape[0]
    
    def forward(self, obs):
        return self.layers(obs)

class ChannelNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class NormAndAct(nn.Module):
    # fix needed, include dimension for norm
    def __init__(self, norm_dim, norm=nn.LayerNorm, act=nn.SiLU):
        super().__init__()
        self.act = nn.Sequential(norm(norm_dim), act())
    
    def forward(self, x):
        return self.act(x)

class DreamerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hiddens, act=nn.SiLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True))
        layers.append(act())
        for _ in range(num_hiddens):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True))
            layers.append(act())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class DreamerEncoderConv(nn.Module):
    # built for 64x64 observations and downscales them to 4x4, can do other sizes but would need to be changed a bit
    def __init__(self, filter_base=8, num_convs=4, kernel_size=4, image_channels=3, input_size=64, act=nn.SiLU, norm=ChannelNorm):
        super().__init__()
        filters = [filter_base * 2 ** i for i in range(num_convs)]
        layers = []
        for i, filter in enumerate(filters):
            layers.append(nn.Conv2d(image_channels if i == 0 else filters[i - 1], filter, kernel_size, 
                                    stride=2, padding=compute_pad(kernel_size, 2), bias=i == len(filters) - 1))
            if i < len(filters) - 1:
                layers.append(NormAndAct(filter, norm, act))
        self.layers = nn.Sequential(*layers)
        size = input_size // (2 ** num_convs)
        self.output_size = (filters[-1], size, size)
    
    def forward(self, x):
        return self.layers(x)
    
class DreamerDecoderConv(nn.Module):
    # does the reverse of the encoder conv, pass in reversed filters
    def __init__(self, filter_base=8, num_convs=4, kernel_size=4, image_channels=3, act=nn.SiLU, norm=ChannelNorm):
        super().__init__()
        filters = [filter_base * 2 ** i for i in reversed(range(num_convs))]
        layers = []
        for i, filter in enumerate(filters):
            layers.append(nn.ConvTranspose2d(filter, image_channels if i == len(filters) - 1 else filters[i + 1], kernel_size, 
                                             stride=2, padding=compute_pad(kernel_size, 2), bias=i == len(filters) - 1))
            if i < len(filters) - 1:
                layers.append(NormAndAct(image_channels if i == len(filters) - 1 else filters[i + 1], norm, act))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class BlockLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=8):
        super().__init__()
        self.networks = nn.ModuleList([nn.Linear(input_size // num_blocks, output_size // num_blocks) for _ in range(num_blocks)])
        self.num_blocks = num_blocks
    
    def forward(self, x):
        output = []
        x_chunks = torch.split(x, x.shape[-1] // self.num_blocks, dim=-1)
        for i in range(self.num_blocks):
            output.append(self.networks[i](x_chunks[i]))
        return torch.cat(output, -1)

class DreamerGRU(nn.Module):
    def __init__(self, hidden_state_size, use_block_linear=True):
        super().__init__()
        if use_block_linear:
            self.layer = BlockLinear(hidden_state_size, hidden_state_size * 3)
        else:
            self.layer = nn.Linear(hidden_state_size, hidden_state_size * 3)
        self.hidden_state_size = hidden_state_size
    
    def forward(self, h):
        x = self.layer(h)
        reset, cand, update = torch.split(x, self.hidden_state_size, -1)
        reset = F.sigmoid(reset)
        cand = F.tanh(reset * cand)
        update = F.sigmoid(update - 1)
        h_new = update * cand + (1 - update) * h
        return h_new

class TargetNetwork(nn.Module):
    def __init__(self, original_network, tau=None, update_freq=None):
        super().__init__()
        self.network = copy.deepcopy(original_network)
        self.original = original_network
        for param in self.network.parameters():
            param.requires_grad = False
        if tau is None and update_freq is None:
            raise RuntimeError("At least one of tau or update frequency should be specified")
        self.tau = tau # esentially the moving average, slowly updates every time
        self.update_freq = update_freq
        self.i = 0
        
    def update(self):
        target_net_state_dict = self.network.state_dict()
        original_net_state_dict = self.original.state_dict()
        if self.tau is not None:
            for key in original_net_state_dict:
                target_net_state_dict[key] = original_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.network.load_state_dict(target_net_state_dict)
        else:
            self.i += 1
            if (self.i % self.update_freq) == 0:
                self.network.load_state_dict(original_net_state_dict)
    
    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    @property
    def net(self):
        return self.network
