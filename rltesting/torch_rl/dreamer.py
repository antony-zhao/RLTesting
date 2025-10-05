from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from rltesting.torch_rl.models import DreamerDecoderConv, DreamerEncoderConv, BlockLinear, MLP, NormAndAct
import argparse
from functools import partial

# These should definitely take in like a config dictionary instead of the current approach, too many variables otherwise

def unimix(x, num_codes, proportion=0.01):
    uniform = torch.ones_like(x) / num_codes
    return x * (1 - proportion) + uniform * proportion

class DreamerEncoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    # hidden_state_size is h_t (the hidden state of the recurrent network)
    # categories is number of rows, codes is the number of columns (softmaxed over codes)
    # hidden dim is just the hidden dim of linear layers
    def __init__(self, config):
        super().__init__()
        if config.obs_type == "image":
            self.encoder = DreamerEncoderConv(config.filter_base, config.num_convs, config.kernel_size, config.num_channels, config.image_size, config.act)
            output_dim = np.prod(self.encoder.output_size)
        else:
            self.encoder = MLP(config.obs_dim, config.hidden_dim, config.hidden_dim, num_hiddens=1, act=partial(NormAndAct, norm_dim=config.hidden_dim, act=config.act))
            output_dim = config.hidden_dim
        self.out = nn.Linear(output_dim + config.hidden_state_size, config.num_latents * config.num_codes)
        self.num_latents = config.num_latents
        self.num_codes = config.num_codes
    
    def forward(self, x, h):
        # x as in the observation specifically, h is the same hidden state
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        x = torch.cat([encoded, h], 1)
        latent = self.out(x)
        latent = latent.reshape(latent.shape[0], self.num_latents, self.num_codes)
        unimixed_probs = unimix(latent, self.num_codes, config.latent_unimix)
        logits = F.softmax(unimixed_probs, -1)
        return logits#, unimixed_probs

class DreamerDecoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    def __init__(self, config):
        super().__init__()
        self.obs_type = config.obs_type
        if config.obs_type == "image":
            self.input_dim = config.output_dim
            self._in = nn.Linear(config.hidden_state_size + config.num_latents * config.num_codes, np.prod(self.input_dim))
            self.decoder = DreamerDecoderConv(config.filter_base, config.num_convs, config.kernel_size, config.num_channels, config.act)
        else:
            self._in = nn.Linear(config.hidden_state_size + config.num_latents * config.num_codes, config.hidden_dim)
            self.decoder = MLP(config.hidden_dim, config.hidden_dim, config.obs_dim, num_hiddens=1, act=partial(NormAndAct, norm_dim=config.hidden_dim, act=config.act))
    
    def forward(self, z, h):
        x = torch.cat([torch.flatten(z, 1), h], -1)
        x = self._in(x)
        if self.obs_type == "image":
            x = x.reshape(-1, *self.input_dim)
        reconstruction = self.decoder(x)
        return reconstruction

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

class RSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_hidden = nn.Linear(config.hidden_state_size, config.hidden_dim)
        self.in_latent = nn.Linear(config.num_latents * config.num_codes, config.hidden_dim)
        self.in_action = nn.Linear(config.action_dim, config.hidden_dim)
        self.act1 = NormAndAct(config.hidden_dim * 3)
        self.mlp = MLP(config.hidden_dim* 3, config.hidden_state_size, config.hidden_dim, num_hiddens=1, act=partial(NormAndAct, norm_dim=config.hidden_dim, act=config.act))
        self.gru = DreamerGRU(config.hidden_state_size, config.use_block_linear)
    
    def forward(self, h, z, a):
        flattened_latent = torch.flatten(z, 1)
        x1 = self.in_hidden(h)
        x2 = self.in_latent(flattened_latent)
        x3 = self.in_action(a) # need to verify that this is a copy of the action with no gradients
        x = torch.cat([x1, x2, x3], -1)
        x = self.act1(x)
        x = self.mlp(x)
        h_new = self.gru(x)
        return h_new

class DreamerWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="just a temporary argparse while I debug the world model, will be moved elsewhere later")
    # specified for Atari specifically by default and using the 200M size model for Dreamer
    parser.add_argument("--obs_type", default="image", choices=["image", "vector"])
    encoder_parser = parser.add_argument_group("encoder")
    vector_parser = parser.add_argument_group("vector")
    encoder_parser.add_argument("--num_channels", default=3, type=int)
    encoder_parser.add_argument("--image_size", default=64, type=int) # images should be square
    encoder_parser.add_argument("--kernel_size", default=4, type=int) # best to keep it 4 or 6
    encoder_parser.add_argument("--filter_base", default=8, type=int) # the base number of filters, which is doubled for each convolutional layer
    encoder_parser.add_argument("--num_convs", default=4, type=int) # total number of convolutions, after which the dimension is size / 2^num_convs, and the final number of filters is filter_base * 2^(num_convs-1)
    vector_parser.add_argument("--obs_dim", default=None, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int) # hidden dims of MLPs
    parser.add_argument("--hidden_state_size", default=8192, type=int) # hidden state of GRU/RSSM
    parser.add_argument("--num_hidden", default=1, type=int) # determines the depth for various MLPs
    parser.add_argument("--action_dim", default=18, type=int)
    parser.add_argument("--num_latents", default=32, type=int) # the number of rows in the latent
    parser.add_argument("--num_codes", default=64, type=int) # the actual dim that's softmaxed over in the latent
    parser.add_argument("--latent_unimix", default=0.01, type=float)
    parser.add_argument("--use_block_linear", default=True, type=bool)
    parser.add_argument("--act", default="silu", choices=["silu", "gelu", "relu"])
    config = parser.parse_args()
    if config.act == "silu":
        config.act = nn.SiLU
    elif config.act == "gelu":
        config.act = nn.GELU
    elif config.act == "relu":
        config.act = nn.ReLU
    if config.obs_type == "image":
        config.image_dim = (config.num_channels, config.image_size, config.image_size)
        size = config.image_size // 2 ** (config.num_convs)
        config.output_dim = (config.filter_base * 2 ** (config.num_convs - 1), size, size)
    
    test_image = torch.randn((1, *config.image_dim))
    test_hidden = torch.zeros((1, config.hidden_state_size))
    test_act = torch.zeros((1, config.action_dim))
    encoder = DreamerEncoder(config)
    decoder = DreamerDecoder(config)
    rssm = RSSM(config)
    test_latent = encoder(test_image, test_hidden)
    test_reconstruction = decoder(test_latent, test_hidden)
    test_next_hidden = rssm(test_hidden, test_latent, test_act)
    print("hi")