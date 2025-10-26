from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from rltesting.torch_rl.models import DreamerDecoderConv, DreamerEncoderConv, MLP, NormAndAct, DreamerGRU
from rltesting.torch_rl.dreamer.utils import *
import argparse
from functools import partial
from torch.distributions import OneHotCategoricalStraightThrough, Independent, Categorical, Normal, kl_divergence

mlp_norm_act = lambda dim, act: partial(NormAndAct, norm_dim=dim, act=act)

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
            self.encoder = MLP(config.obs_dim, config.hidden_dim, config.hidden_dim, num_hiddens=config.num_hiddens, act=mlp_norm_act(config.hidden_dim, config.act))
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
            self.decoder = MLP(config.hidden_dim, config.hidden_dim, config.obs_dim, num_hiddens=config.num_hiddens, act=mlp_norm_act(config.hidden_dim, config.act))
    
    def forward(self, z, h):
        x = torch.cat([torch.flatten(z, 1), h], -1)
        x = self._in(x)
        if self.obs_type == "image":
            x = x.reshape(-1, *self.input_dim)
        reconstruction = self.decoder(x)
        return reconstruction

class RSSM(nn.Module):
    # Also called the sequence model
    def __init__(self, config):
        super().__init__()
        self.in_hidden = nn.Linear(config.hidden_state_size, config.hidden_dim)
        self.in_latent = nn.Linear(config.num_latents * config.num_codes, config.hidden_dim)
        self.in_action = nn.Linear(config.action_dim, config.hidden_dim) # the required input layers for each of the three needed inputs
        self.act1 = NormAndAct(config.hidden_dim * 3)
        self.mlp = MLP(config.hidden_dim * 3, config.hidden_state_size, config.hidden_dim, num_hiddens=config.num_hiddens, act=mlp_norm_act(config.hidden_dim, config.act))
        self.gru = DreamerGRU(config.hidden_state_size, config.use_block_linear)
    
    def forward(self, h, z, a):
        flattened_latent = torch.flatten(z, -2)
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
        self.encoder = DreamerEncoder(config)
        self.decoder = DreamerDecoder(config)
        self.rssm = RSSM(config)
        self.dynamics_predictor = MLP(config.hidden_state_size, config.latent_size, config.hidden_dim, config.num_hiddens, mlp_norm_act(config.hidden_dim, config.act))
        self.reward_predictor = MLP(config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens, mlp_norm_act(config.hidden_dim, config.act))
        self.continue_predictor = MLP(config.state_size, 1, config.hidden_dim, config.num_hiddens, mlp_norm_act(config.hidden_dim, config.act))
        self.rollout_length = config.rollout_length
    
    def imagine_step(self, hidden, latent, action):
        # treat this like how you would a normal environment step
        # state is {z, h}
        state = torch.cat([latent.flatten(-2), hidden])
        continue_ = self.continue_predictor(state)
        reward = self.reward_predictor(state)
        next_hidden = self.rssm(hidden, latent, action) * continue_
        next_latent = self.dynamics_predictor(next_hidden)
        return (next_hidden, next_latent), reward, continue_
    
    def image_rollout(self, hidden, actor):
        # Used for actor critic training
        # returns a sequence of 
        # s_1 - s_t (where s = {h, z})
        # a_1 - a_t
        # r_1 - r_t
        # v_1 - v_t
        # c_t - c_t
        hiddens = []
        latents = []
        actions = []
        for i in range(self.rollout_length):
            probs = self.dynamics_predictor(hidden)
            latent = Independent(OneHotCategoricalStraightThrough(probs), 1).rsample()
            action = actor.choose_action(latent, hidden) # some placeholder function TO BE IMPLEMENTED
            state = torch.cat([latent.flatten(-2), hidden])
            continue_ = self.continue_predictor(state)
            hiddens.append(hidden)
            latents.append(latent)
            actions.append(action)
            if i < self.rollout_length - 1:
                hidden = self.rssm(hidden, latent, action) * continue_
        hiddens = torch.cat(hiddens)
        latents = torch.cat(latents)
        actions = torch.cat(actions)
    
    def world_model_loss(self, obs, actions, rewards, dones, hidden):
        enc_probs = []
        dyn_probs = []
        reconstructions = []
        continue_preds = []
        reward_probs = []
        for i in range(self.rollout_length):
            probs_enc = self.encoder(obs[i], hidden)
            latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1).rsample()
            enc_probs.append(probs_enc)
            probs_dyn = self.dynamics_predictor(hidden)
            # latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
            dyn_probs.append(probs_dyn)
            reconstruction = self.decoder(latent_enc, hidden)
            reconstructions.append(reconstruction)
            state = torch.cat([latent_enc.flatten(-2), hidden])
            reward_prob = self.reward_predictor(state)
            reward_probs.append(reward_prob)
            continue_pred = self.continue_predictor(state)
            continue_preds.append(continue_pred)
            if i < self.rollout_length - 1:
                hidden = self.rssm(hidden, latent_enc, actions[i])
        pred_loss = self.prediction_loss(obs, reconstructions, rewards, reward_probs, dones, continue_preds)
        dyn_loss = self.dynamics_loss(enc_probs, dyn_probs)
        rep_loss = self.representation_loss(enc_probs, dyn_probs)
        loss = pred_loss * 1 + dyn_loss * 1 + rep_loss * 0.1
        return loss, {"prediction loss": pred_loss, "dynamics loss": dyn_loss, "representation loss": "rep_loss"}
        
    
    def prediction_loss(self, obs, reconstruction, reward, reward_probs, dones, continue_prediction):
        reconstruction_error = symlog_squared_error(obs, reconstruction)
        reward_prediction = WeightedAverageOverBins(reward_probs)
        reward_error = reward_prediction.log_prob(reward)
        continue_error = F.binary_cross_entropy((1 - dones), continue_prediction)
        return reconstruction_error + reward_error + continue_error
    
    def dynamics_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
        return torch.clip(kl_divergence(latent_enc.detach(), latent_dyn), max=1)
    
    def representation_loss(self,  probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
        return torch.clip(kl_divergence(latent_enc, latent_dyn.detach()), max=1)  
          
class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config.state_size, config.action_dim, config.hidden_dim)
        self.action_type = config.action_type
        if config.action_type == "continuous":
            self.log_std = nn.Parameter(-torch.ones(config.action_dim))
    
    def policy_dist(self, x):
        logits = self.mlp(x)
        if self.action_type == "discrete":
            action_dist = Categorical(logits=logits)
        else:
            action_dist = Normal(loc=logits, scale=torch.exp(self.log_std))
        return action_dist
    
    def policy_fn(self, x, det=False):
        # actually choosing the action, returns the actual action as well as the log prob of the action
        if not det:
            action_dist = self.policy_dist(x)
        else:
            action_mean = self.mlp(x)
            return action_mean, None, None
        
        if self.action_type == "discrete":
            action = action_dist.sample()
        else:
            action = action_dist.rsample()
            
        return action, action_dist.log_prob(action).sum(-1), action_dist.entropy()
    
    def choose_action(self, x, det=False):
        if not det:
            action_dist = self.policy_dist(x)
        else:
            action_mean = self.mlp(x)
            return action_mean
        
        if self.action_type == "discrete":
            action = action_dist.sample()
        else:
            action = action_dist.rsample()
            
        return action
    
class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config.state_size, config.num_bins, config.hidden_dim)
    
    def forward(self, x):
        logits = self.mlp(x)
        probs = torch.softmax(logits, dim=-1)
        weighted_average = WeightedAverageOverBins(probs)
        return weighted_average.weighted_average(), weighted_average

class DreamerV3:
    def __init__(self, config):
        self.world_model = DreamerWorldModel(config)
        self.actor = Actor(config)
        self.critic = Critic(config)
        # decide whether or not to leave the optimizers in here, if so it might be good to modify some of the other classes to follow a similar 
        # pattern but then again dreamer is so much more different it might be fine
        self.buffer = None
        # buffer needs to account for order in episodes
    
    def train_world_model(self):
        pass
    
    def train_actor_critic(self):
        pass
    
    def add_episode(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="just a temporary argparse while I debug the world model, will be moved elsewhere later")
    # specified for Atari specifically by default and using the 200M size model for Dreamer
    parser.add_argument("--obs_type", default="image", choices=["image", "vector"])
    encoder_parser = parser.add_argument_group("encoder")
    vector_parser = parser.add_argument_group("vector")
    world_model_parser = parser.add_argument_group("world_model")
    encoder_parser.add_argument("--num_channels", default=3, type=int)
    encoder_parser.add_argument("--image_size", default=64, type=int) # images should be square
    encoder_parser.add_argument("--kernel_size", default=4, type=int) # best to keep it 4 or 6
    encoder_parser.add_argument("--filter_base", default=8, type=int) # the base number of filters, which is doubled for each convolutional layer
    encoder_parser.add_argument("--num_convs", default=4, type=int) # total number of convolutions, after which the dimension is size / 2^num_convs, and the final number of filters is filter_base * 2^(num_convs-1)
    vector_parser.add_argument("--obs_dim", default=None, type=int)
    world_model_parser.add_argument("--hidden_dim", default=1024, type=int) # hidden dims of MLPs
    world_model_parser.add_argument("--hidden_state_size", default=8192, type=int) # hidden state of GRU/RSSM
    world_model_parser.add_argument("--num_hiddens", default=1, type=int) # determines the depth for various MLPs
    world_model_parser.add_argument("--action_dim", default=18, type=int)
    world_model_parser.add_argument("--action_type", default="discrete", choices=["discrete", "continuous"])
    world_model_parser.add_argument("--num_latents", default=32, type=int) # the number of rows in the latent
    world_model_parser.add_argument("--num_codes", default=64, type=int) # the actual dim that's softmaxed over in the latent
    world_model_parser.add_argument("--latent_unimix", default=0.01, type=float)
    world_model_parser.add_argument("--use_block_linear", default=True, type=bool)
    world_model_parser.add_argument("--act", default="silu", choices=["silu", "gelu", "relu"])
    world_model_parser.add_argument("--prediction_loss_coeff", default=1, type=float)
    world_model_parser.add_argument("--dynamics_loss_coeff", default=1, type=float)
    world_model_parser.add_argument("--representation_loss_coeff", default=0.1, type=float)
    world_model_parser.add_argument("--free_nats", default=1, type=float)
    world_model_parser.add_arugment("--bin_low", default=-20, type=int)
    world_model_parser.add_argument("--bin_high", default=20, type=int)
    parser.add_argument("--rollout_length", default=16, type=int)
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
    config.latent_size = config.num_latents * config.num_codes
    config.state_size = config.hidden_state_size + config.latent_size
    config.num_bins = (config.bin_high - config.bin_low) + 1 # inclusive of both high and low
    
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