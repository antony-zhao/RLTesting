import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from rltesting.utils.torch_utils import to_numpy
from rltesting.torch_rl.dreamer.dreamer import (
    DreamerEncoderConv, DreamerDecoderConv, 
    DreamerDynamics, DreamerMLP, 
    unimix, transform_obs, WeightedAverageOverBins
)
from torch.distributions import OneHotCategoricalStraightThrough, Independent, Categorical, Normal, kl_divergence, Bernoulli
from mamba_ssm import Mamba2
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import MambaWrapperModel, MambaConfig
from mamba_ssm.utils.generation import InferenceParams, update_graph_cache

def probs_to_dist(probs):
    dist = Independent(OneHotCategoricalStraightThrough(probs), 1)
    return dist

class DramaEncoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    # hidden_state_size is h_t (the hidden state of the recurrent network)
    # categories is number of rows, codes is the number of columns (softmaxed over codes)
    # hidden dim is just the hidden dim of linear layers
    def __init__(self, config):
        super().__init__()
        if config.obs_type == "image":
            self.encoder = DreamerEncoderConv(config.filter_base, config.num_convs, config.kernel_size, config.num_channels, config.image_size, config.act)
            mlp_dim = np.prod(self.encoder.output_size)
        else:
            mlp_dim = config.obs_dim
        self.out = DreamerMLP(mlp_dim, config.latent_size, config.hidden_dim, 
                       num_hiddens=config.num_hiddens_world_model)
        self.config = config
        self.is_image = config.obs_type == "image"
    
    def forward(self, x):
        # x as in the observation specifically, h is the same hidden state
        encoded = self.embed_obs(x)
        out = self.out(encoded)
        return out
    
    def embed_obs(self, x):
        if self.is_image:
            encoded = self.encoder(x).flatten(-3)
            return encoded
        else:
            return x
    
    def compute_latent(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits.reshape(logits.shape[0], self.num_categoricals, self.num_codes), -1)
        unimixed_probs = unimix(probs, self.num_codes, self.config.latent_unimix)
        return unimixed_probs

class DramaDecoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    def __init__(self, config):
        super().__init__()
        self.obs_type = config.obs_type
        if config.obs_type == "image":
            self.input_dim = config.output_dim
            self._in = nn.Linear(config.latent_size, np.prod(self.input_dim))
            self.decoder = DreamerDecoderConv(config.filter_base, config.num_convs, config.kernel_size, config.num_channels, config.act)
        else:
            self._in = None
            self.decoder = DreamerMLP(config.latent_size, config.obs_dim, config.hidden_dim, num_hiddens=config.num_hiddens_world_model)
    
    def forward(self, x):
        if self._in:
            x = self._in(x)
        if self.obs_type == "image":
            x = x.reshape(-1, *self.input_dim)
        reconstruction = self.decoder(x)
        return reconstruction

class DramaWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = DramaEncoder(config) # reusing a lot of these dreamer components just to not rewrite them, since the main point is changing out the backbone
        self.decoder = DramaDecoder(config)
        mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=config.action_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state, 
                    'layer': 'Mamba2'}
                )
        self.mamba = MambaWrapperModel(mamba_config)
        self.dynamics_predictor = DreamerDynamics(config)
        self.reward_predictor = DreamerMLP(config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_world_model)
        self.continue_predictor = DreamerMLP(config.state_size, 1, config.hidden_dim, config.num_hiddens_world_model)
        self.rollout_length = config.rollout_length
        self.config = config
        self.is_image = self.config.obs_type == "image"
        self.bins = torch.linspace(config.bin_low, config.bin_high, config.num_bins).to(config.device)
    
    def recurrent_step(self, latent, hidden, action):
        if self.config.action_type == "discrete":
            action = F.one_hot(action.long(), self.config.action_dim).float()
        next_hidden = self.mamba.step(torch.cat([latent.flatten(-2), action], dim=-1), hidden)
        return next_hidden
    
    def imagine_step(self, latent, hidden, action):
        # treat this like how you would a normal environment step
        with torch.no_grad():
            state = torch.cat([latent, action], -1)
            continue_prob = F.sigmoid(self.continue_predictor(state))
            reward_logits = self.reward_predictor(state)
            reward = WeightedAverageOverBins(self.bins, reward_logits).weighted_average()
            next_hidden = self.mamba.step(latent, hidden, action)
            probs_dyn = self.dynamics_predictor(next_hidden)
            next_latent = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
        return (next_latent, next_hidden), reward, continue_prob
    
    def world_model_loss(self, obs, actions, rewards, dones):
        dyn_probs = []
        transformed_obs = transform_obs(obs, self.is_image).transpose(0, 1).contiguous()
        if self.is_image:
            T, B, C, H, W = transformed_obs.shape
            flat_transform_obs = transformed_obs.reshape(B * T, C, H, W)
        else:
            T, B, dim = transformed_obs.shape
            flat_transform_obs = transformed_obs.reshape(B * T, dim)
        enc_probs = self.encoder.compute_latent(flat_transform_obs)
        latent = probs_to_dist(enc_probs).rsample()
        recurrent_state = self.mamba(latent)
        actions = actions.transpose(0, 1).contiguous()
        rewards = rewards.transpose(0, 1).contiguous()
        dones = dones.transpose(0, 1).contiguous()
        continue_preds = self.continue_predictor(recurrent_state)
        reward_logits = self.reward_predictor(recurrent_state)
        reconstructions = self.decoder.from_state(latent.reshape(T * B, -1)).reshape(transformed_obs.shape)
        dyn_probs = self.dynamics_predictor(recurrent_state)
        pred_loss, loss_dict = self.prediction_loss(transformed_obs, reconstructions, rewards, reward_logits, dones, continue_preds)
        dyn_loss = self.dynamics_loss(enc_probs, dyn_probs)
        rep_loss = self.representation_loss(enc_probs, dyn_probs)
        loss = pred_loss * self.config.prediction_loss_coef + dyn_loss * self.config.dynamics_loss_coef + rep_loss * self.config.representation_loss_coef
        loss_dict["loss/KL divergence"] = to_numpy(dyn_loss)
        return loss, loss_dict, latent.transpose(0, 1).detach()
    
    def prediction_loss(self, obs, reconstruction, reward, reward_logits, dones, continue_logits):
        if self.is_image:
            obs = obs.flatten(2)
            reconstruction = reconstruction.flatten(2)
        reconstruction_error = ((obs - reconstruction) ** 2).sum(-1).mean()
        reward_prediction = WeightedAverageOverBins(self.bins, reward_logits)
        reward_error = -reward_prediction.log_prob(reward, aggregate=False).mean()
        continue_dist = Independent(Bernoulli(logits=continue_logits), 1)
        continue_error = -continue_dist.log_prob(1 - dones.unsqueeze(-1)).mean()
        total_loss = reconstruction_error + reward_error + continue_error * 10
        return total_loss, {"loss/reconstruction loss": to_numpy(reconstruction_error), 
                                                                      "loss/reward loss": to_numpy(reward_error), 
                                                                      "loss/continue loss": to_numpy(continue_error)}
    
    def dynamics_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc.detach()), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1)
        kl_div = kl_divergence(latent_enc, latent_dyn)
        return torch.clip(kl_div, min=self.config.free_nats).mean() 
    
    def representation_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn.detach()), 1)
        kl_div = kl_divergence(latent_enc, latent_dyn)
        return torch.clip(kl_div, min=self.config.free_nats).mean()