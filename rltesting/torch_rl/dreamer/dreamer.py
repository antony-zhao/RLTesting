from torch import nn
import torch
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from rltesting.torch_rl.models import DreamerDecoderConv, DreamerEncoderConv, DreamerMLP, NormAndAct, DreamerGRU, TargetNetwork
from rltesting.torch_rl.dreamer.utils import *
from rltesting.utils.torch_utils import to_numpy
from rltesting.torch_rl.buffers import PerEnvBuffer
from functools import partial
from torch.distributions import OneHotCategoricalStraightThrough, Independent, Categorical, Normal, kl_divergence, Bernoulli
from torch.distributions.utils import probs_to_logits

def make_state(latent, hidden):
    return torch.cat([latent.flatten(-2), hidden], -1)

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
            self.encoder = nn.Linear(config.obs_dim, config.hidden_dim)#DreamerMLP(config.obs_dim, config.hidden_dim, config.hidden_dim, num_hiddens=config.num_hiddens_world_model)
            output_dim = config.hidden_dim
        self.out = DreamerMLP(output_dim + config.hidden_state_size, config.latent_size, config.hidden_dim, 
                       num_hiddens=config.num_hiddens_world_model)
        self.num_categoricals = config.num_categoricals
        self.num_codes = config.num_codes
        self.config = config
        self.is_image = config.obs_type == "image"
    
    def forward(self, x, h):
        # x as in the observation specifically, h is the same hidden state
        embedded = self.embed_observations(x)
        return self.compute_latent(embedded, h)
    
    def embed_observations(self, x):
        # since the encoder computation can be batched without needing to incorporate the hidden state
        encoded = self.encoder(x)
        if self.is_image:
            encoded = torch.flatten(encoded, -3)
        return encoded
    
    def compute_latent(self, embedded, hidden):
        states = torch.cat([embedded, hidden], 1)
        logits = self.out(states)
        probs = torch.softmax(logits.reshape(logits.shape[0], self.num_categoricals, self.num_codes), -1)
        unimixed_probs = unimix(probs, self.num_codes, self.config.latent_unimix)
        return unimixed_probs

class DreamerDecoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    def __init__(self, config):
        super().__init__()
        self.obs_type = config.obs_type
        if config.obs_type == "image":
            self.input_dim = config.output_dim
            self._in = nn.Linear(config.state_size, np.prod(self.input_dim))
            self.decoder = DreamerDecoderConv(config.filter_base, config.num_convs, config.kernel_size, config.num_channels, config.act)
        else:
            self._in = None
            self.decoder = DreamerMLP(config.state_size, config.obs_dim, config.hidden_dim, num_hiddens=config.num_hiddens_world_model)
    
    def from_state(self, x):
        if self._in:
            x = self._in(x)
        if self.obs_type == "image":
            x = x.reshape(-1, *self.input_dim)
        reconstruction = self.decoder(x)
        return reconstruction
    
    def forward(self, z, h):
        x = make_state(z, h)
        reconstruction = self.from_state(x)
        return reconstruction

class DreamerDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dynamics = DreamerMLP(config.hidden_state_size, config.latent_size, config.hidden_dim, config.num_hiddens_world_model)
    
    def forward(self, hidden):
        logits_dyn = self.dynamics(hidden)
        logits_dyn = logits_dyn.reshape(-1, self.config.num_categoricals, self.config.num_codes)
        probs_dyn = torch.softmax(logits_dyn, -1)
        probs_dyn = unimix(probs_dyn, self.config.num_codes, self.config.latent_unimix)
        return probs_dyn

class RSSM(nn.Module):
    # Also called the sequence model
    def __init__(self, config):
        super().__init__()
        self.in_hidden = nn.Linear(config.hidden_state_size, config.hidden_dim)
        self.in_latent = nn.Linear(config.num_categoricals * config.num_codes, config.hidden_dim)
        self.in_action = nn.Linear(config.action_dim, config.hidden_dim) # the required input layers for each of the three needed inputs
        self.act1 = NormAndAct(config.hidden_dim * 3)
        self.mlp = DreamerMLP(config.hidden_dim * 3, config.hidden_state_size, config.hidden_dim, num_hiddens=config.num_hiddens_world_model)
        self.gru = DreamerGRU(config.hidden_state_size, config.use_block_linear)
    
    def forward(self, z, h, a):
        x1 = self.in_hidden(h)
        x2 = self.in_latent(z)
        x3 = self.in_action(a)
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
        self.dynamics_predictor = DreamerDynamics(config)
        self.reward_predictor = DreamerMLP(config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_world_model)
        self.continue_predictor = DreamerMLP(config.state_size, 1, config.hidden_dim, config.num_hiddens_world_model)
        self.initial_hidden = nn.Parameter(torch.zeros(config.hidden_state_size).float())
        self.rollout_length = config.rollout_length
        self.config = config
        self.is_image = self.config.obs_type == "image"
        self.bins = torch.linspace(config.bin_low, config.bin_high, config.num_bins).to(config.device)
        self.h_initial = None
    
    def compute_latents(self, obs_embedding, hidden):
        probs_enc = self.encoder.compute_latent(obs_embedding, hidden)
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1).rsample()
        probs_dyn = self.dynamics_predictor(hidden)
        return probs_enc, latent_enc, probs_dyn
    
    def imagine_step(self, latent, hidden, action):
        # treat this like how you would a normal environment step
        # state is {z, h}
        with torch.no_grad():
            state = torch.cat([latent, hidden], -1)
            continue_prob = F.sigmoid(self.continue_predictor(state))
            continue_ = ((continue_prob) > 0.5).to(continue_prob)
            reward_logits = self.reward_predictor(state)
            reward = WeightedAverageOverBins(self.bins, reward_logits).weighted_average()
            next_hidden = self.rssm(latent, hidden, action) * continue_ + (1 - continue_) * self._get_hidden(action.shape[0]).detach()
            probs_dyn = self.dynamics_predictor(hidden)
            next_latent = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
        return (next_latent, next_hidden), reward, continue_
    
    def dynamic_step(self, obs_embedding, action, done, hidden, is_first):
        h_initial = self._get_hidden(action.shape[0])
        
        if hidden is None:
            hidden = h_initial
        else:
            hidden = hidden * (1 - is_first) + h_initial * is_first

        probs_enc, latent_enc, probs_dyn = self.compute_latents(obs_embedding, hidden)

        state = make_state(latent_enc, hidden)
        done = done.unsqueeze(-1)
        next_hidden = self.rssm(latent_enc.flatten(-2), hidden, action)# * (1 - done) + done * h_initial
        return probs_enc, probs_dyn, state, next_hidden
    
    def _get_hidden(self, batch_size):
        return torch.tanh(self.initial_hidden.expand(batch_size, -1))
        
    def recurrent_step(self, hidden, latent, action):
        if self.config.action_type == "discrete":
            action = F.one_hot(action.long(), self.config.action_dim).float()
        next_hidden = self.rssm(latent.flatten(-2), hidden, action)
        return next_hidden
    
    def world_model_loss(self, obs, actions, rewards, dones):
        enc_probs = []
        dyn_probs = []
        states = []
        transformed_obs = transform_obs(obs, self.is_image)
        if self.is_image:
            B, T, C, H, W = transformed_obs.shape
            flat_transform_obs = transformed_obs.reshape(B * T, C, H, W)
        else:
            B, T, dim = transformed_obs.shape
            flat_transform_obs = transformed_obs.reshape(B * T, dim)
        obs_embeddings = self.encoder.embed_observations(flat_transform_obs)
        obs_embeddings = obs_embeddings.reshape(B, T, -1).transpose(0, 1)
        actions = actions.transpose(0, 1).contiguous()
        rewards = rewards.transpose(0, 1).contiguous()
        dones = dones.transpose(0, 1).contiguous()
        is_first = torch.zeros_like(dones)
        is_first[0] = 1
        is_first[1:] = dones[:-1]
        is_first = is_first.unsqueeze(-1).float()
        hidden = None
        for i in range(T):
            probs_enc, probs_dyn, state, hidden = self.dynamic_step(
                obs_embeddings[i], actions[i], dones[i], hidden, is_first[i] #i == 0
            )
            enc_probs.append(probs_enc)
            dyn_probs.append(probs_dyn)
            states.append(state)
            
        states = torch.stack(states)
        enc_probs = torch.stack(enc_probs)
        dyn_probs = torch.stack(dyn_probs)
        continue_preds = self.continue_predictor(states)
        reward_logits = self.reward_predictor(states)
        reconstructions = self.decoder.from_state(states.reshape(T * B, -1)).reshape(obs.shape)
        pred_loss, loss_dict = self.prediction_loss(transformed_obs, reconstructions, rewards, reward_logits, dones, continue_preds)
        dyn_loss = self.dynamics_loss(enc_probs, dyn_probs)
        rep_loss = self.representation_loss(enc_probs, dyn_probs)
        loss = pred_loss * self.config.prediction_loss_coef + dyn_loss * self.config.dynamics_loss_coef + rep_loss * self.config.representation_loss_coef
        loss_dict["loss/KL divergence"] = to_numpy(dyn_loss)
        return loss, loss_dict, states.transpose(0, 1).detach(), obs_embeddings.transpose(0, 1).detach()
    
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
          
class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.mlp = DreamerMLP(config.state_size, config.action_dim, config.hidden_dim, config.num_hiddens_actor_critic)
        self.action_type = config.action_type
        if config.action_type == "continuous":
            self.log_std = nn.Parameter(-torch.ones(config.action_dim))
        else:
            self.actor_unimix = config.actor_unimix
    
    def policy_dist(self, x):
        logits = self.mlp(x)
        if self.action_type == "discrete":
            probs = torch.softmax(logits, -1)
            unimixed_probs = unimix(probs, self.action_dim, self.actor_unimix)
            logits = probs_to_logits(unimixed_probs)
            action_dist = Categorical(logits=logits)
        else:
            action_dist = Normal(loc=logits, scale=torch.exp(self.log_std))
        return action_dist
    
    def sample_action(self, dist):
        if self.action_type == "discrete":
            action = dist.sample()
        else:
            action = dist.rsample()
        return action
    
    def policy_fn(self, x, det=False):
        # actually choosing the action, returns the actual action as well as the log prob of the action and entropy
        action_dist = self.policy_dist(x)
        if not det:
            action = self.sample_action(action_dist)
        else:
            return action_dist.mode
        return action.detach()
    
class Critic(nn.Module):
    def __init__(self, config, bins):
        super().__init__()
        self.mlp = DreamerMLP(config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_actor_critic)
        self.bins = bins
    
    def forward(self, x):
        logits = self.mlp(x)
        weighted_average = WeightedAverageOverBins(self.bins, logits)
        return weighted_average.weighted_average(), logits

class DreamerV3:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.world_model = DreamerWorldModel(config).to(self.device)
        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config, self.world_model.bins).to(self.device)
        self.init_models()
        self.critic_target = TargetNetwork(self.critic, config.critic_tau)
        self.optim_wm = Adam(self.world_model.parameters(), config.wm_lr, eps=1e-5)
        self.optim_actor = Adam(self.actor.parameters(), config.ac_lr, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), config.ac_lr, eps=1e-5)
        
        act_dim = () if config.action_type == "discrete" else (config.action_dim,)
        if config.obs_type == "image":
            self.is_image = True
            self.buffer = PerEnvBuffer(config.num_envs, [config.image_dim, act_dim, (), ()], dtypes=[np.uint8, np.int32, np.float32, np.bool], buffer_size=1_000_000)
        elif config.obs_type == "vector":
            self.is_image = False
            self.buffer = PerEnvBuffer(config.num_envs, [(config.obs_dim,), act_dim, (), ()], buffer_size=1_000_000)
        else:
            raise NotImplemented
        # buffer needs to account for order in episodes
        self.active_hidden = torch.zeros(config.num_envs, config.hidden_state_size).to(self.device)
        self.eval_hidden = torch.zeros(1, config.hidden_state_size).to(self.device)
        # the history for the environment itself, keeping track of it in here since 
        # it would be a bit weird to have this be in the main part of the program
        
        self.range_ema = None
        self.return_range_tau = config.return_range_tau
        # Used for calculating the range of returns to help normalize the reinforce gradient
        
        self.gamma = config.gamma
        self.lambda_ = config.lambda_
        self.percentiles = config.percentiles

    def choose_action(self, obs, det=False):
        transformed_obs = transform_obs(obs, self.is_image)
        latent_prob = self.world_model.encoder(transformed_obs, self.active_hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, self.active_hidden)
        action = self.actor.policy_fn(state, det)
        return action, latent
    
    def eval_action(self, obs, det=True, reset=False):
        transformed_obs = transform_obs(obs, self.is_image)
        if reset:
            self.eval_hidden = self.world_model._get_hidden(1)
        latent_prob = self.world_model.encoder(transformed_obs, self.eval_hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, self.eval_hidden)
        action = self.actor.policy_fn(state, det)
        self.eval_hidden = self.world_model.recurrent_step(self.eval_hidden, latent, action).detach()
        return to_numpy(action)
    
    def process_sample(self, obs, latent, action, reward, done):
        # do a step in RSSM and store stuff in buffer
        self.buffer.add_sample([obs, to_numpy(action), reward, done])

        continue_ = torch.tensor(1 - done).unsqueeze(1).to(self.device)
        self.active_hidden = (continue_ * self.world_model.recurrent_step(self.active_hidden, latent, action) +
                              (1 - continue_) * self.world_model._get_hidden(self.config.num_envs)).detach()
    
    def imagine_rollout(self, state, steps=None):
        states = []
        actions = []
        action_log_probs = []
        action_entropies = []
        rewards = []
        continues = []
        for _ in range(self.config.rollout_length if steps is None else steps):
            action_dist = self.actor.policy_dist(state)
            action = self.actor.sample_action(action_dist).detach()
            action_prob = action_dist.log_prob(action)
            action_log_probs.append(action_prob)
            action_entropy = action_dist.entropy()
            action_entropies.append(action_entropy)
            if self.config.action_type == "discrete":
                action = F.one_hot(action.long(), self.config.action_dim).float()
            (next_latent, next_hidden), reward, continue_ = self.world_model.imagine_step(state[:, :self.config.latent_size], state[:, self.config.latent_size:], action)
            states.append(state.detach())
            actions.append(action.detach())
            rewards.append(reward.detach())
            continues.append(continue_.squeeze().detach())
            state = torch.concatenate([next_latent.flatten(-2), next_hidden], 1)
        states.append(state.detach())
        return torch.stack(states), torch.stack(rewards), torch.stack(continues), torch.stack(action_log_probs), torch.stack(action_entropies)
    
    def train(self):
        obs, actions, rewards, dones = self.buffer.sample_as_tensors(self.config.device, self.config.sample_batch_size, self.config.sample_seq_len)
        if self.config.action_type == "discrete":
            actions = F.one_hot(actions.long(), self.config.action_dim).float()
        loss_wm, loss_dict, new_states, _ = self.world_model.world_model_loss(obs, actions, rewards, dones)
        self.optim_wm.zero_grad()
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 5)
        self.optim_wm.step()
        
        states, rewards, continues, log_probs, entropy = self.imagine_rollout(new_states.reshape(-1, self.config.state_size))
        continues[0] = (1 - dones.flatten())
            
        loss_critic, returns, values = self.train_critic(states, rewards, continues)
        loss_actor, actor_ent = self.train_actor(returns, values, log_probs, entropy)
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.optim_critic.step()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.optim_actor.step()
        
        loss_dict["loss/actor loss"] = to_numpy(loss_actor)
        loss_dict["loss/critic loss"] = to_numpy(loss_critic)
        loss_dict["loss/actor entropy"] = to_numpy(actor_ent)
        return to_numpy(loss_wm), to_numpy(loss_critic), to_numpy(loss_actor), loss_dict 
    
    def train_actor(self, returns, values, action_log_probs, entropy):
        range_ = torch.quantile(returns, 1 - self.percentiles) - torch.quantile(returns, self.percentiles)
        if self.range_ema is not None:
            self.range_ema = range_ * self.return_range_tau + self.range_ema * (1 - self.return_range_tau)
        else:
            self.range_ema = range_
        
        adv = ((returns - values) / torch.clip(self.range_ema, min=1)).detach()
        actor_loss = -(adv * action_log_probs + entropy * self.config.entropy_coef)
        actor_loss = actor_loss.mean()
        return actor_loss, entropy.mean()
    
    def train_critic(self, states, rewards, continues):
        self.critic_target.update()
        values, value_logits = self.critic(states)
        value_target, _ = self.critic_target(states)
        returns = compute_lambda_returns(values, rewards, continues, self.gamma, self.lambda_)
        value_bins = WeightedAverageOverBins(self.world_model.bins, value_logits[:-1])
        loss = -value_bins.log_prob(returns.detach(), aggregate=False)
        loss -= value_bins.log_prob(value_target.detach()[:-1], aggregate=False)
        loss = loss.mean()
        return loss, returns.detach(), values.detach()[:-1] # returning returns and values for the actor to reuse later
        
    def checkpoint_models(self, folderpath, filename):
        torch.save(self.world_model.state_dict(), f"{folderpath}/world_model-{filename}.pth")
        torch.save(self.actor.state_dict(), f"{folderpath}/actor-{filename}.pth")
        torch.save(self.critic.state_dict(), f"{folderpath}/critic-{filename}.pth")
        self.buffer.save(f"{folderpath}/buffer_{filename}.npz")
    
    def init_models(self):
        self.critic.apply(init_weights)
        self.actor.apply(init_weights)
        self.world_model.reward_predictor.apply(init_weights)
        self.world_model.continue_predictor.apply(init_weights)
        self.world_model.dynamics_predictor.apply(init_weights)
        self.world_model.encoder.apply(init_weights)
        self.world_model.decoder.apply(init_weights)
        self.world_model.rssm.apply(init_weights)
        
        init_last_layer(self.critic, nn.init.zeros_)
        init_last_layer(self.world_model.reward_predictor, nn.init.zeros_)
        init_last_layer(self.actor, nn.init.xavier_uniform_)
        init_last_layer(self.world_model.continue_predictor, nn.init.xavier_uniform_)
        init_last_layer(self.world_model.dynamics_predictor, nn.init.xavier_uniform_)
        init_last_layer(self.world_model.encoder, nn.init.xavier_uniform_)
        init_last_layer(self.world_model.decoder, nn.init.xavier_uniform_)
        init_last_layer(self.world_model.rssm, nn.init.xavier_uniform_)
