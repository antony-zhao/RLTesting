from torch import nn
import torch
from torch.optim import RMSprop, Adam
import numpy as np
import torch.nn.functional as F
from rltesting.torch_rl.models import DreamerDecoderConv, DreamerEncoderConv, MLP, NormAndAct, DreamerGRU, TargetNetwork
from rltesting.torch_rl.dreamer.utils import *
from rltesting.utils.torch_utils import to_numpy
from rltesting.torch_rl.buffers import PerEnvBuffer
import argparse
from functools import partial
from torch.distributions import OneHotCategoricalStraightThrough, Independent, Categorical, Normal, kl_divergence, Bernoulli
from torch.distributions.utils import probs_to_logits

mlp_norm_act = lambda dim, act: partial(NormAndAct, norm_dim=dim, act=act)

make_state = lambda latent, hidden: torch.cat([latent.flatten(-2), hidden], -1)

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
            self.encoder = MLP(config.obs_dim, config.hidden_dim, config.hidden_dim, num_hiddens=1, act=mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
            output_dim = config.hidden_dim
        self.out = MLP(output_dim + config.hidden_state_size, config.latent_size, config.hidden_dim, 
                       num_hiddens=config.num_hiddens_world_model, act=mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
        self.num_categoricals = config.num_categoricals
        self.num_codes = config.num_codes
        self.config = config
    
    def forward(self, x, h):
        # x as in the observation specifically, h is the same hidden state
        embedded = self.embed_observations(x)
        return self.compute_latent(embedded, h)
    
    def embed_observations(self, x):
        # since the encoder computation can be batched without needing to incorporate the hidden state
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, -3)
        return encoded
    
    def compute_latent(self, embedded, hidden):
        states = torch.cat([embedded, hidden], 1)
        logits = self.out(states)
        probs = F.softmax(logits.reshape(logits.shape[0], self.num_categoricals, self.num_codes), -1)
        unimixed_probs = unimix(probs, self.num_codes, self.config.latent_unimix)
        return unimixed_probs

class DreamerDecoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    def __init__(self, config):
        super().__init__()
        self.obs_type = config.obs_type
        if config.obs_type == "image":
            self.input_dim = config.output_dim
            self._in = MLP(config.state_size, np.prod(self.input_dim), config.hidden_dim, 
                       num_hiddens=config.num_hiddens_world_model, act=mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
            self.decoder = DreamerDecoderConv(config.filter_base, config.num_convs, config.kernel_size, config.num_channels, config.act)
        else:
            self._in = nn.Linear(config.hidden_state_size + config.num_categoricals * config.num_codes, config.hidden_dim)
            self.decoder = MLP(config.hidden_dim, config.hidden_dim, config.obs_dim, num_hiddens=config.num_hiddens_world_model, act=mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
    
    def from_state(self, state):
        x = self._in(state)
        if self.obs_type == "image":
            x = x.reshape(-1, *self.input_dim)
        reconstruction = self.decoder(x)
        return reconstruction
    
    def forward(self, z, h):
        x = make_state(z, h)
        reconstruction = self.from_state(x)
        return reconstruction

class RSSM(nn.Module):
    # Also called the sequence model
    def __init__(self, config):
        super().__init__()
        self.in_hidden = nn.Linear(config.hidden_state_size, config.hidden_dim)
        self.in_latent = nn.Linear(config.num_categoricals * config.num_codes, config.hidden_dim)
        self.in_action = nn.Linear(config.action_dim, config.hidden_dim) # the required input layers for each of the three needed inputs
        self.act1 = NormAndAct(config.hidden_dim * 3)
        self.mlp = MLP(config.hidden_dim * 3, config.hidden_state_size, config.hidden_dim, num_hiddens=config.num_hiddens_world_model, 
                       act=mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
        self.gru = DreamerGRU(config.hidden_state_size, config.use_block_linear)
    
    def forward(self, z, h, a):
        x1 = self.in_hidden(h)
        x2 = self.in_latent(z)
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
        self.dynamics_predictor = MLP(config.hidden_state_size, config.latent_size, config.hidden_dim, config.num_hiddens_world_model, mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
        self.reward_predictor = MLP(config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_world_model, mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False, final_act=nn.Softmax(-1))
        self.continue_predictor = MLP(config.state_size, 1, config.hidden_dim, config.num_hiddens_world_model, mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
        self.initial_hidden = nn.Parameter(torch.zeros(config.hidden_state_size).float())
        self.rollout_length = config.rollout_length
        self.config = config
        self.is_image = self.config.obs_type == "image"
        self.bins = torch.linspace(config.bin_low, config.bin_high, config.num_bins).to(config.device)
    
    def compute_latents(self, obs_embedding, hidden):
        probs_enc = self.encoder.compute_latent(obs_embedding, hidden)
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1).rsample()
        logits_dyn = self.dynamics_predictor(hidden)
        logits_dyn = logits_dyn.reshape(-1, self.config.num_categoricals, self.config.num_codes)
        probs_dyn = torch.softmax(logits_dyn, -1)
        probs_dyn = unimix(probs_dyn, self.config.num_codes, self.config.latent_unimix)
        return probs_enc, latent_enc, probs_dyn
    
    def imagine_step(self, latent, hidden, action):
        # treat this like how you would a normal environment step
        # state is {z, h}
        with torch.no_grad():
            state = torch.cat([latent, hidden], -1)
            continue_prob = F.sigmoid(self.continue_predictor(state))
            continue_ = ((continue_prob) > 0.5).to(continue_prob)
            reward_probs = self.reward_predictor(state)
            reward = WeightedAverageOverBins(self.bins, reward_probs).weighted_average()
            next_hidden = self.rssm(latent, hidden, action) * continue_ + (1 - continue_) * self._get_hidden(action.shape[0]).detach()
            logits_dyn = self.dynamics_predictor(hidden)
            logits_dyn = logits_dyn.reshape(-1, self.config.num_categoricals, self.config.num_codes)
            probs_dyn = torch.softmax(logits_dyn, -1)
            probs_dyn = unimix(probs_dyn, self.config.num_codes, self.config.latent_unimix)
            next_latent = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
        return (next_latent, next_hidden), reward, continue_
    
    def dynamic_step(self, obs_embedding, action, hidden, done, is_first):
        # following a similar way the way sheeprl implements this but also not exactly the same
        initial_hidden = self._get_hidden(action.shape[0])
        if is_first:
            hidden = initial_hidden
        probs_enc, latent_enc, probs_dyn = self.compute_latents(obs_embedding, hidden)
        state = make_state(latent_enc, hidden)
        reward_prob = self.reward_predictor(state)
        continue_pred = self.continue_predictor(state)
        done = done.unsqueeze(-1)
        next_hidden = self.rssm(latent_enc.flatten(-2), hidden, action) * (1 - done) + done * initial_hidden
        return probs_enc, probs_dyn, reward_prob, continue_pred, state, next_hidden
    
    def _get_hidden(self, batch_size):
        h_0 = torch.tanh(self.initial_hidden.expand(batch_size, -1))
        logits_dyn = self.dynamics_predictor(h_0)
        logits_dyn = logits_dyn.reshape(-1, self.config.num_categoricals, self.config.num_codes)
        probs_dyn = torch.softmax(logits_dyn, -1)
        probs_dyn = unimix(probs_dyn, self.config.num_codes, self.config.latent_unimix)
        z_0 = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).mode
        return self.rssm(z_0.flatten(-2), h_0, torch.zeros(batch_size, self.config.action_dim).to(self.config.device))
    
    def recurrent_step(self, hidden, latent, action):
        if self.config.action_type == "discrete":
            action = F.one_hot(action, self.config.action_dim).float()
        next_hidden = self.rssm(latent.flatten(-2), hidden, action)
        return next_hidden
    
    def world_model_loss(self, obs, actions, rewards, dones, hidden):
        enc_probs = []
        dyn_probs = []
        continue_preds = []
        reward_probs = []
        states = []
        if self.is_image:
            obs = obs / 255.0 - 0.5
        transformed_obs = symlog(obs)
        T, B, C, H, W = transformed_obs.shape
        obs_embeddings = self.encoder.embed_observations(transformed_obs.reshape(T * B, C, H, W))
        obs_embeddings = obs_embeddings.reshape(T, B, -1)
        for i in range(T):
            probs_enc, probs_dyn, reward_prob, continue_pred, state, hidden = self.dynamic_step(
                obs_embeddings[i], actions[i], hidden, dones[i], i == 0)
            enc_probs.append(probs_enc)
            dyn_probs.append(probs_dyn)
            states.append(state.detach())
            reward_probs.append(reward_prob)
            continue_preds.append(continue_pred)
            
        states = torch.stack(states)
        enc_probs = torch.stack(enc_probs)
        dyn_probs = torch.stack(dyn_probs)
        continue_preds = torch.stack(continue_preds)
        reward_probs = torch.stack(reward_probs)
        reconstructions = self.decoder.from_state(states.reshape(T * B, -1)).reshape(obs.shape)
        pred_loss, loss_dict = self.prediction_loss(obs, reconstructions, rewards, reward_probs, dones, continue_preds)
        dyn_loss = self.dynamics_loss(enc_probs, dyn_probs)
        rep_loss = self.representation_loss(enc_probs, dyn_probs)
        loss = pred_loss * self.config.prediction_loss_coef + dyn_loss * self.config.dynamics_loss_coef + rep_loss * self.config.representation_loss_coef
        loss_dict["KL divergence"] = to_numpy(dyn_loss)
        return loss, loss_dict, states
    
    def prediction_loss(self, obs, reconstruction, reward, reward_probs, dones, continue_logits):
        reconstruction_error = symlog_squared_error(obs.flatten(2), reconstruction.flatten(2))
        reward_prediction = WeightedAverageOverBins(self.bins, reward_probs)
        reward_error = reward_prediction.log_prob(reward)
        continue_dist = Independent(Bernoulli(logits=continue_logits), 1)
        continue_error = continue_dist.log_prob(1 - dones.unsqueeze(-1)).mean()
        total_loss = reconstruction_error - reward_error - continue_error
        return total_loss, {"reconstruction_loss": to_numpy(reconstruction_error), 
                                                                      "reward_loss": to_numpy(-reward_error), 
                                                                      "continue_loss": to_numpy(-continue_error)}
    
    def dynamics_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc.detach()), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1)
        return torch.clip(kl_divergence(latent_enc, latent_dyn), min=1).mean()
    
    def representation_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn.detach()), 1)
        return torch.clip(kl_divergence(latent_enc, latent_dyn), min=1).mean()
          
class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.mlp = MLP(config.state_size, config.action_dim, config.hidden_dim, config.num_hiddens_actor_critic, mlp_norm_act(config.hidden_dim, config.act), use_bias_hidden=False, skip_connections=False)
        self.action_type = config.action_type
        if config.action_type == "continuous":
            self.log_std = nn.Parameter(-torch.ones(config.action_dim))
    
    def policy_dist(self, x):
        logits = self.mlp(x)
        if self.action_type == "discrete":
            probs = F.softmax(logits, -1)
            unimixed_probs = unimix(probs, self.action_dim)
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
        self.mlp = MLP(config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_actor_critic, mlp_norm_act(config.hidden_dim, config.act), 
                       use_bias_hidden=False, skip_connections=False, final_act=nn.Softmax(-1))
        self.bins = bins
    
    def forward(self, x):
        probs = self.mlp(x)
        weighted_average = WeightedAverageOverBins(self.bins, probs)
        return weighted_average.weighted_average(), probs

class DreamerV3:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.world_model = DreamerWorldModel(config).to(self.device)
        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config, self.world_model.bins).to(self.device)
        self.init_models()
        self.critic_target = TargetNetwork(self.critic, config.critic_tau)
        self.optim_wm = Adam(self.world_model.parameters(), config.lr, eps=1e-5)
        self.optim_actor = Adam(self.actor.parameters(), config.lr, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), config.lr, eps=1e-5)
        # decide whether or not to leave the optimizers in here, if so it might be good 
        # to modify some of the other classes to follow a similar 
        # pattern but then again dreamer is so much more different it might be fine
        act_dim = () if config.action_type == "discrete" else (config.action_dim,)
        if config.obs_type == "image":
            self.buffer = PerEnvBuffer(config.num_envs, [config.image_dim, act_dim, (), ()], dtypes=[np.uint8, np.int32, np.float32, np.float32, np.bool], buffer_size=1_000_000)
        elif config.obs_type == "vector":
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
        latent_prob = self.world_model.encoder(obs, self.active_hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, self.active_hidden)
        action = self.actor.policy_fn(state, det)
        return action, latent
    
    def eval_action(self, obs, det=False):
        latent_prob = self.world_model.encoder(obs, self.eval_hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, self.eval_hidden)
        action = self.actor.policy_fn(state, det)
        self.eval_hidden = self.world_model.recurrent_step(self.eval_hidden, latent, action).detach()
        return to_numpy(action)
    
    def process_sample(self, obs, latent, action, reward, done):
        # do a step in RSSM and store stuff in buffer
        self.buffer.add_sample([obs, to_numpy(action), reward, done])
        continue_ = torch.tensor(1 - done).unsqueeze(1).to(self.device)
        self.active_hidden = (continue_ * self.world_model.recurrent_step(self.active_hidden, latent, action) + (1 - continue_) * self.world_model._get_hidden(self.config.num_envs)).detach()
    
    def imagine_rollout(self, state):
        states = []
        actions = []
        action_log_probs = []
        action_entropies = []
        rewards = []
        dones = []
        for _ in range(self.config.rollout_length):
            action_dist = self.actor.policy_dist(state)
            action = self.actor.sample_action(action_dist)
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
            dones.append((1 - continue_).squeeze().detach())
            state = torch.concatenate([next_latent.flatten(-2), next_hidden], 1).detach()
        # print(action_dist.probs)
        return torch.stack(states), torch.stack(rewards), torch.stack(dones), torch.stack(action_log_probs), torch.stack(action_entropies)
    
    def train(self):
        obs, actions, rewards, dones = self.buffer.sample_as_tensors(self.config.device, self.config.sample_batch_size, self.config.sample_seq_len)
        hidden_start = torch.zeros(self.config.sample_batch_size, self.config.hidden_state_size).to(self.device) #states[0, :, self.config.latent_size:]
        if self.config.action_type == "discrete":
            actions = F.one_hot(actions.long(), self.config.action_dim).float()
        loss_wm, loss_dict, new_states = self.world_model.world_model_loss(obs, actions, rewards, dones, hidden_start)
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1000)
        self.optim_wm.step()
        self.optim_wm.zero_grad()
        # loss_critic1, _, _ = self.train_critic(new_states, rewards, dones)
        
        states, rewards, imagined_dones, log_probs, entropy = self.imagine_rollout(new_states.reshape(-1, self.config.state_size))
        # imagined_dones[0] = dones.flatten()
        with torch.no_grad():
            weights = (torch.cumprod((1 - imagined_dones) * self.config.gamma, dim=0) / self.config.gamma).detach()
        loss_critic2, returns, values = self.train_critic(states, rewards, imagined_dones, weights)
        loss_actor, actor_ent = self.train_actor(returns, values[:-1], log_probs[:-1], entropy[:-1], weights[:-1])
        
        loss_critic = loss_critic2 #+ 0.3 * loss_critic1
        loss_critic.backward()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.optim_actor.step()
        self.optim_actor.zero_grad()
        self.optim_critic.step()
        self.optim_critic.zero_grad()
        loss_dict["actor loss"] = to_numpy(loss_actor)
        loss_dict["critic loss"] = to_numpy(loss_critic)
        loss_dict["actor entropy"] = to_numpy(actor_ent)
        return to_numpy(loss_wm), to_numpy(loss_critic), to_numpy(loss_actor), loss_dict 
        
    def train_world_model(self, obs, actions, rewards, dones, hidden):
        loss, loss_dict = self.world_model.world_model_loss(obs, actions, rewards, dones, hidden)
        return loss, loss_dict
    
    def train_actor(self, returns, values, action_log_probs, entropy, weights):
        range_ = torch.quantile(returns, 1 - self.percentiles) - torch.quantile(returns, self.percentiles)
        if self.range_ema is not None:
            self.range_ema = range_ * self.return_range_tau + self.range_ema * (1 - self.return_range_tau)
        else:
            self.range_ema = range_
        
        actor_loss = -torch.mean(weights * (((returns - values) / torch.clip(self.range_ema, min=1)).detach() * action_log_probs + entropy * self.config.entropy_coef))
        return actor_loss, entropy.mean()
    
    def train_critic(self, states, rewards, dones, weights):
        self.critic_target.update(self.critic)
        values, value_probs = self.critic(states)
        value_target, _ = self.critic_target(states)
        continues = (1 - dones)
        returns = compute_lambda_values(values[1:], rewards[1:], continues[1:], self.gamma, self.lambda_)
        value_bins = WeightedAverageOverBins(self.world_model.bins, value_probs[:-1])
        loss = -value_bins.log_prob(returns.detach(), aggregate=False)
        loss -= value_bins.log_prob(value_target[:-1], aggregate=False)
        loss = (loss * weights[:-1]).mean()
        return loss, returns.detach(), values.detach() # returning returns and values for the actor to reuse later
        
    def checkpoint_models(self, folderpath, filename):
        torch.save(self.world_model.state_dict(), f"{folderpath}/world_model-{filename}.pth")
        torch.save(self.actor.state_dict(), f"{folderpath}/actor-{filename}.pth")
        torch.save(self.actor.state_dict(), f"{folderpath}/critic-{filename}.pth")
    
    def init_models(self):
        self.critic.apply(init_weights)
        self.actor.apply(init_weights)
        self.world_model.reward_predictor.apply(init_weights)
        self.world_model.continue_predictor.apply(init_weights)
        self.world_model.dynamics_predictor.apply(init_weights)
        self.world_model.encoder.apply(init_weights)
        self.world_model.decoder.apply(init_weights)
        self.world_model.rssm.apply(init_weights)
        
        # init_last_layer(self.critic, nn.init.zeros_)
        # init_last_layer(self.world_model.reward_predictor, nn.init.zeros_)
        # init_last_layer(self.actor, nn.init.xavier_uniform_)
        # init_last_layer(self.world_model.continue_predictor, nn.init.xavier_uniform_)
        # init_last_layer(self.world_model.dynamics_predictor, nn.init.xavier_uniform_)
        # init_last_layer(self.world_model.encoder, nn.init.xavier_uniform_)
        # init_last_layer(self.world_model.decoder, nn.init.xavier_uniform_)
        # init_last_layer(self.world_model.rssm, nn.init.xavier_uniform_)
        init_last_layer(self.critic, 0)
        init_last_layer(self.world_model.reward_predictor, 0)
        init_last_layer(self.actor, 1)
        init_last_layer(self.world_model.continue_predictor, 1)
        init_last_layer(self.world_model.dynamics_predictor, 1)
        init_last_layer(self.world_model.encoder, 1)
        init_last_layer(self.world_model.decoder, 1)
        # init_last_layer(self.world_model.rssm, 1)


if __name__ == "__main__":
    import gymnasium as gym
    from matplotlib import pyplot as plt
    from rltesting.utils.logger import Logger

    import ale_py
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.atari_wrappers import NoopResetEnv, FireResetEnv, EpisodicLifeEnv
    from rltesting.torch_rl.buffers import PerEnvBuffer
    import argparse
    import imageio
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    parser = argparse.ArgumentParser(description="just a temporary argparse while I debug the world model, will be moved elsewhere later")
    # specified for Atari specifically by default and using the 200M size model for Dreamer
    parser.add_argument("--obs_type", default="image", choices=["image", "vector", "multi"]) # need to handle multi for some robotics environments/experiments
    encoder_parser = parser.add_argument_group("encoder")
    vector_parser = parser.add_argument_group("vector")
    world_model_parser = parser.add_argument_group("world_model")
    encoder_parser.add_argument("--num_channels", default=3, type=int)
    encoder_parser.add_argument("--image_size", default=64, type=int) # images should be square
    encoder_parser.add_argument("--kernel_size", default=4, type=int) # best to keep it 4 or 6
    encoder_parser.add_argument("--filter_base", default=64, type=int) # the base number of filters, which is doubled for each convolutional layer
    encoder_parser.add_argument("--num_convs", default=4, type=int) # total number of convolutions, after which the dimension is size / 2^num_convs, and the final number of filters is filter_base * 2^(num_convs-1)
    vector_parser.add_argument("--obs_dim", default=None, type=int) # specify both this and image stuff for multi
    world_model_parser.add_argument("--hidden_dim", default=1024, type=int) # hidden dims of MLPs
    world_model_parser.add_argument("--hidden_state_size", default=8192, type=int) # hidden state of GRU/RSSM
    world_model_parser.add_argument("--num_hiddens_world_model", default=1, type=int) # determines the depth for MLPs in the world model
    world_model_parser.add_argument("--num_hiddens_actor_critic", default=2, type=int)
    world_model_parser.add_argument("--action_dim", default=4, type=int)
    world_model_parser.add_argument("--action_type", default="discrete", choices=["discrete", "continuous"])
    world_model_parser.add_argument("--num_categoricals", default=32, type=int) # the number of rows in the latent
    world_model_parser.add_argument("--num_codes", default=64, type=int) # the actual dim that's softmaxed over in the latent
    world_model_parser.add_argument("--latent_unimix", default=0.01, type=float)
    world_model_parser.add_argument("--use_block_linear", default=True, type=bool)
    world_model_parser.add_argument("--act", default="silu", choices=["silu", "gelu", "relu"])
    world_model_parser.add_argument("--prediction_loss_coef", default=1, type=float)
    world_model_parser.add_argument("--dynamics_loss_coef", default=0.5, type=float)
    world_model_parser.add_argument("--representation_loss_coef", default=0.1, type=float)
    world_model_parser.add_argument("--free_nats", default=1, type=float)
    world_model_parser.add_argument("--bin_low", default=-20, type=int)
    world_model_parser.add_argument("--bin_high", default=20, type=int)
    world_model_parser.add_argument("--num_bins", default=255, type=int)
    parser.add_argument("--sample_batch_size", default=16, type=int)
    parser.add_argument("--sample_seq_len", default=64, type=int)
    parser.add_argument("--imagination_batch_size", default=1024, type=int)
    parser.add_argument("--rollout_length", default=16, type=int)
    parser.add_argument("--critic_tau", default=0.02, type=float)
    parser.add_argument("--critic_imagination_loss_coef", default=1, type=float)
    parser.add_argument("--critic_replay_loss_coef", default=0.3, type=float)
    parser.add_argument("--entropy_coef", default=3e-4, type=float)
    parser.add_argument("--return_range_tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.997, type=float)
    parser.add_argument("--lambda_", default=0.95, type=float)
    parser.add_argument("--percentiles", default=0.05, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", default=8e-5, type=float)
    parser.add_argument("--num_envs", default=32, type=int)
    parser.add_argument("--replay_ratio", default=32, type=int) # batch_size * seq_len / timesteps_in_env = replay ratio, so default of 64 * 16 / num_envs * timesteps = 16
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
    config.latent_size = config.num_categoricals * config.num_codes
    config.state_size = config.hidden_state_size + config.latent_size
    config.train_every = config.sample_batch_size * config.sample_seq_len // (config.num_envs * config.replay_ratio)

    def make_env(gym_id, eval=False):
        def thunk():
            gym.register_envs(ale_py)
            env = gym.make(gym_id, render_mode="rgb_array")#, full_action_space=True)
            env = NoopResetEnv(env, noop_max=30)
            env = FireResetEnv(env)
            # if not eval:
            #     env = EpisodicLifeEnv(env)
            env = gym.wrappers.ResizeObservation(env, (64, 64))
            env = gym.wrappers.TransformObservation(env, lambda x: x.transpose(2, 0, 1), observation_space=gym.spaces.Box(0, 255, config.image_dim))
            env = gym.wrappers.TimeLimit(env, 10000)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk

    num_envs = config.num_envs
    env = make_vec_env(make_env('ALE/Breakout-v5'), num_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
    eval_env = make_vec_env(make_env(f'ALE/Breakout-v5', True), 1)
    dreamer = DreamerV3(config)
    logger = Logger("logs/dreamer-v3")
    
    def eval(dreamer, eval_env):
        done = False
        obs = eval_env.reset()
        total_reward = 0
        num_completed = 0
        frames = []
        while not done:
            action = dreamer.eval_action(torch.tensor(obs).float().to(config.device))
            obs, reward, done, infos = eval_env.step(action)
            total_reward += reward
            frames.append(eval_env.render())
            for info in infos:
                if 'episode' in info.keys():
                    total_reward += info['episode']['r']
                    num_completed += 1
        dreamer.eval_hidden = torch.zeros(1, 8192).to(config.device)
        return total_reward, frames
    
    losses_wm = []
    losses_actor = []
    losses_critic = []
    losses_dict = {}
    obs = env.reset()
    timestep = 0
    for i in range(50_000_000 // config.num_envs):
        timestep += config.num_envs
        if i % 100 == 0:
            print(i)
        action, latent = dreamer.choose_action(torch.tensor(obs).float().to(config.device))
        next_obs, reward, done, info = env.step(action)
        dreamer.process_sample(obs, latent, action, reward, done)
        if True in done:
            indices = np.where(done)[0]
            temp_rew = []
            for index in indices:
                temp_rew.append(info[index]["episode"]["r"])
            logger.add_scalar("Train Reward", np.mean(temp_rew))
            logger.write(timestep)
        if i % 500 == 0:
            eval_reward, frames = eval(dreamer, eval_env)
            logger.add_scalar("Eval Reward", eval_reward)
            imageio.mimwrite(f'gifs/Breakout_{i}.gif', frames[::4], loop=0, fps=20)
        if i % config.train_every == 0 and i > 1000:
            loss_wm, loss_critic, loss_actor, loss_dict = dreamer.train()
            logger.add_scalar("World Model Loss", loss_wm)
            logger.add_metrics(loss_dict)
            logger.write(timestep)
        obs = next_obs