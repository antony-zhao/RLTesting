from rltesting.torch_rl.buffers import PerEnvTrajectoryBuffer
from rltesting.torch_rl.contrastive_rl.crl import ScalingMLP, SafeTanhNormal
from rltesting.torch_rl.dreamer.dreamer import (
    DreamerWorldModel, Critic, WeightedAverageOverBins, 
    transform_obs, compute_lambda_returns, make_state, 
    init_weights, init_last_layer, unimix, probs_to_logits
)
from rltesting.torch_rl.models import TargetNetwork
from rltesting.torch_rl.utils import to_numpy
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F
from torch.distributions import OneHotCategoricalStraightThrough, Independent, Categorical

class PolicyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_size = config.state_size
        self.action_dim = config.action_dim
        self.action_type = config.action_type
        if self.action_type == "continuous":
            self.policy_mu = ScalingMLP(self.state_size * 2, self.action_dim, config.num_blocks, config.hidden_dim, config.act)
            self.policy_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        else:
            self.policy = ScalingMLP(self.state_size * 2, self.action_dim, config.num_blocks, config.hidden_dim, config.act)
            self.actor_unimix = config.actor_unimix
    
    def policy_dist(self, x, goal=None):
        if goal is None:
            goal = torch.zeros_like(x)
        x = torch.concat([x, goal], -1)
        logits = self.policy(x)
        if self.action_type == "discrete":
            probs = torch.softmax(logits, -1)
            unimixed_probs = unimix(probs, self.action_dim, self.actor_unimix)
            logits = probs_to_logits(unimixed_probs)
            action_dist = Categorical(logits=logits)
        else:
            action_dist = SafeTanhNormal(loc=logits, scale=torch.exp(self.policy_logstd))
        return action_dist
    
    def sample_action(self, dist):
        if self.action_type == "discrete":
            action = dist.sample()
        else:
            action = dist.rsample()
        return action
    
    def policy_fn(self, x, goal=None, det=False):
        # actually choosing the action, returns the actual action as well as the log prob of the action and entropy
        action_dist = self.policy_dist(x, goal)
        if not det:
            action = self.sample_action(action_dist)
        else:
            return action_dist.mode
        return action.detach()
        
class ContrastiveCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_size = config.state_size
        self.action_dim = config.action_dim
        self.g_encoder = ScalingMLP(config.state_size, config.repr_dim, config.num_blocks, config.hidden_dim, config.act)
        self.sa_encoder = ScalingMLP(config.state_size + config.action_dim, config.repr_dim, config.num_blocks, config.hidden_dim, config.act)
    
    def logits_matrix(self, obs, action, obs_f):
        sa_repr = self.sa_encoder(torch.concat([obs, action], -1))
        g_repr = self.g_encoder(obs_f)
        logits = -torch.sqrt(torch.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, dim=-1))
        return logits
    
    def q_values(self, obs, action, obs_f):
        sa_repr = self.sa_encoder(torch.concat([obs, action], -1))
        g_repr = self.g_encoder(obs_f)
        if len(sa_repr.shape) == 3:
            g_repr = g_repr[:, None, :]
        logits = -torch.sqrt(torch.sum((sa_repr - g_repr) ** 2, dim=-1))
        return logits

class DreamerV3:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.world_model = DreamerWorldModel(config).to(self.device)
        self.actor = PolicyModel(config).to(self.device)
        self.critic = Critic(config, self.world_model.bins).to(self.device)
        self.contrastive_critic = ContrastiveCritic(config).to(self.device)
        self.critic_target = TargetNetwork(self.critic, config.critic_tau)
        self.optim_wm = Adam(self.world_model.parameters(), config.wm_lr, eps=1e-5)
        self.optim_actor = Adam(self.actor.parameters(), config.ac_lr, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), config.ac_lr, eps=1e-5)
        self.optim_contrastive_critic = Adam(self.contrastive_critic.parameters(), config.ac_lr, eps=1e-5)
        self.log_alpha_crl = nn.Parameter(torch.tensor(0.0).to(self.device))
        self.alpha_optim = Adam([self.log_alpha_crl], 3e-4)
        self.init_models()
        
        if config.obs_type == "image":
            self.is_image = True
            self.buffer = PerEnvTrajectoryBuffer(config.num_envs, buffer_size=1_000_000)
        elif config.obs_type == "vector":
            self.is_image = False
            self.buffer = PerEnvTrajectoryBuffer(config.num_envs, buffer_size=1_000_000)
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
        self.action_type = config.action_type
        self.num_actions = config.action_dim
        self.use_alpha = config.use_alpha
        self.penalty = config.penalty

    def obs_to_state(self, obs, hidden=None):
        if hidden is None:
            hidden = self.world_model._get_hidden(obs.shape[0])
        transformed_obs = transform_obs(obs, self.is_image)
        latent_prob = self.world_model.encoder(transformed_obs, hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, hidden)
        return state, latent
    
    def choose_action(self, obs, goal=None, det=False):
        state, latent = self.obs_to_state(obs, self.active_hidden)
        goal_state = self.obs_to_state(goal)[0] if goal is not None else None
        action = self.actor.policy_fn(state, goal_state, det)
        return action, latent
    
    def eval_action(self, obs, goal=None, det=True, reset=False):
        if reset:
            self.eval_hidden = self.world_model._get_hidden(1)
        state, latent = self.obs_to_state(obs, self.eval_hidden)
        goal_state = self.obs_to_state(goal)[0] if goal is not None else None
        action = self.actor.policy_fn(state, goal_state, det)
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
        sample, future_obs, _ = self.buffer.sample_with_goals_as_tensors(self.config.device, self.config.sample_batch_size, self.config.sample_seq_len)
        obs, actions, rewards, dones = sample
        if self.config.action_type == "discrete":
            actions = F.one_hot(actions.long(), self.config.action_dim).float()
        loss_wm, loss_dict, new_states = self.world_model.world_model_loss(obs, actions, rewards, dones)
        self.optim_wm.zero_grad()
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 5)
        self.optim_wm.step()
        
        future_states, _ = self.obs_to_state(future_obs)
        # states, rewards, continues, log_probs, entropy = self.imagine_rollout(new_states.reshape(-1, self.config.state_size))
        # continues[0] = (1 - dones.flatten())
            
        # loss_critic, returns, values = self.reinforce_critic_loss(states, rewards, continues)
        # loss_actor, actor_ent = self.reinforce_actor_loss(returns, values, log_probs, entropy)
        new_states_start = new_states[:, 0]
        loss_critic, _ = self.contrastive_critic_loss(new_states_start, actions[:, 0], future_states)
        loss_actor, _ = self.contrastive_actor_loss(new_states_start, future_states)
        
        self.optim_critic.zero_grad()
        self.optim_contrastive_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.optim_critic.step()
        self.optim_contrastive_critic.step()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.optim_actor.step()
        
        loss_dict["loss/actor loss"] = to_numpy(loss_actor)
        loss_dict["loss/critic loss"] = to_numpy(loss_critic)
        # loss_dict["loss/actor entropy"] = to_numpy(actor_ent)
        return to_numpy(loss_wm), to_numpy(loss_critic), to_numpy(loss_actor), loss_dict 
    
    def train_actor(self, returns, values, log_probs, entropy, states, goal_states):
        r_actor_loss, actor_ent = self.reinforce_actor_loss(returns, values, log_probs, entropy)
        c_actor_loss, c_actor_metrics = self.contrastive_actor_loss(states, goal_states)
        actor_loss = r_actor_loss + c_actor_loss
        return actor_loss, actor_ent, c_actor_metrics
    
    def train_critic(self, states, actions, rewards, continues, future_states):
        r_critic_loss, returns, values = self.reinforce_critic_loss(states, rewards, continues)
        c_critic_loss, c_critic_metrics = self.contrastive_critic_loss(states, actions, future_states)
        critic_loss = r_critic_loss + c_critic_loss
        return critic_loss, returns, values, c_critic_metrics
    
    def reinforce_actor_loss(self, returns, values, action_log_probs, entropy):
        range_ = torch.quantile(returns, 1 - self.percentiles) - torch.quantile(returns, self.percentiles)
        if self.range_ema is not None:
            self.range_ema = range_ * self.return_range_tau + self.range_ema * (1 - self.return_range_tau)
        else:
            self.range_ema = range_
        
        adv = ((returns - values) / torch.clip(self.range_ema, min=1)).detach()
        actor_loss = -(adv * action_log_probs + entropy * self.config.entropy_coef)
        actor_loss = actor_loss.mean()
        return actor_loss, entropy.mean()
    
    def reinforce_critic_loss(self, states, rewards, continues):
        self.critic_target.update()
        values, value_logits = self.critic(states)
        value_target, _ = self.critic_target(states)
        returns = compute_lambda_returns(values, rewards, continues, self.gamma, self.lambda_)
        value_bins = WeightedAverageOverBins(self.world_model.bins, value_logits[:-1])
        loss = -value_bins.log_prob(returns.detach(), aggregate=False)
        loss -= value_bins.log_prob(value_target.detach()[:-1], aggregate=False)
        loss = loss.mean()
        return loss, returns.detach(), values.detach()[:-1] # returning returns and values for the actor to reuse later
        
    def contrastive_actor_loss(self, states, goal_states):
        states = torch.cat([states, states], dim=0)
        random_goals = torch.roll(goal_states, shifts=1, dims=0)
        goal_states = torch.cat([goal_states, random_goals], dim=0)
        B = states.shape[0]
        
        action_dist = self.actor.policy_dist(states, goal_states)
        alpha = torch.exp(self.log_alpha_crl).detach()
        
        if self.action_type == "continuous":
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1)
            q_values = self.contrastive_critic.q_values(states, action, goal_states)
            if self.use_alpha:
                loss = (-q_values + alpha * log_prob).mean()
            else:
                loss = -q_values.mean()
        else:
            probs = action_dist.probs
            log_probs = action_dist.logits
            obs_expanded = states.repeat_interleave(self.num_actions, dim=0)
            one_hots = torch.eye(self.num_actions, device=states.device).repeat(B, 1)

            q_values = self.contrastive_critic.q_values(obs_expanded, one_hots, goal_states)
            if self.use_alpha:
                alpha = torch.exp(self.log_alpha_crl).detach()
                loss = (probs * (alpha * log_probs - q_values)).sum(-1).mean()
            else:
                loss = (probs * -q_values).sum(-1).mean()
                
        return loss, {"alpha": to_numpy(alpha), "contrastive_entropy": to_numpy(action_dist.entropy().mean())}
    
    def contrastive_critic_loss(self, states, actions, future_states):
        logits = self.contrastive_critic.logits_matrix(states, actions, future_states)
        n = logits.shape[0]
        logits_pos = logits.diag().mean()
        logits_neg = (logits.sum() - logits.diag().sum()) / (n * n - n)
        loss = -torch.diag(logits) + torch.logsumexp(logits, dim=1)
        loss = loss.mean()
        logsumexp_reg = self.penalty * torch.logsumexp(logits + 1e-6, dim=1).pow(2).mean()
        return loss + logsumexp_reg, {
            "logits_pos": to_numpy(logits_pos),
            "logits_neg": to_numpy(logits_neg)
        }
    
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