from rltesting.torch_rl.buffers import PerEnvTrajectoryBuffer
from rltesting.torch_rl.contrastive_rl.crl import ScalingMLP, SafeTanhNormal
from rltesting.torch_rl.dreamer.dreamer import (
    DreamerWorldModel, Critic, WeightedAverageOverBins, DreamerV3,
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
        self.latent_size = config.latent_size
        self.contrastive_state_size = config.contrastive_state_size
        self.action_dim = config.action_dim
        self.action_type = config.action_type
        if self.action_type == "continuous":
            self.policy_mu = ScalingMLP(self.contrastive_state_size + self.latent_size, self.action_dim, config.num_blocks, config.hidden_dim, config.act)
            self.policy_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        else:
            self.policy = ScalingMLP(self.contrastive_state_size + self.latent_size, self.action_dim, config.num_blocks, config.hidden_dim, config.act)
            self.actor_unimix = config.actor_unimix
    
    def policy_dist(self, x, goal=None):
        if goal is None:
            goal = torch.zeros((x.shape[0], self.latent_size)).to(x.device)
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
        self.latent_size = config.obs_dim
        self.action_dim = config.action_dim
        self.g_encoder = ScalingMLP(self.latent_size, config.repr_dim, config.num_blocks, config.hidden_dim, config.act)
        self.sa_encoder = ScalingMLP(self.latent_size + config.action_dim, config.repr_dim, config.num_blocks, config.hidden_dim, config.act)
    
    def logits_matrix(self, obs, action, obs_f):
        sa_repr = self.sa_encoder(torch.concat([obs, action], -1))
        g_repr = self.g_encoder(obs_f)
        logits = -torch.sqrt(torch.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, dim=-1))
        return logits
    
    def q_values(self, obs, action, obs_f):
        sa_repr = self.sa_encoder(torch.concat([obs, action], -1))
        g_repr = self.g_encoder(obs_f)
        if sa_repr.shape[0] != g_repr.shape[0]:
            g_repr = g_repr[:, None, :]
            sa_repr = sa_repr.view(g_repr.shape[0], self.action_dim, -1)
        logits = -torch.sqrt(torch.sum((sa_repr - g_repr) ** 2, dim=-1))
        return logits

class DreamerCRL(DreamerV3):
    def __init__(self, config, obs_to_goal=None):
        super().__init__(config)
        self.actor = PolicyModel(config).to(self.device)
        self.contrastive_critic = ContrastiveCritic(config).to(self.device)
        
        self.optim_contrastive_critic = Adam(self.contrastive_critic.parameters(), config.contrastive_lr, eps=1e-5)
        self.optim_contrastive_actor = Adam(self.actor.parameters(), config.contrastive_lr, eps=1e-5)
        self.log_alpha_crl = nn.Parameter(torch.tensor(-2.3).to(self.device))
        self.alpha_optim = Adam([self.log_alpha_crl], 3e-4)
        self.obs_to_goal = obs_to_goal
        
        self.buffer = PerEnvTrajectoryBuffer(config.num_envs, buffer_size=1_000_000)
        
        self.target_entropy = -config.action_dim * 0.5 if config.action_type == "continuous" else 0.5 * np.log(config.action_dim)
        self.use_alpha = config.use_alpha
        self.penalty = config.penalty
    
    def choose_action(self, obs, goal=None, det=False):
        state, latent = self.obs_to_contrastive_state(obs, self.active_hidden)
        goal_state = self.obs_to_state(goal)[0][:, :self.config.latent_size] if goal is not None else None
        action = self.actor.policy_fn(state, goal_state, det)
        return action, latent
    
    def eval_action(self, obs, goal=None, det=True, reset=False):
        if reset:
            self.eval_hidden = self.world_model._get_hidden(obs.shape[0])
        state, latent = self.obs_to_contrastive_state(obs, self.eval_hidden)
        goal_state = self.obs_to_state(goal)[0][:, :self.config.latent_size] if goal is not None else None
        # dist = self.actor.policy_dist(state, goal_state)
        # print(f"action probs: {dist.probs[0].detach().cpu().numpy().round(3)}")
        action = self.actor.policy_fn(state, goal_state, det)
        self.eval_hidden = self.world_model.recurrent_step(self.eval_hidden, latent, action).detach()
        return to_numpy(action)
    
    def train(self):
        sample, future_obs, _ = self.buffer.sample_with_goals_as_tensors(self.config.device, self.config.sample_batch_size, self.config.sample_seq_len)
        obs, actions, rewards, dones = sample
        if self.config.action_type == "discrete":
            actions = F.one_hot(actions.long(), self.config.action_dim).float()
        loss_wm, loss_dict, new_states, encoder_repr = self.world_model.world_model_loss(obs, actions, rewards, dones)
        self.optim_wm.zero_grad()
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 5)
        self.optim_wm.step()
        
        loss_critic = loss_actor = torch.tensor(0.0)
        actor_metrics = critic_metrics = {}
        
        if self.buffer.size > 50_000:
            transformed_obs = transform_obs(obs, self.is_image)
            encoder_repr = self.world_model.encoder.embed_observations(transformed_obs).detach()
            # Sequence-based CRL from world model states
            hidden_state = new_states[:, :, self.config.latent_size:]
            contrastive_state = torch.cat([encoder_repr, hidden_state], -1)
            result = self.extract_crl_pairs(contrastive_state, actions, dones, obs)
            if result[0] is not None:
                crl_states, crl_actions, goal_latents, crl_obs, crl_future_obs, crl_goal_obs = result
                
                if crl_states is not None:
                    N = crl_states.shape[0]
                    perm = torch.randperm(N, device=crl_states.device)
                    mini_batch_size = 256
                    
                    for start in range(0, N, mini_batch_size):
                        idx = perm[start:start + mini_batch_size]
                        mb_states = crl_states[idx]
                        mb_goals = goal_latents[idx]
                        mb_obs = crl_obs[idx]
                        mb_goal_obs = crl_goal_obs[idx]
                        mb_actions = crl_actions[idx]
                        
                        loss_actor, actor_metrics = self.contrastive_actor_loss(mb_states, mb_goals, mb_obs, mb_goal_obs)
                        self.optim_contrastive_actor.zero_grad()
                        loss_actor.backward()
                        self.optim_contrastive_actor.step()
                        
                        loss_critic, critic_metrics = self.contrastive_critic_loss(mb_obs, mb_actions, mb_goal_obs)
                        self.optim_contrastive_critic.zero_grad()
                        loss_critic.backward()
                        self.optim_contrastive_critic.step()
        
            # # Buffer-based CRL for extra critic/actor updates
            for _ in range(self.config.crl_steps_per_train):
                sample_buf, future_obs_buf, _ = self.buffer.sample_with_goals_as_tensors(self.config.device, self.config.crl_batch_size)
                buf_obs, buf_actions, _, _ = sample_buf
                if self.config.action_type == "discrete":
                    buf_actions = F.one_hot(buf_actions.long(), self.config.action_dim).float()
                
                # Actor state from RSSM (cold-encoded, no history)
                with torch.no_grad():
                    buf_state, _ = self.obs_to_contrastive_state(buf_obs)
                buf_state = buf_state.detach()
                
                # Goal obs and goal latent
                buf_goal_obs = torch.as_tensor(self.obs_to_goal(to_numpy(future_obs_buf)), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    buf_goal_state, _ = self.obs_to_state(buf_goal_obs)
                buf_goal_latent = buf_goal_state[:, :self.config.latent_size].detach()
                
                # loss_actor, actor_metrics = self.contrastive_actor_loss(buf_state, buf_goal_latent, buf_obs, buf_goal_obs)
                loss_critic, critic_metrics = self.contrastive_critic_loss(buf_obs, buf_actions, buf_goal_obs)
                
                # self.optim_contrastive_actor.zero_grad()
                # loss_actor.backward()
                # self.optim_contrastive_actor.step()
                
                self.optim_contrastive_critic.zero_grad()
                loss_critic.backward()
                self.optim_contrastive_critic.step()
        
            loss_dict["loss/actor loss"] = to_numpy(loss_actor)
            loss_dict["loss/critic loss"] = to_numpy(loss_critic)
    
        return to_numpy(loss_wm), to_numpy(loss_critic), to_numpy(loss_actor), loss_dict | actor_metrics | critic_metrics
    
    def obs_to_contrastive_state(self, obs, hidden=None):
        """Returns (contrastive_state, latent) where contrastive_state uses
        continuous encoder embeddings instead of one-hot latents."""
        if hidden is None:
            hidden = self.world_model._get_hidden(obs.shape[0])
        transformed_obs = transform_obs(obs, self.is_image)
        encoder_repr = self.world_model.encoder.embed_observations(transformed_obs)
        latent_prob = self.world_model.encoder.compute_latent(encoder_repr, hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = torch.cat([encoder_repr, hidden], -1)
        return state, latent
    
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
        
    def contrastive_actor_loss(self, states, goal_latents, obs, future_obs):
        states = torch.cat([states, states], dim=0)
        random_goals = torch.roll(goal_latents, shifts=1, dims=0)
        goal_latents = torch.cat([goal_latents, random_goals], dim=0)
        obs = torch.cat([obs, obs], dim=0)
        random_obs_f = torch.roll(future_obs, shifts=1, dims=0)
        future_obs = torch.cat([future_obs, random_obs_f], dim=0)
        B = states.shape[0]
        
        action_dist = self.actor.policy_dist(states, goal_latents)
        alpha = torch.exp(self.log_alpha_crl).detach()
        
        if self.action_type == "continuous":
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1)
            q_values = self.contrastive_critic.q_values(obs, action, future_obs)
            if self.use_alpha:
                loss = (-q_values + alpha * log_prob).mean()
            else:
                loss = -q_values.mean()
        else:
            probs = action_dist.probs
            log_probs = action_dist.logits
            obs_expanded = obs.repeat_interleave(self.num_actions, dim=0)
            one_hots = torch.eye(self.num_actions, device=states.device).repeat(B, 1)

            q_values = self.contrastive_critic.q_values(obs_expanded, one_hots, future_obs)
            if self.use_alpha:
                loss = (probs * (alpha * log_probs - q_values)).sum(-1).mean()
            else:
                loss = (probs * -q_values).sum(-1).mean()
                
        return loss, {"alpha": to_numpy(alpha), "contrastive_entropy": to_numpy(action_dist.entropy().mean())}
    
    def contrastive_critic_loss(self, obs, actions, future_obs):
        logits = self.contrastive_critic.logits_matrix(obs, actions, future_obs)
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
    
    def alpha_loss(self, states, goal_states):
        action_dist = self.actor.policy_dist(states, goal_states)
        if self.action_type == "continuous":
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1)
        else:
            log_prob = (action_dist.probs * action_dist.logits).sum(-1)
        alpha = torch.exp(self.log_alpha_crl)
        alpha_loss = alpha * (-log_prob - self.target_entropy).detach()
        return (alpha_loss).mean()
    
    def checkpoint_models(self, folderpath, filename):
        super().checkpoint_models(folderpath, filename)
        torch.save(self.contrastive_critic.state_dict(), f"{folderpath}/contrastive_critic-{filename}.pth")
    
    def extract_crl_pairs(self, states, actions, dones, obs):
        B, T, D = states.shape
        
        modified_dones = dones.clone()
        modified_dones[:, -1] = 1
        
        continues = (1 - modified_dones).flip(dims=[1])
        cs = continues.cumsum(dim=1)
        reset_vals = cs * (1 - continues)
        cummax_reset = reset_vals.cummax(dim=1).values
        count = cs - cummax_reset
        max_future = count.flip(dims=[1]).long()
        
        valid_mask = max_future[:, :-1] > 0
        
        offsets = torch.randint(1, T, (B, T - 1), device=states.device)
        offsets = torch.min(offsets, max_future[:, :-1].clamp(min=1))
        
        batch_idx = torch.arange(B, device=states.device).unsqueeze(1).expand(-1, T - 1)
        time_idx = torch.arange(T - 1, device=states.device).unsqueeze(0).expand(B, -1)
        future_time = time_idx + offsets
        
        # RSSM states for actor
        current_s = states[:, :-1][valid_mask]
        current_a = actions[:, :-1][valid_mask]
        
        # Raw obs for critic
        current_obs = obs[:, :-1][valid_mask]
        future_obs_raw = obs[batch_idx, future_time][valid_mask]
        
        # Goal latent for actor conditioning (cold-encoded)
        future_goal_obs = torch.as_tensor(self.obs_to_goal(to_numpy(future_obs_raw)), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            future_goal_state, _ = self.obs_to_state(future_goal_obs)
        future_goal_latent = future_goal_state[:, :self.config.latent_size].detach()
        
        if current_s.shape[0] == 0:
            return None, None, None, None, None, None
        
        return current_s, current_a, future_goal_latent, current_obs, future_obs_raw, future_goal_obs