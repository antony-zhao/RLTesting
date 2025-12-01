from rltesting.torch_rl.models import TargetNetwork
from rltesting.torch_rl.buffers import ReplayBuffer
import numpy as np
import torch
from torch.nn import functional as F

to_numpy = lambda x: x.detach().cpu().numpy()

class DQN:
    def __init__(self, q_network, num_actions, buffer=None, obs_dim=None, obs_type=None, tau=0.001, lr=1e-4, eps_decay=0.95, eps_min=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_actions = num_actions
        self.device = device
        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = ReplayBuffer([(obs_dim,), (), (), (obs_dim,), ()], dtypes=[obs_type, np.int32, np.float32, obs_type, np.bool], buffer_size=100_000)
        self.q_network = q_network
        self.q_target = TargetNetwork(self.q_network, tau=tau).to(device)
        self.opt = torch.optim.AdamW(self.q_network.parameters(), lr=lr)
        self.eps = 1
        self.decay = eps_decay
        self.eps_min = eps_min
    
    def eps_decay(self):
        self.eps = max(self.eps * self.decay, self.eps_min)
        
    def choose_action(self, obs, det=False):
        obs = torch.tensor(obs).float().to(self.device)
        if not det and np.random.rand() < self.eps:
            action = np.random.randint(self.num_actions)
        else:
            action = to_numpy(self.q_network.choose_action(obs))
        return action
    
    def learn(self, batch_size=256, double_dqn=False): # batch size of 1 for learning from last sample otherwise 256
        self.q_target.update()
        obs, action, reward, next_obs, done = self.buffer.sample(batch_size)
        obs = torch.tensor(obs).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        if not double_dqn:
            q_target = reward + 0.99 * torch.max(self.q_target(next_obs).detach(), 1)[0] * (1 - done)
        else:
            q_target = reward + 0.99 * self.q_target(next_obs).detach().gather(1, torch.argmax(self.q_network(next_obs).detach(), 1, keepdim=True)).squeeze() * (1 - done)
        q_values = self.q_network(obs).gather(1, action.unsqueeze(-1)).squeeze()
        loss = F.mse_loss(q_target, q_values)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    
    def process_sample(self, obs, action, rew, next_obs, done):
        if done:
            self.eps_decay()
        self.buffer.add_sample([obs, action, rew, next_obs, done])
    