import numpy as np
import torch

class ReplayBuffer:
    # The vanilla replay buffer
    def __init__(self, buffer_shapes=[], dtypes=None, buffer_size=1_000_000, prioritized=False, done_index=-1):
        self.buffers = []
        self.temp_buffer = [[] for _ in range(len(buffer_shapes))]
        self.index = 0
        self.buffer_size = buffer_size
        self.total = 0
        if dtypes is not None:
            for shape, dtype in zip(buffer_shapes, dtypes):
                self.buffers.append(np.empty((buffer_size,) + shape, dtype=dtype))
        else:
            for shape in buffer_shapes:
                self.buffers.append(np.empty((buffer_size,) + shape))
        self.weights = np.zeros(buffer_size) if prioritized else None
        self.done_index = done_index
    
    def sample(self, batch_size, seq_len=1):
        # can do a thing where the valid indices go from [0, pos - seq_len) and [pos, buffer_size-seq_len] to prevent the
        # out of order sampling, instead of needing to wait for episode to terminate
        if self.weights is None:
            if self.total < self.buffer_size:
                indices = np.random.randint(0, min(self.total - seq_len, self.buffer_size), size=batch_size)
            else:
                valid_indices = list(range(0, self.index - seq_len)) + list(range(self.index, self.buffer_size))
                indices = np.random.choice(valid_indices, size=batch_size)
        if seq_len > 1:
            indices = np.expand_dims(indices, axis=-1) + np.arange(seq_len)
            indices %= self.buffer_size
            indices = indices.transpose()
        sampled = [buffer[indices] for buffer in self.buffers]
        if batch_size > 0  and sampled[1].max() >= 18:
            print('hi')
        return sampled
    
    def add_sample(self, sample):
        for i, buffer in enumerate(self.buffers):
            buffer[self.index] = sample[i]
        self.index = (self.index + 1) % self.buffer_size
        self.total += 1
    
    def add_sample_until_episode_terminal(self, sample):
        # use if the ordering of the data matters (i.e. recurrent) so that there won't be a weird 
        # transition between an incomplete and an old episode.
        for i in range(len(self.buffers)):
            self.temp_buffer[i].append(sample[i])
        if sample[-1]:
            for i in range(len(self.buffers)):
                self.temp_buffer[i] = np.stack(self.temp_buffer[i])
            self.add_samples(self.temp_buffer)
            self.temp_buffer = [[] for _ in range(len(self.buffers))]
    
    def add_samples(self, data_samples):
        # used mainly for adding full episodes but can also handle small rollouts
        # assumes data_samples is a list of num_steps * dimension 
        num_steps = data_samples[0].shape[0]
        indices = (self.index + np.arange(num_steps)) % self.buffer_size
        for i, buffer in enumerate(self.buffers):
            buffer[indices] = data_samples[i]
        self.index = (self.index + num_steps) % self.buffer_size
        self.total += num_steps
    
    def modify_indices(self, buffer_num, indices, data):
        # might not be needed for this but would be good for prioritized replay or dreamerv3
        raise NotImplemented
    
    def modify_weights(self, indices, weights):
        raise NotImplemented
    
    def store_as_dataset(self, filename):
        pass
    
class PerEnvBuffer:
    def __init__(self, num_envs, buffer_shapes=[], dtypes=None, buffer_size=1_000_000, prioritized=False):
        self.buffers = [ReplayBuffer(buffer_shapes, dtypes, buffer_size // num_envs, prioritized) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.num_items = len(buffer_shapes)
    
    def add_sample(self, sample, idxs=None):
        if idxs is None:
            idxs = list(range(self.num_envs))
        for i in idxs:
            self.buffers[i].add_sample([s[i] for s in sample])
    
    def add_sample_until_episode_terminal(self, sample):
        for i in range(self.num_envs):
            self.buffers[i].add_sample_until_episode_terminal([s[i] for s in sample])
    
    def sample(self, batch_size, seq_len=1):
        per_env_batch = np.bincount(np.random.randint(0, self.num_envs, batch_size))
        per_env_samples = [self.buffers[i].sample(batch_sizes, seq_len) for i, batch_sizes in enumerate(per_env_batch)]
        samples = []
        for i in range(self.num_items):
            samples.append(np.concatenate([sample[i] for sample in per_env_samples], axis=0 if seq_len == 1 else 1))
        
        return samples
    
    def sample_as_tensors(self, device, batch_size, seq_len=1):
        samples = self.sample(batch_size, seq_len)
        
        samples_tensor = [torch.tensor(sample).to(device).float() for sample in samples]
        return samples_tensor
        