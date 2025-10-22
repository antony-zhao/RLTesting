import numpy as np

class ReplayBuffer:
    # The vanilla replay buffer
    def __init__(self, buffer_shapes=[], dtypes=None, buffer_size=1_000_000, prioritized=False):
        self.buffers = []
        self.temp_buffer = []
        self.index = 0
        self.buffer_size = buffer_size
        if dtypes is not None:
            for shape, dtype in zip(buffer_shapes, dtypes):
                self.buffers.append(np.empty(shape, dtype=dtype))
        else:
            for shape in buffer_shapes:
                self.buffers.append(np.empty(shape))
        self.weights = np.zeros(buffer_size) if prioritized else None
    
    def sample(self, batch_size, seq_len=1):
        if self.weights is None:
            indices = np.random.randint(min(self.index, self.buffer_size), size=batch_size)
        indices = np.expand_dims(indices, axis=-1) + np.arange(seq_len)
        indices %= self.buffer_size
        sampled = [buffer[indices] for buffer in self.buffers]
        return sampled
    
    def add_sample(self, data):
        for i, buffer in enumerate(self.buffers):
            buffer[self.index] = data[i]
        self.index = (self.index + 1) % self.buffer_size
    
    def add_sample_until_episode_terminal(self, data, episode_terminal=False):
        self.temp_buffer.append(data)
        if episode_terminal:
            self.add_samples(self.temp_buffer)
        self.temp_buffer = []
    
    def add_samples(self, data_samples):
        # used mainly for adding full episodes but can also handle small rollouts
        # assumes data_samples is a list of num_steps * dimension 
        num_steps = data_samples[0].shape[0]
        indices = (self.index + np.arange(num_steps)) % self.buffer_size
        for i, buffer in enumerate(self.buffers):
            buffer[indices] = data_samples[i]
        self.index = (self.index + num_steps) % self.buffer_size
    
    def modify_indices(self, buffer_num, indices, data):
        # might not be needed for this but would be good for prioritized replay or dreamerv3
        raise NotImplemented
    
    def modify_weights(self, indices, weights):
        raise NotImplemented
    
    def store_as_dataset(self, filename):
        pass
    
class PerEnvBuffer:
    pass