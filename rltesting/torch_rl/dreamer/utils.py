import torch
import torch.nn.functional as F
import numpy as np

def unimix(x, num_codes, proportion=0.01):
    uniform = torch.ones_like(x) / num_codes
    return x * (1 - proportion) + uniform * proportion

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def symlog_squared_error(y, y_hat):
    return 1/2 * (symlog(y) - y_hat) ** 2

def sample_latent(probs):
    pass

class WeightedAverageOverBins:
    def __init__(self, probs=None, logits=None, low=-20, high=20, forward=symexp, backward=symlog):
        self.probs = torch.softmax(logits, -1) if probs is None else probs
        self.bins = torch.linspace(low, high, high - low + 1)
        self.low = low
        self.high = high
        self.forward = forward
        self.backward = backward
    
    def weighted_average(self):
        weighted_average = self.probs @ self.bins
        return self.forward(weighted_average)
    
    def two_hot(self, vals):
        # assumes bins are spaced 1 apart
        index_1 = vals.type(torch.long) - self.low
        index_2 = index_1 + torch.sign(vals).type(torch.long)
        proportion_2 = torch.abs(vals) - torch.abs(vals).type(torch.int)
        proportion_1 = 1 - proportion_2
        one_hot_1 = F.one_hot(index_1, len(self.bins))
        one_hot_2 = F.one_hot(index_2, len(self.bins))
        two_hot_encoded = proportion_1.unsqueeze(-1) * one_hot_1 + proportion_2.unsqueeze(-1) * one_hot_2
        return two_hot_encoded
    
    def log_prob(self, vals):
        # basically just the loss
        target = self.two_hot(self.backward(vals))
        return -target @ torch.log(self.probs)
    
