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
    return (0.5 * (symlog(y) - y_hat) ** 2).sum(-1).mean()

def sample_latent(probs):
    pass

class WeightedAverageOverBins:
    def __init__(self, bins, probs=None, logits=None, forward=symexp, backward=symlog):
        self.probs = torch.softmax(logits, -1) if probs is None else probs
        self.bins = bins
        self.forward = forward
        self.backward = backward
    
    def weighted_average(self):
        weighted_average = self.probs @ self.bins
        return self.forward(weighted_average)
    
    def two_hot(self, vals):
        index_1 = (self.bins.repeat(vals.shape[0], 1) < vals.unsqueeze(1)).sum(-1) - 1
        index_2 = index_1 + 1
        b_k = self.bins[index_1]
        b_k2 = self.bins[index_2]
        proportion_2 = torch.abs(b_k - vals) / torch.abs(b_k2 - b_k)
        proportion_1 = torch.abs(b_k2 - vals) / torch.abs(b_k2 - b_k)
        one_hot_1 = F.one_hot(index_1, len(self.bins))
        one_hot_2 = F.one_hot(index_2, len(self.bins))
        two_hot_encoded = proportion_1.unsqueeze(-1) * one_hot_1 + proportion_2.unsqueeze(-1) * one_hot_2
        return two_hot_encoded
    
    def log_prob(self, vals):
        # basically just the loss
        target = self.two_hot(self.backward(vals))
        return (target * torch.log(self.probs)).sum(-1).mean()
    
