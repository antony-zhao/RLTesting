import torch
import numpy as np

def unimix(x, num_codes, proportion=0.01):
    uniform = torch.ones_like(x) / num_codes
    return x * (1 - proportion) + uniform * proportion

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * torch.exp(torch.abs(x) - 1)

def symlog_squared_error(y, y_hat):
    return 1/2 * (symlog(y) - y_hat) ** 2

class WeightedAverageOverBins:
    def __init__(self, low=-20, high=20, forward=symexp, backward=symlog):
        self.bins = torch.linspace(low, high)
        self.low = low
        self.high = high
        self.forward = forward
        self.backward = backward
    
    def forward(self, logits):
        probs = torch.softmax(logits, -1)
        weighted_average = probs @ self.bins.T
        return self.forward(weighted_average)
    
    def two_hot(self, vals):
        index_1 = int(vals) - self.low
        index_2 = index_1 + 1
        proportion_2 = vals - int(vals)
        proportion_1 = 1 - proportion_2
        two_hot_encoded = torch.zeros((vals.shape[0], self.high - self.low))
        two_hot_encoded[index_1] = proportion_1
        two_hot_encoded[index_2] = proportion_2
        return two_hot_encoded
    
    def log_prob(self, vals, logits):
        # basically just the loss
        target = self.two_hot(vals)
        return -target @ torch.log(torch.softmax(logits, -1))
