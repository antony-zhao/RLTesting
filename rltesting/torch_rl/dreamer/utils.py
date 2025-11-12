import torch
from torch import nn
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
    return ((y - y_hat) ** 2).sum(-1).mean()

def init_last_layer(model, scale):
    # init func should be a partial func with everything already specified, might be a better way to do it but this is just what I know
    for module in model.modules():
        if isinstance(module, (nn.Linear)):#, nn.Conv2d)):#, nn.ConvTranspose2d)):
            last_layer = module
    # init_func(last_layer.weight)
    if isinstance(last_layer, nn.Linear):
        in_num = last_layer.in_features
        out_num = last_layer.out_features
        denoms = (in_num + out_num) / 2.0
        scale = scale / denoms
        limit = np.sqrt(3 * scale)
        nn.init.uniform_(last_layer.weight.data, a=-limit, b=limit)
        if hasattr(last_layer.bias, "data"):
            last_layer.bias.data.fill_(0.0)
    elif isinstance(last_layer, nn.LayerNorm):
        last_layer.weight.data.fill_(1.0)
        if hasattr(last_layer.bias, "data"):
            last_layer.bias.data.fill_(0.0)
    
def init_weights(m):
    # if isinstance(module, (nn.Linear, nn.Conv2d)):
    #     nn.init.xavier_normal_(module.weight)
    #     if hasattr(module.bias, "data"):
    #         module.bias.data.fill_(0.0)
    # elif isinstance(module, nn.LayerNorm):
    #     module.weight.data.fill_(1.0)
    #     if hasattr(module.bias, "data"):
    #         module.bias.data.fill_(0.0)
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

def compute_lambda_returns(values, rewards, continues, gamma, lambda_):
    returns = torch.empty_like(values)
    for i in reversed(range(len(values))):
        if i < len(values) - 1:
            returns[i] = rewards[i] + gamma * continues[i] * ((1 - lambda_) * values[i] + lambda_ * returns[i + 1])
        else:
            returns[i] = values[-1]
    return returns

def compute_lambda_values(
    values, rewards, continues, gamma, lmbda: float = 0.95,
):
    continues *= gamma
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret

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
        index_1 = (self.bins.expand(vals.shape + (-1,)) < vals.unsqueeze(-1)).sum(-1) - 1
        index_2 = index_1 + 1
        b_k = self.bins[index_1]
        b_k2 = self.bins[index_2]
        proportion_2 = torch.abs(b_k - vals) / torch.abs(b_k2 - b_k)
        proportion_1 = torch.abs(b_k2 - vals) / torch.abs(b_k2 - b_k)
        one_hot_1 = F.one_hot(index_1, len(self.bins))
        one_hot_2 = F.one_hot(index_2, len(self.bins))
        two_hot_encoded = proportion_1.unsqueeze(-1) * one_hot_1 + proportion_2.unsqueeze(-1) * one_hot_2
        return two_hot_encoded
    
    def log_prob(self, vals, aggregate=True):
        # basically just the loss
        target = self.two_hot(self.backward(vals))
        log_probs = (target * torch.log(self.probs)).sum(-1)
        if aggregate:
            return log_probs.mean()
        return log_probs
    
