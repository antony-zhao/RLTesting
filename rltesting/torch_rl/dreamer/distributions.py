import torch
from torch.distributions import OneHotCategoricalStraightThrough, Independent

class DreamerLatentDist:
    def __init__(self, probs=None, logits=None):
        self.dist = Independent(OneHotCategoricalStraightThrough(probs=probs, logits=logits), 1)
    
    def sample(self):
        return self.dist.sample()
    
    def deterministic(self):
        return self.dist.mode()
    