from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

class Logger:
    def __init__(self, logdir):
        date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.logdir = f"{logdir}/{date}"
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.scalars = {}
        self.timestep = 0
    
    def add_scalar(self, key, value):
        self.scalars[key] = value
        
    def add_metrics(self, metrics):
        self.scalars.update(metrics)
        
    def write(self, timestep):
        for key, value in self.scalars.items():
            self.writer.add_scalar(key, np.array(value), timestep)
        
        self.writer.flush()
        self.scalars = {}