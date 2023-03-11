import torch
import numpy as np
from abc import abstractmethod


class Env:
    # Abstract class for environments
    @abstractmethod
    def __init__(self):
        pass

    def step(self):
        pass


class SineEnv(Env):
    def __init__(self, parms, dtype):
        self.dtype = dtype
        self.sigma = parms.sigma
        self.min_x = parms.min_x
        self.max_x = parms.max_x
        self.train_size = parms.train_size

    def __f(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        epsilon = torch.randn(*x.shape) * sigma
        return 10 * torch.sin(2 * np.pi * (x)) + epsilon.to(x.device)

    def reset(self):
        self.x = torch.linspace(
            self.min_x/4, self.max_x/4, self.train_size).reshape(-1, 1).type(self.dtype)
        self.y = self.__f(self.x, sigma=self.sigma).type(self.dtype)

        self.unobs_x = torch.linspace(
            self.min_x, self.max_x, self.train_size*4).reshape(-1, 1).type(self.dtype)
        self.unobs_y = self.__f(
            self.unobs_x, sigma=self.sigma).type(self.dtype)
        
        self.top1_idxs = torch.argsort(self.unobs_y, dim=0, descending=True)[:int(self.train_size*0.01)].cpu()
        self.top5_idxs = torch.argsort(self.unobs_y, dim=0, descending=True)[:int(self.train_size*0.05)].cpu()
        self.top10_idxs = torch.argsort(self.unobs_y, dim=0, descending=True)[:int(self.train_size*0.1)].cpu()

        self.observed_top1_count = 0
        self.observed_top5_count = 0
        self.observed_top10_count = 0
        
        self.prev_action_set = set()
        
    def step(self, action):
        if action < 0 or action > self.unobs_x.shape[0] or action in self.prev_action_set:
            return False

        self.x = torch.cat([self.x, self.unobs_x[action: action+1]], dim=0)
        self.y = torch.cat([self.y, self.unobs_y[action: action+1]], dim=0)
        
        self.prev_action_set.add(action)
        if action in self.top1_idxs:
            self.observed_top1_count += 1
        if action in self.top5_idxs:
            self.observed_top5_count += 1
        if action in self.top10_idxs:
            self.observed_top10_count += 1
            
        return True
    
    def get_percentage_observed_topK(self, k):
        assert k in [1, 5, 10]
        if k == 1:
            return self.observed_top1_count / self.top1_idxs.shape[0]
        elif k == 5:    
            return self.observed_top5_count / self.top5_idxs.shape[0]
        elif k == 10:
            return self.observed_top10_count / self.top10_idxs.shape[0]
        
    def get_average_observed_y(self):
        return torch.mean(
            self.unobs_y[torch.tensor(list(self.prev_action_set))]
            ).item()

    def rollout(self, burnin: bool = False, num_steps: int = 1):
        x = torch.linspace(
            self.min_x, self.max_x, num_steps).reshape(-1, 1).type(self.dtype)
        if burnin:
            x = x / 4
        y = self.__f(x, sigma=self.sigma).type(self.dtype)

        return x, y
