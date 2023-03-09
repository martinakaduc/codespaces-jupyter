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

    def step(self, action):
        if action < 0 or action > self.unobs_x.shape[0]:
            return False

        self.x = torch.cat([self.x, self.unobs_x[action: action+1]], dim=0)
        self.y = torch.cat([self.y, self.unobs_y[action: action+1]], dim=0)
        return True

    def rollout(self, burnin: bool = False, num_steps: int = 1):
        x = torch.linspace(
            self.min_x, self.max_x, num_steps).reshape(-1, 1).type(self.dtype)
        if burnin:
            x = x / 4
        y = self.__f(x, sigma=self.sigma).type(self.dtype)

        return x, y
