import torch
import numpy as np
from abc import abstractmethod


class Env:
    # Abstract class for environments
    def __init__(self, parms, dtype):
        self.dtype = dtype
        self.min_x = parms.min_x
        self.max_x = parms.max_x
        self.train_size = parms.train_size
        self.allow_replacement = parms.allow_replacement

    @abstractmethod
    def _f(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def reset(self):
        self.x = (
            torch.linspace(self.min_x / 4, self.max_x / 4, self.train_size)
            .reshape(-1, 1)
            .type(self.dtype)
        )
        self.y = self._f(self.x).type(self.dtype)

        self.unobs_x = (
            torch.linspace(self.min_x, self.max_x, self.train_size * 4)
            .reshape(-1, 1)
            .type(self.dtype)
        )

        self.unobs_y_true = self._f(self.unobs_x).type(self.dtype)

        self.top1_idxs = torch.argsort(self.unobs_y_true, dim=0, descending=True)[
            : int(self.train_size * 0.01)
        ].cpu()
        self.top5_idxs = torch.argsort(self.unobs_y_true, dim=0, descending=True)[
            : int(self.train_size * 0.05)
        ].cpu()
        self.top10_idxs = torch.argsort(self.unobs_y_true, dim=0, descending=True)[
            : int(self.train_size * 0.1)
        ].cpu()

        self.y_in_top1 = torch.zeros(self.unobs_y_true.shape[0], dtype=torch.bool)
        self.y_in_top1[self.top1_idxs] = True
        self.y_in_top5 = torch.zeros(self.unobs_y_true.shape[0], dtype=torch.bool)
        self.y_in_top5[self.top5_idxs] = True
        self.y_in_top10 = torch.zeros(self.unobs_y_true.shape[0], dtype=torch.bool)
        self.y_in_top10[self.top10_idxs] = True

        self.observed_top1_count = 0
        self.observed_top5_count = 0
        self.observed_top10_count = 0

        self.prev_action_set = set()
        self.prev_y = list()

    def step(self, action):
        if action < 0 or action > self.unobs_x.shape[0]:
            return False

        unobs_y = self._f(self.unobs_x[action : action + 1])
        self.x = torch.cat([self.x, self.unobs_x[action : action + 1]], dim=0)
        self.y = torch.cat([self.y, unobs_y], dim=0)

        if action not in self.prev_action_set:
            self.observed_top1_count += self.y_in_top1[action].item()
            self.observed_top5_count += self.y_in_top5[action].item()
            self.observed_top10_count += self.y_in_top10[action].item()

        self.prev_y.append(unobs_y[0][0].item())

        if self.allow_replacement:
            self.prev_action_set.add(action)

        else:
            self.unobs_x = torch.cat(
                [self.unobs_x[:action], self.unobs_x[action + 1 :]], dim=0
            )
            self.unobs_y_true = torch.cat(
                [self.unobs_y_true[:action], self.unobs_y_true[action + 1 :]], dim=0
            )
            self.y_in_top1 = torch.cat(
                [self.y_in_top1[:action], self.y_in_top1[action + 1 :]], dim=0
            )
            self.y_in_top5 = torch.cat(
                [self.y_in_top5[:action], self.y_in_top5[action + 1 :]], dim=0
            )
            self.y_in_top10 = torch.cat(
                [self.y_in_top10[:action], self.y_in_top10[action + 1 :]], dim=0
            )

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
        return sum(self.prev_y) / len(self.prev_y)

    def rollout(self, num_steps: int = 1):
        x = (
            torch.linspace(self.min_x, self.max_x, num_steps)
            .reshape(-1, 1)
            .type(self.dtype)
        )

        y = self._f(x).type(self.dtype)

        return x, y


class SineEnv(Env):
    def __init__(self, parms, dtype):
        super(SineEnv, self).__init__(parms=parms, dtype=dtype)
        self.scale = parms.scale
        self.sigma = parms.sigma

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn(*x.shape) * self.sigma
        return self.scale * torch.sin(2 * np.pi * (x)) + epsilon.to(x.device)
