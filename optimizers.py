import torch
import numpy as np
from typing import Generator
from torch.optim import Optimizer


class SGLD(Optimizer):
    def __init__(
        self,
        params: Generator[torch.Tensor, None, None],
        lr: float,
        weight_decay: float = 0.0,
        glr: str = "var",
    ):
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        self.glr = glr

        super(SGLD, self).__init__(params, defaults)

    def step(self) -> torch.Tensor:
        loss = None
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p += weight_decay * p.data

                langevin_noise = torch.empty(p.data.size()).normal_().to(d_p.device)
                if self.glr == "var":
                    p.data += (
                        -group["lr"] * 0.5 * d_p + np.sqrt(group["lr"]) * langevin_noise
                    )
                elif self.glr == "std":
                    p.data += -group["lr"] * 0.5 * d_p + group["lr"] * langevin_noise
                else:
                    raise NotImplementedError

        return loss
