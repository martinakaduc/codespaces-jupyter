import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Callable, List
from torch.optim import lr_scheduler, Adam
from torch.optim import Optimizer

from utils import calc_uncert


def train(
    model: nn.Module,
    loss_fct: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1,
    optim: Optimizer = None,
    burnin_iter: int = 0,
    mcmc_iter: int = 50,
    glr: str = "var",
    weight_decay: float = 0.0,
    lr: float = 1e-3,
    num_epochs: int = 1000,
    gamma: float = 0.996,
) -> list:
    losses, models = [], []
    if optim is not None:
        optim = optim(model.parameters(), lr=lr, weight_decay=weight_decay, glr=glr)
    else:
        optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_sched = lr_scheduler.ExponentialLR(optim, gamma)

    m = int(len(x) / batch_size)
    pbar = tqdm(range(num_epochs))
    model.train()
    for epoch in pbar:
        idx = torch.randperm(len(x))
        total_loss = 0

        for batch_idx in range(1, m + 1):
            optim.zero_grad()

            out = model(x[idx[(batch_idx - 1) * batch_size : batch_idx * batch_size]])

            loss = loss_fct(
                out, y[idx[(batch_idx - 1) * batch_size : batch_idx * batch_size]]
            )

            loss.backward()
            optim.step()
            lr_sched.step()

            total_loss += loss.item()

        losses.append(total_loss)

        if epoch > burnin_iter and (epoch + 1) % mcmc_iter == 0:
            models.append(deepcopy(model))

        pbar.set_description("loss: %.6f" % total_loss)

    return losses, models


def predict(models: List[nn.Module], x: torch.Tensor) -> torch.Tensor:
    y_preds = []
    
    # with torch.no_grad():
    for model in models:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
            
        y_pred = model(x)
        y_preds.append(y_pred.unsqueeze(0))

    y_mean = torch.cat(y_preds, dim=0)[:, :, 0].mean(dim=0)

    if y_preds[0].shape[-1] > 1:
        ale, epi, uncert = calc_uncert(y_preds, reduction=None)
    else:
        ale = torch.tensor([0.0])
        epi = uncert = torch.var(torch.cat(y_preds, dim=0)[:, :, 0], dim=0)

    return y_mean, ale, epi, uncert
