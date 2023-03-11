import torch
import random
import numpy as np
import torch.nn as nn
from typing import List, Tuple
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


def calc_uncert(preds: List[torch.Tensor], reduction: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = torch.cat(preds, dim=0)
    epi = torch.var(preds[:, :, 0], dim=0)
    ale = torch.mean(preds[:, :, 1], dim=0)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean(), epi.mean(), uncert.mean()
    else:
        return ale, epi, uncert


def init_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-4, b=4)


def plot_uncert(x_test: torch.Tensor, y_pred_mean: torch.Tensor, x_train: torch.Tensor, y_train: torch.Tensor,
                ale: torch.Tensor, epi: torch.Tensor, uncert: torch.Tensor, save_file: str = None):

    x_test, y_pred_mean, x_train, y_train = x_test.cpu(
    ), y_pred_mean.cpu(), x_train.cpu(), y_train.cpu()
    ale, epi, uncert = torch.sqrt(ale.cpu()), torch.sqrt(
        epi.cpu()), torch.sqrt(uncert.cpu())

    no_std = 2

    fig, ax = plt.subplots()
    ax.plot(x_test, y_pred_mean, color='#D1895C', label='Predictive mean')
    ax.scatter(x_train, y_train, color='black', label='Training data')
    ax.fill_between(x_test.flatten(),
                    gaussian_filter1d(y_pred_mean + no_std *
                                      (ale + epi), sigma=5),
                    gaussian_filter1d(y_pred_mean - no_std *
                                      (ale + epi), sigma=5),
                    color='#6C85B6',
                    alpha=0.3, label='Aleatoric uncertainty')
    ax.fill_between(x_test.flatten(),
                    gaussian_filter1d(y_pred_mean + no_std * epi, sigma=5),
                    gaussian_filter1d(y_pred_mean - no_std * epi, sigma=5),
                    color='#6C85B6',
                    alpha=0.5, label='Epistemic uncertainty')
    ax.set_xlabel(r'$x$', fontsize=17)
    ax.set_ylabel(r'$y$', fontsize=17)
    ax.legend()

    if save_file is not None:
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches='tight')

    plt.close(fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lineplot(xs, ys, xlabel, ylabel, ylim=None, save_file=None):
    if len(xs) == 1:
        plt.scatter(xs, ys)
    else:
        plt.plot(xs, ys)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(0, ylim)

    if save_file is not None:
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches='tight')

    plt.close()
