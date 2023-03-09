import torch
import random

from experiments import predict


def thompson_sampling(models, unobs_x):
    model = random.choice(models)
    y_pred_mean, ale, epi, uncert = predict(models=[model], x=unobs_x)
    selected_idx = torch.argmax(y_pred_mean)
    return selected_idx


def uncertainty_sampling(models, unobs_x):
    y_pred_mean, ale, epi, uncert = predict(models=models, x=unobs_x)
    selected_idx = torch.argmax(epi)
    return selected_idx


def random_sampling(unobs_x):
    return torch.randint(0, unobs_x.shape[0], size=(1,))
