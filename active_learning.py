import torch
import random

from experiments import predict


def actor(type, models, unobs_x):
    if type == "random":
        return random_sampling(unobs_x)
    elif type == "uncertainty":
        return uncertainty_sampling(models, unobs_x)
    elif type == "thompson":
        return thompson_sampling(models, unobs_x)
    elif type == "hentropy_search":
        return hentropy_search(models, unobs_x)
    else:
        raise NotImplementedError


def hentropy_search(models, unobs_x):
    X = torch.randn(
        256, unobs_x.shape[1], device=unobs_x.device, dtype=unobs_x.dtype
    ) * 10
    
    X.requires_grad_(True)
    optimizer = torch.optim.Adam([X], lr=1e-3)
    for _ in range(10000):
        tX = torch.sigmoid(X) * 4 - 2
        y_pred_mean, ale, epi, uncert = predict(models=models, x=tX)
        loss = - y_pred_mean.mean()
        loss.backward()
        print("Acqf Loss:", loss.item(), end="\r", flush=True)
        optimizer.step()
        optimizer.zero_grad()
        
    print()
    tX = torch.sigmoid(X) * 4 - 2
    y_pred_mean, ale, epi, uncert = predict(models=models, x=tX)
    selected_idx = torch.argmax(y_pred_mean)
    print("Selected:", tX[selected_idx])
    
    # Select unobserved point that is closest to the selected point
    dist = torch.norm(tX[selected_idx] - unobs_x, dim=1)
    selected_idx = torch.argmin(dist)
    return selected_idx.item()
    

def thompson_sampling(models, unobs_x):
    model = random.choice(models)
    y_pred_mean, ale, epi, uncert = predict(models=[model], x=unobs_x)
    selected_idx = torch.argmax(y_pred_mean)
    return selected_idx.item()


def uncertainty_sampling(models, unobs_x):
    y_pred_mean, ale, epi, uncert = predict(models=models, x=unobs_x)
    selected_idx = torch.argmax(epi)
    return selected_idx.item()


def random_sampling(unobs_x):
    return torch.randint(0, unobs_x.shape[0], size=(1,)).item()
