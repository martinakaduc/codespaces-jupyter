import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_features: int = 1, out_features: int = 1):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(in_features, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, out_features)

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
