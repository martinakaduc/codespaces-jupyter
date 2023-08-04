import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features, hiddem_dim: int = 32):
        super(Model, self).__init__()

        self.encoder = FeatureExtractor(in_features, hiddem_dim)
        self.decoder = nn.Linear(hiddem_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FeatureExtractor(nn.Sequential):           
    def __init__(self, in_features, hiddem_dim, gp_dim=None):                                      
        super(FeatureExtractor, self).__init__()        
        self.add_module('linear1', torch.nn.Linear(in_features, hiddem_dim))
        self.add_module('relu1', torch.nn.ReLU())                            
        self.add_module('linear2', torch.nn.Linear(hiddem_dim, hiddem_dim))     
        self.add_module('relu2', torch.nn.ReLU())
        
        if gp_dim is not None:
            self.add_module('linear3', torch.nn.Linear(hiddem_dim, gp_dim))
            
                                                 
class HybridGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, **fe_kwargs):
            super(HybridGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=fe_kwargs["gp_dim"])),
                num_dims=fe_kwargs["gp_dim"], grid_size=100
            )
            self.feature_extractor = FeatureExtractor(**fe_kwargs)
            
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        
            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)