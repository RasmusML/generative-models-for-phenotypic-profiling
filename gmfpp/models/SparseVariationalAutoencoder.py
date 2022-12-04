import numpy as np
from torch import nn, Tensor
import torch
from torch.distributions import Distribution, Exponential, Cauchy, HalfCauchy, Normal
from gmfpp.models.PrintSize import *
from typing import List, Set, Dict, Tuple, Optional, Any

def ReparameterizedSpikeAndSlab_sample(mu, log_sigma, log_gamma):
    eps = torch.empty_like(log_sigma.exp()).normal_()
    eta = torch.empty_like(log_sigma.exp()).normal_()
    selector = nn.functional.sigmoid(log_gamma.exp() + eta -1)    
    return selector * (mu + eps.mul(log_sigma.exp()))

class SparseVariationalAutoencoder(nn.Module):
   
    def __init__(self, input_shape, latent_features: int) -> None:
        super(SparseVariationalAutoencoder, self).__init__()
        #print("Init SVAE input_shape, latent_features: ", input_shape, latent_features)
        self.input_shape = getattr(input_shape, "tolist", lambda: input_shape)()
        #print("Init SVAE self.input_shape: ", tuple(self.input_shape))
        
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.observation_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3*latent_features)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1*self.observation_features),
            nn.Unflatten(1,self.input_shape)
        )
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        #print("Init SVAE end")

    def observation(self, z:Tensor) -> Tensor:
        """return the distribution `p(x|z)`"""
        mu = self.decoder(z)
        mu = mu.view(-1, *self.input_shape) # reshape the output
        return mu
    
    def forward(self, x) -> Dict[str, Any]:
        # flatten the input
        #x = x.reshape(x.size(0), -1)
        h_z = self.encoder(x)
        qz_mu, qz_log_sigma, qz_log_gamma = h_z.chunk(3, dim=-1)

        #print("mu.shape", mu.shape) # should be dim batch, x, y, channel
        #print("log_sigma.shape", log_sigma.shape) # should be dim batch, x, y, channel
        #print("log_gamma.shape", log_gamma.shape) # should be dim batch, x, y, channel, #latentvar
        z = ReparameterizedSpikeAndSlab_sample(qz_mu, qz_log_sigma, qz_log_gamma)
        x_hat = self.observation(z)
        #print("x_hat.shape", x_hat.shape) 
        
        return {'x_hat': x_hat, 
                'z': z, 
                'qz_log_gamma': qz_log_gamma, 
                'qz_mu': qz_mu, 
                'qz_log_sigma':qz_log_sigma}
    