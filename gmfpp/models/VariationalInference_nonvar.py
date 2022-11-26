from gmfpp.utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any



def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    flat = view_flat_samples(x)
    return flat.sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta
        self.loss = nn.MSELoss(reduction='none')
        
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = model(x)

        x_hat, pz, qz, z = [outputs[k] for k in ["x_hat", "pz", "qz", "z"]]
        mse_loss = ((x_hat - x)**2).mean(axis=[1,2,3])
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        kl = log_qz - log_pz
        #elbo = log_px - kl
        beta_elbo = -mse_loss - self.beta*kl
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'mse_loss':mse_loss, 'kl': kl}
            
        return loss, diagnostics, outputs
      
