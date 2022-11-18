from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus, relu
from torch.distributions import Distribution, Normal
from torch.utils.data import DataLoader

from gmfpp.utils.data_preparation import *
from gmfpp.utils.data_transformers import *
from gmfpp.utils.plotting import *

from gmfpp.models.ReparameterizedDiagonalGaussian import *
from gmfpp.models.CytoVariationalAutoencoder import *
from gmfpp.models.VariationalAutoencoder import *
from gmfpp.models.ConvVariationalAutoencoder import *
from gmfpp.models.VariationalInference import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)

path = get_server_directory_path()

metadata = read_metadata(path + "metadata.csv")
metadata = metadata[:100] # @TODO: figure what to do loading the imabes below gets _very_ slow after 50_000 images
print("loaded metadata")

relative_paths = get_relative_image_paths(metadata)
image_paths = [path + relative for relative in relative_paths]
images = load_images(image_paths, verbose=True)
print("loaded images")

train_set = prepare_raw_images(images)
normalize_channels_inplace(train_set)
print("normalized images")

channel_first = view_channel_dim_first(train_set)
for i in range(channel_first.shape[0]):
    channel = channel_first[i]
    print("channel {} interval: [{:.2f}; {:.2f}]".format(i, torch.min(channel), torch.max(channel)))


# VAE
image_shape = np.array([3, 68, 68], dtype=np.int32)
latent_features = 256
vae = CytoVariationalAutoencoder(image_shape, latent_features)
#vae = VariationalAutoencoder(image_shape, latent_features)

beta = 1
vi = VariationalInference(beta=beta)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

num_epochs = 3000
batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# move the model to the device
vae = vae.to(device)
#vi = vi.to(device)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=10e-4)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

# training..

for epoch in range(num_epochs):
    print(f"epoch: {epoch}/{num_epochs}")
    
    training_epoch_data = defaultdict(list)
    vae.train()
    
    for x in train_loader:
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1000)
        
        optimizer.step()
        
        # gather data for the current batch
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
    
    print("training | elbo: {:2f}, log_px: {:.2f}, kl: {:.2f}:".format(np.mean(training_epoch_data["elbo"]), np.mean(training_epoch_data["log_px"]), np.mean(training_epoch_data["kl"])))
    
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
    
    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()
        
        # Just load a single batch from the test loader
        '''x, y = next(iter(test_loader))'''
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
        
    print("validation | elbo: {:2f}, log_px: {:.2f}, kl: {:.2f}:".format(np.mean(validation_data["elbo"]), np.mean(validation_data["log_px"]), np.mean(validation_data["kl"])))    
    

print("finished training!")

create_directory("images")

vae.eval() # because of batch normalization

n = 10
for i in range(n):
    x = train_set[i]
    
    x = x[None,:,:,:]
    
    outputs = vae(x.cuda())
    px = outputs["px"]
    
    x_reconstruction = px.sample()
    x_reconstruction = x_reconstruction[0]
    
    save_image(x_reconstruction.cpu(), "images/x{}_reconstruction.npy".format(i))
    save_image(x.cpu(), "images/x{}.npy".format(i))

print("saved images")
print("script done!")