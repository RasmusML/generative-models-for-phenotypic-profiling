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

######### Utilities #########

def get_clock_time():
    from time import gmtime, strftime
    result = strftime("%H:%M:%S", gmtime())
    return result
    
def cprint(s: str):
    clock = get_clock_time()
    print("{} | {}".format(clock, s))


torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}")


######### loading data #########

path = get_server_directory_path()
#path = "data/all/"

metadata = read_metadata(path + "metadata.csv")
#metadata = metadata[:100] # @TODO: figure what to do loading the imabes below gets _very_ slow after 50_000 images
cprint("loaded metadata")

cprint("loading images")
relative_paths = get_relative_image_paths(metadata)
image_paths = [path + relative for relative in relative_paths]
images = load_images(image_paths, verbose=False)
cprint("loaded images")

train_set = prepare_raw_images(images)
normalize_channels_inplace(train_set)
cprint("normalized images")


######### VAE Configs #########
image_shape = np.array([3, 68, 68])
latent_features = 256

vae = CytoVariationalAutoencoder(image_shape, latent_features)
#vae = VariationalAutoencoder(image_shape, latent_features)
vae = vae.to(device)

beta = 1.
vi = VariationalInference(beta=beta)


######### Training Configs #########
num_epochs = 30
batch_size = 32

learning_rate = 1e-3
weight_decay = 10e-4

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


######### Training #########

training_data = defaultdict(list)
validation_data = defaultdict(list)

for epoch in range(num_epochs):
    cprint(f"epoch: {epoch}/{num_epochs}")
    
    training_epoch_data = defaultdict(list)
    vae.train()
    
    for x in train_loader:
        x = x.to(device)
        
        loss, diagnostics, outputs = vi(vae, x)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1000)
        optimizer.step()
        
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
    
    cprint("training | elbo: {:2f}, log_px: {:.2f}, kl: {:.2f}:".format(np.mean(training_epoch_data["elbo"]), np.mean(training_epoch_data["log_px"]), np.mean(training_epoch_data["kl"])))
    
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
    
    
    with torch.no_grad():
        vae.eval()
        
        # Just load a single batch from the test loader
        '''x, y = next(iter(test_loader))'''
        x = x.to(device)
        
        loss, diagnostics, outputs = vi(vae, x)
        
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
        
    cprint("validation | elbo: {:2f}, log_px: {:.2f}, kl: {:.2f}:".format(np.mean(validation_data["elbo"]), np.mean(validation_data["log_px"]), np.mean(validation_data["kl"])))    
    

cprint("finished training")


######### Save VAE parameters #########
create_directory("dump/parameters")
torch.save(vae.state_dict(), "dump/parameters/vae_parameters.pt")


######### extract a few images already #########
create_directory("dump/images")

vae.eval() # because of batch normalization

n = 10
for i in range(n):
    x = train_set[i]    
    x = x[None,:,:,:]
    x = x.to(device)
   
    outputs = vae(x)
    px = outputs["px"]
    
    x_reconstruction = px.sample()
    x_reconstruction = x_reconstruction[0]
    
    save_image(x_reconstruction.cpu(), "dump/images/x{}_reconstruction.npy".format(i))
    save_image(x.cpu(), "dump/images/x{}.npy".format(i))

cprint("saved images")
cprint("script done.")