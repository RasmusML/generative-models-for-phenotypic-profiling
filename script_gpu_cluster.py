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
from gmfpp.utils.utils import *

######### Utilities #########




torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}")


######### loading data #########

#path = get_server_directory_path()
path = "data/all/"

metadata = read_metadata(path + "metadata.csv")
metadata = metadata[:100] # @TODO: figure what to do loading the imabes below gets _very_ slow after 50_000 images
cprint("loaded metadata")

cprint("loading images")
relative_paths = get_relative_image_paths(metadata)
image_paths = [path + relative for relative in relative_paths]
images = load_images(image_paths, verbose=True, log_every=10000)
mapping = get_MOA_mappings(metadata)
cprint("loaded images")

train_set = images
normalize_channels_inplace(train_set)
cprint("normalized images")


######### VAE Configs #########
cprint("VAE Configs")
image_shape = np.array([3, 68, 68])
latent_features = 256

vae = CytoVariationalAutoencoder(image_shape, latent_features)
#vae = VariationalAutoencoder(image_shape, latent_features)
vae = vae.to(device)

beta = 1.
vi = VariationalInference(beta=beta)


######### Training Configs #########
cprint("Training Configs")
num_epochs = 10
batch_size = 32

learning_rate = 1e-3
weight_decay = 10e-4

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


######### VAE Training #########
cprint("VAE Training")

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
cprint("Save VAE parameters")
create_directory("dump/parameters")
torch.save(vae.state_dict(), "dump/parameters/vae_parameters.pt")
torch.save(validation_data, "dump/parameters/validation_data.pt")
torch.save(training_data, "dump/parameters/training_data.pt")




######### extract a few images already #########
cprint("Extract a few images already")
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




############ Classifier training ############
class NeuralNetwork(nn.Module):
    
    def __init__(self, n_classes: int = 13):
        super(NeuralNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes))

    def forward(self, x):
        logits = self.net(x)
        return logits

    
# VAE
image_shape = np.array([3, 68, 68])
latent_features = 256
#vae = CytoVariationalAutoencoder(image_shape, latent_features) # @TODO: load trained parameters
vae.eval()

# Classifier
N_classes = len(mapping)
classifier = NeuralNetwork(N_classes).to(device)

# Training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2)

num_epochs = 10
batch_size = 32

def count_num_correct(y_pred, y_true):
    return torch.sum(y_pred == y_true).item()

train_loss = []
train_accuracy = []

validation_loss = []
validation_accuracy = []

for epoch in range(num_epochs):
    cprint(f"epoch: {epoch}/{num_epochs}")    

    train_epoch_loss = []
    train_epoch_accuracy = []
    
    classifier.train()
    
    train_correct = 0
    train_num_predictions = 0
    
    for x, y in train_loader:
        x = x.to(device)
        
        outputs = vae(x)
        z = outputs["z"]
        
        prediction_prob = classifier(z)
        loss = loss_fn(prediction_prob, y)
        
        train_epoch_loss.append(loss.item())
        
        N = len(x)
        train_num_predictions += N
        pred = torch.argmax(prediction_prob, dim=1)
        train_correct += count_num_correct(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_mean_loss = np.mean(train_epoch_loss)
    train_loss.append(epoch_mean_loss)
    train_accuracy.append(train_correct / train_num_predictions)
    
    cprint("training | loss: {:.2f}".format(epoch_mean_loss))
    
    
    validation_epoch_loss = []
    classifier.eval()
    
    validation_correct = 0
    validation_num_predictions = 0
    


    
    with torch.no_grad():
        vae.eval()
        
        # Just load a single batch from the test loader
        '''x, y = next(iter(test_loader))'''
        x = x.to(device)
        outputs = vae(x)
        z = outputs["z"]

        prediction_prob = classifier(z)
        loss = loss_fn(prediction_prob, y)
        
        validation_epoch_loss.append(loss.item())
        pred = torch.argmax(prediction_prob, dim=1)
        validation_correct += count_num_correct(pred, y)

    print("validation | loss: {:.2f}".format(validation_epoch_loss))


