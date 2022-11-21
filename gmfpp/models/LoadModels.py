from gmfpp.models.CytoVariationalAutoencoder import *
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict


def LoadVAEmodel(datetime):
    validation_data = torch.load("dump/parameters/validation_data_{}.pt".format(datetime))
    training_data = torch.load("dump/parameters/training_data_{}.pt".format(datetime))
    VAE_settings = torch.load("dump/parameters/VAE_settings_{}.pt".format(datetime))
    vae = CytoVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    vae.load_state_dict(torch.load("dump/parameters/vae_parameters_{}.pt".format(datetime)))
    return vae, validation_data, training_data, VAE_settings

def initVAEmodel(latent_features= 256,
                    beta = 1.,
                    num_epochs = 1000,
                    batch_size = 32,
                    learning_rate = 1e-3,
                    weight_decay = 10e-4,
                    image_shape = np.array([3, 68, 68])):

    VAE_settings = {
        'latent_features' : latent_features,
        'beta' : beta,
        'num_epochs' : num_epochs,
        'batch_size' : batch_size,
        'learning_rate' : learning_rate,
        'weight_decay' : weight_decay,
        'image_shape' : image_shape
        }
    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    vae = CytoVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    return vae, validation_data, training_data, VAE_settings
    
    