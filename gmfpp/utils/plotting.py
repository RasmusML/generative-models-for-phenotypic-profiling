from typing import List, Set, Dict, Tuple, Optional, Any
import torch
import matplotlib.pyplot as plt

from gmfpp.utils.data_transformers import view_as_image_plot_format, clip_image_to_zero_one

def plot_image(image: torch.Tensor, clip: bool = True, file=None, title=None):
    image = image.clone()

    if clip:
        image = clip_image_to_zero_one(image)

    plot_image = view_as_image_plot_format(image)
    plt.imshow(plot_image)

    if file==None:
        plt.show()
    else: 
        plt.savefig(file)

    plt.close()


def plot_image_channels(image: torch.Tensor, clip: bool = True, colorized: bool = True, file=None, title=None):
    image = image.clone()

    if clip:
        image = clip_image_to_zero_one(image)

    fig, axs = plt.subplots(1, 4, figsize=(14,6))
    if not title == None: fig.suptitle(title, fontsize=14)

    channel_names = ["DNA", "F-actin", "B-tubulin"]
    for i, name in enumerate(channel_names):
        if colorized:
            channel_image = torch.zeros_like(image)
            channel_image[i] = image[i]
        else:
            channel_image = image[i]
            channel_image = channel_image[None,:,:].expand(3, -1, -1)

        ax = axs[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        ax.imshow(view_as_image_plot_format(channel_image))

    ax = axs[3]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Combined")
    ax.imshow(view_as_image_plot_format(image))

    if file==None:
        plt.show()
    else: 
        plt.savefig(file)

    plt.close()


def plot_VAE_performance(elbo, log_px, kl, file=None, title=None):
    fig, axs = plt.subplots(1, 3, figsize=(14,6), constrained_layout = True)
    fig.suptitle(title, fontsize=16)
    
    ax1 = axs[0]
    ax1.grid()
    ax1.plot(log_px)
    ax1.set_ylabel("elbo")
    ax1.set_xlabel("epoch")
    
    ax2 = axs[1]
    ax2.grid()
    ax2.plot(log_px)
    ax2.set_ylabel("log p(x)")
    ax2.set_xlabel("epoch")
    
    ax3 = axs[2]
    ax3.grid()
    ax3.plot(kl)
    ax3.set_ylabel("KL-divergence")
    ax3.set_xlabel("epoch")
    
    if file == None:
        plt.show()
    else: 
        plt.savefig(file)
    
    plt.close()

def plot_control_vs_target_cells(vae, validation_data, training_data, VAE_settings, val_set_obs_1, val_set_obs_2):
    #Select 2 observations from the validation set
    x0, _ = validation_set[val_set_obs_1]
    x1, _ = validation_set[val_set_obs_2]
    
    #Get latents variables
    outputs0 = vae(x0[None,:,:,:])
    outputs1 = vae(x1[None,:,:,:])
    
    z0 = outputs0["z"].detach().numpy()
    z1 = outputs1["z"].detach().numpy()
    
    #Create 10 sets of latent variables from linear interpolations of the 2 images' latent variables
    zs = np.linspace(z0, z1, num=10)
    zs = torch.tensor(zs)
    
    #Calculate Cosine similarity
    cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    cos_list = []
    
    for i in range(len(zs)):
      input1 = vae.observation_model(zs[i]).sample()[0]
      benchmark = vae.observation_model(zs[-1]).sample()[0]
      cos_list.append(cos_similarity(input1, benchmark).mean().numpy().round(2))

    #Plotting
    fig = plt.figure(figsize=(25, 10))

    # setting values to rows and column variables
    rows = 1
    columns = len(zs)

    #Add subplot at the 1st position
    for i in range(columns):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow((torch.permute(vae.observation_model(zs[i]).sample()[0], (1, 2, 0))* 255).numpy().astype(np.uint8))
        plt.axis('off')
        plt.title(cos_list[i])
        if i == range(columns):
            plt.savefig('dump/parameters/control_vs_target_cells_plots_{}.pt'.format(datetime))
    
    plt.close()
    

