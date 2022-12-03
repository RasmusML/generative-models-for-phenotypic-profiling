from typing import List, Set, Dict, Tuple, Optional, Any
import torch
import matplotlib.pyplot as plt

from gmfpp.utils.data_transformers import view_as_image_plot_format, clip_image_to_zero_one

def img_saturate(img):
    return img / torch.max(img)

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

 
def plot_VAE_performance(plotdata, file=None, title=None):
    keys = plotdata.keys()
    fig, axs = plt.subplots(1, len(keys), figsize=(14,6), constrained_layout = True)
    fig.suptitle(title, fontsize=16)
    for no, key in enumerate(keys):
        ax = axs[no]
        ax.grid()
        ax.plot(plotdata[key])
        ax.set_ylabel(key)
        ax.set_xlabel("epoch")
        
    if file == None:
        plt.show()
    else: 
        plt.savefig(file)
    
    plt.close()
