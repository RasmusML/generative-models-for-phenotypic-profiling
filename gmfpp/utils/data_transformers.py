from typing import List, Set, Dict, Tuple, Optional, Any
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd


def view_flat_samples(data: torch.Tensor) -> torch.Tensor:
    """
        in:  [samples, channels, height, width]
        out: [samples, channels * height * width]
    """  
    return data.reshape(data.size(0), -1)

def view_channel_dim_first(data: torch.Tensor) -> torch.Tensor:
    """ 
        Swap sample and channel order.
        in:  [samples, channels, height, width]
        out: [channels, samples, height, width]
    """
    return torch.permute(data, dims=(1,0,2,3))

def normalize_channels_inplace(data: torch.Tensor):
    """ 
        Normalize by max channel across all images: X_i / max(X_i)
        input shape: [sample, channel, height, width]
    """
    view = view_channel_dim_first(data)
    for i in range(view.shape[0]):
        view[i] /= torch.max(view[i])

def normalize_every_image_channels_seperately_inplace(images: torch.Tensor):
    """ 
        As the original paper: X_ij / max(X_ij)
        input shape: [sample, channel, height, width] 
    """
    flat_images = images.reshape(images.size(0), 3, -1)
    
    max_values, _ =  torch.max(flat_images, dim=-1)
    view_max_expanded = max_values[:,:,None].expand(images.size(0), 3, 4624)
    
    flat_images /= view_max_expanded
    
def normalize_channels_by_max_inplace(data: torch.Tensor):
    """ 
        X / 40,000
        input shape: [sample, channel, height, width] 
    """
    data /= 40_000
    
 
def img_saturate(img: torch.Tensor) -> torch.Tensor:
    return img / torch.max(img)

def clip_image_to_zero_one(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image, 0., 1.)

def view_as_image_plot_format(image: torch.Tensor) -> torch.Tensor:
    ''' 
        in: (channel, height, width) 
        out:(height, width, channel)
    '''
    return image.permute(1,2,0)
    
class SingleCellDataset(Dataset):
    
    def __init__(self, metadata: pd.DataFrame, images: torch.Tensor, label_to_id: Dict[str, int]):
        self.metadata = metadata
        self.label_to_id = label_to_id
        self.images = images
        
    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        image_id = row["Single_Cell_Image_Id"]
        image = self.images[image_id]
        
        label_name = row["moa"]
        label = self.label_to_id[label_name]
        
        return image, label