from typing import List, Set, Dict, Tuple, Optional, Any
import torch
import numpy as np

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
    """ input shape: [sample, channel, height, width] """
    view = view_channel_dim_first(data)
    for i in range(view.shape[0]):
        view[i] /= torch.max(view[i])

def prepare_raw_images(images: List[np.ndarray]) -> torch.Tensor:
    """
        in:  [sample, height, width, channel]
        out: [sample, channel, height, width]
    """
    result = torch.tensor(np.array(images, dtype=np.float32)) 
    result = result.permute(0, 3, 1, 2)
    return result

def clip_image_to_zero_one(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image, 0., 1.)

def view_as_image_plot_format(image: torch.Tensor) -> torch.Tensor:
    ''' 
        in: (channel, height, width) 
        out:(height, width, channel)
    '''
    return image.permute(1,2,0)
