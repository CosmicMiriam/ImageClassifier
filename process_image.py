import numpy as np
import torch
from PIL import Image

def process_image(image, normalize = True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 256, 256
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image.thumbnail(size)

    # The crop method from the Image module takes four coordinates as input.
    # The right can also be represented as (left+width)
    # and lower can be represented as (upper+height).
    (left, upper, right, lower) = (16, 16, 240, 240)
    pil_image = pil_image.crop((left, upper, right, lower))
    
    np_image = np.array(pil_image) / 255
    if (normalize):
        np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 1, 0))
    
    # transform numpy to torch sensor
    np_image = torch.from_numpy(np_image).float()
    
    return np_image