# Imports here
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import datasets, transforms
from torch import nn, optim

import torchvision.models as models
import time

import numpy as np

import json

from PIL import Image
from collections import OrderedDict

def initialize(image_dir, batch_size):
    data_dir = image_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    print("Initialized input folders: train => {} - valid => {} - test => {}".format(train_dir, valid_dir, test_dir))
    print()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean, std)
    resize = transforms.Resize(256)
    data_loader_batch_size = batch_size        
    
    # Define transforms for the training data and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize])

    valid_test_transforms = transforms.Compose([resize,
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = data_loader_batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = data_loader_batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = data_loader_batch_size, shuffle = True)
    
    return train_loader, valid_loader, test_loader, train_data