#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# PROGRAMMER: Davide Zordan
# PURPOSE: Train a model
#
# Use following arguments:
#      python train.py --dir <directory with images> --arch <model>
#             --checkpoint <file name for saving train data> --learning_rate <learning rate value> 
#             --epochs <epochs number> --batchsize <batch size> --trainsteps <trainsteps> --gpu
#   Example call:
#    python train.py --dir flowers/ --arch densenet121 --checkpoint checkpoint.pth --learning_rate 0.001 --epochs 3 --batchsize 64 --trainsteps 3 --gpu
##

# Imports python modules
from time import time, sleep
import torch

# Imports print functions that check the lab
# from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from initialize import initialize
from build_model import build_model
from load_categories import load_categories
from train_network import train_network
from test_network import test_network
from save_checkpoint import save_checkpoint

def main():
    start_time = time()
    
    in_args = get_input_args()

    print("********************************")
    print("Training network using the following parameters:")
    print(in_args)
    print("********************************")
    print()
   
    learning_rate = in_args.learningrate
    dropout = in_args.dropout
    train_epochs = in_args.epochs
    train_steps = in_args.trainsteps

    categories = load_categories(in_args.categories)
    output_size = len(categories)
    
    model_type = in_args.arch

    # Initialize device
    device = "cpu"
    if (in_args.gpu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device initialized: {}".format(device))
    print()
    
    # Initialize the loaders
    train_loader, valid_loader, test_loader, train_data = initialize(in_args.dir, in_args.batchsize)
    
    # Initialize the model
    model, criterion, optimizer = build_model(model_type, device, output_size, dropout, learning_rate)
    
    # Train the network
    train_network(train_epochs, train_steps, train_loader, test_loader, model, device, optimizer, criterion)
    
    # Test network accuracy
    test_network(test_loader, device, model, criterion)
    
    # Save checkpoint
    save_checkpoint(model, model_type, train_data, train_epochs, optimizer, in_args.checkpoint, in_args.savedir)
    
    end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Call to main function to run the program
if __name__ == "__main__":
    main()