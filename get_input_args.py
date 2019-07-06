#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    """
    #  python train.py --dir <directory with images> --arch <model>
    #    --checkpoint <file name for saving train data> --learning_rate <learning rate value> 
    #    --epochs <epochs number> --batchsize <batch size> --trainsteps <trainsteps> --gpu True
    #    --dropout <train dropout> --categories <categories file name> --savedir <checkpoint dir> 
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type = str, default = 'flowers', help = 'path to the folder of flowers images') 
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'CNN Model Architecture') 
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Archive with train data') 
    parser.add_argument('--learningrate', type = float, default = 0.001, help = 'Learning rate') 
    parser.add_argument('--epochs', type = int, default = 3, help = 'Train epochs') 
    parser.add_argument('--batchsize', type = int, default = 64, help = 'Train batch size') 
    parser.add_argument('--trainsteps', type = int, default = 3, help = 'Train steps')  
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'Train dropout')  
    parser.add_argument('--gpu', type = bool, default = True, help = 'Uses gpu for calculation')
    parser.add_argument('--categories', type = str, default = 'cat_to_name.json', help = 'Categories to names file')
    parser.add_argument('--savedir', type = str, default = 'checkpoints', help = 'Checkpoints save dir')
    
    return parser.parse_args()