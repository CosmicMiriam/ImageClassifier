#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    """
#      python predict.py --image <path/to/the/image> --savedir <checkpoints dir>
#             --checkpoint <file name for loading train data> --categories <categories file name>
#             --gpu <use gpu True/False> --topk <number of classes to be returned>
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type = str, default='image.jpg', help = 'path to the folder of flowers images') 
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Archive with train data') 
    parser.add_argument('--gpu', type = bool, default = True, help = 'Uses gpu for calculation')
    parser.add_argument('--categories', type = str, default = 'cat_to_name.json', help = 'Categories to names file')
    parser.add_argument('--savedir', type = str, default = 'checkpoints', help = 'Checkpoints load dir')
    parser.add_argument('--topk', type = int, default = 3, help = 'Number of top classes to be returned')
    
    return parser.parse_args()