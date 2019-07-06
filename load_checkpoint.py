import torch

from build_model import build_model

def load_checkpoint(filename, savedir, output_size, device):
   """Load a model checkpoint from a file

    Parameters:
    filename (string): the name of the file
    savedir (string): the directory containing the file
    output_size (int): output size of the network
    device (string): cpu or cuda

    Returns:
    model, criterion, optimizer, class_to_idx
   """
     
   filename = savedir + "/" + filename
    
   checkpoint = torch.load(filename)
    
   model, criterion, optimizer = build_model(checkpoint['model_type'], device, output_size, checkpoint['dropout'], checkpoint['learning_rate'], checkpoint['hidden_units'])
    
   model.load_state_dict(checkpoint['state_dict'])
    
   class_to_idx = checkpoint['class_to_idx']
    
   return model, criterion, optimizer, class_to_idx