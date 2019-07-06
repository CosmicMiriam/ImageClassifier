import torch

def save_checkpoint(model, model_type, train_data, epochs, optimizer, filename, savedir):
    """Save checkpoint

    Parameters:
    model (object): the trained model
    model_type (string): the model type
    train_data (object): the trained data
    epochs (int): the train epochs
    optimizer (object): the train optimizer
    filename (string): the checkpoint filename
    savedir (string): the checkpoint save dir
   """
    
    filename = savedir + "/" + filename
    print("Saving checkpoint: {}".format(filename))
    print()
    
    class_to_idx = train_data.class_to_idx
    checkpoint = {
              'model_type': model_type,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer.state_dict': optimizer.state_dict,
              'class_to_idx': class_to_idx
             }

    torch.save(checkpoint, filename)