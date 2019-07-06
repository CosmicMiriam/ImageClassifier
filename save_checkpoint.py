import torch

def save_checkpoint(model, model_type, train_data, epochs, optimizer, filename, savedir, dropout, learning_rate):
    """Save checkpoint

    Parameters:
    model (object): the trained model
    model_type (string): the model type
    train_data (object): the trained data
    epochs (int): the train epochs
    optimizer (object): the train optimizer
    filename (string): the checkpoint filename
    savedir (string): the checkpoint save dir
    dropout: (float): dropout used in training
    learning_rate: (float): learning rate used during training
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
              'class_to_idx': class_to_idx,
              'dropout': dropout,
              'learning_rate': learning_rate
             }

    torch.save(checkpoint, filename)