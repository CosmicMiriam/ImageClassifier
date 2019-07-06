import torchvision.models as models
from torch import nn, optim

from collections import OrderedDict

def build_model(model_type_input, device, output, dropout_value, learning_rate_value):
    """Build a model using a pretrained one

    Parameters:
    model_type_input (string): the model type - alexnet, vgg16 or resnet

    Returns:
    object:the model
    object:the loss criterion
    object:the optimizer
   """

    print('Initializing model: {}'.format(model_type_input))
    print()
    
    model_type_vgg16 = 'vgg16'
    model_type_alexnet = 'alexnet'
    model_type_densenet121 = 'densenet121'
    
    ## Initialize alexnet model
    if (model_type_input == model_type_alexnet):    
        model_new = models.alexnet(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model_new.parameters():
            param.requires_grad = False

        model_new.classifier = nn.Sequential(
         nn.Dropout(dropout_value),
         nn.Linear(9216, 2048),
         nn.ReLU(),
         nn.Linear(2048, output),
         nn.LogSoftmax(dim = 1))
        
        # Only train the classifier parameters, feature parameters are frozen
        custom_optimizer = optim.Adam(model_new.classifier.parameters(), learning_rate_value)
    
    ## Initialize vgg16 model
    if (model_type_input == model_type_vgg16):    
        model_new = models.vgg16(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model_new.parameters():
            param.requires_grad = False

        model_new.classifier = nn.Sequential(
         nn.Dropout(dropout_value),
         nn.Linear(25088, 2048),
         nn.ReLU(),
         nn.Linear(2048, 512),
         nn.ReLU(),
         nn.Linear(512, output),
         nn.LogSoftmax(dim = 1))
        
        # Only train the classifier parameters, feature parameters are frozen
        custom_optimizer = optim.Adam(model_new.classifier.parameters(), learning_rate_value)
    
    ## Initialize densenet model
    if (model_type_input == model_type_densenet121):
        model_new = models.densenet121(pretrained=True)
    
        # Freeze parameters so we don't backprop through them
        for param in model_new.parameters():
            param.requires_grad = False
    
        model_new.classifier = nn.Sequential(OrderedDict([
              ('dropout', nn.Dropout(dropout_value)),
              ('fc1', nn.Linear(1024, 500)),
              ('relu', nn.ReLU()),
              ('fc2', nn.Linear(500, output)),
              ('output', nn.LogSoftmax(dim=1))
              ]))
        
        # Only train the classifier parameters, feature parameters are frozen
        custom_optimizer = optim.Adam(model_new.classifier.parameters(), learning_rate_value)
    
    model_new.to(device);
    
    return model_new, nn.NLLLoss(), custom_optimizer