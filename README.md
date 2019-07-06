# Image classifier
An image classifier built using Python and PyTorch

References:
- Dataset images from: [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- [Udacity AI Nanodegree starter project](https://github.com/udacity/aipnd-project)


# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, I've trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. I've used [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import datasets, transforms
from torch import nn, optim

import torchvision.models as models
import time

import numpy as np

from PIL import Image
from collections import OrderedDict
```

## Load the data

`torchvision` has been used to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data can be [downloaded here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, transformations are applied such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. The input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this the images have been cropped to the appropriate size.

The pre-trained networks were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets I've normalized the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

normalize = transforms.Normalize(mean, std)
resize = transforms.Resize(256)
data_loader_batch_size = 64
learning_rate = 0.001
dropout = 0.5
train_epochs = 3
train_steps = 3
hidden_units = 512

model_type_vgg16 = 'vgg16'
model_type_alexnet = 'alexnet'
model_type_densenet121 = 'densenet121'

model_type = model_type_densenet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
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
```


```python
# Show sample images
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def show_images(data):
    """Show images from the input datasets for testing purposes

    Parameters:
    data (DataLoader): the data loader to visualise
   """
    data_iter = iter(data)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for ii in range(4):
        ax = axes[ii]
        imshow(images[ii], ax=ax)

print("Train / Valid / Test data:")
show_images(train_loader)
show_images(valid_loader)
show_images(test_loader)
```

    Train / Valid / Test data:



![png](output_5_1.png)



![png](output_5_2.png)



![png](output_5_3.png)


### Label mapping

Labels have been loaded from the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

output_size = len(cat_to_name)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

The classifier performs the following steps:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html): the example uses vgg16, densenet121 and alexnet
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters


```python
def build_model(model_type_input, device, output, dropout_value, learning_rate_value, hidden_units):
    """Build a model using a pretrained one

    Parameters:
    model_type_input (string): the model type - alexnet, vgg16 or resnet

    Returns:
    object:the model
    object:the loss criterion
    object:the optimizer
   """

    print('Initializing model: {}'.format(model_type_input))
    
    ## Initialize alexnet model
    if (model_type_input == model_type_alexnet):    
        model_new = models.alexnet(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model_new.parameters():
            param.requires_grad = False

        model_new.classifier = nn.Sequential(
         nn.Dropout(dropout_value),
         nn.Linear(9216, hidden_units),
         nn.ReLU(),
         nn.Linear(hidden_units, output),
         nn.LogSoftmax(dim = 1))
    
    ## Initialize vgg16 model
    if (model_type_input == model_type_vgg16):    
        model_new = models.vgg16(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model_new.parameters():
            param.requires_grad = False

        model_new.classifier = nn.Sequential(
         nn.Dropout(dropout_value),
         nn.Linear(25088, hidden_units),
         nn.ReLU(),
         nn.Linear(hidden_units, output),
         nn.LogSoftmax(dim = 1))
    
    ## Initialize densenet model
    if (model_type_input == model_type_densenet121):
        model_new = models.densenet121(pretrained=True)
    
        # Freeze parameters so we don't backprop through them
        for param in model_new.parameters():
            param.requires_grad = False
    
        model_new.classifier = nn.Sequential(OrderedDict([
              ('dropout', nn.Dropout(dropout_value)),
              ('fc1', nn.Linear(1024, hidden_units)),
              ('relu', nn.ReLU()),
              ('fc2', nn.Linear(hidden_units, output)),
              ('output', nn.LogSoftmax(dim=1))
              ]))
        
    # Only train the classifier parameters, feature parameters are frozen
    custom_optimizer = optim.Adam(model_new.classifier.parameters(), learning_rate_value)
    model_new.to(device);
    
    return model_new, nn.NLLLoss(), custom_optimizer
```


```python
# Initialize the model
model, criterion, optimizer = build_model(model_type, device, output_size, dropout, learning_rate, hidden_units)

# Train the network
def train_network(epochs = 5, steps = 3):
    """Train the network

    Parameters:
    epochs (int): epochs number
    steps  (int): the number of steps
   """
    train_losses, test_losses = [], []
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(test_loader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()

    #Print training and validation loss
    plt.plot(train_losses, label = 'Training loss')
    plt.plot(test_losses, label = 'Validation loss')
    plt.legend(frameon = False)
    
train_network(train_epochs, train_steps)
```

    Initializing model: densenet121


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
    Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.torch/models/densenet121-a639ec97.pth
    100%|██████████| 32342954/32342954 [00:00<00:00, 90704427.59it/s]


    Epoch 1/3.. Train loss: 3.196.. Test loss: 4.461.. Test accuracy: 0.032
    Epoch 1/3.. Train loss: 4.374.. Test loss: 4.258.. Test accuracy: 0.043
    Epoch 1/3.. Train loss: 4.202.. Test loss: 3.972.. Test accuracy: 0.157
    Epoch 1/3.. Train loss: 3.983.. Test loss: 3.629.. Test accuracy: 0.291
    Epoch 1/3.. Train loss: 3.723.. Test loss: 3.243.. Test accuracy: 0.290
    Epoch 1/3.. Train loss: 3.346.. Test loss: 2.888.. Test accuracy: 0.418
    Epoch 1/3.. Train loss: 3.184.. Test loss: 2.477.. Test accuracy: 0.458
    Epoch 1/3.. Train loss: 2.763.. Test loss: 2.200.. Test accuracy: 0.540
    Epoch 1/3.. Train loss: 2.557.. Test loss: 1.943.. Test accuracy: 0.594
    Epoch 1/3.. Train loss: 2.349.. Test loss: 1.718.. Test accuracy: 0.635
    Epoch 2/3.. Train loss: 2.119.. Test loss: 1.529.. Test accuracy: 0.690
    Epoch 2/3.. Train loss: 1.886.. Test loss: 1.401.. Test accuracy: 0.684
    Epoch 2/3.. Train loss: 1.902.. Test loss: 1.307.. Test accuracy: 0.724
    Epoch 2/3.. Train loss: 1.855.. Test loss: 1.182.. Test accuracy: 0.750
    Epoch 2/3.. Train loss: 1.715.. Test loss: 1.110.. Test accuracy: 0.760
    Epoch 2/3.. Train loss: 1.783.. Test loss: 1.082.. Test accuracy: 0.727
    Epoch 2/3.. Train loss: 1.648.. Test loss: 0.975.. Test accuracy: 0.794
    Epoch 2/3.. Train loss: 1.515.. Test loss: 0.962.. Test accuracy: 0.799
    Epoch 2/3.. Train loss: 1.644.. Test loss: 0.837.. Test accuracy: 0.808
    Epoch 2/3.. Train loss: 1.454.. Test loss: 0.865.. Test accuracy: 0.805
    Epoch 3/3.. Train loss: 1.414.. Test loss: 0.838.. Test accuracy: 0.810
    Epoch 3/3.. Train loss: 1.463.. Test loss: 0.736.. Test accuracy: 0.821
    Epoch 3/3.. Train loss: 1.354.. Test loss: 0.773.. Test accuracy: 0.815
    Epoch 3/3.. Train loss: 1.397.. Test loss: 0.727.. Test accuracy: 0.830
    Epoch 3/3.. Train loss: 1.369.. Test loss: 0.800.. Test accuracy: 0.785
    Epoch 3/3.. Train loss: 1.425.. Test loss: 0.694.. Test accuracy: 0.815
    Epoch 3/3.. Train loss: 1.394.. Test loss: 0.670.. Test accuracy: 0.839
    Epoch 3/3.. Train loss: 1.398.. Test loss: 0.637.. Test accuracy: 0.853
    Epoch 3/3.. Train loss: 1.334.. Test loss: 0.612.. Test accuracy: 0.864
    Epoch 3/3.. Train loss: 1.276.. Test loss: 0.625.. Test accuracy: 0.844
    Epoch 3/3.. Train loss: 1.217.. Test loss: 0.573.. Test accuracy: 0.870



![png](output_10_3.png)


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. The following function runs the test images through the network and measures the accuracy, the same way as validation.


```python
def test_network(loader, device_test):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device_test), labels.to(device_test)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(loader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(loader)))
    
test_network(test_loader, device)
```

    Test Loss: 0.573..  Test Accuracy: 0.865


## Save the checkpoint

Now that your network is trained, it can be saved for loading it later and making predictions.

```python
# print("Our model: \n\n", model, '\n')
# print("The state dict keys: \n\n", model.state_dict().keys())

class_to_idx = train_data.class_to_idx
checkpoint = {
              'model_type': model_type,
              'state_dict': model.state_dict(),
              'epochs': train_epochs,
              'optimizer.state_dict': optimizer.state_dict,
              'class_to_idx': class_to_idx,
              'dropout': dropout,
              'learning_rate': learning_rate
             }

torch.save(checkpoint, 'checkpoint.pth')
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    dropout = checkpoint['dropout']
    
    learning_rate = checkpoint['learning_rate']
    
    model, criterion, optimizer = build_model(checkpoint['model_type'], device, output_size, dropout, learning_rate, hidden_units)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    class_to_idx = checkpoint['class_to_idx']
    
    return model, criterion, optimizer, class_to_idx

model, criterion, optimizer, class_to_idx = load_checkpoint('checkpoint.pth')
```

    Initializing model: densenet121


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.


# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

```python
def process_image(image, normalize = True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 256, 256
    
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

# Test correct visualisation
imshow(process_image(train_dir + "/1/image_06734.jpg"))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa75c223940>




![png](output_18_1.png)

## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, checkpoint_file_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model, criterion, optimizer, classes_dictionary = load_checkpoint(checkpoint_file_name)
    
    # Implement the code to predict the class from an image file
    tensor_image = process_image(image_path)
    
    tensor_image = tensor_image.to(device)
    
    inv_classes_dictionary = {v: k for k, v in classes_dictionary.items()}
        
    # Calculate the class probabilities (softmax) for img
    model.eval()
    with torch.no_grad():
        tensor_image.unsqueeze_(0)
        logps=model(tensor_image)
        ps=torch.exp(logps)
        probs, classes = ps.topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        
        return_classes = []
        for c in classes:
            return_classes.append(inv_classes_dictionary[c])
        
    return probs, return_classes
    
probs, classes = predict(test_dir + "/1/image_06743.jpg", 'checkpoint.pth', 5)

print(probs)
print(classes)
```

    Initializing model: densenet121


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.


    [0.4922772943973541, 0.22364234924316406, 0.08935745805501938, 0.04865694046020508, 0.021168140694499016]
    ['1', '83', '70', '86', '53']


## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs.

```python
# Display an image along with the top 5 classes
def view_classify(img_path, checkpoint_file):
    ''' Function for viewing an image and it's predicted classes.
    '''
    
    number_classes = 5
    
    tensor_image = process_image(img_path, False)   
    probs, classes = predict(img_path, checkpoint_file)
    
    img = mpimg.imread(img_path)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(8,8), nrows=2)

    probs = np.flip(probs, axis = 0)
    classes = np.flip(classes, axis = 0)
    labels = [cat_to_name[cl] for cl in classes]
    
    np_image = tensor_image.numpy().squeeze()
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(cat_to_name[classes[number_classes-1]])
    
    ax2.barh(np.arange(len(probs)), probs)
    ax2.set_aspect(0.185)
    ax2.set_yticks(np.arange(number_classes))
    ax2.set_yticklabels(labels)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1)

    plt.tight_layout()
```


```python
view_classify(test_dir + "/1/image_06743.jpg", "checkpoint.pth")
```

    Initializing model: densenet121


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.



![png](output_24_2.png)



```python
view_classify(test_dir + "/16/image_06657.jpg", "checkpoint.pth")
```

    Initializing model: densenet121


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.



![png](output_25_2.png)


