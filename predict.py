# PROGRAMMER: Davide Zordan
# PURPOSE: Predict the category of an image given a specific checkpoint containing a trained model
#
# Use following arguments:
#      python predict.py --image <path/to/the/image> --savedir <checkpoints dir>
#             --checkpoint <file name for loading train data> --categories <categories file name>
#             --gpu <use gpu True/False> --topk <number of classes to be returned>
#   Example call:
#    python predict.py --image flowers/test/1/image_06743.jpg --checkpoint checkpoint-densenet.pth --savedir checkpoints --gpu True --categories cat_to_name.json --topk 3
##

# Imports python modules
from time import time, sleep
import torch

# Imports functions created for this program
from get_input_args_predict import get_input_args
# from initialize import initialize
from load_checkpoint import load_checkpoint
from load_categories import load_categories
from process_image import process_image
# from test_network import test_network

def main():
    start_time = time()
    
    in_args = get_input_args()

    print("********************************")
    print("Predict image category using the following parameters:")
    print(in_args)
    print("********************************")
    print()

    # Loading categories
    categories = load_categories(in_args.categories)
    output_size = len(categories)
    print("Loaded {} categories from {}".format(output_size, in_args.categories))
    print()

    # Initialize device
    device = "cpu"
    if (in_args.gpu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device initialized: {}".format(device))
    print()
    
    # Load the checkpoint
    model, criterion, optimizer, classes_dictionary = load_checkpoint(in_args.checkpoint, in_args.savedir, output_size, device)
    print()
    print("Loaded checkpoint: {}".format(in_args.checkpoint))
    print()
    
    topk = in_args.topk
    image_path = in_args.image
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
    
    print("**************************************")
    print("Top {} most likely classes for {}:".format(topk, image_path))
    print()
    for i in range(0, len(return_classes)):
        print("{} - Probability: {}%".format(categories[return_classes[i]], round(probs[i] * 100, 1)))
    
    end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Call to main function to run the program
if __name__ == "__main__":
    main()