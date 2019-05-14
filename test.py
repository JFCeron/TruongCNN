"""
Evaluates the results of train.py on its test set, calculating the images if needed

@author: Bananin
"""
import os
import time
import pdb
import numpy as np
import pandas as pd
# custom constants for this project
import constants
# image dataset generation
from signal2img import *
# custom dataset class and neural network
from EegDataset import EegDataset
from TruongNet import TruongNet
#from TruongNet_old import TruongNet
# torch deep learning
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn

def main():
    # command line parameters
    params = constants.parseCommandLineParams()
    # evaluation metric names (for results table)
    metric_names = constants.test_metric_names

    # the folders associated with this parametrizarion (in 'preproccesed' and 'results')
    params_folder = "W="+str(params.W)+ "_O=("+('%.2f'%params.O0)+","+('%.2f'%params.O1)+")_w="\
        +str(params.w)+"_o="+('%.2f'%params.o)+"_maxPerClass="+str(params.maxPerClass)+"_ch["+str(params.ch).replace(" ","")+"]/"

    # exit script if test_metrics.csv exists
    if os.path.isfile("results/"+params_folder+"/test_responses.csv"):
        print("Test metrics file exists, exiting script")
        exit()

    # generate the image dataset if it doesn't exist
    test_root = "preprocessed/"+params_folder+"Pat"+params.patient+"Test"
    if not os.path.exists(test_root):
        generate_images(data_subfolder="Pat"+params.patient+"Test", W=params.W, O=(params.O0,params.O1),w=params.w,
            o=params.o, channels=[int(c) for c in params.ch.split(",")], maxPerClass=params.maxPerClass)
    print("Image dataset ready")

    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("WARNING: GPU not available")
        device = torch.device("cpu")

    # generate Pytorch Datasets and Dataloaders
    test_dataset = EegDataset(test_root)
    test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch, shuffle=True, num_workers=30)
    print("DataLoaders ready")

    # load one image to extract dimensions for TruongNet instanciation
    img0, label0 = test_dataset.__getitem__(0)
    # CNN class instance
    model = TruongNet(img0.shape[0], img0.shape[1], img0.shape[2])
    del img0, label0

    # loss weights are added to optimize for F2-measure
    criterion = nn.CrossEntropyLoss(reduction="sum", weight=torch.Tensor([params.weight0,params.weight1]).to(device))

    # load best (not last!) model for testing
    try:
        best_state = torch.load("results/"+params_folder+"best_model.pth.tar")
        model.load_state_dict(best_state)
        model.to(device)
        print("Best model loaded")
    except:
        print("Best model not available or corrupted, cannot test")
        exit()

    # build matrix with a row for each test image, with columns stating true label and our output
    column_names = ["label","p0","p1"]
    test_responses = pd.DataFrame([], columns=column_names)
    softmax = nn.Softmax(dim=1)
    model.eval() # batchnorm or dropout layers will work in eval mode
    with torch.no_grad(): # speeds up computations but you won’t be able to backprop
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            # Run the forward pass
            outputs = model(images)
            batch_responses = softmax(outputs.data).cpu().numpy()
            batch_responses = np.append(labels.numpy().reshape((len(labels),1)), batch_responses, axis=1)
            batch_responses = pd.DataFrame(batch_responses, columns=column_names)
            test_responses = test_responses.append(batch_responses, ignore_index=True)

    # store metrics
    test_responses.to_csv("results/"+params_folder+"test_responses.csv", index=False)

if __name__ == "__main__":
    main()
