"""
Evaluates the results of train.py on its test set, calculating the images if needed

@author: Bananin
"""
import os
import time
import numpy as np
import pandas as pd
# custom constants for this project
import constants
# image dataset generation
from signal2img import *
# custom dataset class and neural network
from EegDataset import EegDataset
from TruongNet import TruongNet
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
    if os.path.isfile("results/"+params_folder+"/test_metrics.csv"):
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

    # calculate metrics on test set
    eval_start = time.time()
    lossXsample,tp,fp,tn,fn = metrics(model, test_loader, device, criterion)
    precision = float("inf") if tp+fp == 0 else tp/(tp+fp)
    recall = float("inf") if tp+fn == 0 else tp/(tp+fn)
    Fbeta = (1+params.beta**2)*(precision*recall)/((precision*params.beta**2)+recall)
    print("Test metrics:   Loss per sample: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F_beta: {:.2f}"
        .format(lossXsample,precision,recall,Fbeta))
    eval_duration = time.time()-eval_start
    print("Evaluation duration: {:.2f}".format(eval_duration))

    # store metrics
    test_metrics = pd.DataFrame([[lossXsample,tp,fp,tn,fn,precision,recall,Fbeta,eval_duration]],columns=metric_names)
    test_metrics.to_csv("results/"+params_folder+"test_metrics.csv", index=False)

def metrics(model, loader, device, criterion, max_samples=float("inf")):
    """
    Report loss, precision and recall of a model on data from a loader
    Leave max_samples unspecified to iterate through complete dataloader
    """
    loss = tp = tn = fp = fn = 0
    model.eval() # batchnorm or dropout layers will work in eval mode
    with torch.no_grad(): # speeds up computations but you won’t be able to backprop
        n_evaluated = 0
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            # Run the forward pass
            outputs = model(images)
            predictions = torch.max(outputs.data, 1)[1]
            # aggregate partial statistics on this mini-batch
            loss += criterion(outputs, labels).item()
            tp += torch.sum(predictions*labels).item()
            fp += torch.sum(predictions*(1-labels)).item()
            tn += torch.sum((1-predictions)*(1-labels)).item()
            fn += torch.sum((1-predictions)*labels).item()

            # limit the number of evaluated samples to make computation faster
            n_evaluated += images.shape[0]
            if n_evaluated >= max_samples:
                print("Samples evaluated: "+str(n_evaluated))
                break

    print("tp={:.2f}, fp={:.2f}, tn={:.2f}, fn={:.2f}".format(tp,fp,tn,fn))
    # calculate precision and recall
    precision = float("inf") if tp+fp == 0 else tp/(tp+fp)
    recall = float("inf") if tp+fn == 0 else tp/(tp+fn)
    lossXsample = loss/n_evaluated

    return lossXsample, tp, fp, tn, fn

if __name__ == "__main__":
    main()
