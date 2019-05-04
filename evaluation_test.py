import os
import time
import argparse
import numpy as np
import pdb
# image dataset generation
from signal2img import *
# custom dataset class and neural network
from EegDataset import EegDataset
from TruongNet import TruongNet
# torch deep learning
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn

# parse command line parameters
parser = argparse.ArgumentParser()
# data subset for this analysis
parser.add_argument("-data", type=str, default="Pat3Train")
# image time range and overlap for negatives and positives
parser.add_argument("-W", type=int, default=30)
parser.add_argument("-S0", type=float, default=0)
parser.add_argument("-S1", type=float, default=5.08/6)
# STFT small window time range and overlap
parser.add_argument("-w", type=float, default=1)
parser.add_argument("-s", type=float, default=2/3)
# channels encoded in a string
parser.add_argument("-ch", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
# neural network hyperparameters
parser.add_argument("-batch", type=int, default=16)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-miniepochs", type=int, default=100)
# object that stores all parameters
params = parser.parse_args()

# generate the image dataset if it doesn't exist
img_dataset_root = "preprocessed/"+params.data+"/W="+str(params.W)+  \
    "_S=("+('%.2f'%params.S0)+","+('%.2f'%params.S1)+")_w="+str(params.w)+  \
    "_s="+('%.2f'%params.s)+"_ch["+str(params.ch).replace(" ","")+"]/"
if not os.path.exists(img_dataset_root):
    "Generating dataset at "+img_dataset_root
    canales = [int(i) for i in params.ch.split(",")]
    generar_imagenes(data_subfolder=params.data, W=params.W, S=(params.S0,params.S1),
    w=params.w, s=params.s, canales=canales)
print("Image dataset ready")

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate Pytorch Datasets and Dataloaders
train_dataset = EegDataset(img_dataset_root+"/train")
train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch, shuffle=True, num_workers=20)
val_dataset = EegDataset(img_dataset_root+"/val")
val_loader = DataLoader(dataset=val_dataset, batch_size=params.batch, shuffle=True, num_workers=20)
test_dataset = EegDataset(img_dataset_root+"/test")
test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch, shuffle=True, num_workers=20)
print("DataLoaders ready")
# load one image to extract dimensions for TruongNet instanciation
img0, label0 = train_dataset.__getitem__(0)

# CNN class instance
model = TruongNet(img0.shape[0], img0.shape[1], img0.shape[2])
model.to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
# decay learning rate when valaidation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

def metrics(model, loader, device, criterion, max_batches=float("inf")):
    """
    Report loss, precision and recall of a model on data from a loader
    Leave max_batches unspecified to iterate through complete dataloader
    """
    loss = tp = tn = fp = fn = 0
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        # Run the forward pass
        outputs = model(images)
        predictions = torch.max(outputs.data, 1)[1]
        # aggregate partial statistics on this mini-batch
        loss += criterion(outputs, labels)
        tp += torch.sum(predictions*labels).item()
        fp += torch.sum(predictions*(1-labels)).item()
        tn += torch.sum((1-predictions)*(1-labels)).item()
        fn += torch.sum((1-predictions)*labels).item()
        # limit the number of batches to make computation faster
        if i >= max_batches:
            print("Last batch "+str(i))
            break

    # calculate precision and recall
    precision = float("inf") if tp+fp == 0 else tp/(tp+fp)
    recall = float("inf") if tp+fn == 0 else tp/(tp+fn)

    return loss, precision, recall

pdb.set_trace()
