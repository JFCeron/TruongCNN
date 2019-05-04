# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 00:22:36 2019

@author: Bananin
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
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

def main():
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
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-sch_patience", type=int, default=6)
    parser.add_argument("-es_patience", type=int, default=10)
    # printing parameters
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
    print("DataLoaders ready")

    # load one image to extract dimensions for TruongNet instanciation
    img0, label0 = train_dataset.__getitem__(0)
    # CNN class instance
    model = TruongNet(img0.shape[0], img0.shape[1], img0.shape[2])
    model.to(device)
    del img0, label0
    # loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    # decay learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

    # Train the model
    print("Training started")
    train_start = time.time()
    total_steps = len(train_loader)
    metric_names = ["train_lossXsample","train_tp","train_fp","train_tn","train_fn","train_f1",
                    "val_lossXsample","val_tp","val_fp","val_tn","val_fn","val_f1"]
    all_metrics = pd.DataFrame([], columns=metric_names)

    for epoch in range(1, params.epochs+1):
        # record gradients
        model.train()
        epoch_start = time.time()
        miniepoch_start = epoch_start # a mini-epoch is params.miniepochs mini-batches
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop and perform Adam optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # partial training report (once every mini-epoch)
            if i % params.miniepochs == 1:
                print("Epoch [{}/{}], Step [{}/{}], Loss per sample: {:.4f}, Duration: {:.2f}"
                    .format(epoch, params.epochs, i+1, total_steps, loss.item()/params.batch, time.time()-miniepoch_start))
                miniepoch_start = time.time()
        print("Epoch [{}/{}] duration: {:.2f}".format(epoch, params.epochs, time.time()-epoch_start))

        # training epoch over; free up memory for evaluation
        del images, labels, outputs, loss
        torch.cuda.empty_cache()

        # record validation loss, precision and recall
        eval_start = time.time()
        # limit the number of evaluated samples for speed
        n_eval_samples = len(val_dataset)
        train_lossXsample,train_tp,train_fp,train_tn,train_fn = metrics(model, train_loader, device, criterion, n_eval_samples)
        train_precision = float("inf") if train_tp+train_fp == 0 else train_tp/(train_tp+train_fp)
        train_recall = float("inf") if train_tp+train_fn == 0 else train_tp/(train_tp+train_fn)
        train_f1 = 2*(train_precision*train_recall)/(train_precision+train_recall)
        print("Train metrics: Loss per sample: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}"
            .format(train_lossXsample,train_precision,train_recall,train_f1))
        val_lossXsample,val_tp,val_fp,val_tn,val_fn = metrics(model, val_loader, device, criterion, n_eval_samples)
        val_precision = float("inf") if val_tp+val_fp == 0 else val_tp/(val_tp+val_fp)
        val_recall = float("inf") if val_tp+val_fn == 0 else val_tp/(val_tp+val_fn)
        val_f1 = 2*(val_precision*val_recall)/(val_precision+val_recall)
        print("Val metrics:   Loss per sample: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}"
            .format(val_lossXsample,val_precision,val_recall,val_f1))
        print("Evaluation duration: {:.2f}".format(time.time()-eval_start))
        # does learning rate need decay?
        scheduler.step(val_lossXsample*n_eval_samples)

        # store metrics to make graphs
        epoch_metrics = pd.DataFrame([[train_lossXsample,train_tp,train_fp,train_tn,train_fn,train_f1,
                        val_lossXsample,val_tp,val_fp,val_tn,val_fn,val_f1]], columns=metric_names)
        all_metrics = all_metrics.append(epoch_metrics)

        # store parameters unless validation F1 is on the rise
        f1_history = list(all_metrics["val_f1"])
        best_epoch = f1_history.index(max(f1_history)) + 1
        if best_epoch == epoch:
            # store parameters
            # store optimizer
        # early stop if f1 is not decreasing anymore
        if epoch - params.es_patience <= best_epoch:
            # early stop


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
            loss += criterion(outputs, labels)
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
