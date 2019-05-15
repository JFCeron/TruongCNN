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
    # parse command line parameters
    params = constants.parseCommandLineParams()
    # evaluation metric names (for results table)
    metric_names = constants.train_metric_names
    pdb.set_trace()
    # the folders associated with this parametrizarion (in 'preproccesed' and 'results')
    params_folder = "W="+str(params.W)+ "_O=("+('%.2f'%params.O0)+","+('%.2f'%params.O1)+")_w="\
        +str(params.w)+"_o="+('%.2f'%params.o)+"_maxPerClass="+str(params.maxPerClass)+"_ch["+str(params.ch).replace(" ","")+"]/"

    # generate the image dataset if it doesn't exist
    train_root = "preprocessed/"+params_folder+"Pat"+params.patient+"Train"
    if not os.path.exists(train_root):
        generate_images(data_subfolder="Pat"+params.patient+"Train", W=params.W, O=(params.O0,params.O1),w=params.w,
        o=params.o, channels=[int(c) for c in params.ch.split(",")], maxPerClass=params.maxPerClass)
    val_root = "preprocessed/"+params_folder+"Pat"+params.patient+"Val"
    if not os.path.exists(val_root):
        generate_images(data_subfolder="Pat"+params.patient+"Val", W=params.W, O=(params.O0,params.O1),w=params.w,
        o=params.o, channels=[int(c) for c in params.ch.split(",")])
    print("Image dataset ready")

    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("WARNING: GPU not available")
        device = torch.device("cpu")

    # generate Pytorch Datasets and Dataloaders
    train_dataset = EegDataset(train_root)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch, shuffle=True, num_workers=30)
    val_dataset = EegDataset(val_root)
    val_loader = DataLoader(dataset=val_dataset, batch_size=params.batch, shuffle=True, num_workers=30)
    print("DataLoaders ready")

    # load one image to extract dimensions for TruongNet instanciation
    img0, label0 = train_dataset.__getitem__(0)
    # CNN class instance
    model = TruongNet(img0.shape[0], img0.shape[1], img0.shape[2])
    del img0, label0

    # loss and optimizer; weights are added to optimize for F2-measure
    criterion = nn.CrossEntropyLoss(reduction="sum", weight=torch.Tensor([params.weight0,params.weight1]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    # decay learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params.sch_patience, verbose=True)

    # load last (not best!) checkpoint (model, optimizer, scheduler) and execution metrics if they exist
    try:
        all_metrics = pd.read_csv("results/"+params_folder+"train_metrics.csv")
        checkpoint = torch.load("results/"+params_folder+"checkpoint.pth.tar")
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded: Resuming from epoch {}".format(checkpoint['epoch']))
    except:
        print("No check points available; starting from scratch")
        all_metrics = pd.DataFrame([], columns=metric_names)
        model.to(device)
        start_epoch = 1
    try:
        os.makedirs("results/"+params_folder)
    except:
        pass

    # Train the model
    print("Training started")
    train_start = time.time()
    total_steps = len(train_loader)

    for epoch in range(start_epoch, params.epochs+1):
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
        epoch_duration = time.time()-epoch_start
        print("Epoch [{}/{}] duration: {:.2f}".format(epoch, params.epochs, epoch_duration))

        # training epoch over; free up memory for evaluation
        del images, labels, outputs, loss
        torch.cuda.empty_cache()

        # record validation loss, precision and recall
        eval_start = time.time()
        # limit the number of evaluated samples for speed
        n_eval_samples = len(val_dataset)
        # training metrics
        train_lossXsample,train_tp,train_fp,train_tn,train_fn = metrics(model, train_loader, device, criterion, n_eval_samples)
        train_precision = float("inf") if train_tp+train_fp == 0 else train_tp/(train_tp+train_fp)
        train_recall = float("inf") if train_tp+train_fn == 0 else train_tp/(train_tp+train_fn)
        train_Fbeta = (1+params.beta**2)*(train_precision*train_recall)/((train_precision*params.beta**2)+train_recall)
        print("Train metrics: Loss per sample: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F_beta: {:.2f}"
            .format(train_lossXsample,train_precision,train_recall,train_Fbeta))
        # validation metrics
        val_lossXsample,val_tp,val_fp,val_tn,val_fn = metrics(model, val_loader, device, criterion, n_eval_samples)
        val_precision = float("inf") if val_tp+val_fp == 0 else val_tp/(val_tp+val_fp)
        val_recall = float("inf") if val_tp+val_fn == 0 else val_tp/(val_tp+val_fn)
        val_Fbeta = (1+params.beta**2)*(val_precision*val_recall)/((val_precision*params.beta**2)+val_recall)
        print("Val metrics:   Loss per sample: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F_beta: {:.2f}"
            .format(val_lossXsample,val_precision,val_recall,val_Fbeta))
        eval_duration = time.time()-eval_start
        print("Evaluation duration: {:.2f}".format(eval_duration))

        # does learning rate need decay?
        scheduler.step(val_lossXsample*n_eval_samples)

        # store metrics to make graphs
        epoch_metrics = pd.DataFrame([[train_lossXsample,train_tp,train_fp,train_tn,train_fn,train_precision,
                        train_recall,train_Fbeta,val_lossXsample,val_tp,val_fp,val_tn,val_fn,val_precision,
                        val_recall,val_Fbeta,epoch_duration,eval_duration]],columns=metric_names)
        all_metrics = all_metrics.append(epoch_metrics, ignore_index=True)
        all_metrics.to_csv("results/"+params_folder+"train_metrics.csv", index=False)

        # store a checkpoint in case of system failure
        torch.save({'epoch':epoch+1,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict()}, "results/"+params_folder+"checkpoint.pth.tar")

        # keep best model (only the model) in a separate file
        Fbeta_history = list(all_metrics["val_Fbeta"])
        best_epoch = Fbeta_history.index(max(Fbeta_history)) + 1
        if best_epoch == epoch:
            torch.save(model.state_dict(), "results/"+params_folder+"best_model.pth.tar")

        # early stop if F_beta is not decreasing anymore
        if best_epoch <= epoch - params.es_patience:
            print("Early stop at epoch {}".format(epoch))
            best_loss = all_metrics.loc[best_epoch-1, "val_lossXsample"]
            best_Fbeta = all_metrics.loc[best_epoch-1, "val_Fbeta"]
            best_tp = all_metrics.loc[best_epoch-1, "val_tp"]
            best_fp = all_metrics.loc[best_epoch-1, "val_fp"]
            best_fn = all_metrics.loc[best_epoch-1, "val_fn"]
            best_precision = float("inf") if best_tp+best_fp == 0 else best_tp/(best_tp+best_fp)
            best_recall = float("inf") if best_tp+best_fn == 0 else best_tp/(best_tp+best_fn)
            print("Best metrics: Loss per sample: {:.4f}, Precision: {:.2f}, Recall: {:.2f}, F_beta: {:.2f}"
                .format(best_loss,best_precision,best_recall,best_Fbeta))
            break

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
