# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:27:54 2019

Splits the training sets into Train and Val. Also appends the correct
label to each test .csv to make it more similar to the training files

The original dataset brings test labels in a separate pdf file. 

@author: Bananin
"""

import os
import pandas as pd
import numpy as np

val_ratio = 0.3

# split each PatXTrain into PatXTrain and PatXVal
train_folders = [f.path for f in os.scandir("dataset") if f.is_dir() and "train" in f.path.lower()]
for folder in train_folders:
    # make new folder for validation segments
    val_folder = folder.replace("Train","Val")
    os.makedirs(val_folder)
    # move val_ratio of the training segments there
    files = np.array([f.path for f in os.scandir(folder)])
    to_val = np.random.binomial(n=1, p=val_ratio, size=len(files)) == 1
    for val_file in files[to_val]:
        os.rename(val_file, val_file.replace("Train","Val"))

# fix test set names
test_folders = [f.path for f in os.scandir("dataset") if f.is_dir() and "test" in f.path.lower()]
test_labels = pd.read_csv("dataset/test_labels.csv", dtype="str")
test_labels.set_index(keys="image", drop=True, inplace=True)

for folder in test_folders:
    files = [f.path for f in os.scandir(folder)]
    image_names = [f.split("\\")[-1].split(".")[0] for f in files]
    for i in range(len(files)):
        label = test_labels.loc[image_names[i],"class"]
        new_name = files[i].replace("0.csv",str(label)+".csv")
        os.rename(files[i], new_name)
        
