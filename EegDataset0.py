# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:45:43 2019

@author: Bananin
"""
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import numpy as np
import os
import re

class EegDataset(Dataset):
    def __init__(self, path2imgs, train_not_val, train_ratio):
        """
        Args:
            path2imgs (string): path to npy images of eeg STFT
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # keep the root path, image paths, labels and number of images
        self.path = path2imgs
        img_paths = np.array([path2imgs+"/"+f for f in os.listdir(path2imgs) if re.match(".*\.npy", f)])
        labels = np.array([bool(re.match(".*1\.csv.*", f)) for f in img_paths], dtype="long")

        # keep only training or validation subset
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos
        all_positives = np.where(labels)[0]
        all_negatives = np.where(labels == False)[0]

        if train_not_val:
            positives = all_positives[0:int(n_pos*train_ratio)]
            negatives = all_negatives[0:int(n_neg*train_ratio)]
        else:
            positives = all_positives[int(n_pos*train_ratio):]
            negatives = all_negatives[int(n_neg*train_ratio):]

        kept_indices = np.concatenate((positives, negatives))
        np.random.shuffle(kept_indices) # shuffle the training samples to remove temporal correlation
        self.img_paths = img_paths[kept_indices]
        self.labels = labels[kept_indices]
        self.data_len = len(self.img_paths)

    def __getitem__(self, index):
        # load a single image
        single_image_path = self.img_paths[index]
        single_image_label = self.labels[index]
        img_as_npy = np.load(single_image_path)
        img_as_tensor = self.to_tensor(img_as_npy)
        return (img_as_tensor.float(), torch.tensor(single_image_label, dtype=torch.long))

    def __len__(self):
        return self.data_len
