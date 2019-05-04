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
    def __init__(self, path2imgs):
        """
        Args:
            path2imgs (string): path to npy images of eeg STFT
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # keep the root path, image paths, labels and number of images
        self.path = path2imgs
        self.img_paths = np.array([path2imgs+"/"+f for f in os.listdir(path2imgs) if re.match(".*\.npy", f)])
        self.labels = np.array([bool(re.match(".*1\.csv.*", f)) for f in self.img_paths], dtype="long")
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
