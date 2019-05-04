# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 00:22:36 2019

@author: Bananin
"""

from EegDataset import EegDataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pdb

# preprocessing parameters
W = 30
SOBRE = (20,0)
w = 3
sobre = 2
conjunto = "Train"
paciente = 1
canales = ["ch0","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15"]
path2imgs = "imagenes/W="+str(W)+"_SP="+str(SOBRE[0])+"_SN="+str(SOBRE[1])+"_w="+str(w)+"_s="+str(sobre)+"_canales="+str(canales)+"/Pat"+str(paciente)+conjunto+"/"
# cnn hyperparameters
num_epochs = 5
batch_size = 32
learning_rate = 0.001

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate Pytorch Datasets and Dataloaders
train_dataset = EegDataset(path2imgs, train_not_val=True, train_ratio=0.8)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = EegDataset(path2imgs, train_not_val=False, train_ratio=0.8)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # out_channels = number of convolutions to apply on this layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=len(canales), out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc1 = nn.Sequential(
            nn.Linear(64*1*37, 256), # 37 comes from frequency resolution, 256 from paper
            nn.Sigmoid()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax()
            )
    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # flatten
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        pdb.set_trace()
        return out

# CNN class instance
model = ConvNet()
model.to(device)
# model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
