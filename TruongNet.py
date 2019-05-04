"""
More general implementation of the network in 'Convolutional neural networks for
seizure prediction using intracranial and scalp electroencephalogram' by Truong,
Nguyen, Kuhlmann, Reza, Yang, Ippolito and Kavehei
"""
import torch.nn as nn
import pdb

class TruongNet(nn.Module):

    def __init__(self, c, t, f):
        """
        c : number of eeg channels in images
        t : length of temporal axis (vertical)
        f : length of frequency axis (horizonal)
        """
        super(TruongNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=c),
            # out_channels = number of convolutions to apply on this layer
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # sizes of fully connected layers depend on sample image dimensions
        height = (((t//4 - 2)//2) - 2)//2
        width = (((f//4 - 2)//2) - 2)//2
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            # input images at this point have 64 channels (form previous layer out)
            # we keep the last fc layer's size as in Truong (256)
            nn.Linear(64*height*width, 256),
            nn.Sigmoid()
            )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
            # no softmax activation because Pytorch's CrossEntropyLoss includes it
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # flatten
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
