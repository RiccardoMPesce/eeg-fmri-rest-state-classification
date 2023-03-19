import nibabel
import mne

import importlib
import json
import wandb

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch import nn, optim, functional
from torch.nn import functional as F


class Conv1DBaseNet(nn.Module):
    """
    Baseline 1D CNN for EEG Classification
    """
    def __init__(self, args):
        # init args
        super(Conv1DBaseNet, self).__init__()
        args_defaults = {"in_channels": 8, "num_classes": 64, "verbose": False, "dropout": 0.25}
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        self.architecture = "CNN-1D"
        
        self.convolutions = nn.Sequential(

            # Layer 1
            nn.Conv1d(self.in_channels, 32, 64, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout),

            # Layer 2
            nn.Conv1d(32, 32, 48, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout),

            # Layer 3
            nn.Conv1d(32, 64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),

            # Layer 4
            nn.Conv1d(64, 64, 16, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),

            # Layer 5
            nn.Conv1d(64, 128, 8, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout)
        )

        self.pooling = nn.AdaptiveAvgPool1d(102)

        self.fully_connected = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.num_classes),
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        if self.verbose:
            print(x.size())

        if len(x.shape) == 4:
            x = x.squeeze(0)

        x = self.convolutions(x)

        if self.verbose:
            print(x.size())

        x = self.pooling(x)

        if self.verbose:
            print(x.size())

        B = x.size(0)
        x = x.view(B, -1)
        logits = self.fully_connected(x)

        if self.verbose:
            print(logits.size())    

        return logits


class ConvNet3D(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.architecture = "CNN-3D"
        self.layer1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(5))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer3 = nn.Flatten()
        self.layer4 = nn.Sequential(
            nn.Linear(2496,2),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out