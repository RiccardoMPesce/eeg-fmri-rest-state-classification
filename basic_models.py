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


class ConvNet1D(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool1d(5))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, dilation=2),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer3 = nn.Flatten()
        self.layer4 = nn.Sequential(
            nn.Linear(2496, 512),
            nn.Linear(512, 2),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class ConvNet3D(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
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