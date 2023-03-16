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



class EEGMRIDataset(Dataset):
    def __init__(self, directory_path, use_cwl=None, he_pump=None):
        self.dataset_files = list(Path(directory_path).glob("*"))

        self.use_cwl = use_cwl
        
        if use_cwl is not None:
            if use_cwl:
                self.dataset_files = [f for f in self.dataset_files if "no_cwl" not in f.name]
            else:
                self.dataset_files = [f for f in self.dataset_files if "no_cwl" in f.name]

        if he_pump is not None:
            if he_pump:
                self.dataset_files = [f for f in self.dataset_files if "hpump-on" in f.name]
            else:
                self.dataset_files = [f for f in self.dataset_files if "hpump-off" in f.name]

    def __len__(self):
        return len(self.dataset_files) 

    def __getitem__(self, idx):
        f_path = self.dataset_files[idx]
        
        with open(f_path, "r") as f_p:
            observation = json.load(f_p)
        
        label = 1 if observation["label"] == "eo" else 0
        fmri = torch.Tensor(observation["fmri"])
        eeg = torch.Tensor(observation["eeg"])

        if eeg.shape[1] < 2000:
            eeg = torch.cat((eeg, torch.zeros((eeg.shape[0], 50))), axis=1)

        if eeg.shape[0] not in (32, 38):
            if self.use_cwl:
                eeg = torch.cat((eeg, torch.zeros((38 - eeg.shape[0], eeg.shape[1]))), axis=0)
            else:
                eeg = torch.cat((eeg, torch.zeros((32 - eeg.shape[0], eeg.shape[1]))), axis=0)

        fmri = fmri.reshape(1, fmri.shape[0], fmri.shape[1], fmri.shape[2])

        return eeg, fmri, label