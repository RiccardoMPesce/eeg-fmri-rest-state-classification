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

def extract_relevant_markers_from_eeg(eeg_file, kind):
    recording_metadata = list(zip(eeg_file.annotations.description.tolist(), eeg_file.annotations.onset.tolist()))
    clean_recs_markers = [(rec[0], int(rec[1] * 1000)) for rec in recording_metadata if rec[0] in ("eeo", "eec", "beo", "bec", "mri")]
    
    clean_markers = clean_recs_markers[:]

    t_r = 1.95 if kind == "trio" else 2.00

    mri_intervals = []

    last_annotation = None

    for (a, t) in clean_markers:
        if a in ("beo", "eeo", "bec", "eec"):
            last_annotation = a
        else:
            if last_annotation == "beo" and a == "mri":
                mri_intervals += [((t, t + int(t_r * 1000)), "eo")]
            elif last_annotation == "bec" and a == "mri":
                mri_intervals += [((t, t + int(t_r * 1000)), "ec")]
            else:
                pass
    
    return mri_intervals


def retrive_times_fmri_cwl(raw):
    """
    Extract information from annottation raw file about fmri frames.
    This information importatn for further interpo
    
    -----
    Input
    Raw is file from EEG set.
    Retrive fMRI time annotation. When occurs recordings in seconds
    It is useful for aligning EEG and fMRI data 
    
    Output: 
    times_fmri - np 
        array of times in ms .
    """
    
    times_fmri = []
    for annot in raw.annotations:
        if annot['description'] == "mri":
            times_fmri.append(annot['onset'])

    times_fmri = np.array(times_fmri)
    times_fmri = times_fmri * 1000  # seconds to milliseconds
    
    return times_fmri


def dump_dataset_to_json(mri_images, eeg_files, database_base_path):
    couples = zip(mri_images, eeg_files)

    for mri_file, eeg_file in couples:
        eeg = mne.io.read_raw_eeglab(eeg_file)
        mri = nibabel.load(mri_file)

        filename = eeg_file.name.replace(".set", "")
        kind = "trio" if "trio" in filename else "verio"

        intervals = extract_relevant_markers_from_eeg(eeg, kind=kind)

        chunk_size = min(len(intervals), mri.get_fdata().shape[-1])

        ds = {
            "eeg": [],
            "fmri": [],
            "label": []
        }

        ds_no_cwl = {
            "eeg": [],
            "fmri": [],
            "label": []
        }

        print(f"Shape of data: {eeg.get_data().shape}")

        eeg_data = eeg.get_data()
        eeg_data_no_cwl = eeg.get_data(picks=[ch for ch in eeg.ch_names if "cw" not in ch.lower()])
        mri_data = mri.get_fdata()
        fmri_data = list(a.tolist() for a in np.array_split(mri_data, mri_data.shape[-1], axis=3))

        for (start, end), label in (intervals if len(intervals) <= chunk_size else intervals[:chunk_size]):
            eeg_chunk = eeg_data[:, start:end].tolist()
            eeg_chunk_no_cwl = eeg_data_no_cwl[:, start:end].tolist()

            ds["label"] += [label]
            ds_no_cwl["label"] += [label]

            ds["eeg"] += [eeg_chunk]
            ds_no_cwl["eeg"] += [eeg_chunk_no_cwl]

        ds["fmri"] = fmri_data if len(fmri_data) < chunk_size else fmri_data[:chunk_size]
        ds_no_cwl["fmri"] = fmri_data if len(fmri_data) < chunk_size else fmri_data[:chunk_size]

        ds_file = Path(database_base_path) / (filename + "_dataset.json")
        ds_file_no_cwl = Path(database_base_path) / (filename + "_no_cwl_dataset.json")

        with open(ds_file, "w") as ds_file:
            json.dump(ds, ds_file)

        with open(ds_file_no_cwl, "w") as ds_file_no_cwl:
            json.dump(ds_no_cwl, ds_file_no_cwl)

        print("Dumped ", filename)
        
        eeg_data = None
        
        eeg = None
        mri = None
        ds = {}


def melt_json(folder_path, out_file_name, use_cwl=True, dump_every=2):
    melted = []

    files = [f for f in Path(folder_path).glob("*dataset*") if "no_cwl" not in f.name] if use_cwl else [f for f in Path(folder_path).glob("*dataset*") if "no_cwl" in f.name]
    
    Path(out_file_name).touch(exist_ok=True)

    for count_processed, f in enumerate(files):
        print(f"Processing {f.name}")

        with open(f, "r") as in_f:
            ds = json.load(in_f)

        with open(out_file_name, "w") as out_f:
            keys = list(ds.keys())

            size = len(keys)

            for i in range(size):
                for key in keys:
                    melted = [{
                        key: ds[key][i]
                    }]

            if count_processed % dump_every == 0:
                json.dump(melted, out_f)

    return len(melted)


def dump_json_by_step(mri_files, eeg_files, dataset_base_path):
    ds = []
    ds_no_cwl = []
    couples = zip(mri_files, eeg_files)
    
    for mri_file, eeg_file in couples:
        eeg = mne.io.read_raw_eeglab(eeg_file)
        mri = nibabel.load(mri_file)

        filename = eeg_file.name.replace(".set", "")
        kind = "trio" if "trio" in filename else "verio"

        intervals = extract_relevant_markers_from_eeg(eeg, kind=kind)

        chunk_size = min(len(intervals), mri.get_fdata().shape[-1])

        print(f"Shape of data: {eeg.get_data().shape}")

        eeg_data = eeg.get_data()
        eeg_data_no_cwl = eeg.get_data(picks=[ch for ch in eeg.ch_names if "cw" not in ch.lower()])
        mri_data = mri.get_fdata()
        fmri_data = list(a.tolist() for a in np.array_split(mri_data, mri_data.shape[-1], axis=3))

        entry = {}
        entry_no_cwl = {}

        for (start, end), label in (intervals if len(intervals) <= chunk_size else intervals[:chunk_size]):
            eeg_chunk = eeg_data[:, start:end].tolist()
            eeg_chunk_no_cwl = eeg_data_no_cwl[:, start:end].tolist()

            entry["label"] = label
            entry_no_cwl["label"] = label

            entry["eeg"] = eeg_chunk
            entry_no_cwl["eeg"] = eeg_chunk_no_cwl

        entry["fmri"] = fmri_data if len(fmri_data) < chunk_size else fmri_data[:chunk_size]
        entry_no_cwl["fmri"] = fmri_data if len(fmri_data) < chunk_size else fmri_data[:chunk_size]

        ds_file = Path(dataset_base_path) / "dataset_melted.json"
        ds_file_no_cwl = Path(dataset_base_path) / "no_cwl_dataset_melted.json"

        ds += [entry]
        ds_no_cwl += [entry_no_cwl]

        with open(ds_file, "w") as ds_file:
            json.dump(ds, ds_file)

        with open(ds_file_no_cwl, "w") as ds_file_no_cwl:
            json.dump(ds_no_cwl, ds_file_no_cwl)

        print("Dumped ", filename)


def dump_dataset_by_interval(eeg_files, mri_files, dataset_base_path):
    (dataset_base_path / "by_interval").mkdir(exist_ok=True)

    couples = zip(eeg_files, mri_files)

    for mri_file, eeg_file in couples:
        eeg = mne.io.read_raw_eeglab(eeg_file)
        mri = nibabel.load(mri_file)

        filename = eeg_file.name.replace(".set", "")
        kind = "trio" if "trio" in filename else "verio"

        intervals = extract_relevant_markers_from_eeg(eeg, kind=kind)

        chunk_size = min(len(intervals), mri.get_fdata().shape[-1])

        eeg_data = eeg.get_data()
        eeg_data_no_cwl = eeg.get_data(picks=[ch for ch in eeg.ch_names if "cw" not in ch.lower()])
        mri_data = mri.get_fdata()
        fmri_data = list(a.tolist() for a in np.array_split(mri_data, mri_data.shape[-1], axis=3))

        for i, ((start, end), label) in enumerate((intervals if len(intervals) <= chunk_size else intervals[:chunk_size])):
            entry = {}
            entry_no_cwl = {}

            eeg_chunk = eeg_data[:, start:end].tolist()
            eeg_chunk_no_cwl = eeg_data_no_cwl[:, start:end].tolist()

            entry["label"] = label
            entry_no_cwl["label"] = label

            entry["eeg"] = eeg_chunk
            entry_no_cwl["eeg"] = eeg_chunk_no_cwl

            entry["fmri"] = fmri_data[i] 
            entry_no_cwl["fmri"] = fmri_data[i] 

            with open(dataset_base_path / "by_interval" / f"{filename}_s{start}_e{end}", "w") as ds_file:
                json.dump(entry, ds_file)

            with open(dataset_base_path / "by_interval" / f"{filename}_no_cwl_s{start}_e{end}", "w") as ds_file_no_cwl:
                json.dump(entry_no_cwl, ds_file_no_cwl)