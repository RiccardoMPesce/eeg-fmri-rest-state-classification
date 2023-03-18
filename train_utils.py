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



def train_loop_eeg(model, loaders, optimizer, loss_function, accuracy_json, loss_json, checkpoint_prefix, n_epochs, device, lr=0.001, debug=False, save_every=2):
    # Init WandDB
    wandb.init(
    # set the wandb project where this run will be logged
        project = "EEG-FMRI-Rest-State-Classification",
        
        # track hyperparameters and run metadata
        config = {
        "learning_rate": lr,
        "architecture": model.architecture,
        "epochs": n_epochs,
        }
    )
    
    # Initialize training, validation, test losses and accuracy list
    accuracy_json_file_path = Path(accuracy_json)
    loss_json_file_path = Path(loss_json)

    if accuracy_json_file_path.exists() and accuracy_json_file_path.is_file():
        with open(accuracy_json_file_path, "r+") as accuracy_json_f:
            accuracies_per_epoch = json.load(accuracy_json_f)
            print(f"Loaded accuracy dictionary at {accuracy_json_file_path}")
    else:
        accuracies_per_epoch = {"train": [], "val": [], "test": []}

    if loss_json_file_path.exists() and loss_json_file_path.is_file():
        with open(loss_json_file_path, "r+") as loss_json_f:
            losses_per_epoch = json.load(loss_json_f)
            print(f"Loaded loss dictionary at {loss_json_file_path}")
    else:
        losses_per_epoch = {"train": [], "val": [], "test": []}

    starting_epoch = 0

    checkpoint_path = Path(checkpoint_prefix) 

    checkpoint_path.mkdir(exist_ok=True)

    # Check for the latest weights
    checkpoints = [checkpoint.name for checkpoint in checkpoint_path.glob("*.pth") if checkpoint.is_file()]

    if len(checkpoints) > 0:
        epochs = [int(s.replace(".pth", "")) for s in checkpoints]

        latest_state_dict_path = checkpoint_path / f"{max(epochs)}.pth"

        # Loading state dict
        model.load_state_dict(torch.load(latest_state_dict_path, map_location=device))
        print(f"Loaded weights generated at epoch {max(epochs)}")

        starting_epoch = max(epochs)

    best_accuracy = 0
    best_accuracy_val = 0
    best_epoch = 0
    
    predicted_labels = [] 
    correct_labels = []

    print(f"Training starting at epoch {starting_epoch + 1}")
    for epoch in range(starting_epoch + 1, n_epochs + starting_epoch + 1):
        # Initialize loss/accuracy variables
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}
        
        # Adjust learning rate for SGD
        """
        if OPTIMIZER == "SGD":
            lr = LEARNING_RATE * (LR_DECAY ** (epoch // LEARNING_RATE_DECAY_EVERY))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        """
        
        # Process each split
        for split in ("train", "val", "test"):
            # Set network mode
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)
            
            # Process all split batches
            for i, (input, _, target) in enumerate(loaders[split]):

                # Move model to device
                model = model.to(device)
                
                # Move tensors to device
                input = input.to(device)
                target = target.to(device)
                
                if debug:
                    print(input.device)

                # Forward
                output = model(input)

                # Compute loss
                loss = loss_function(output, target)
                losses[split] += loss.item()
                
                # Compute accuracy
                _, pred = output.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracy = correct / input.data.size(0)   
                accuracies[split] += accuracy
                counts[split] += 1
                
                # Backward and optimize
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Print info at the end of the epoch
        if accuracies["val"] / counts["val"] >= best_accuracy_val:
            best_accuracy_val = accuracies["val"] / counts["val"]
            best_accuracy = accuracies["test"] / counts["test"]
            best_epoch = epoch

        train_loss = losses["train"] / counts["train"]
        train_accuracy = accuracies["train"] / counts["train"]
        validation_loss = losses["val"] / counts["val"]
        validation_accuracy = accuracies["val"] / counts["val"]
        test_loss = losses["test"] / counts["test"]
        test_accuracy = accuracies["test"] / counts["test"]

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print("INFO")
        print(f"- Model: {model.__class__.__name__} - epoch {epoch}")
        print("STATS")
        print(f"- Training: Loss {train_loss:.4f}, Accuracy {train_accuracy:.4f} " 
            f"- Validation: Loss {validation_loss:.4f}, Accuracy {validation_accuracy:.4f} "
            f"- Test: Loss {test_loss:.4f}, Accuracy {test_accuracy:.4f}")
        print(f"Best Test Accuracy at maximum Validation Accuracy (validation_accuracy = {best_accuracy_val:.4f}) is {best_accuracy:.4f} at epoch {best_epoch}\n")

        losses_per_epoch["train"].append(train_loss)
        losses_per_epoch["val"].append(validation_loss)
        losses_per_epoch["test"].append(test_loss)
        accuracies_per_epoch["train"].append(train_accuracy)
        accuracies_per_epoch["val"].append(validation_accuracy)
        accuracies_per_epoch["test"].append(test_accuracy)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_path / f"{epoch}.pth")
            
            with open(accuracy_json_file_path, "w+") as accuracy_json_f:
                json.dump(accuracies_per_epoch, accuracy_json_f)
            
            with open(loss_json_file_path, "w+") as loss_json_f:
                json.dump(losses_per_epoch, loss_json_f)

    # At the end of training, save
    torch.save(model.state_dict(), checkpoint_path / f"{epoch}.pth")
    
    # Stop Wandb
    wandb.finish()



def train_loop_fmri(model, loaders, optimizer, loss_function, accuracy_json, loss_json, checkpoint_prefix, n_epochs, device, lr=0.001, debug=False, save_every=2):
    # Init WandDB
    wandb.init(
    # set the wandb project where this run will be logged
        project = "EEG-FMRI-Rest-State-Classification",
        
        # track hyperparameters and run metadata
        config = {
        "learning_rate": lr,
        "architecture": model.architecture,
        "epochs": n_epochs,
        }
    )
    
    # Initialize training, validation, test losses and accuracy list
    accuracy_json_file_path = Path(accuracy_json)
    loss_json_file_path = Path(loss_json)

    if accuracy_json_file_path.exists() and accuracy_json_file_path.is_file():
        with open(accuracy_json_file_path, "r+") as accuracy_json_f:
            accuracies_per_epoch = json.load(accuracy_json_f)
            print(f"Loaded accuracy dictionary at {accuracy_json_file_path}")
    else:
        accuracies_per_epoch = {"train": [], "val": [], "test": []}

    if loss_json_file_path.exists() and loss_json_file_path.is_file():
        with open(loss_json_file_path, "r+") as loss_json_f:
            losses_per_epoch = json.load(loss_json_f)
            print(f"Loaded loss dictionary at {loss_json_file_path}")
    else:
        losses_per_epoch = {"train": [], "val": [], "test": []}

    starting_epoch = 0

    checkpoint_path = Path(checkpoint_prefix) 

    checkpoint_path.mkdir(exist_ok=True)

    # Check for the latest weights
    checkpoints = [checkpoint.name for checkpoint in checkpoint_path.glob("*.pth") if checkpoint.is_file()]

    if len(checkpoints) > 0:
        epochs = [int(s.replace(".pth", "")) for s in checkpoints]

        latest_state_dict_path = checkpoint_path / f"{max(epochs)}.pth"

        # Loading state dict
        model.load_state_dict(torch.load(latest_state_dict_path, map_location=device))
        print(f"Loaded weights generated at epoch {max(epochs)}")

        starting_epoch = max(epochs)

    best_accuracy = 0
    best_accuracy_val = 0
    best_epoch = 0
    
    predicted_labels = [] 
    correct_labels = []

    print(f"Training starting at epoch {starting_epoch + 1}")
    for epoch in range(starting_epoch + 1, n_epochs + starting_epoch + 1):
        # Initialize loss/accuracy variables
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}
        
        # Adjust learning rate for SGD
        """
        if OPTIMIZER == "SGD":
            lr = LEARNING_RATE * (LR_DECAY ** (epoch // LEARNING_RATE_DECAY_EVERY))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        """
        
        # Process each split
        for split in ("train", "val", "test"):
            # Set network mode
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)
            
            # Process all split batches
            for i, (_, input, target) in enumerate(loaders[split]):

                # Move model to device
                model = model.to(device)
                
                # Move tensors to device
                input = input.to(device)
                target = target.to(device)
                
                if debug:
                    print(input.device)

                # Forward
                output = model(input.squeeze())

                # Compute loss
                loss = loss_function(output, target)
                losses[split] += loss.item()
                
                # Compute accuracy
                _, pred = output.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracy = correct / input.data.size(0)   
                accuracies[split] += accuracy
                counts[split] += 1
                
                # Backward and optimize
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Print info at the end of the epoch
        if accuracies["val"] / counts["val"] >= best_accuracy_val:
            best_accuracy_val = accuracies["val"] / counts["val"]
            best_accuracy = accuracies["test"] / counts["test"]
            best_epoch = epoch

        train_loss = losses["train"] / counts["train"]
        train_accuracy = accuracies["train"] / counts["train"]
        validation_loss = losses["val"] / counts["val"]
        validation_accuracy = accuracies["val"] / counts["val"]
        test_loss = losses["test"] / counts["test"]
        test_accuracy = accuracies["test"] / counts["test"]

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print("INFO")
        print(f"- Model: {model.__class__.__name__}")
        print("STATS")
        print(f"- Training: Loss {train_loss:.4f}, Accuracy {train_accuracy:.4f} " 
            f"- Validation: Loss {validation_loss:.4f}, Accuracy {validation_accuracy:.4f} "
            f"- Test: Loss {test_loss:.4f}, Accuracy {test_accuracy:.4f}")
        print(f"Best Test Accuracy at maximum Validation Accuracy (validation_accuracy = {best_accuracy_val:.4f}) is {best_accuracy:.4f} at epoch {best_epoch}\n")

        losses_per_epoch["train"].append(train_loss)
        losses_per_epoch["val"].append(validation_loss)
        losses_per_epoch["test"].append(test_loss)
        accuracies_per_epoch["train"].append(train_accuracy)
        accuracies_per_epoch["val"].append(validation_accuracy)
        accuracies_per_epoch["test"].append(test_accuracy)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_path / f"{epoch}.pth")
            
            with open(accuracy_json_file_path, "w+") as accuracy_json_f:
                json.dump(accuracies_per_epoch, accuracy_json_f)
            
            with open(loss_json_file_path, "w+") as loss_json_f:
                json.dump(losses_per_epoch, loss_json_f)

    # At the end of training, save
    torch.save(model.state_dict(), checkpoint_path / f"{epoch}.pth")

    # Stop Wandb
    wandb.finish()