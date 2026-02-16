import os
import numpy as np
import torch
from torch.utils.data import DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def DataOutPath(main_path, spectral_type, noise, num_poles, sID, s, xi, tag):
    """Construct output path based on parameters."""
    if noise:
        data_path = f"{main_path}/Data/datasets/half-filled-{spectral_type}/inputs-1/Gbins_s{s:.0e}_xi{xi}.csv"
        output_path = f"{main_path}/VAE_Library/{tag}_out_{spectral_type}_numpoles{num_poles}_s{s:.0e}_xi{xi}-{sID}"
        
        print("" * 50)
        print(f"Using dataset with noise")
        print("" * 50)
        print(f"Special ID: {sID} | S: {s:.0e} | XI: {xi} | NUM_POLES: {num_poles} | SPECTRAL_TYPE: {spectral_type}")
        
    else:
        data_path = f"{main_path}/Data/datasets/half-filled-{spectral_type}/Gbins_boot_means_no_noise.csv"
        output_path = f"{main_path}/VAE_Library/{tag}_out_{spectral_type}_numpoles{num_poles}_no_noise-{sID}"
        
        print("" * 50)
        print(f"Using dataset with no noise")
        print("" * 50)
        print(f"Special ID: {sID} | NUM_POLES: {num_poles} | SPECTRAL_TYPE: {spectral_type}")
        
    return data_path, output_path

def MakeOutPath(output_path):
    """Create output directories if they don't exist."""
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/model", exist_ok=True)
    os.makedirs(f"{output_path}/losses", exist_ok=True)
    os.makedirs(f"{output_path}/poles_residues", exist_ok=True)
    os.makedirs(f"{output_path}/Greens", exist_ok=True)

# Function to calculate standard deviation of dataset
def STD(dataset):
    """Calculate the standard deviation of the dataset."""
    data_array = dataset.data.numpy()
    std = np.std(data_array, axis=0, ddof=1) # sample standard deviation
    return torch.tensor(std, dtype=torch.float32).to(DEVICE)

# Function for train-test split
def TrainTestSplit(dataset, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    assert train_size + val_size + test_size == len(dataset), "Sizes do not sum up to dataset length"
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, test_dataset

def LoadData(train_dataset, val_dataset, test_dataset, batch_size=32, shuffle=True):
    """Create a DataLoader for the given dataset."""
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader

def PrintDataSize(dataset, train_dataset, val_dataset, test_dataset, train_loader):
    """Print the sizes of the datasets."""
    print(f"\nFull Dataset shape: {dataset.data.shape}")
    print(f"Train Dataset shape: {train_dataset.dataset.data[train_dataset.indices].shape}")
    print(f"Val Dataset shape: {val_dataset.dataset.data[val_dataset.indices].shape}")
    print(f"Test Dataset shape: {test_dataset.dataset.data[test_dataset.indices].shape}")
    print(f"Number of batches for training: {len(train_loader)}\n")