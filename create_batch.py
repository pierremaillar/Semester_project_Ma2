import numpy as np
import pandas as pd
import torch

def split_dataset(dataset, train_ratio, test_ratio, val_ratio):
    """
    Split the dataset into train, test, and validation sets.

    Args:
        dataset (pd.DataFrame): Input data.
        train_ratio (float): Percentage of the dataset to use for training.
        test_ratio (float): Percentage of the dataset to use for testing.
        val_ratio (float): Percentage of the dataset to use for validation.

    Returns:
        all are torch.Tensor:
        Training set, training set labels, test set, test set labels, validation set, validation set labels.
    """
    # Calculate the number of samples for each set
    num_samples = len(dataset)
    num_train = int(num_samples * train_ratio)
    num_test = int(num_samples * test_ratio)
    num_val = num_samples - num_train - num_test
    
    # Shuffle the dataset
    shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    # Split the dataset into train, test, and validation sets
    trainset = shuffled_dataset[:num_train]
    testset = shuffled_dataset[num_train:num_train+num_test]
    valset = shuffled_dataset[num_train+num_test:]
    
    # Convert sets to PyTorch tensors
    train_set = torch.tensor(trainset.iloc[:, 1:].values, dtype=torch.float32)
    test_set = torch.tensor(testset.iloc[:, 1:].values, dtype=torch.float32)
    val_set = torch.tensor(valset.iloc[:, 1:].values, dtype=torch.float32)
    
    # Convert labels to PyTorch tensors
    train_set_label = torch.tensor(pd.to_numeric(trainset.iloc[:, 0].values), dtype=torch.long)
    test_set_label = torch.tensor(pd.to_numeric(testset.iloc[:, 0].values), dtype=torch.long)
    val_set_label = torch.tensor(pd.to_numeric(valset.iloc[:, 0].values), dtype=torch.long)


    return train_set, train_set_label, test_set, test_set_label, val_set, val_set_label