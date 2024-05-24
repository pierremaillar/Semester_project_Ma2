import numpy as np
import torch
import pandas as pd


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
    
    # Shuffle the dataset
    dataset_values = dataset.sample(frac=1).reset_index(drop=True)
    
    # Split the dataset into train, test, and validation sets
    trainset = dataset_values.iloc[:num_train]
    testset = dataset_values.iloc[num_train:num_train+num_test]
    valset = dataset_values.iloc[num_train+num_test:]


    # Convert sets to PyTorch tensors
    train_set = torch.tensor(trainset.iloc[:, 1:].values.astype(bool), dtype=torch.bool)
    test_set = torch.tensor(testset.iloc[:, 1:].values.astype(bool), dtype=torch.bool)
    val_set = torch.tensor(valset.iloc[:, 1:].values.astype(bool), dtype=torch.bool)

    # Convert labels to PyTorch tensors
    train_set_label = torch.tensor(trainset.iloc[:, 0].values.astype(int), dtype=torch.int8)
    test_set_label = torch.tensor(testset.iloc[:, 0].values.astype(int), dtype=torch.int8)
    val_set_label = torch.tensor(valset.iloc[:, 0].values.astype(int), dtype=torch.int8)

    return train_set, train_set_label, test_set, test_set_label, val_set, val_set_label
