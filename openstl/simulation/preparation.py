import os
import numpy as np


def normalize_data_min_max(dataset, vmin, vmax):
    """
    Normalizes the dataset using min-max scaling to a range of [0, 1].

    Arguments:
    - dataset: The dataset to normalize.
    - min_val: The minimum value of the input data.
    - max_val: The maximum value of the input data.

    Returns:
    - The normalized data scaled to range [0, 1].
    """

    return (dataset - vmin) / (vmax - vmin)

def train_val_test_split_files(data_folder, train_ratio, val_ratio, num_samples=None):
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]

    if num_samples is not None:
        file_paths = np.random.choice(file_paths, min(num_samples, len(file_paths)), replace=False).tolist()

    np.random.shuffle(file_paths)

    train_size = int(len(file_paths) * train_ratio)
    val_size = int(len(file_paths) * val_ratio)

    train_files = file_paths[:train_size] if train_ratio > 0 else []
    val_files = file_paths[train_size:train_size + val_size] if val_ratio > 0 else []
    test_files = file_paths[train_size + val_size:] if train_ratio + val_ratio < 1 else []

    splits = {}
    if train_files:
        splits['train'] = train_files
    if val_files:
        splits['val'] = val_files
    if test_files:
        splits['test'] = test_files

    return splits

def random_samples_split_data(dataset, num_random_samples, total_length):
    """
    Splits each sample in the dataset into multiple samples.

    Arguments:
    - dataset: The dataset to split.
    - num_random_samples: The number of random samples to create from each sample.
    - total_length: The length of the new samples.

    Returns:
    - A new dataset with each sample split into multiple samples.
    """

    if type(dataset) is dict:
        new_dataset = {}
        for split in ['train', 'val', 'test']:
            new_dataset[split] = split_sample(dataset[split], num_random_samples, total_length)
    else:
        new_dataset = split_sample(dataset, num_random_samples, total_length)

    return new_dataset

def split_sample(sample, num_random_samples, total_length):
    new_samples = np.zeros((num_random_samples * sample.shape[0], total_length, *sample.shape[2:]), dtype=sample.dtype)

    for i in range(sample.shape[0]):
        for j in range(num_random_samples):
            start = np.random.randint(0, sample.shape[1] - total_length)
            new_samples[i * num_random_samples + j] = sample[i, start:start + total_length, ...]

    return new_samples