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


def train_val_test_split_files(data, train_ratio, val_ratio, num_samples=None):
    train_ratio = round(train_ratio, 3)
    val_ratio = round(val_ratio, 3)

    if num_samples is not None:
        data = np.random.choice(data, min(num_samples, len(data)), replace=False).tolist()

    np.random.shuffle(data)

    train_size = round(len(data) * train_ratio)
    val_size = round(len(data) * val_ratio)

    train_data = data[:train_size] if train_ratio > 0 else []
    val_data = data[train_size:train_size + val_size] if val_ratio > 0 else []
    test_data = data[train_size + val_size:] if train_ratio + val_ratio < 1 else []

    splits = {}
    if train_data:
        splits['train'] = {
            'ratio': train_ratio,
            'samples': train_data
        }
    if val_data:
        splits['validation'] = {
            'ratio': val_ratio,
            'samples': val_data
        }
    if test_data:
        splits['test'] = {
            'ratio': round(1 - train_ratio - val_ratio, 3),
            'samples': test_data
        }

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


def normalize_samples(datafolder, vmin, vmax):
    for root, dirs, files in os.walk(datafolder):
        for file in files:
            if file.endswith(".npy"):
                data = np.load(os.path.join(root, file))
                norm_data = normalize_data_min_max(data, vmin, vmax)
                np.save(os.path.join(root, file), norm_data)


def load_files(datafolder, num_samples, sample_start_index, total_length):
    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]
    folders = np.random.choice(folders, min(num_samples, len(folders)), replace=False)

    data = []
    for i, unique_id in enumerate(folders):
        files = [f for f in os.listdir(os.path.join(datafolder, unique_id)) if f.endswith('.npy')]
        file_count = len(files)

        if total_length > 0 and file_count < total_length:
            print(f"Skipping {unique_id} due to insufficient data.")
            continue

        start_index = sample_start_index
        if start_index == -1:
            start_index = np.random.randint(0, file_count - total_length)

        final_files = []
        for j in range(start_index, start_index + total_length):
            final_files.append(os.path.join(datafolder, unique_id, f"{j}.npy"))

        data.append(final_files)

    return data
