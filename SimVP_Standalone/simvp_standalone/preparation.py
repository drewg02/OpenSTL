import os
import shutil

import numpy as np
from tqdm import tqdm


def normalize_data_min_max(dataset, vmin, vmax):
    """
    Normalizes the dataset using min-max scaling to a range of [0, 1].

    Arguments:
    - dataset: The dataset to normalize.
    - vmin: The minimum value of the input data.
    - vmax: The maximum value of the input data.

    Returns:
    - The normalized data scaled to range [0, 1].
    """
    if vmax == vmin:
        raise ValueError("vmax and vmin must be different values to avoid division by zero.")

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


def load_files(datafolder, num_samples, sample_start_index, total_length, verbose=True):
    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]
    folders = np.random.choice(folders, min(num_samples, len(folders)), replace=False)

    progress_iterator = folders
    if verbose:
        progress_iterator = tqdm(progress_iterator, desc="Loading samples")

    data = []
    for i, unique_id in enumerate(progress_iterator):
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


def check_samples(datafolders, verbose=True):
    """
    Checks samples

    Arguments:
    - datafolder: The list of datafolders to check.
    - verbose: If True, prints the progress of the generation.

    Returns: None
    """

    hashes = {}

    folders = []
    for datafolder in datafolders:
        folders += [os.path.join(datafolder, f) for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]

    progress_iterator = folders
    if verbose:
        progress_iterator = tqdm(progress_iterator, desc="Checking samples")

    for folder in progress_iterator:
        unique_id = folder.split('/')[-1]
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        if len(files) < 1:
            continue

        hash = unique_id.split('_')[0]
        if hashes.get(hash):
            first_file = hashes[hash]
            second_file = f'{folder}/0.npy'

            first_data = np.load(first_file)
            second_data = np.load(second_file)

            print(f"Hash collision: {hash}")
            print(f"First file: {first_file}")
            print(f"Second file: {second_file}")
            if not np.array_equal(first_data, second_data):
                print("Data is not equal.\n")
            else:
                print("Data is equal, cancelling.")
                return False

        hashes[hash] = f'{folder}/0.npy'

    return True


def copy_samples(datafolders, new_datafolder, move=True, verbose=True):
    """
    Copies samples from multiple datafolders to a new datafolder.

    Arguments:
    - datafolders: The list of datafolders to move samples from.
    - new_datafolder: The new datafolder to move the samples to.
    - verbose: If True, prints the progress of the generation.

    Returns: None
    """

    folders = []
    for datafolder in datafolders:
        folders += [os.path.join(datafolder, f) for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]

    progress_iterator = folders
    if verbose:
        progress_iterator = tqdm(progress_iterator, desc="Moving samples")

    for folder in progress_iterator:
        unique_id = folder.split('/')[-1]

        if move:
            shutil.move(folder, new_datafolder)
        else:
            dest_folder = os.path.join(new_datafolder, unique_id)

            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Copy entire directory
            shutil.copytree(folder, dest_folder, dirs_exist_ok=True)