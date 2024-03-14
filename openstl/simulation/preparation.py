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


def train_val_test_split_data(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the dataset into training, validation, and test sets.

    Arguments:
    - dataset: The dataset to split.
    - train_ratio: Proportion of the dataset to include in the training set.
    - val_ratio: Proportion of the dataset to include in the validation set.

    Returns:
    - A dictionary with 'train', 'val', and 'test' keys, each of which contains their respective split of the data.
    """
    np.random.shuffle(dataset)
    train_size = round(len(dataset) * train_ratio)
    val_size = round(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size

    return {
        'train': dataset[:train_size],
        'val': dataset[train_size:train_size + val_size],
        'test': dataset[-test_size:]
    }


def X_Y_split_data(dataset, pre_seq_length):
    """
    Saves the split data into a numpy pickle file in input and output pairs (X and Y respectively).

    Arguments:
    - dataset: A dictionary with 'train', 'val', and 'test' keys, each of which contains their respective split of the data.
    - pre_seq_length: The length of the sequence that is the input for the model.
    - file_path: The path to the file where the data will be saved.
    """

    data = {}
    if type(dataset) is dict:
        for split in ['train', 'val', 'test']:
            data['X_' + split] = dataset[split][:, :pre_seq_length, ...]
            data['Y_' + split] = dataset[split][:, pre_seq_length:, ...]
    else:
        data['X'] = dataset[:, :pre_seq_length, ...]
        data['Y'] = dataset[:, pre_seq_length:, ...]

    return data


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

    print(new_samples)

    return new_samples