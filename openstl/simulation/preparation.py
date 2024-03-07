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
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
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