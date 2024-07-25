import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(sample):
    data = []
    for file in sample:
        if isinstance(file, str):
            if file.endswith('.npy'):
                loaded_file = np.load(file)
            elif file.endswith('.npz'):
                loaded_file = np.load(file)['data']
            elif file.endswith('.pkl'):
                with open(file, 'rb') as f:
                    loaded_file = pickle.load(f)
            else:
                raise ValueError("Unsupported file format")
        else:
            loaded_file = np.array(file)
        data.append(loaded_file)
    return np.stack(data)


class SimVP_Dataset(Dataset):
    def __init__(self, data, pre_seq_length, aft_seq_length):
        self.data = data
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.data['samples'])

    def __getitem__(self, idx):
        sample = self.data['samples'][idx]

        data = load_data(sample)

        x_sequence = data[:self.pre_seq_length]
        y_sequence = data[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length]

        x_sequence = torch.tensor(x_sequence).float()
        y_sequence = torch.tensor(y_sequence).float()

        x_sequence = x_sequence.unsqueeze(1)
        y_sequence = y_sequence.unsqueeze(1)

        return x_sequence, y_sequence
