import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
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

        data = [np.load(file) for file in sample]
        data = np.stack(data)

        x_sequence = data[:self.pre_seq_length]
        y_sequence = data[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length]

        x_sequence = torch.tensor(x_sequence).float()
        y_sequence = torch.tensor(y_sequence).float()

        x_sequence = x_sequence.unsqueeze(1)
        y_sequence = y_sequence.unsqueeze(1)

        return x_sequence, y_sequence
