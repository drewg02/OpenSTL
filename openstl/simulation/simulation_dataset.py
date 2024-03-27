import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, file_paths, pre_seq_length, aft_seq_length):
        self.file_paths = file_paths
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)

        x_sequence = data[:self.pre_seq_length]
        y_sequence = data[self.pre_seq_length:self.pre_seq_length + self.aft_seq_length]

        x_sequence = torch.tensor(x_sequence).float()
        y_sequence = torch.tensor(y_sequence).float()

        # data is currently [10, 64, 64] (10 time steps, 64x64 image) but we need to add a channel dimension
        x_sequence = x_sequence.unsqueeze(1)
        y_sequence = y_sequence.unsqueeze(1)

        return x_sequence, y_sequence