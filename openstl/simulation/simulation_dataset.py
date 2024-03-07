import torch
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
    def __init__(self, X, Y):
        super(SimulationDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels