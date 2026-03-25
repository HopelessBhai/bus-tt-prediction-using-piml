"""PyTorch Dataset wrappers for sequential and tabular inputs."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SeqCtxDataset(Dataset):
    """For PhyLSTM / LSTM: returns (x_seq, x_ctx, y)."""

    def __init__(self, x_seq: np.ndarray, x_ctx: np.ndarray, y: np.ndarray):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.x_ctx = torch.tensor(x_ctx, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_ctx[idx], self.y[idx]


class TabularDataset(Dataset):
    """For ANN / PINN: returns (x, y)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
