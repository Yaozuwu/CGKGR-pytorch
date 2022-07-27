import scipy.sparse as sp
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, _utils
import os

class RecoDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset


    def __getitem__(self, idx):
        user_indices = self.dataset[idx, 0]
        item_indices = self.dataset[idx, 1]
        labels = self.dataset[idx, 2]
        return user_indices, item_indices, labels

    def __len__(self):
        return len(self.dataset)


