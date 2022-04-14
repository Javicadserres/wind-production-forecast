import pandas as pd
import torch 


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the training.

    Parameters
    ----------
    input: list or numpy.array
    target: list or numpy.array
    """
    def __init__(self, inputs, target):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self,idx):
        return self.inputs[idx], self.target[idx]
