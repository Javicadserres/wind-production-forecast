import torch.nn as nn


class QuantileNet(nn.Module):
    """
    Neural network.
    """
    def __init__(self):
        super(QuantileNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(126, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        return self.layers(x)