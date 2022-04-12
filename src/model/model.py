import torch.nn as nn


class QuantileNet(nn.Module):
    """
    Neural network.
    """
    def __init__(self, n_inputs=129, n_outputs=2):
        super(QuantileNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, n_outputs)
        )

    def forward(self, x):
        return self.layers(x)

class LinearNet(nn.Module):
    """
    Linear neural network to test as a 
    benchmark the Quantile Network.
    """
    def __init__(self, n_inputs=129, n_outputs=2):
        super(LinearNet, self).__init__()
        self.layers = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        return self.layers(x)