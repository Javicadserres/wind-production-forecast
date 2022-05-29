import torch.nn as nn


class QuantileNet(nn.Module):
    """
    MLP neural network.

    Parameters
    ----------
    n_inputs: int
    n_outputs: int
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
        """
        Forward propagation.

        Parameters
        ----------
        x: torch.tensor
        """
        return self.layers(x)

class LinearNet(nn.Module):
    """
    Linear model as a benchmark for other complex networks.

    Parameters
    ----------
    n_inputs: int
    n_outputs: int
    """
    def __init__(self, n_inputs=129, n_outputs=2):
        super(LinearNet, self).__init__()
        self.layers = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x: torch.tensor
        """
        return self.layers(x)

class LSTM(nn.Module):
    """
    NN model with a LSTM layer.

    Parameters
    ----------
    input_size: int
        This is the number of features.
    out_size: int
        Number of targets.
    hidden_size: int, default=1
    n_layers: int, default=1
    """
    def __init__(self, input_size, out_size, hidden_size=50, n_layers=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True   
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x: torch.tensor
        """
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.lin(out)

        return out

class AttentionLSTM(nn.Module):
    """
    LSTM using a multihead attention mechanism.

    Parameters
    ----------
    embed_dim: int
    out_size: int
    hidden_size: int, default=20    
    n_layers: int, default=2
    """
    def __init__(self, embed_dim, out_size, hidden_size=20, n_layers=2):
        super(AttentionLSTM, self).__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out, weights = self.att(X, X, X)
        out, (h, c) = self.lstm(out)
        out = out[:, -1, :]
        out = self.lin(out)
        return out