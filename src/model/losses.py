import torch 
from torch import nn

class QuantileLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensort
    """
    def __init__(self, quantiles):
        super(QuantileLoss,self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        
    def forward(self,pred ,target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        upper =  self.quantiles * error
        lower = (self.quantiles - 1) * error 

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss