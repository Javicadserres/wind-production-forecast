import pandas as pd

from model.scores import AverageCoverageError, IntervalScorePaper
from model.losses import SmoothPinballLoss

def get_scores(y_pred, target, quantiles):
    """
    Get prediction scores.
    
    Parameters
    ----------
    y_pred: torch.tensor
    target: torch.tensor
    quantiles: torch.tensor
    
    Returns
    -------
    scores: pd.DataFrame
    """
    # final validation loss
    criterion = SmoothPinballLoss(quantiles)
    qs = criterion(y_pred, target)

    # interval score
    iscore = IntervalScorePaper(quantiles)
    interval_score, sharpness = iscore.forward(y_pred, target)

    # average coverage error
    acerror = AverageCoverageError(quantiles)
    ace = acerror.forward(y_pred, target)
    
    scores = pd.Series(
        [qs.item(), interval_score, sharpness, ace],
        index=['QS', 'IS', 'Sharpnees', 'ACE'],
    )
    return scores