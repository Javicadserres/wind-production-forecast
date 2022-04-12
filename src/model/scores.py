import torch

class AverageCoverageError(object):
    """
    Computes the average coverage error.

    Parameters
    ----------
    quantiles : list
        List of quantiles to be used.
    """
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.n_pi = int(len(quantiles) / 2)
        self.pinc = self.get_pinc()

    def get_pinc(self):
        """
        Returns the prediction interval nominal confindence.

        Parameters
        ----------
        quantiles : torch.tensor

        Returns
        -------
        pinc: torch.tensor
        """
        reverse = torch.flip(self.quantiles, dims=[0])
        pinc = reverse[:self.n_pi]- self.quantiles[:self.n_pi]

        return pinc

    def get_picp(self):
        """
        Returns the prediction interval confidence probability

        Parameters
        ----------
        pred : torch.tensor
        target : torch.tensor

        Returns
        -------
        picp : torch tensor
        """
        higher = torch.ge(self.target, self.pred_low)
        lower = torch.le(self.target, self.pred_high)
        
        self.picp = (higher * lower * 1.).mean(dim=0)
            
        return self.picp
    
    def forward(self, pred, target):
        """
        Returns the average coverage error
        
        Parameters
        ----------
        pred: torch.tensor
        target: torch.tensor
        
        Returns
        -------
        ace: int
            Average coverage error
        """
        self.pred = pred
        self.target = target
        self.get_pred_bounds()
        
        # prediction interval confidence probability
        picp = self.get_picp()
        
        # average coverage error
        ace = 100 * torch.abs(picp - self.pinc).mean().item()
        
        return ace
    
    def get_pred_bounds(self):
        """
        Get prediction bounds high and low.
        
        Parameters
        ----------
        pred: torch.tensor

        """
        self.pred_low = self.pred[:, :self.n_pi]
        self.pred_high = torch.flip(
            self.pred[:, self.n_pi:], dims=[1]
        )

class IntervalScore(AverageCoverageError):
    """
    Computes the Interval Score.

    Parameters
    ----------
    quantiles : list
        List of quantiles to be used.
    """
    def __init__(self, quantiles):
        super(IntervalScore, self).__init__(quantiles)
    
    def forward(self, pred, target):
        """
        Returns the interval score.
        
        Parameters
        ----------
        pred: torch.tensor
        target: torch.tensor
        
        Returns
        -------
        interval_score: int
            Interval score
        sharpness_score: int
            Sharpness score
        """
        self.pred = pred
        self.target = target
        self.get_pred_bounds()
        
        # sharpness
        self.sharpness = (self.pred_high - self.pred_low)
        self.sharpness_score = self.sharpness.mean().item()
        
        # score and penalization
        score = -2 * (1 - self.pinc) * self.sharpness  
        penalize = self.get_penalization()

        # interval score
        self.interval_score = (score.mean() + penalize.mean()).item()

        return self.interval_score, self.sharpness_score
    
    def get_penalization(self):
        """
        Get penalization when the target is outside the
        predicted interval.
        
        Parameters
        ----------
        pred: torch.tensor
        target: torch.tensor
        
        Returns
        -------
        penalize: int
            Penalization part in the interval score.
        """      
        spread_low = self.pred_low - self.target
        spread_high = self.target - self.pred_high
        
        penalize = - 4 * torch.max(spread_low, spread_high).clip(0)
        
        return penalize

class IntervalScorePaper(AverageCoverageError):
    """
    Computes the Interval Score.

    Parameters
    ----------
    quantiles : list
        List of quantiles to be used.
    """
    def __init__(self, quantiles):
        super(IntervalScorePaper, self).__init__(quantiles)
    
    def forward(self, pred, target):
        """
        Returns the interval score.
        
        Parameters
        ----------
        pred: torch.tensor
        target: torch.tensor
        
        Returns
        -------
        interval_score: int
            Interval score
        sharpness_score: int
            Sharpness score
        """
        self.pred = pred
        self.target = target
        self.get_pred_bounds()
        
        # sharpness
        self.sharpness = (self.pred_high - self.pred_low)
        
        # score and penalization
        sharpness_score =  self.sharpness.mean().item()
        penalize = self.get_penalization()

        # interval score
        self.interval_score = (sharpness_score + penalize.mean()).item()

        return self.interval_score, sharpness_score
    
    def get_penalization(self):
        """
        Get penalization when the target is outside the
        predicted interval.
        
        Parameters
        ----------
        pred: torch.tensor
        target: torch.tensor
        
        Returns
        -------
        penalize: int
            Penalization part in the interval score.
        """      
        spread_low = self.pred_low - self.target
        spread_high = self.target - self.pred_high
        
        scale = 3 / (1 - self.pinc)
        penalize = scale * torch.max(spread_low, spread_high).clip(1)
        
        return penalize