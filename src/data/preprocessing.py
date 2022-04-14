import torch 

from torch.utils.data import DataLoader
from .dataset import Dataset


class FrameTorch(object):
    """
    Class to facility the dataframe to torch for the
    training.
    
    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing the inputs and the label
        to be used in the model. 
    target_name: str
        Name of the target in the dataframe.

    Attributes
    ----------
    data_train: torch.tensor
    data_val: torch.tensor
    data_test: torch.tensor

    """
    def __init__(self, data, target_name='production'):
        # move target to the end
        _target = data.pop(target_name)
        data[target_name] = _target

        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.n = len(data)
    
    def split_data(self, train_size=0.8, val_size=0.1, test_size=0.1):
        """
        Split the data into train, val and test.
        
        Parameters
        ----------
        train_size: int
        val_size: int
        test_size: int
        """
        # split sizes
        n_train = int(self.n * train_size)
        n_val = int(self.n * val_size)
        n_test = self.n - (n_train + n_val)
        
        # datasets
        self.data_train = self.data[:n_train, :]
        self.data_val = self.data[n_train:(n_train + n_val), :]
        self.data_test = self.data[-n_test:, :]
        
    def scale_data(self, method):
        """
        Scales the data with a given method.
        
        Parameters
        ----------
        method: sklearn.preprocessing
        """
        self.scaler = method()
        self.scaler.fit(self.data_train)

        self.data_train = self.scaler.transform(self.data_train)
        self.data_val = self.scaler.transform(self.data_val)   
        self.data_test = self.scaler.transform(self.data_test)
    
    def data_to_loader(self, batch_size=64, slide=None):
        """
        Converts data to DataLoader pytorch module.
        
        Parameters
        ----------
        batch_size: int
        slide: int, default=None
            This parameter is used for recurrent networks
            and set the number of windows you want to look
            back.
        """
        # train, val and test
        train_loader = self._get_loader(self.data_train, slide)
        val_loader = self._get_loader(self.data_val, slide)
        test_loader = self._get_loader(self.data_test, slide)
        
        return train_loader, val_loader, test_loader

    @staticmethod
    def slide_data(inputs, outputs, slide):
        """
        Slide data for recurrent networks.

        Parameters
        ----------
        inputs: torch.tensor, numpy.array or list
        slide: int
            This parameter is used for recurrent networks
            and set the number of windows you want to look
            back.

        Returns
        -------
        slide_inputs: list
        outputs: list
            Corrected outputs
        """
        slide_inputs = []
        steps = range(inputs.shape[0] - slide)

        for step in steps:
            slide_input = inputs[step : step+slide]
            slide_inputs.append(slide_input.tolist())
        
        outputs = outputs[slide:].tolist()

        return slide_inputs, outputs

    def _get_loader(self, data, slide=None):
        """
        Get loader.
        """
        # inputs and target
        inputs = data[:, :-1]
        target = data[:, -1:]

        if slide is not None:
            inputs, target = self.slide_data(inputs, target, slide)
        
        # loader
        dataset = Dataset(inputs, target)
        loader = DataLoader(dataset=dataset, batch_size=64)
        
        return loader
