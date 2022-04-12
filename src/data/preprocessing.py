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
        to be used in the model. Note: The label needs 
        to be in the last column.
    """
    def __init__(self, data):
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
        
    def data_to_loader(self, batch_size=64):
        """
        Converts data to DataLoader pytorch module.
        
        Parameters
        ----------
        batch_size: int
        """
        # train, val and test
        train_loader = self._get_loader(self.data_train)
        val_loader = self._get_loader(self.data_val)
        test_loader = self._get_loader(self.data_test)
        
        return train_loader, val_loader, test_loader
        
    def _get_loader(self, data):
        """
        Get loader.
        """
        # inputs and target
        inputs = data[:, :-1]
        target = data[:, -1:]
        
        # loader
        dataset = Dataset(inputs, target)
        loader = DataLoader(dataset=dataset, batch_size=64)
        
        return loader
