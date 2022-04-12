"""Class to read and process the .grib files"""
import xarray as xr
import numpy as np
from conventions import GridNames as gn

class GribData:
    """
    Reads .grib file and converts the data into a dataframe.
    Parameters
    ----------
    file_path : str
        Path where the .grib is located.
    data_type : str
        Type of data, it can be 'an', 'es' or 'em' for 
        reanalysis and ensemble respectively.

    Atributes
    ---------
    self.data : pandas.DataFrame
    self.grib_data : xarray

    """
    def __init__(self, file_path, data_type):
        """
        """
        _types = ['an', 'em', 'es']
        if data_type in _types:
            self.data_type = data_type
        else:
            _error = f"data_type '{data_type}' not in {_types}."
            raise KeyError(_error)
            
        self.file_path = file_path
        self.data = None
        
    def convert(self):
        """
        Transforms the grib into a pandas.dataframe with a set
        of transformations to facilitate the analysis.
        """
        self.data = self._read_grib()
        self.data = self._process_data()
        
        return self.data
        
    def _read_grib(self):
        """
        Reads .grib data.
        """
        self.grib_data = xr.open_dataset(
            self.file_path, 
            engine='cfgrib', 
            backend_kwargs={'filter_by_keys':{'dataType': self.data_type}}
        )
        
        return self.grib_data.to_dataframe()
    
    def _process_data(self):
        """
        Adds the id corresponding to an exact longitude and 
        latitude and calculates the norm for the 10m and 100m.
        
        Returns
        -------
        data : pandas.DataFrame
        """
        # create an id for latitude and longitude
        self.data = self.data.reset_index(level=[gn.LAT, gn.LON])
        self.data[gn.LATLON] = (
            self.data[gn.LAT].astype(str) 
            + self.data[gn.LON].astype(str)
        )

        self.data = self.data[gn.COLS]
        
        if self.data_type in ['an', 'em']:
            self.data = self.get_norm(self.data)
        
        return self.data

    def get_norm(self, data):
        """
        Calculates the norm
        """
        # create the norm for 10m and 100m
        data[gn.NORM10] = (
            np.sqrt(data[gn.U10].pow(2)  + data[gn.V10].pow(2))
        )
        data[gn.NORM100] = (
            np.sqrt(data[gn.U100].pow(2) + data[gn.V100].pow(2))
        )
        
        return data