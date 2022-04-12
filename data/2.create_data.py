"""Reads .grib data and creates a dataframe to use in the model."""
import pandas as pd

from grib_reader import GribData
from pathlib import Path

PATH = Path.cwd()
PATH_DATA = PATH / 'files'

# READ THE .GRIB FILES FOR ENSEMBLEM MEAN (EM) AND SPREAD SPREAD (EM)
data_list_mean = []
data_list_spread = []

for year in range(2015, 2019):
    _file_path = f'{year}_datos.grib'
    file_path = PATH_DATA / _file_path
    
    # mean data
    gribdata = GribData(file_path=file_path, data_type='em')
    _data = gribdata.convert()
    data_list_mean.append(_data)
    
    # spread data
    gribdata = GribData(file_path=file_path, data_type='es')
    _data = gribdata.convert()
    data_list_spread.append(_data)

# CONCATENATE ALL THE YEARS FORM EM AND ES
data_mean = pd.concat(data_list_mean)
data_mean = data_mean.reset_index().pivot(
    index='time', columns='latitude_longitude'
)
data_mean.columns = data_mean.columns.map('_mean_'.join)

data_spread = pd.concat(data_list_spread)
data_spread = data_spread.reset_index().pivot(
    index='time', columns='latitude_longitude'
)
data_spread.columns = data_spread.columns.map('_spread_'.join)

# MERGE EM AND ES
data_total = pd.merge(
    data_mean, data_spread, left_index=True, right_index=True
)
data_total['day'] = pd.to_datetime(data_total.index).day
data_total['month'] = pd.to_datetime(data_total.index).month
data_total['hour'] = pd.to_datetime(data_total.index).hour

# GET ENERGY PRODUCTION OF CADIZ
prod_file = PATH_DATA / 'energyproductioncadiz.csv'
output = pd.read_csv(prod_file, sep=';', index_col='datetime')
output.columns = ['production']
# we need to shift the data because the output is in UTC+1
output.index = output.index.astype('datetime64[ns]')
# we start from 2015
output = output.loc['2015':]
# since ensemble only has data every 3 hours, we aggregate
#  the production 3 hours
output = output.groupby(pd.Grouper(freq='3H')).sum()

# MERGE THE VARIABLES WITH THE LABEL AND SAVE THE DATA
final_data = pd.merge(
    data_total, output, left_index=True, right_index=True
)
final_data.to_csv(PATH_DATA / 'data.csv')