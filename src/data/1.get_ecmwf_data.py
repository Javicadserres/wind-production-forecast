"""Gets data from ECMWF"""
import cdsapi
from conventions import ECMWF


def get_data(year, file_name):
    """
    API request for ECMWF data.

    Parameters
    ----------
    year : int
    file_name : str 
    """
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': ECMWF.PRODUCT_TYPE,
            'format': 'grib',
            'variable': ECMWF.VARIABLES,
            'year': year,
            'month': ECMWF.MONTHS,
            'day': ECMWF.DAYS,
            'time': ECMWF.HOURS,
            'area': ECMWF.AREA,
        },
        file_name
    )


if __name__== "__main__":
    start, end = [2015, 2022]
 
    for year in range(start, end):
        file_name = f'{year}_datos.grib'
        get_data(year=year, file_name=file_name)