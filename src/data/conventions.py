class ECMWF:
    PRODUCT_TYPE = ['ensemble_mean', 'ensemble_spread'] #'reanalysis'
    VARIABLES = [
        '100m_u_component_of_wind', 
        '100m_v_component_of_wind', 
        '10m_u_component_of_wind',
        '10m_v_component_of_wind', 
        '2m_temperature', 
        'surface_pressure'
    ]
    MONTHS = [
        '01', '02', '03', '04', '05', '06',
        '07', '08', '09', '10', '11', '12',
    ]
    DAYS = [
        '01', '02', '03', '04', '05', '06',
        '07', '08', '09', '10', '11', '12',
        '13', '14', '15', '16', '17', '18',
        '19', '20', '21', '22', '23', '24',
        '25', '26', '27', '28', '29', '30',
        '31',
    ]
    HOURS = [
        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
    ]
    AREA = [37.1, -6.45, 36,-5]


class GridNames: 
    LAT = 'latitude'
    LON = 'longitude'
    LATLON = 'latitude_longitude'
    NORM10 = 'norm_10'
    NORM100 = 'norm_100'
    U10 = 'u10'
    V10 = 'v10'
    U100 = 'u100'
    V100 = 'v100'
    
    COLS = [
        'latitude_longitude', 
        'u100', 
        'v100', 
        'u10', 
        'v10', 
        't2m', 
        'sp', 
    ]
    