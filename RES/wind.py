"""
Wind resource analysis module for renewable energy source modeling.

This module provides functions for processing wind data from multiple sources:
- Global Wind Atlas (GWA) data for wind speed scaling
- ERA5 reanalysis data for time series generation
- Geographic coordinate transformations and wind speed interpolation

Key functionality includes:
- Extracting wind speeds at specific coordinates from raster data
- Scaling ERA5 wind speeds using Global Wind Atlas reference data
- Converting between geographic and grid coordinates for wind data processing

Dependencies:
    - atlite: For renewable energy data processing
    - numpy: For numerical computations
    - pandas: For data manipulation

Example:
    import wind
    
    # Get wind speeds at asset locations
    wind_coords = wind.get_wind_coords(assets_df, wind_atlas_data, geojson_data)
    
    # Scale wind speeds using GWA data
    scaled_wind = wind.scale_wind(asset_row, era5_wind_data)
"""

import numpy as np
import pandas as pd

def get_speed(row, xaxis, yaxis, data):
    """
    Function to get wind speed at a specific latitude and longitude from the Global Wind Atlas data.
    """
    #Get indices of the nearest pixels
    xIdx = np.searchsorted(xaxis, row['longitude'], side='left')
    yIdx = len(yaxis) - np.searchsorted(yaxis, row['latitude'], side='left', sorter=np.arange(len(yaxis)-1, -1, -1))

    return data[yIdx][xIdx] #Return the wind speed at the indices

#Generate a data frame that matches wind speeds from Global Wind Atlas to latitude/longitude values for scaling the cutout speeds

def get_wind_coords(assets:pd.DataFrame, 
                    wind_atlas, 
                    wind_geojson)-> pd.DataFrame:

    """
    Paramters:
        assets (pd.DataFrame): Data frame containing wind asset locations.
        wind_atlas: The Global Wind Atlas wind speed data from the .tif file.
        wind_geojson: The Global Wind Atlas geojson data which creates the shape for the region

    Returns:
        _type_: _description_

    """
    #Store longitude and latitude values in a list for processing.
    longitudes = [wind_geojson[i][0][j][0] for i in range(len(wind_geojson)) for j in range(len(wind_geojson[i][0]))] #[lon, lat], choose index 0
    latitudes = [wind_geojson[i][0][j][1] for i in range(len(wind_geojson)) for j in range(len(wind_geojson[i][0]))] #[lon, lat], choose index 1

    #Get latitude and longitude values to construct a bounding box for the wind speed data in latitude longitude format
    west = min(longitudes); north = max(latitudes) #Upper left corner
    east = max(longitudes); south = min(latitudes) #Lower right corner

    #Get x and y axis as linearly spaced longitudes and latitudes from the values calculated above
    xaxis = np.linspace(west, east, wind_atlas.shape[1])
    yaxis = np.linspace(north, south, wind_atlas.shape[0])

    #Match speeds of turbines to Global Wind Atlas
    wind_coords:pd.DataFrame = assets.apply(lambda x: get_speed(x, xaxis, yaxis, wind_atlas), axis=1)

    return wind_coords


def get_XY(row, wnd) -> list:
    """
    Function to get INDEX values of the square in the ERA5 data array is
    Used in generate_wind_ts()
    Parameters:
        row (pd.Series): A row from the wind assets data frame containing 'x' and 'y' coordinates.
        wnd: The ERA5 wind data cutout object. 
    Returns:
        list: A list containing the x and y indices corresponding to the coordinates in the ERA5 data.
    """
    x = 0
    y = 0
    for i in range(wnd.x.size):
        if row['x'] == wnd.x.values[i]:
            x = i
            break
    
    for j in range(wnd.y.size):
        if row['y'] == wnd.y.values[j]:
            y = j
            break

    return [x, y]


def scale_wind(row, wnd):
    """
    Function to scale the wind speeds on the ERA5 data array
    Used in generate_wind_ts()
    
    Parameters:     
        row: Some row in the wind_assets.csv data frame
        wind: cutout.data.wnd100m
    NOTE: 
        Modications made here 2023-10-25, since the flag parameter should not be used to dictate whether scaling occurs. Now the GWA scaling is used by default.
    """ 

    wind_at_location = wnd.sel(x=row['x'], y=row['y']).values
    scaled = wind_at_location * row['GWA wind speed'] / np.mean(wind_at_location)
    return scaled
