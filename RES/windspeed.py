import geopandas as gpd
import atlite
from RES import utility
import numpy as np
import pandas as pd

def impute_ERA5_windspeed_to_Cells(
        cutout:atlite.Cutout, 
        province_grid_cells:gpd.GeoDataFrame)->gpd.GeoDataFrame:
        """
        For each grid cells, this function finds the yearly mean windspeed from the windspeed timeseries and imputes to the cell dataframe.
        """
        
        print(f"Calculating yearly mean windspeed and imputing to provincial Grid Cells named as 'windspeed_ERA5'")

        # Calculate yearly mean windspeed
        wnd_ymean_df = cutout.data.wnd100m.groupby('time.year').mean('time').to_dataframe(name='windspeed_ERA5').reset_index()

        # Create a GeoDataFrame for spatial join
        wnd_ymean_gdf = gpd.GeoDataFrame(wnd_ymean_df, geometry=gpd.points_from_xy(wnd_ymean_df['x'], wnd_ymean_df['y']))
        wnd_ymean_gdf.crs = province_grid_cells.crs

        # Perform spatial join and drop unnecessary columns
        province_grid_cells = (
            gpd.sjoin(province_grid_cells.rename(columns={'x': 'x_bc', 'y': 'y_bc'}), 
                      wnd_ymean_gdf, 
                      predicate='intersects')
            .drop(columns=['x_bc', 'y_bc', 'lon', 'lat','index_right','year'])
        )
        
        # Handle potential duplicate indices
        # Resetting index to ensure unique index after join
        province_grid_cells = province_grid_cells.reset_index(drop=True)
        province_grid_cells = province_grid_cells.drop_duplicates(subset=['geometry'])
        province_grid_cells=utility.assign_cell_id(province_grid_cells)

        return province_grid_cells

    
def scale_wind(row:pd.Series, 
               wnd,
               method=2):
    """
    Scales wind speeds from ERA5 data based on a specified method.
    This function adjusts the wind speeds in the ERA5 data array using scaling factors 
    derived from a row of wind asset data. It is typically used in the `generate_wind_ts()` 
    function to prepare wind speed time series data.
    
    Args:
        row (pd.Series): A row from the `wind_assets.csv` DataFrame containing wind asset 
            information, including coordinates (`x`, `y`) and scaling factors 
            (`GWA_wind_speed` or `windspeed_gwa`).
        wnd (xarray.DataArray): The ERA5 wind speed data array (e.g., `cutout.data.wnd100m`).
        method (int, optional): The scaling method to use. Defaults to 2. 
            - Method 1: Scales using `row['GWA_wind_speed']` (preferred for specific project points).
            - Method 2: Scales using `row['windspeed_gwa']` (preferred for ERA5 grid cells, default).
    Returns:
        np.ndarray: The scaled wind speed values at the specified location.
    Notes:
        - The function selects wind speed data at the location specified by `row['x']` 
          and `row['y']` using the `wnd.sel()` method.
        - The scaling factor is applied by dividing the wind speed at the location by 
          its mean and multiplying by the corresponding scaling factor from the row.

    """     
    wind_at_location = wnd.sel(x=row['x'], y=row['y']).values
    
    if method==1:
        scaled = wind_at_location * row['GWA_wind_speed'] / np.mean(wind_at_location) # method1 , preferably for specific project points
    if method==2:
        scaled = wind_at_location * row['windspeed_gwa'] / np.mean(wind_at_location) # method2, preferably for ERA5 grid cells
        
    return scaled


def rescale_cutout_windspeed(cutout: atlite.Cutout,
                            wind_assets:pd.DataFrame):


    #The wind data is the main target here
    wnd = cutout.data.wnd100m

    #Select the nearest squares on the ERA5 grid for each wind turbine farm
    nearest = cutout.data.sel({'x': wind_assets.x.values, 'y': wind_assets.y.values}, 'nearest').coords
    x_near = nearest.get('x').values
    y_near = nearest.get('y').values

    #Put these matched squares into the dataframe to line them up nice
    wind_assets['x'] = x_near
    wind_assets['y'] = y_near

    #Get scaled wind values and x-y coords
    scaled_wind = wind_assets.apply(lambda x: scale_wind(x, wnd), axis=1)
    xy = wind_assets.apply(lambda x: get_XY(x, wnd), axis=1)

    #Combine scaled_wind and xy an reformat a bit
    scaled_wind_xy = pd.DataFrame(data={'Wind': scaled_wind, 'XY': xy}).dropna().reset_index()#.drop(columns='index')

    #This will be used to hold the data array values to be assigned back to the cutout's dataset
    to_data_array = wnd

    #Writing the data in scaledWindXY into toDataArray
    for i in range(scaled_wind_xy.index.size):
        #to_data_array[time, y, x], scaledWind.Wind contains the scaled hourly wind at location y, x
        to_data_array[:, scaled_wind_xy.loc[i].XY[1], scaled_wind_xy.loc[i].XY[0]] = scaled_wind_xy.loc[i].Wind

    #Overwriting the wind data in the cutout with the scaled data
    wnd.data = to_data_array
    
    return cutout



def get_speed(row, xaxis, yaxis, data):
    """
    Function to return wind speed of closest pixel for a turbine
    Used in get_wind_coords()
    
    Args:
        row = Some row in the wind_assets.csv data frame
        xaxis = Linear space ranging from westmost point on wind_atlas to eastmost point on wind_atlas
        yaxis = Linear space ranging from northmost point on wind_atlas to southmost point on wind_atlas
        data = The wind_atlas .tif data
    """
    #Get indices of the nearest pixels
    # xIdx = np.searchsorted(xaxis, row['longitude'], side='left')
    xIdx = np.searchsorted(xaxis, row['x'], side='left')
    # yIdx = len(yaxis) - np.searchsorted(yaxis, row['latitude'], side='left', sorter=np.arange(len(yaxis)-1, -1, -1))
    yIdx = len(yaxis) - np.searchsorted(yaxis, row['y'], side='left', sorter=np.arange(len(yaxis)-1, -1, -1))

    return data[yIdx][xIdx] #Return the wind speed at the indices


#Generate a data frame that matches wind speeds from Global Wind Atlas to latitude/longitude values for scaling the cutout speeds

def get_wind_coords(assets, wind_atlas, wind_geojson):
    
    """    
    assets = The data frame for wind_assets.csv
    wind_atlas = The Global Wind Atlas wind speed data from the .tif file
    wind_geojson = The Global Wind Atlas geojson data which creates the shape for BC

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
    wind_coords = assets.apply(lambda x: get_speed(x, xaxis, yaxis, wind_atlas), axis=1)

    return wind_coords


def get_XY(row, wnd):
    '''
    Function to get INDEX values of the square in the ERA5 data array is
    Used in generate_wind_ts()
    row = Some row in the wind_assets.csv data frame
    wnd = cutout.wnd100m.data
    '''
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