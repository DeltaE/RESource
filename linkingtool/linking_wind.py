# Script for Wind Module utility Functions
# Prepared by : Md Eliasinul Islam
# Affiliation : Delta E+ lab, Simon Fraser University
# Version : 1.0
# Release : 2024

import os,yaml
from scipy.spatial import cKDTree
import pandas as pd
from scipy.spatial import cKDTree
import geopandas as gpd
import rioxarray as rxr
import atlite
from shapely.geometry import box,Point
import logging as log
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import linkingtool.linking_utility as utility
from tqdm import tqdm
import numpy as np

# Function for GWA data download
def download_GWA_data(parent_direct,url_source,raster_file_name):
    file_path=os.path.join(parent_direct,raster_file_name)
    response = requests.get(url_source)

    if response.status_code == 200:
        # Download the file and save it with the defined name
        with open(file_path, 'wb') as file:
            file.write(response.content)
        log.info(f"> Global Wind Atlas (GWA) raster file downloaded successfully and saved as {raster_file_name}.\n")
    else:
        log.info("> Failed to download the file from the URL. Checkand update URL in Unser Config file at '{user_config_file}'")

    return True

def calculate_ERA5_cell_area(gdf,availability_column,actual_area_ROI):
    #actual_area_ROI = Actual area of Region of Interest in Sq. km
    gdf = gdf.to_crs(epsg=2163) 

    gdf_with_area_column=gdf.copy()

    if 'land_area_sq_km' not in gdf_with_area_column.columns:
        gdf_with_area_column ['land_area_sq_km']=gdf_with_area_column['geometry'].area / 1e6 #sqkm
        gdf_with_area_column['eligible_land_area']= gdf_with_area_column['land_area_sq_km']* gdf_with_area_column[availability_column] 
    else:
        gdf_with_area_column['eligible_land_area']= gdf_with_area_column['eligible_land_area']* gdf_with_area_column[availability_column] 
    
    _total_eligible_land_area = int(gdf_with_area_column['eligible_land_area'].sum()) #sq. km

    print(
        f'\nRegion of Interest : BC\n',
        f"> Eligible Land Area of the  Grid Cells = {{:,}} Km²\n".format(_total_eligible_land_area),
        f"> Actual Land Area : {{:,}} km²\n".format(actual_area_ROI)
    )

    #Restore the projection for other usage
    gdf_with_area_column=gdf_with_area_column.to_crs(epsg=4326)
    
    return gdf_with_area_column


def IEC_class_from_pixel_values(gwa_IEC_class_layers_df,pixel_values_column_name):
    # Mapping pixel values to IEC class
    class_mapping = {0: 'III', 1: 'II', 2: 'I', 3: 'T', 4: 'S'}
    df=gwa_IEC_class_layers_df
    # Apply mapping, set 'none' for other values
    df[pixel_values_column_name] = df[pixel_values_column_name].map(class_mapping).fillna('none')
    df.reset_index(inplace=True)
    df.drop(columns=['spatial_ref','band'], inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # geometry = [Point(xy) for xy in zip(gwa_IEC_class_layers_df['x'], df['y'])]
    # gwa_IEC_class_layers_gdf = gpd.GeoDataFrame(df, geometry=geometry)

    log.info(f"IEC Classes mapped for cells. Class mapping from Pixel value as described in URL : https://globalwindatlas.info/en/about/dataset")
    return df
    
def visualize_GWA_data(gwa_dataframe, data, color, xlabel):

    ####### Parameters:
    # gwa_dataframe: The DataFrame containing the data to be visualized.
    # data: str, The column name from the DataFrame to be plotted.
    # color: The color of the histogram bars.
    # xlabel: The label for the x-axis of the plo


    # Create a histogram plot using Seaborn
    sns.histplot(gwa_dataframe[data], color=color, kde=True)

    # Check if the data column is related to windspeed and update xlabel accordingly
    if 'windspeed' in data:
        xlabel = 'windspeed (m/s)'
    else:
        xlabel = xlabel

    # Set x-axis label, y-axis label, and plot title
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(f'{data} Distribution of GWA Cells')

    # Adjust layout, add grid, and save the plot as an SVG file
    plt.tight_layout()
    plt.grid(True)
    save_to=f'vis/Wind/{data}.svg'
    plt.savefig(save_to)

    # Close the current figure to free up resources
    plt.close()

    # Log an information message indicating the successful creation and saving of the plot
    return log.info(f"{data} Distribution of All GWA Cells - Created and saved in Visualization directory")


import geopandas as gpd
from shapely.geometry import box
import logging

log = logging.getLogger(__name__)

def calculate_common_parameters_GWA_cells(GWA_dataframe, wind_cap_per_km2, cell_resolution_arc_deg=0.002499):
    """
    Calculate common parameters for GWA cells including land area and potential wind capacity.

    Parameters:
    - GWA_dataframe (pd.DataFrame): DataFrame containing GWA cell coordinates ('x', 'y').
    - wind_cap_per_km2 (float): Wind capacity per square kilometer.
    - cell_resolution_arc_deg (float): Resolution of the cell in degrees (default 0.002499).

    Returns:
    - pd.DataFrame: Updated GWA_dataframe with land area and potential capacity.
    - gpd.GeoDataFrame: GeoDataFrame for a single GWA cell geometry.
    """
    res = cell_resolution_arc_deg
    log.info("Creating Polygon Geometry for GWA Cells for a Single Cell")

    # Check for required columns
    if not {'x', 'y'}.issubset(GWA_dataframe.columns):
        log.error("GWA_dataframe must contain 'x' and 'y' columns.")
        raise ValueError("GWA_dataframe must contain 'x' and 'y' columns.")

    # Extract centroid coordinates for the first row
    x, y = GWA_dataframe[['x', 'y']].iloc[0]

    # Generate the geometry for a single cell
    geom_single_cell = box(x - res / 2, y - res / 2, x + res / 2, y + res / 2)

    # Create the GeoDataFrame for a single cell
    GWA_single_cell_gdf = gpd.GeoDataFrame(geometry=[geom_single_cell], crs="EPSG:4326")

    # Change the CRS for area calculation
    GWA_single_cell_gdf = GWA_single_cell_gdf.to_crs(epsg=2163)

    # Calculate the land area for the cell and add it as a new column
    GWA_single_cell_gdf['land_area_sq_km'] = GWA_single_cell_gdf['geometry'].area / 1e6

    # Restore the projection for other usage
    GWA_single_cell_gdf = GWA_single_cell_gdf.to_crs(epsg=4326)

    log.info("Imputing Land Area, Potential Capacity for all GWA Cells")
    GWA_dataframe['land_area_sq_km'] = GWA_single_cell_gdf['land_area_sq_km'].iloc[0]
    GWA_dataframe['potential_capacity'] = wind_cap_per_km2 * GWA_single_cell_gdf['land_area_sq_km'].iloc[0]

    return GWA_dataframe, GWA_single_cell_gdf


def update_ERA5_params_from_mapped_GWA_cells(era5_cells_gdf,mapped_GWA_cells_gdf):

    log.info(f"Updating ERA5 mean windspeed with mapped GWA Cells' for each cell, labeled as 'windspeed_GWA'")

    mean_windspeed_by_bcgrid = mapped_GWA_cells_gdf.groupby('ERA5_cell_index')['windspeed'].mean()
    mean_CF_by_bcgrid = mapped_GWA_cells_gdf.groupby('ERA5_cell_index')['CF_IEC3'].mean()
    
    era5_cells_gdf = era5_cells_gdf.copy()
    # Map the mean windspeed values to the corresponding bcgrid_id in the DataFrame
    era5_cells_gdf['windspeed_GWA'] = era5_cells_gdf.index.map(mean_windspeed_by_bcgrid)
    era5_cells_gdf['CF_mean_GWA'] = era5_cells_gdf.index.map(mean_CF_by_bcgrid)

    # # era5_cells_gdf_filtered = era5_cells_gdf_filtered.loc[era5_cells_gdf_filtered['windspeed_GWA'].notna()]
    # era5_cells_gdf_filtered.to_pickle('data/Processed_data/Wind/era5_cells_gdf_filtered.pkl')
    
    era5_cells_gdf_updated=era5_cells_gdf

    log.info(f"ERA5 mean windspeed and CF updated and labeled as 'windspeed_GWA' , 'CF_mean_GWA' ")
    return era5_cells_gdf_updated

def create_GWA_all_cells_polygon(dataframe,cell_resolution=0.002499):
    res = cell_resolution
    log.info("Creating Polygon Geometry for GWA Cells for a Single Cell")

    dataframe['geometry'] = dataframe.apply(lambda row: box(row['x'] - res/2, row['y'] - res/2, row['x'] + res/2, row['y'] + res/2), axis=1)

    return dataframe


def map_GWAcells_to_ERA5cells(
    gwa_cells_gdf: gpd.GeoDataFrame,
    era5_cells_gdf: gpd.GeoDataFrame,
    tech_cap_per_km2: float
    )->tuple:
    """
    Maps GWA cells to ERA5 cells, calculating potential capacity, and filtering ERA5 cells 
    that contain at least one GWA cell.
    """
    era5_cells_gdf=era5_cells_gdf.reset_index()
    # Perform the spatial join to map GWA cells to ERA5 cells
    gwa_cells_mapped_gdf = gpd.sjoin(
        gwa_cells_gdf, 
        era5_cells_gdf[['cell','Region', 'geometry', 'Region_ID']], 
        how='inner', 
        predicate='intersects'
    )
    era5_cells_gdf=era5_cells_gdf.set_index('cell')
    # Reset the index to avoid duplicate indices, while keeping the index_right column
    gwa_cells_mapped_gdf = gwa_cells_mapped_gdf.reset_index(drop=True)

    # Rename the 'index_right' column for clarity (optional)
    gwa_cells_mapped_gdf = gwa_cells_mapped_gdf.rename(columns={'cell': 'ERA5_cell_index'})

    # Filter ERA5 cells that have at least one GWA cell
    GWA_unique_era5_cells = gwa_cells_mapped_gdf['ERA5_cell_index'].unique()
    era5_cells_gdf_filtered = era5_cells_gdf.loc[GWA_unique_era5_cells]

    gwa_cells_mapped_gdf=gwa_cells_mapped_gdf.to_crs(epsg=4326)
    max_cap_GWA_cell = gwa_cells_mapped_gdf['potential_capacity'].max()

    # Distribute potential capacity to GWA cells within each ERA5 cell
    for province_index in era5_cells_gdf_filtered.index:
        cell_mask = gwa_cells_mapped_gdf['ERA5_cell_index'] == province_index
        if cell_mask.any():
            # Calculate the distributed capacity per GWA cell
            calculated_cap = era5_cells_gdf_filtered.loc[province_index, 'potential_capacity'] / cell_mask.sum()
            
            # Assign the distributed capacity to each GWA cell in the ERA5 cell
            gwa_cells_mapped_gdf.loc[cell_mask, 'potential_capacity'] = calculated_cap

    # Output the total potential capacity for both ERA5 and GWA cells for verification
    print(f'Filtered Sites: Total Wind Potential (ERA5 Cells): {round(era5_cells_gdf_filtered.potential_capacity.sum() / 1000, 2)} GW')
    print(f'Filtered Sites: Total Wind Potential (GWA Cells): {round(gwa_cells_mapped_gdf.potential_capacity.sum() / 1000, 2)} GW')

    return gwa_cells_mapped_gdf, era5_cells_gdf_filtered

'''
////replaced by new func////

def map_GWAcells_to_ERA5cells(
        gwa_cells_gdf: gpd.GeoDataFrame,
        era5_cells_gdf: gpd.GeoDataFrame,
        tech_cap_per_km2: float,
        batch_process_size: int = 1000):
    """
    Finds the associated GWA cells under each ERA5 cell and manages computation memory resources by batching.
    """
    batch_size = batch_process_size  # Adjust the batch size based on your system's capacity
    
    gwa_batches = [gwa_cells_gdf.iloc[i:i + batch_size] for i in range(0, len(gwa_cells_gdf), batch_size)]

    log.info(f"Mapping {len(gwa_cells_gdf)} GWA Cells to {len(era5_cells_gdf)} ERA5 Cells")
    log.warning("!! Memory intensive process initiated !!")
    
    gwa_cells_mapped_dfs = []
    
    # for gwa_batch in gwa_batches:
    #     gwa_cells_mapped_df_batch = gpd.sjoin(gwa_batch, era5_cells_gdf[['Region', 'geometry', 'Region_ID']], how='inner', predicate='intersects')
    #     gwa_cells_mapped_dfs.append(gwa_cells_mapped_df_batch)
    #     gwa_batch = None  # Clear the memory

    # gwa_cells_mapped_gdf = gpd.GeoDataFrame(pd.concat(gwa_cells_mapped_dfs), crs=gwa_cells_gdf.crs)
    # gwa_cells_mapped_gdf = gwa_cells_mapped_gdf.rename(columns={'cell': 'ERA5_cell_index'})
    
    for i, gwa_batch in enumerate(gwa_batches):
        gwa_batch = gwa_batch.reset_index(drop=True)  # Reset index in each batch
        gwa_batch['batch_id'] = i  # Add a batch_id
        gwa_cells_mapped_df_batch = gpd.sjoin(gwa_batch, era5_cells_gdf[['Region', 'geometry', 'Region_ID']], how='inner', predicate='intersects')
        gwa_cells_mapped_df_batch['unique_index'] = gwa_cells_mapped_df_batch.index.astype(str) + '_' + gwa_batch['batch_id'].astype(str)
        gwa_cells_mapped_df_batch = gwa_cells_mapped_df_batch.set_index('unique_index')
        gwa_cells_mapped_dfs.append(gwa_cells_mapped_df_batch)
        gwa_batch = None  # Clear the memory

        gwa_cells_mapped_gdf = gpd.GeoDataFrame(pd.concat(gwa_cells_mapped_dfs), crs=gwa_cells_gdf.crs)

    log.info("Filtering the ERA5 Cells which contain at least one GWA Cell")
    GWA_unique_era5_cells = gwa_cells_mapped_gdf['ERA5_cell_index'].unique().tolist()

    era5_cells_gdf_filtered = era5_cells_gdf[era5_cells_gdf.index.isin(GWA_unique_era5_cells)]

    log.info(f'{len(era5_cells_gdf_filtered)} out of {len(era5_cells_gdf)} ERA5 Cells Kept for further processing. \
             {len(era5_cells_gdf) - len(era5_cells_gdf_filtered)} Cells filtered out due to non-availability of at least 1 GWA Cell within')

    gwa_cells_mapped_gdf, GWA_single_cell_gdf = calculate_common_parameters_GWA_cells(gwa_cells_mapped_gdf, tech_cap_per_km2)

    GWA_single_cell_gdf['potential_capacity'] = tech_cap_per_km2 * GWA_single_cell_gdf['land_area_sq_km']
    max_cap_GWA_cell = GWA_single_cell_gdf['potential_capacity'].iloc[0]

    for province_index in era5_cells_gdf_filtered.index:
        cell_mask = gwa_cells_mapped_gdf.ERA5_cell_index == province_index
        calculated_cap_series  = era5_cells_gdf_filtered.loc[province_index, 'potential_capacity'] / len(gwa_cells_mapped_gdf[cell_mask])

        # if calculated_cap > max_cap_GWA_cell:
        #     gwa_cells_mapped_gdf.loc[cell_mask, 'potential_capacity'] = max_cap_GWA_cell
        # else:
        #     gwa_cells_mapped_gdf.loc[cell_mask, 'potential_capacity'] = calculated_cap
            
        # Compare the Series with max_cap_GWA_cell and update gwa_cells_mapped_gdf accordingly
        updated_cap = calculated_cap_series.clip(upper=max_cap_GWA_cell)
        single_updated_cap = updated_cap.values[0] if len(updated_cap) > 0 else updated_cap
        gwa_cells_mapped_gdf.loc[cell_mask, 'potential_capacity'] = single_updated_cap

    print(f'Filtered Sites: Total Wind Potential (ERA5 Cells): {round(era5_cells_gdf_filtered.potential_capacity.sum() / 1000, 2)} GW')
    print(f'Filtered Site: Total Wind Potential (GWA Cells): {round(gwa_cells_mapped_gdf.potential_capacity.sum() / 1000, 2)} GW')

    gwa_cells_mapped_gdf = create_GWA_all_cells_polygon(gwa_cells_mapped_gdf)

    return gwa_cells_mapped_gdf, era5_cells_gdf_filtered
'''
def impute_ERA5_windspeed_to_Cells(
        cutout:atlite.Cutout, 
        province_grid_cells:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    """
    For each grid cells, this function finds the yearly mean windspeed from the windspeed timeseries and imputes to the cell dataframe.
    """
    
    log.info(f"Calculating yearly mean windspeed and imputing to provincial Grid Cells named as 'windspeed_ERA5'")

    # Calculate yearly mean windspeed
    wnd_ymean_df = cutout.data.wnd100m.groupby('time.year').mean('time').to_dataframe(name='windspeed_ERA5').reset_index()

    # Create a GeoDataFrame for spatial join
    wnd_ymean_gdf = gpd.GeoDataFrame(wnd_ymean_df, geometry=gpd.points_from_xy(wnd_ymean_df['x'], wnd_ymean_df['y']))
    wnd_ymean_gdf.crs = province_grid_cells.crs

    # Perform spatial join and drop unnecessary columns
    province_grid_cells = (
        gpd.sjoin(province_grid_cells.rename(columns={'x': 'x_bc', 'y': 'y_bc'}), wnd_ymean_gdf, predicate='intersects')
        .drop(columns=['x_bc', 'y_bc', 'lon', 'lat'])
    )
    
    # Handle potential duplicate indices
    # Resetting index to ensure unique index after join
    province_grid_cells = province_grid_cells.reset_index(drop=True)
    
    province_grid_cells = province_grid_cells.drop_duplicates(subset=['geometry'])
    province_grid_cells=utility.assign_regional_cell_ids(province_grid_cells,'Region','cell')

    return province_grid_cells


# def rescale_ERA5_cutout_windspeed_with_mapped_GWA_cells(
#         cutout:atlite.Cutout,
#         era5_cells_gdf:gpd.GeoDataFrame):
    
#     """
#     Rescales the ERA5 windspeed timeseries with the scaling factor obtained from mean_windspeed of mapped GWA cells under each ERA5 grid cells. 

#     Background: GWA cells contains a static data for each cell, whereas ERA5 cells contains timeseries (8760hrs) data for a cell. GWA cells have higher resolution (~250m) from ERA5 cells (~30km) and 
#     literture has found the GWA data to be more accurate due to windspeed variance across ~30km range. But GWA doesn't provide a hourly temporal representation of that windspeed. Hence, for further calculation we have rescaled the ERA5 cells timeseries with GWA cells' windspeed.
#     while rescalling, the windspeed from associated GWA cells mapped with each ERA5 cells have been accounted. Especially the mean_windspeed (thus the mean_CapacityFactor) calculations have significant difference due to this rescaling. Particularly to avoid over/under estimation of the windspeed distribution.
#     """

#     log.info("Scaling the ERA5 Cell's Windspeed with mapped GWA Cells within each Cell")
#     log.warning(f"!! Memory Intensive Calculation in Progress...")
#     log.warning("!! Windspeed Only for the Coordinates associated to filtered ERA5 Cells's shall be scaled")

#     def scale_wind(row, wnd):
        
#         if row['potential_capacity'] >0:
#             #Scale the wind speeds at this location on the ERA5 data array
#             wind_at_ERA5 = wnd.sel(x=row['x'], y=row['y']).values    #Time series
#             scaling_factor= row['windspeed_GWA'] / row['windspeed_ERA5'] # Scalar
#             scaled = wind_at_ERA5 * scaling_factor
#             return scaled
#         else:
#             #Do nothing
#             return None
        
#     def get_XY(row, wnd):
        
#         x = 0
#         y = 0
#         for i in range(wnd.x.size):
#             if row['x'] == wnd.x.values[i]:
#                 x = i
#                 break
        
#         for j in range(wnd.y.size):
#             if row['y'] == wnd.y.values[j]:
#                 y = j
#                 break

#         return [x, y]

#     wnd = cutout.data.wnd100m
#     era5_cells = era5_cells_gdf   # Potential Sites

#     #Get scaled wind values and x-y coords
#     scaled_wind = era5_cells.apply(lambda x: scale_wind(x, wnd), axis=1)
#     log.info("Extracting the Windspeed of ERA5 Grid")
#     xy = era5_cells.apply(lambda x: get_XY(x, wnd), axis=1)

#     #Combine scaled_wind and xy an reformat a bit
#     scaled_wind_xy = pd.DataFrame(data={'Wind': scaled_wind, 'XY': xy}).dropna().reset_index().drop(columns='cell')

#     #This will be used to hold the data array values to be assigned back to the cutout's dataset
#     to_data_array = wnd

#     #Writing the data in scaledWindXY into toDataArray
#     for i in range(scaled_wind_xy.index.size):
#         #to_data_array[time, y, x], scaledWind.Wind contains the scaled hourly wind at location y, x
#         to_data_array[:, scaled_wind_xy.loc[i].XY[1], scaled_wind_xy.loc[i].XY[0]] = scaled_wind_xy.loc[i].Wind

#     log.info("Overwriting the scaled windspeed to ERA5 CUTOUT")
#     #Overwriting the wind data in the cutout with the scaled data
#     wnd.data = to_data_array
#     log.info("Scaled Windspeed data for filtered ERA5 Cells loaded to Cutout.")

#     return cutout

def rescale_ERA5_cutout_windspeed_with_mapped_GWA_cells(
    cutout: atlite.Cutout,
    era5_cells_gdf: gpd.GeoDataFrame
):
    """
    Rescales the ERA5 windspeed timeseries with the scaling factor obtained from mean_windspeed 
    of mapped GWA cells under each ERA5 grid cell.
    """

    log.info("Scaling the ERA5 Cell's Windspeed with mapped GWA Cells within each Cell")
    log.warning(f"!! Memory Intensive Calculation in Progress...")
    log.warning("!! Windspeed Only for the Coordinates associated to filtered ERA5 Cells shall be scaled")

    wnd = cutout.data.wnd100m
    era5_cells = era5_cells_gdf.copy()  # Work on a copy to avoid modifying the original dataframe

    # Create arrays for indexing
    x_values = wnd.x.values
    y_values = wnd.y.values
    x_index = {x: idx for idx, x in enumerate(x_values)}
    y_index = {y: idx for idx, y in enumerate(y_values)}

    # Extract and align data
    era5_cells['x_idx'] = era5_cells['x'].map(x_index)
    era5_cells['y_idx'] = era5_cells['y'].map(y_index)

    # Drop rows where x or y index mapping failed
    era5_cells = era5_cells.dropna(subset=['x_idx', 'y_idx'])

    # Calculate scaled winds
    era5_cells['scaled_wind'] = era5_cells.apply(
        lambda row: row['windspeed_GWA'] / row['windspeed_ERA5'] * cutout.data.wnd100m[:, int(row['y_idx']), int(row['x_idx'])] 
        if row['potential_capacity'] > 0 else None, axis=1
    )

    # Ensure no NaN values
    era5_cells = era5_cells.dropna(subset=['scaled_wind'])

    # Convert to numpy arrays for faster indexing
    scaled_winds = np.zeros_like(wnd.data)
    for _, row in era5_cells.iterrows():
        x_idx = int(row['x_idx'])
        y_idx = int(row['y_idx'])
        scaled_winds[:, y_idx, x_idx] = row['scaled_wind']

    # Overwrite the wind data in the cutout with the scaled data
    log.info("Overwriting the scaled windspeed to ERA5 CUTOUT")
    cutout.data.wnd100m.data = scaled_winds

    log.info("Scaled Windspeed data for filtered ERA5 Cells loaded to Cutout.")
    return cutout


def find_grid_nodes_GWA_cells(
        buses_gdf:gpd.GeoDataFrame,
        gwa_cells_mappedwithERA5_gdf:gpd.GeoDataFrame,
        # era5_cells_gdf_with_mean_CF:gpd.GeoDataFrame,
        grid_node_proximity_filter:float)->gpd.GeoDataFrame:
    
    """
    takes in the provincial grid nodes, a proximity filter ( <= km) and the GWA cells dataframe. Calcuates the nearest nodde and distance to that node, imputes the 
    data to the GWA cells dataframe and returns as geodataframe. 
    """

    gwa_cells_gdf=gwa_cells_mappedwithERA5_gdf
    # era5_cells_gdf=era5_cells_gdf_with_mean_CF

    buses_gdf.sindex
    # Create a KDTree for bus station geometries
    bus_tree = cKDTree(buses_gdf['geometry'].apply(lambda x: (x.x, x.y)).tolist())

    log.info(">> Calculating Nearest Grid Nodes for BC Grid Cells...")

    # Add empty columns to gwa_cells_mapped_gdf for the results
    gwa_cells_gdf['nearest_station'] = None
    gwa_cells_gdf['nearest_station_distance_km'] = None

    ERA5_cells = gwa_cells_mappedwithERA5_gdf['ERA5_cell_index'].unique().tolist()

    pbar = tqdm(total=len(ERA5_cells), desc="Calculating Nearest Stations")

    i = 0
    for ERA5_cell in ERA5_cells:
        mask = gwa_cells_gdf['ERA5_cell_index'] == ERA5_cell
        gwa_cells_mapped_gdf_x = gwa_cells_gdf.loc[mask]

        # Apply find_nearest_station to each row in the subset and fill in the columns
        gwa_cells_gdf.loc[mask, ['nearest_station', 'nearest_station_distance_km']] = gwa_cells_mapped_gdf_x.apply(
            lambda row: utility.find_nearest_station(row['geometry'], buses_gdf=buses_gdf, bus_tree=bus_tree),
            axis=1
        ).tolist()
        
        i += 1
        pbar.update(1)  # Update the progress bar
        pbar.set_postfix(cell=f"{ERA5_cell}")

    # Close the tqdm bar
    pbar.close()

    # gwa_cells_gdf['CF_mean'] = gwa_cells_gdf['ERA5_cell_index'].map(era5_cells_gdf['CF_mean_GWA'])

   
    proximity_to_nodes_mask=gwa_cells_gdf['nearest_station_distance_km']<=grid_node_proximity_filter 
    Gwa_cells_df_node_filtered=gwa_cells_gdf[proximity_to_nodes_mask]

    log.info(f"GWA Cells Filtered based on Proximity to Tx Nodes \n\
    Size: {len(Gwa_cells_df_node_filtered)}\n")
   
    return Gwa_cells_df_node_filtered


def create_CF_timeseries_df(cutout,start_date,end_date,geodataframe_sites,turbine_model,turbine_config_file,Site_index='cell',config='OEDB'):

    _layout_MW=utility.create_layout_for_generation(cutout,geodataframe_sites,'potential_capacity')
    
    log.info(f"Calculating Generation timeseries for BC Grid Cells as per the Layout Capacity (MW)...")
    # turbine_config=atlite.resource.get_windturbineconfig(turbine_model)
    if config=='atlite':
        hub_height_turbine=atlite.resource.get_windturbineconfig(turbine_model)['hub_height']
        turbine_config = atlite.resource.get_windturbineconfig(turbine_model, {"hub_height": 100})
        log.info(f"Selected Wind Turbine  Model : {turbine_model} @ {hub_height_turbine}m Hub Height")
    else:
        # Specify the path to your YAML file
        config_file = turbine_config_file #'configs/3.2M114_NES.yaml'

        # Read the YAML file into a dictionary
        with open(config_file, 'r') as file:
            turbine_config = yaml.safe_load(file)
            hub_height_turbine=turbine_config['hub_height']
            log.info(f"Selected Wind Turbine  Model : {turbine_config['name']} @ {hub_height_turbine}m Hub Height")

    wind = cutout.wind(
        turbine=turbine_config,
        show_progress=True,
        capacity_factor=True,
        layout=_layout_MW,
        add_cutout_windspeed=True,
        shapes=geodataframe_sites.geometry,
        per_unit=True,
    )  # Xarray

    # Convert Xarray to Dataframe
    sites = wind[Site_index].to_pandas()

    # start_date = configs['cutout']['start_date']
    # end_date = configs['cutout']['end_date']

    datetime_index = pd.date_range(start=start_date , end=end_date, freq='h')
    CF_ts_df = pd.DataFrame(index=datetime_index)

    # List to store DataFrames for each site
    site_dfs = []

    for site in sites:
        site_df = pd.DataFrame({site: wind.sel(time=slice(start_date, end_date), **{Site_index: site}).values}, index=datetime_index)
        site_dfs.append(site_df)

    # Concatenate all site-specific DataFrames into one DataFrame
    CF_ts_df = pd.concat(site_dfs, axis=1)
    # Assign the mean values to a new column 'CF_mean' in bc_grid_cells
    log.info(f"Calculating CF mean from the {len(CF_ts_df)} data points for each Cell ...")
    geodataframe_sites['CF_mean_atlite'] = CF_ts_df.mean()

    return geodataframe_sites,CF_ts_df

def clip_bc_data_from_GWA(GWA_raster_file,column_name,bounding_box):

    # Open the GeoTIFF file using rioxarray
    data = rxr.open_rasterio(GWA_raster_file).rename(column_name)
    
    min_x,max_x,min_y,max_y=bounding_box.values()

    # Clip the data to the specified bounding box
    bc_data = data.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)

    gwa_df = bc_data.to_dataframe()
    log.info(f"{column_name} data @ 100m hub height ; clipped for BC Region. Size: {len(gwa_df)}")

    return gwa_df #dataframe

def filter_GWA_cells(GWA_raster_path,bounding_box,data_low_bound,data_high_bound,column_name):

    gwa_df = clip_bc_data_from_GWA(GWA_raster_path,column_name,bounding_box)
    mask = (gwa_df[column_name] >= data_low_bound) & (gwa_df[column_name] <= data_high_bound)
    gwa_df_filtered = gwa_df[mask].sort_values(by=column_name, ascending=False)

    # Assigning a Cell id for the high resolution data to map with the lower resoltiuon BC Grid Cells.
  
    gwa_df_filtered.reset_index(inplace=True)
    # Clean the Data Fields (Columns) of the data frame
    gwa_df_filtered=gwa_df_filtered.loc[:, ( 'x', 'y',column_name)].astype('float32')

    # gwa_df_filtered.to_pickle(save_to)
    log.info(f'{column_name} data filtered with the filter range : {data_low_bound}-{data_high_bound}. Size: {len(gwa_df_filtered)}')
    log.warning(f"!! You can Configure the {column_name} filter in User config file and restart the data prepration with updated filter !!")
    return gwa_df_filtered

# '''
def create_timeseries_for_Cluster(
        Clustered_sites, 
        dissolved_indices, 
        gwa_cells_scored, 
        province_grid_CF_ts_df):
    
    regions = list(Clustered_sites.Region)
    
    region_cluster_dict = {}  # Initialize a dictionary to store median DataFrames for each (region, cluster) pair
    all_GWA_cells_cluster_ts = {}  # Initialize a dictionary to store GWA_cells_CF_ts DataFrames for all regions
    i=0
    for region in regions:
        i+=1
        log.info(f"Creating Timeseries for {region} Region")
        mask = Clustered_sites['Region'] == region
        clusters = list(Clustered_sites[mask].Cluster_No)

        region_median_values_df = pd.DataFrame()  # Initialize an empty DataFrame for the current region
        region_GWA_cells_cluster_ts = pd.DataFrame()  # Initialize an empty DataFrame for the GWA_cells_CF_ts of the current region

        for cluster in clusters:
            log.info (f" >> Creating Timeseries for cluster {cluster} / {len(clusters)}")
            GWA_cells = dissolved_indices[region][cluster]
            dfs_to_concat = []

            for i in range(len(GWA_cells)):
                GWA_cell_id = dissolved_indices[region][cluster][i]
                ERA5_cell_id = gwa_cells_scored.loc[gwa_cells_scored.index[i], 'ERA5_cell_index']

                if isinstance(ERA5_cell_id, pd.Series):
                    ERA5_cell_id = ERA5_cell_id.iloc[0]
                else:
                    ERA5_cell_id = ERA5_cell_id

                data = province_grid_CF_ts_df[ERA5_cell_id].values.tolist()
                index = province_grid_CF_ts_df.index
                column = GWA_cell_id

                GWA_cells_CF_ts = pd.DataFrame(data, index=index, columns=[column])
                dfs_to_concat.append(GWA_cells_CF_ts)

            GWA_cells_cluster_ts = pd.concat(dfs_to_concat, axis=1)
            column_name = f'{region}_{cluster}'

            # Calculate median values for the subset
            cluster_median_df = GWA_cells_cluster_ts.mean(axis=1)
            cluster_median_df = cluster_median_df.rename(column_name)

            # Append the median values as a new column to region_median_values_df
            region_median_values_df = pd.concat([region_median_values_df, cluster_median_df], axis=1)
            region_median_values_df.index = pd.to_datetime(region_median_values_df.index)

            # Append GWA_cells_CF_ts DataFrame to region_GWA_cells_cluster_ts
            region_GWA_cells_cluster_ts = pd.concat([region_GWA_cells_cluster_ts, GWA_cells_cluster_ts], axis=1)

        # Store the DataFrame for the current region in the dictionary
        region_cluster_dict[region] = region_median_values_df
        all_GWA_cells_cluster_ts[region] = region_GWA_cells_cluster_ts

    # Concatenate all DataFrames for each region into a single DataFrame
    cf_ts_cluster_df = pd.concat(region_cluster_dict.values(), axis=1)
    all_GWA_cells_cluster_ts_df=pd.concat(all_GWA_cells_cluster_ts.values(), axis=1)
    
    return cf_ts_cluster_df, all_GWA_cells_cluster_ts_df
# ''' 

def find_representative_ERA5Cluster_names(GWA_cell_id_str):
    split_parts=GWA_cell_id_str.split('_')
    ERA5Cluster_id='_'.join(split_parts[:2])
    return ERA5Cluster_id
