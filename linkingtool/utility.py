# Script for Utility Functions for all modules

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
from shapely.geometry import box
import logging as log
import json
from shapely.geometry import box,Point

import pickle

def dict_to_pickle(
        my_dictionary:dict,
        save_to_path:str):
    """
    Takes dictionary file and saves to given local path as pickle file. Returns a NONE as
    """
    with open(save_to_path,'wb') as file:
        pickle.dump(my_dictionary,file)
    # return None

def pickle_to_dict(pickle_file_path):
    with open(pickle_file_path,'rb') as file:
        my_dictionary=pickle.load(file)
    return my_dictionary

def create_blank_yaml(file_path):
        with open(file_path, 'w'):
            pass

def save_dict_datafile(dictionary,save_to):
    with open(save_to,'w') as json_file:
        json.dump(dictionary,json_file)
        return log.info(f" saved as '{save_to}")
    
def load_dict_datafile(json_file_path:str)->dict:
    with open(json_file_path,'r') as json_file:
        dictionary_:dict=json.load(json_file)
        return dictionary_

def check_LocalCopy_and_run_function(
        directory_path:str, 
        function_to_run:function, 
        force_update:bool=False)->bool:
    """
    Check if a directory exists. If it does, execute the provided function.

    Parameters:
    - directory_path (str): The path to the directory.
    - function_to_run (callable): The function to execute if the directory exists.

    Returns:
    - bool: True if the directory exists and the function is executed, False otherwise.
    """
    if force_update:
        output=function_to_run()
        log.info(f"Forcefully ran '{function_to_run.__name__}' on '{directory_path}'.")
        return output
    else:
        if not os.path.exists(directory_path):
            output=function_to_run()
            log.info(f"Directory '{directory_path}' created.")
            return output
        else:
            # log.info(f"Directory '{directory_path}' found locally.")
            return log.info(f"Directory '{directory_path}' found locally.")

# Function to Create pre-requisite Directories
def create_directories(path):
    """
    Takes in the path string and creates the directories if non-existent.
    """

    if not os.path.exists(path):
        os.mkdir(path)
        log.info(f"Directory created: {path}")
        return True
    else:
        #do nothing
        log.info(f"Directory existis locally: {path}")
        return False
    
# Function to Load User Configuration File
def load_config(file_path):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

# Function to Extract BC Grid Cells from ERA5 Cutout and GADM Regional Boundary
def extract_BC_grid_cells_within_Regions(cutout,gadm_regions_gdf):

    log.info(f"Extracting BC Grid Cells from ERA5 Cutout and GADM Region Boundaries...")

    bc_grid_cells = gpd.sjoin(cutout.grid, gadm_regions_gdf, predicate='intersects')
    bc_grid_cells.drop(columns=['index_right'],inplace=True)

    return bc_grid_cells

def calculate_cell_area(gdf,column_name,region_name,actual_area_ROI):
    #actual_area_ROI = Actual area of Region of Interest in Sq. km
    gdf = gdf.to_crs(epsg=2163) 
    gdf_with_area_column=gdf.copy()
    gdf_with_area_column[column_name] = gdf_with_area_column['geometry'].area / 1e6  # Converts to square kilometers
    _total_land_area = int(gdf_with_area_column[column_name].sum()) #sq. km

    print(
        f'Region of Interest : {region_name}\n\
            > Available Land Area of the  Grid Cells = {{:,}} Km²\n\
            > Actual Land Area : {actual_area_ROI} km² \n'.format(_total_land_area))

    #Restore the projection for other usage
    gdf_with_area_column=gdf_with_area_column.to_crs(epsg=4326)
    return gdf_with_area_column

# Function to Generate Cell Index from Region name
def assign_regional_cell_ids(cells_dataframe, Source_Column, index_name):
    unique_values = cells_dataframe[Source_Column].unique()

    # If there's only one unique value, return the original DataFrame
    if len(unique_values) == 1:
        return cells_dataframe

    region_dfs = []

    for x in unique_values:
        _mask = cells_dataframe[Source_Column] == x
        _x_df_ = cells_dataframe[_mask].reset_index(drop=True)

        # Create a new column with given index name, for each region
        _x_df_[index_name] = [f'{x}_{index + 1}' for index in range(len(_x_df_))]

        region_dfs.append(_x_df_)

    # Concatenate all site-specific DataFrames into one DataFrame
    dataframe_with_cell_ids = pd.concat(region_dfs, ignore_index=True)

    # Set the index to the newly created column if it exists
    if index_name in dataframe_with_cell_ids.columns:
        dataframe_with_cell_ids.set_index(index_name, inplace=True)

    return dataframe_with_cell_ids



def create_layout_for_generation(cutout,cells_gdf,capacity_column):
    log.info(f"Creating Layout for PV generation from BC Grid Cells...")
    wind_layout_MW = cutout.layout_from_capacity_list(
        cells_gdf, col=capacity_column)
    wind_layout_MW = wind_layout_MW.where(wind_layout_MW != 0, drop=True)

    return wind_layout_MW  #xarray


def find_nearest_station(cell_geometry, buses_gdf, bus_tree):
    _, index = bus_tree.query((cell_geometry.centroid.x, cell_geometry.centroid.y))
    nearest_bus_row = buses_gdf.iloc[index]
    distance_km = cell_geometry.distance(nearest_bus_row['geometry']) * 111.32  # Degrees to km conversion
    nearest_station_name = nearest_bus_row['name']
    return nearest_station_name, distance_km


# def prepare_cutout(module,start_date, end_date,bounding_box,save_to_path):

#     min_x,max_x,min_y,max_y=bounding_box.values()

#     cutout = atlite.Cutout(
#         path=save_to_path,
#         module=module,
#         x=slice(min_x,max_x), # Longitude
#         y=slice(min_y,max_y), # Latitude
#         time=slice(start_date,end_date)
#     )

#     cutout.prepare()
#     return True

def get_cutout_path(region_code:str,
                    cutout_config:dict):
    '''
    This function return the unique name based on the region and start/end year
    for a cutout. 
    return: file path + name for the cutout described by selections in the
    cutout configuration.
    '''
    # Create file_path name with custom year_+date
    start_year = cutout_config["snapshots"]["start"][0][:4]
    end_year = cutout_config["snapshots"]["end"][0][:4]
    prefix = cutout_config['root'] + region_code
    if start_year == end_year:
        suffix = start_year
        file = "_".join([prefix, suffix + ".nc"])
    else: # multi_year_file
        suffix = "_".join([start_year, end_year])
        file_path = "_".join([prefix, suffix + ".nc"])

    return file_path

def create_era5_cutout(
    region_code:str,
    bounding_box:dict, 
    cutout_config:dict):
    '''
    This function creates a cutout based on data for era5.
    '''
    # Extract parameters from configuration file
    dx,dy = cutout_config["dx"], cutout_config['dy']
    time_horizon = slice(cutout_config["snapshots"]['start'][0],
                        cutout_config["snapshots"]['end'][0])

    # get path + filename for the cutout
    file_path = get_cutout_path(region_code,cutout_config)
    min_x,max_x,min_y,max_y=bounding_box.values()
    # Create the cutout based on bounds found from above
    cutout = atlite.Cutout(path=file_path,
                    module=cutout_config["module"],
                    # x=slice(bounds['west_lon'] - dx, bounds['east_lon'] + dx),
                    # y=slice(bounds['south_lat'] - dy, bounds['north_lat'] + dy ),
                    x=slice(min_x-dx,max_x+dx), # Longitude
                    y=slice(min_y-dy,max_y+dy), # Latitude
                    dx=dx,
                    dy=dy,
                    time=time_horizon)

    cutout.prepare()
    return True




def select_top_sites(all_scored_sites_gdf, max_wind_capacity):
    print(f">>> Selecting TOP Sites to for {max_wind_capacity} GW Capacity Investment in BC...")
    """
    Select the top wind sites based on potential capacity and a maximum wind limit.

    Parameters:
    - sites_gdf: GeoDataFrame containing wind cell and bucket information.
    - max_wind_capacity: Maximum allowable wind capacity in GW.

    Returns:
    - selected_sites: GeoDataFrame with the selected top wind sites.
    """
    print(f"{'_'*50}")
    print(f"Selecting the Top Ranked Sites to invest in {max_wind_capacity} GW PV in BC")
    print(f"{'_'*50}\n")

    # Initialize variables
    selected_rows = []
    total_capacity = 0.0

    top_sites = all_scored_sites_gdf.copy()

    if top_sites['potential_capacity'].iloc[0] < max_wind_capacity * 1000:
        # Iterate through the sorted GeoDataFrame
        for index, row in top_sites.iterrows():
            # Check if adding the current row's capacity exceeds Max_wind
            if total_capacity + row['potential_capacity'] <= max_wind_capacity * 1000:
                selected_rows.append(index)  # Add the row to the selection
                # Update the total capacity
                total_capacity += row['potential_capacity']
            # If adding the current row's capacity would exceed Max_wind, stop the loop
            else:
                break

        # Create a new GeoDataFrame with the selected rows
        top_sites = top_sites.loc[selected_rows]

        # Apply the additional logic
        mask = all_scored_sites_gdf['Site_ID'] > top_sites['Site_ID'].max()
        selected_additional_sites = all_scored_sites_gdf[mask].head(1)
        
        remaining_capacity = max_wind_capacity * 1000 - top_sites['potential_capacity'].sum()

        if remaining_capacity > 0:
            
            # selected_additional_sites['capex'] = capex_wind* remaining_capacity
            print(f"\n!! Note: The Last cluster originally had {round(selected_additional_sites['potential_capacity'].iloc[0] / 1000,2)} GW potential capacity."
                 f"To fit the maximum capacity investment of {max_wind_capacity} GW, it has been adjusted to {round(remaining_capacity / 1000,2)} GW\n")
            
            selected_additional_sites['potential_capacity'] = remaining_capacity
        # Concatenate the DataFrames
        top_sites = pd.concat([top_sites, selected_additional_sites])
    else:
        original_capacity = all_scored_sites_gdf['potential_capacity'].iloc[0]

        print(f"\n!! Note: The first cluster originally had {round(original_capacity / 1000,2)} GW potential capacity."
              f"To fit the maximum capacity investment of {max_wind_capacity} GW, it has been adjusted. \n")

        top_sites = top_sites.iloc[:1]  # Keep only the first row
        # Adjust the potential_capacity of the first row
        top_sites.at[top_sites.index[0], 'potential_capacity'] = max_wind_capacity * 1000

    return top_sites  # gdf


def print_module_title(text,Length_Char_inLine=60):
    print(f"{Length_Char_inLine*'_'}\n"
        f"{5*' ' }{text}\n"
        f"{Length_Char_inLine*'_'}")