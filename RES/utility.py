# Script for Utility Functions for all modules

# Prepared by : Md Eliasinul Islam
# Affiliation : Delta E+ lab, Simon Fraser University
# Version : 1.0
# Dev_year : 2024-2025
import yaml
import os
import requests
from typing import Optional
from colorama import Fore, Style
import pandas as pd
import geopandas as gpd 
import logging as log
import json
import pickle
import datetime
from pathlib import Path
import geojson as gj
import rasterio as rio
import numpy as np
from RES.logger import setup_logger
import logging
logger = setup_logger(__name__, level=logging.DEBUG)

now = datetime.datetime.now()
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

def print_update(level: int=None,
                 message: str="--",
                 alert:Optional[bool]=False):
    if level is not None:
        if level == 1:
            color = Fore.YELLOW
            prefix="└"
        elif level == 2:
            color = Fore.CYAN
            prefix=" └"
        elif level == 3:
            color = Fore.LIGHTCYAN_EX + Style.DIM
            prefix="  └"
        elif level > 3:
            color = Fore.LIGHTBLACK_EX + Style.DIM
            prefix="  └─"
        elif alert:
            level=2
            color = Fore.RED
            prefix=" └ X "
    else:
        color = Fore.LIGHTMAGENTA_EX + Style.DIM
        prefix=" ─"
    
    print(f"{color}{prefix}> {message}{Style.RESET_ALL}")

def print_module_title(text, Length_Char_inLine=60):
    print(f"{Fore.LIGHTCYAN_EX}{Length_Char_inLine * '_'}{Style.RESET_ALL}\n"
          f"{Fore.LIGHTGREEN_EX}{5 * ' '}{text}{Style.RESET_ALL}\n"
          f"{Fore.LIGHTCYAN_EX}{Length_Char_inLine * '_'}{Style.RESET_ALL}")
    
def print_banner(message: str):
    line = "*" * len(message)
    print(f"{Fore.GREEN}{Style.BRIGHT}{line}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{line}{Style.RESET_ALL}")

def print_info(info:str):
    print(f"{Fore.LIGHTBLACK_EX}{Style.BRIGHT}ℹ️  {info}{Style.RESET_ALL}")

def load_geojson_file(geojson_file_path:str|Path)->list:
    """
    Loads a GeoJSON file and extracts the coordinates from its geometry.

    Args:
        geojson_file_path (str | Path): The file path to the GeoJSON file.

    Returns:
        list: A list of coordinates extracted from the GeoJSON file's geometry.

    Raises:
        FileNotFoundError: If the specified GeoJSON file does not exist.
        JSONDecodeError: If the file is not a valid GeoJSON format.
        KeyError: If the 'geometry' or 'coordinates' keys are missing in the GeoJSON data.
    """
    with open(geojson_file_path) as f:
            coords_list = gj.load(f)['geometry']['coordinates']
            f.close()
            return coords_list
        
# Function to generate a unique index from region name and coordinates
def assign_cell_id(cells: gpd.GeoDataFrame, 
                  source_column: str = 'Region', 
                  index_name: str = 'cell') -> gpd.GeoDataFrame:
    """
    Assigns unique cell IDs to each region in the specified GeoDataFrame.

    Parameters:
    cells (gpd.GeoDataFrame): Input GeoDataFrame containing spatial data with 'x' and 'y' coordinates.
    source_column (str): Column name in the GeoDataFrame that contains regional names.
    index_name (str): Name for the new index column to be created.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame with a new column of unique cell IDs for each region.
    """
    # Ensure the source column exists
    if source_column not in cells.columns:
        print(f"'{source_column}' does not exist in the GeoDataFrame.")
        raise ValueError(f"'{source_column}' does not exist in the GeoDataFrame.")

    # Remove spaces in the region names for consistency
    cells[source_column] = cells[source_column].str.replace(" ", "", regex=False)

    # Check if 'x' and 'y' coordinates exist
    if 'x' not in cells.columns or 'y' not in cells.columns:
        raise ValueError("Columns 'x' and 'y' must exist in the GeoDataFrame.")

    # Generate unique cell IDs using a combination of the region name and coordinates
    cells[index_name] = (
        cells.apply(
            lambda row: f"{row[source_column]}_{row['x']}_{row['y']}",
            axis=1
        )
    )

    # Set the index to the newly created column
    cells.set_index(index_name, inplace=True)

    return cells

def ensure_path(save_to: str | Path) -> Path:
    """
    Ensures that the given argument is a Path object. If the user provides a string,
    it converts it to a Path object to facilitate operations like directory creation.
    
    ## Args:
    - save_to (str | Path): The path input, either as a string or a Path object.

    ## Returns:
    - Path: The input converted (if necessary) to a Path object.
    """
    if not isinstance(save_to, Path):
        Warning(f">> Given instance for 'destination (save_to)' is of type: {type(save_to)}. Converting it to a Path")
        save_to = Path(save_to)
    return save_to


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

# print_update(level=2, message=f"{__name__}| ❌ ")
             
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
        return print_update(level=2, message=f"{__name__}| Dictionary datafile saved as '{save_to}")
    
def load_dict_datafile(json_file_path:str)->dict:
    with open(json_file_path,'r') as json_file:
        dictionary_:dict=json.load(json_file)
        return dictionary_

def check_LocalCopy_and_run_function(
        directory_path:str, 
        function_to_run:str, 
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
        print_update(level=2, message=f"{__name__}| Forcefully ran '{function_to_run.__name__}' on '{directory_path}'.")
        return output
    else:
        if not os.path.exists(directory_path):
            output=function_to_run()
            print_update(level=2, message=f"{__name__}| Directory '{directory_path}' created.")
            return output
        else:
            # log.info(f"Directory '{directory_path}' found locally.")
            print_update(level=2, message=f"{__name__}| Directory '{directory_path}' found locally.")

# Function to Load User Configuration File
def load_config(file_path):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data


def download_data(source_URL: str, file_path: str) -> str:
    """
    Downloads a file from a given URL and saves it to the specified file path.
    
    Parameters:
        source_URL (str): URL of the file to download.
        file_path (str): Path where the downloaded file will be saved.
    
    Returns:
        str: The file path if download is successful; otherwise, an instruction message.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Send HTTP GET request
        response = requests.get(source_URL, headers=headers, timeout=30)
        
        # Check if the request was successful
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print_update(level=3,message=f"{__name__}| ✔ File downloaded successfully and saved as {file_path}")
            return file_path
        else:
            print_update(level=2, message=f"{__name__}| ❌ Failed to download the file. Status code: {response.status_code}")
            return print_update(level=2, message=f"{__name__}| Please download the data manually from {source_URL} and save it to {file_path}")
    except requests.RequestException as e:
        print_update(level=3,message=f"{__name__}| ❌ An error occurred while downloading the file: {e}")
        return print_update(level=2, message=f"{__name__}| Please download the data manually from {source_URL} and save it to {file_path}")


def load_raster_file(raster_path:str|Path)->np.ndarray :
    raster_path=Path(raster_path)
    with rio.open(raster_path) as f:
        raster_data: np.ndarray = f.read(1)
        f.close()  
    return raster_data

