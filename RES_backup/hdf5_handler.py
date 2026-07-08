import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import h5py
import pandas as pd
from colorama import Fore, Style
from shapely.geometry.base import BaseGeometry
from shapely.wkt import dumps, loads

import RES.utility as utils


class DataHandler:
    """HDF5-based data storage manager for geospatial renewable energy datasets.
    
    Provides efficient storage and retrieval of large DataFrame and GeoDataFrame
    datasets using HDF5 format. Handles geometry serialization for spatial data
    and implements caching mechanisms for workflow optimization.
    
    Parameters
    ----------
    hdf_file_path : Path
        Path to the HDF5 storage file
    silent_initiation : bool, default True
        Suppress initialization messages
    show_structure : bool, default False
        Display HDF5 file structure on initialization
        
    Attributes
    ----------
    store : Path
        Path to the HDF5 storage file
    geom_columns : list
        Column names containing geometry data for special handling
    """
    def __init__(self, hdf_file_path: Path = None, silent_initiation: Optional[bool] = True, 
                 show_structure: Optional[bool] = False):
        """Initialize HDF5 data handler.
        
        Args:
            hdf_file_path: Path to HDF5 storage file
            silent_initiation: Suppress initialization messages
            show_structure: Display file structure after initialization
        """
        try:
            if hdf_file_path is None:
                warnings.warn("Store path not set during initialization. Define store path when calling methods.")
            else:
                self.store = Path(hdf_file_path)
                self.geom_columns = ['geometry', 'nearest_connection_point','centroid']
                if not silent_initiation:
                    utils.print_update(level=2, message=f"Store initialized: {hdf_file_path}")
                if show_structure:
                    self.show_tree(self.store)
                
        except Exception as e:
            warnings.warn(f"Error initializing data handler: {e}")
               
    def to_store(self, data: pd.DataFrame, key: str, hdf_file_path: Path = None, 
                 force_update: bool = False):
        """Save DataFrame or GeoDataFrame to HDF5 storage.
        
        Handles geometry serialization for spatial data and implements
        intelligent updates to avoid data duplication.
        
        Args:
            data: DataFrame or GeoDataFrame to store
            key: Storage key identifier
            hdf_file_path: Optional override for storage file path
            force_update: Force overwrite of existing data
            
        Raises:
            TypeError: If data is not a DataFrame or GeoDataFrame
        """
        
        if hdf_file_path is not None:
            self.store = Path(hdf_file_path)
        
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            self.data_new = data.copy()
        # Proceed with saving to HDF5
        else:
            raise TypeError(f"{__name__}| ❌ to be stored 'data' must be a DataFrame or GeoDataFrame.")

        store = pd.HDFStore(self.store, mode='a')  # Open store in append mode ('a')

        try:
            if key not in store or force_update:
                # Handle GeoDataFrame geometry if present
                if 'geometry' in self.data_new.columns:
                    if isinstance(self.data_new['geometry'].iloc[0], BaseGeometry):
                        self.data_new['geometry'] = self.data_new['geometry'].apply(dumps)
                if 'nearest_connection_point' in self.data_new.columns:
                    if isinstance(self.data_new['nearest_connection_point'].iloc[0], BaseGeometry):
                        self.data_new['nearest_connection_point'] = self.data_new['nearest_connection_point'].apply(dumps)
                # Save the modified data to HDF5
                self.data_new.to_hdf(self.store, key=key)
                utils.print_update(level=3,message=f"{__name__}|💾 Data (GeoDataFrame/DataFrame) saved to {self.store} with key '{key}'")
            else:
                # Read existing data from HDF5
                self.data_ext = store.get(key)

                # Add new columns to the existing DataFrame if not present
                for column in self.data_new.columns:
                    if not data.empty and column not in self.data_ext.columns:
                        self.data_ext[column] = self.data_new[column]

                # Update the existing DataFrame in HDF5
                self.updated_data = self.data_ext
                for geom_col in self.geom_columns:
                    if geom_col in self.updated_data.columns:
                        if isinstance(self.updated_data[geom_col].iloc[0], BaseGeometry):
                            self.updated_data[geom_col] = self.updated_data[geom_col].apply(dumps)
                            utils.print_update(level=4,message=f"{__name__}| 💾 Updated key :'{key}' with column: '{geom_col}'")
                    

                self.updated_data.to_hdf(self.store, key=key)
                utils.print_update(level=3,message=f"{__name__}| 💾 Updated '{key}' saved to {self.store} with key '{key}'")
        
        finally:
            store.close()

    def from_store(self, key: str):
        """Load data from HDF5 storage with geometry reconstruction.
        
        Automatically handles geometry deserialization for spatial datasets
        and returns appropriate DataFrame or GeoDataFrame objects.
        
        Args:
            key: Storage key identifier
            
        Returns:
            DataFrame or GeoDataFrame with reconstructed geometry columns
            
        Raises:
            KeyError: If storage key is not found
        """
        try:
            with pd.HDFStore(self.store, 'r') as store:
                if key not in store:
                    utils.print_update(level=3,message=f"{__name__}| ❌ Error: Key '{key}' not found in {self.store}")
                    return None

                # Load the data
                self.data = pd.read_hdf(self.store, key)

                # Rename 'geometry' back to 'geometry' and convert WKT to geometry if applicable
                for geom_col in self.geom_columns:
                    if geom_col in self.data.columns:
                        self.data[geom_col] = self.data[geom_col].apply(loads)
                        return gpd.GeoDataFrame(self.data, geometry='geometry', crs='EPSG:4326')

                # If not geometry, return the regular DataFrame
                if key == 'timeseries':
                    utils.print_info({__name__}|"'timeseries' key access suggestions: use '.solar' to access Solar-timeseries and '.wind' for Wind-timeseries.")
                
                return self.data
        except Exception as e:
            utils.print_update(level=3,message=f"{__name__}| ❌ Error loading data from store: {e}")
            return None

    def refresh(self):
        """
        Initialize a new DataHandler instance with the current store path.
        This method is useful for reloading the DataHandler with the same store path without needing to reinitialize the entire class.
        
        Parameters:
            None
            
        Returns:
            DataHandler: A new instance of DataHandler with the same store path.
        """
        return DataHandler(self.store, silent_initiation=True, show_structure=False)
    
    @staticmethod
    def show_tree(store_path,
                  show_dataset:bool=False):
        """
        This method provides a structured view of the keys and datasets within the HDF5 file, allowing users to understand its organization.
        
        parameters:
            store_path (Path): Path to the HDF5 file.   
            show_dataset (bool): If True, also show datasets within the groups.
        Raises:
            Exception: If there is an error reading the file.
        Returns:
            None: This method prints the structure to the console.

        """
        def print_structure(name, obj, indent=""):
            """Helper function to recursively print the structure."""
            if isinstance(obj, h5py.Group):
                print(f"{indent}{Fore.LIGHTBLUE_EX}[key]{Style.RESET_ALL} {Fore.LIGHTGREEN_EX}{name}{Style.RESET_ALL}")
                # Iterate through the group's keys and call recursively
                for sub_key in obj.keys():
                    print_structure(f"{name}/{sub_key}", obj[sub_key], indent + "  └─ ")
            elif show_dataset and isinstance(obj, h5py.Dataset):
                print(f"{indent}[Dataset] {name} - Shape: {obj.shape}, Type: {obj.dtype}")

        try:
            with h5py.File(store_path, 'r') as f:
                utils.print_module_title(f"{__name__}|🗄️ Structure of HDF5 file: {store_path}")
                for key in f.keys():
                    print_structure(key, f[key])
                print("\n")
                utils.print_update(level=1,message="To access the data : ")
                utils.print_update(level=2,message="<datahandler instance>.from_store('<key>')")
        except Exception as e:
            utils.print_update(message=f"{__name__}|  ❌ Error reading file: {e}",alert=True)

    def clean_store(self):
        """
        Cleans the HDF5 store by removing all keys and datasets.
        
        Parameters:
            None
        """
        # Remove all keys and datasets from the HDF5 store
        with h5py.File(self.store, "a") as hdf_file:
            keys_to_delete = list(hdf_file.keys())
            for key in keys_to_delete:
                del hdf_file[key]
        utils.print_update(level=3, message=f"{__name__}|🗑️ All keys have been deleted from the store: {self.store}")
    
    @staticmethod
    def del_key(store_path,
                key_to_delete:str):
        
        """
        Deletes a specific key from the HDF5 file.
        
        Parameters:

            store_path (Path): Path to the HDF5 file.
            key_to_delete (str): The key to delete from the HDF5 file.
        Raises:
            KeyError: If the key does not exist in the HDF5 file.
        Returns:        
            None: This method prints the status of the deletion operation.
        Example:
            >>> DataHandler.del_key(Path('data.h5'), 'my_key')
            This will delete 'my_key' from the 'data.h5' file if it exists
        """
        
        # Open the HDF5 file in read/write mode
        with h5py.File(store_path, "r+") as hdf_file:
            # Check if the key exists in the file
            if key_to_delete in hdf_file:
                del hdf_file[key_to_delete]
                utils.print_update(level=3,message=f"{__name__}|Key '{key_to_delete}' has been deleted.Store status:\n")
                DataHandler(store_path).show_tree(store_path)
            else:
                utils.print_update(level=3,message=f"{__name__}|Key '{key_to_delete}' not found in the file. Store status:\n")
                DataHandler(store_path).show_tree(store_path)

