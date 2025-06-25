import pandas as pd
import geopandas as gpd
import h5py
from shapely.wkt import loads, dumps
from shapely.wkt import dumps as wkt_dumps
from shapely.geometry import base
from shapely.geometry.base import BaseGeometry
from pathlib import Path
import warnings
from colorama import Fore, Style
from typing import Optional
import RES.utility as utils
import tables

class DataHandler:
    def __init__(self,
                 hdf_file_path:Path=None,
                 silent_initiation:Optional[bool]=True,
                 show_structure:Optional[bool]=False):
        """
        Initialize the DataHandler with the file path.

        :param store: Path to the HDF5 file.
        """
        try:
            if hdf_file_path is None:
                warnings.warn(f">> Store has not been set during initialization. Please define the store path during applying DataHandler methods")
            else:
                self.store = Path(hdf_file_path)
                if not silent_initiation:
                    utils.print_update(level=1,message=f">> Store initialized with the given path: {hdf_file_path}")
                if show_structure:
                    self.show_tree(self.store)
                
        except Exception as e:
            warnings.warn(f"Error reading file: {e}")
               
    def to_store(self,
                 data: pd.DataFrame | gpd.GeoDataFrame, 
                 key: str,
                 hdf_file_path:Path=None,
                 force_update: bool = False):
        """
        Save the DataFrame or GeoDataFrame to an HDF5 file.

        :param data: The DataFrame or GeoDataFrame to save.
        :param key: Key for saving the DataFrame to the HDF5 file.
        :param force_update: If True, force update the data even if it exists.
        """
        if hdf_file_path is not None:
            self.store = Path(hdf_file_path)
        
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            self.data_new = data.copy()
        # Proceed with saving to HDF5
        else:
            raise TypeError(">> to be stored 'data' must be a DataFrame or GeoDataFrame.")

        store = pd.HDFStore(self.store, mode='a')  # Open store in append mode ('a')

        try:
            if key not in store or force_update:
                # Handle GeoDataFrame geometry if present
                # if 'geometry' in self.data_new.columns:
                #     if isinstance(self.data_new['geometry'].iloc[0], BaseGeometry):
                #         self.data_new['geometry'] = self.data_new['geometry'].apply(dumps)

                # Convert any columns containing geometry objects to WKT strings
                for col in self.data_new.columns:
                    if self.data_new[col].apply(lambda x: isinstance(x, BaseGeometry)).any():
                        self.data_new[col] = self.data_new[col].apply(lambda x: wkt_dumps(x) if isinstance(x, BaseGeometry) else str(x))
                    elif self.data_new[col].apply(lambda x: isinstance(x, str) and x.startswith('POINT')).any():
                        # Likely a geometry string: cast to plain string to avoid PyTables issues
                        self.data_new[col] = self.data_new[col].astype(str)

                # Save the modified data to HDF5
                self.data_new.to_hdf(self.store, key=key)
                print(f">> Data (GeoDataFrame/DataFrame) saved to {self.store} with key '{key}'")
            else:
                # Read existing data from HDF5
                self.data_ext = store.get(key)

                # Add new columns to the existing DataFrame if not present
                for column in self.data_new.columns:
                    if not data.empty and column not in self.data_ext.columns:
                        self.data_ext[column] = self.data_new[column]

                for col in self.data_ext.columns:
                    if self.data_ext[col].apply(lambda x: isinstance(x, BaseGeometry)).any():
                        self.data_ext[col] = self.data_ext[col].apply(lambda x: wkt_dumps(x) if isinstance(x, BaseGeometry) else str(x))
                    elif self.data_ext[col].apply(lambda x: isinstance(x, str) and x.startswith('POINT')).any():
                        # Likely a geometry string: cast to plain string to avoid PyTables issues
                        self.data_ext[col] = self.data_ext[col].astype(str)
                
                # Update the existing DataFrame in HDF5
                self.updated_data = self.data_ext
                self.updated_data.to_hdf(self.store, key=key)
                print(f">> Updated '{key}' saved to {self.store} with key '{key}'")
        
        finally:
            store.close()


    def from_store(self, key: str):
        """
        Load data from the HDF5 store and handle geometry conversion.
        Cleans and removes corrupted keys if loading fails due to attribute errors.
        """
        import tables

        try:
            with pd.HDFStore(self.store, 'r') as store:
                if key not in store:
                    print(f"Error: Key '{key}' not found in {self.store}")
                    return None

                self.data = pd.read_hdf(self.store, key)

        except (AttributeError, tables.exceptions.HDF5ExtError) as e:
            print(f">> Warning: Failed to load key '{key}' due to corruption: {e}")
            # Attempt to delete the corrupted key
            with pd.HDFStore(self.store, 'a') as store:
                try:
                    store.remove(key)
                    print(f">> Corrupted key '{key}' removed from {self.store}")
                except Exception as cleanup_err:
                    print(f">> Cleanup failed: could not remove key '{key}': {cleanup_err}")
            return None

        # Convert WKT to geometry objects in any column that looks like WKT
        for col in self.data.columns:
            if self.data[col].dtype == object:
                if self.data[col].apply(lambda x: isinstance(x, str) and x.strip().startswith(('POINT', 'LINESTRING', 'POLYGON', 'MULTIPOLYGON'))).any():
                    try:
                        self.data[col] = self.data[col].apply(lambda x: loads(x) if isinstance(x, str) else x)
                    except Exception as geom_err:
                        print(f">> Warning: Failed to convert '{col}' to geometry: {geom_err}")

        if 'geometry' in self.data.columns and self.data['geometry'].apply(lambda x: isinstance(x, BaseGeometry)).any():
            return gpd.GeoDataFrame(self.data, geometry='geometry', crs='EPSG:4326')

        if key == 'timeseries':
            print(f">>> 'timeseries' key access suggestions: use '.solar' to access Solar-timeseries and '.wind' for Wind-timeseries.")

        return self.data




    def refresh(self):
         return DataHandler(self.store, silent_initiation=True, show_structure=False)
     
    @staticmethod
    def show_tree(store_path,
                  show_dataset:bool=False):
        """
        Recursively print the hierarchy of an HDF5 file.

        :param file_path: Path to the HDF5 file.
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
                utils.print_module_title(f"Structure of HDF5 file: {store_path}")
                for key in f.keys():
                    print_structure(key, f[key])
                print("\n")
                utils.print_update(level=1,message="To access the data : ")
                utils.print_update(level=2,message="<datahandler instance>.from_store('<key>')")
        except Exception as e:
            utils.print_update(message=f"Error reading file: {e}",alert=True)

            
    @staticmethod
    def del_key(store_path,
                key_to_delete:str):
        # Open the HDF5 file in read/write mode
        with h5py.File(store_path, "r+") as hdf_file:
            # Check if the key exists in the file
            if key_to_delete in hdf_file:
                del hdf_file[key_to_delete]
                print(f"Key '{key_to_delete}' has been deleted.Store status:\n")
                DataHandler(store_path).show_tree(store_path)
            else:
                print(f"Key '{key_to_delete}' not found in the file. Store status:\n")
                DataHandler(store_path).show_tree(store_path)
