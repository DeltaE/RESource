import pandas as pd
import geopandas as gpd
import h5py
from shapely.wkt import loads, dumps
from shapely.geometry.base import BaseGeometry
from pathlib import Path
import warnings

class DataHandler:
    def __init__(self,
                 hdf_file_path:Path=None):
        """
        Initialize the DataHandler with the file path.

        :param store: Path to the HDF5 file.
        """
        try:
            if hdf_file_path is None:
                warnings.warn(">> Store has not been set during initialization. Please define the store path during applying DataHandler methods")
            else:
                self.store = Path(hdf_file_path)
                # print(f">> Store initialized with the given path: {hdf_file_path}")
                
        except Exception as e:
            warnings.warn(f"Error reading file: {e}")


    def to_store(self,
                data: pd.DataFrame | gpd.GeoDataFrame, 
                key: str,
                hdf_file_path: Path = None,
                force_update: bool = False):
        """
        Save the DataFrame or GeoDataFrame to an HDF5 file.
        Automatically converts any geometry-type columns to WKT strings.
        Updates existing data, adds new columns if needed.

        :param data: The DataFrame or GeoDataFrame to save.
        :param key: Key for saving the DataFrame to the HDF5 file.
        :param hdf_file_path: Optional path to the HDF5 file.
        :param force_update: If True, overwrite the existing data at the given key.
        """
        if hdf_file_path is not None:
            self.store = Path(hdf_file_path)

        if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            raise TypeError(">> to be stored 'data' must be a DataFrame or GeoDataFrame.")

        self.data_new = data.copy()

        # Convert all geometry-type columns to WKT
        for col in self.data_new.columns:
            if self.data_new[col].apply(lambda x: isinstance(x, BaseGeometry)).any():
                print(f">> Converting geometry column '{col}' to WKT")
                self.data_new[col] = self.data_new[col].apply(lambda x: dumps(x) if isinstance(x, BaseGeometry) else x)

        store = pd.HDFStore(self.store, mode='a')

        try:
            if key not in store or force_update:
                self.data_new.to_hdf(self.store, key=key)
                print(f">> Data saved to {self.store} with key '{key}'")
            else:
                self.data_ext = store.get(key)

                # Merge: update overlapping data and add new columns
                self.data_ext.update(self.data_new)  # updates overlapping values

                # Add any new columns from new data
                new_cols = self.data_new.columns.difference(self.data_ext.columns)
                for col in new_cols:
                    self.data_ext[col] = self.data_new[col]

                # Save the updated + extended DataFrame
                self.data_ext.to_hdf(self.store, key=key)
                print(f">> Updated and extended '{key}' saved to {self.store}")

        finally:
            store.close()





    def from_store(self, 
                   key: str):
        """
        Load data from the HDF5 store and handle geometry conversion.
        
        :param key: Key for loading the DataFrame or GeoDataFrame.
        :return: DataFrame or GeoDataFrame based on the data loaded.
        """
        
        with pd.HDFStore(self.store, 'r') as store:
            if key not in store:
                print(f"Error: Key '{key}' not found in {self.store}")
                return None

            # Load the data
            self.data = pd.read_hdf(self.store, key)

            # Rename 'geometry' back to 'geometry' and convert WKT to geometry if applicable
            if 'geometry' in self.data.columns :
                if not isinstance(self.data['geometry'].iloc[0], BaseGeometry):
                    self.data['geometry'] = self.data['geometry'].apply(loads)
                return gpd.GeoDataFrame(self.data, geometry='geometry', crs='EPSG:4326')

            # If not geometry, return the regular DataFrame
            if key == 'timeseries':
                print(">>> 'timeseries' key access suggestions: use '.solar' to access Solar-timeseries and '.wind' for Wind-timeseries.")
            
            return self.data


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
                print(f"{indent}[Group] {name}")
                # Iterate through the group's keys and call recursively
                for sub_key in obj.keys():
                    print_structure(f"{name}/{sub_key}", obj[sub_key], indent + "  └─ ")
            elif show_dataset and isinstance(obj, h5py.Dataset):
                print(f"{indent}[Dataset] {name} - Shape: {obj.shape}, Type: {obj.dtype}")

        try:
            with h5py.File(store_path, 'r') as f:
                print(f"Structure of HDF5 file: {store_path}")
                for key in f.keys():
                    print_structure(key, f[key])
        except Exception as e:
            print(f"Error reading file: {e}")
            
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

