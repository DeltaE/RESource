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
                warnings.warn(f">> Store has not been set during initialization. Please define the store path during applying DataHandler methods")
            else:
                self.store = Path(hdf_file_path)
                # print(f">> Store initialized with the given path: {hdf_file_path}")
                
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
                if 'geometry' in self.data_new.columns:
                    if isinstance(self.data_new['geometry'].iloc[0], BaseGeometry):
                        self.data_new['geometry'] = self.data_new['geometry'].apply(dumps)

                # Save the modified data to HDF5
                self.data_new.to_hdf(self.store, key=key)
                print(f">> Data (GeoDataFrame/DataFrame) saved to {self.store} with key '{key}'")
            else:
                # Read existing data from HDF5
                self.data_ext = store.get(key)

                
                # Ensure columns are unique before updating
                self.data_new = self.data_new.loc[:, ~self.data_new.columns.duplicated()]

                # Align indices to ensure proper updates
                self.data_new = self.data_new.reindex(self.data_ext.index, fill_value=None)

                # Update only existing columns
                self.data_ext.update(self.data_new)

                # Save updated data
                self.data_ext.to_hdf(self.store, key=key)
                print(f">> Updated '{key}' saved to {self.store}")

        
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

