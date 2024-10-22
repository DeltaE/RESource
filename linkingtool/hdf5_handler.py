import pandas as pd
import geopandas as gpd
import h5py
from shapely.wkt import loads, dumps
from shapely.geometry.base import BaseGeometry
from linkingtool.AttributesParser import AttributesParser

class DataHandler(AttributesParser):
    def __init__(self, store: str):
        """
        Initialize the DataHandler with the file path.

        :param store_path: Path to the HDF5 file.
        """
        super().__init__()
        self.store = store
    
    def show(self):
        """
        Display all keys (groups) in the HDF5 store.
        """
        try:
            with h5py.File(self.store, 'r') as f:
                print(f"Data 'keys' found in store {self.store}:")
                for key in f.keys():
                    print(f"- {key}")
        except Exception as e:
            self.log.error(f"Error opening the store: {e}")

    def to_store(self, data: pd.DataFrame | gpd.GeoDataFrame, key: str, force_update: bool = False):
        """
        Save the DataFrame or GeoDataFrame to an HDF5 file.

        :param data: The DataFrame or GeoDataFrame to save.
        :param key: Key for saving the DataFrame to the HDF5 file.
        """
        self.data_new = data.copy()
        store = pd.HDFStore(self.store, mode='a')  # Open store in append mode ('a')

        try:
<<<<<<< HEAD
            if key not in store or force_update:
                # Handle GeoDataFrame geometry if present
                if 'geometry' in self.data_new.columns:
                    if isinstance(self.data_new['geometry'].iloc[0], BaseGeometry):
                        self.data_new['geometry'] = self.data_new['geometry'].apply(dumps)

                # Save the modified data to HDF5
                self.data_new.to_hdf(self.store, key=key)
                self.log.info(f"Data (GeoDataFrame/DataFrame) saved to {self.store} with key '{key}'")
            else:
                # Read existing data from HDF5
                self.data_ext = store.get(key)

                # Add new columns to the existing DataFrame if not present
                for column in self.data_new.columns:
                    if column not in self.data_ext.columns:
                        self.data_ext[column] = self.data_new[column]
=======
            if 'geometry' in self.data.columns:
                
                if isinstance(self.data['geometry'].iloc[0], BaseGeometry):
                    # Convert the geometry to WKT format before saving
                    self.data['geometry'] = self.data['geometry'].apply(dumps)
                else:
                    pass
                # Save to HDF5
                self.data.to_hdf(self.store, key=key) # mode='w', format='table', index=False
                self.log.info(f"Data (GeoDataFrame) saved to {self.store} with key '{key}'")
                
            else :
                self.data.to_hdf(self.store, key=key) # , mode='w', format='table', index=False
                self.log.info(f"Data (DataFrame) saved to {self.store} with key '{key}'")
>>>>>>> beb6b426000d0e551bb15eab82f64341cb038acf

                # Update the existing DataFrame in HDF5
                self.updated_data = self.data_ext
                self.updated_data.to_hdf(self.store, key=key)
                self.log.info(f"Updated data saved to {self.store} with key '{key}'")
        
        finally:
            store.close()

    def from_store(self, key: str):
        """
        Load data from the HDF5 store and handle geometry conversion.
        
        :param key: Key for loading the DataFrame or GeoDataFrame.
        :return: DataFrame or GeoDataFrame based on the data loaded.
        """
        # try:
        with pd.HDFStore(self.store, 'r') as store:
            if key not in store:
                self.log.error(f"Key '{key}' not found in {self.store}")
                return None
<<<<<<< HEAD

            # Load the data
            self.data = pd.read_hdf(self.store, key)

            # Rename 'geometry_wkt' back to 'geometry' and convert WKT to geometry
            if 'geometry' in self.data.columns:
                self.data['geometry'] = self.data['geometry'].apply(loads)
                return gpd.GeoDataFrame(self.data, geometry='geometry', crs=self.get_default_crs())

            # If not geometry, return the regular DataFrame
            if key == 'timeseries':
                print(f">>> 'timeseries' key access suggestions: use '.solar' to access Solar-timeseries and '.wind' for Wind-timeseries.")
            
            return self.data

        # finally:
        #     store.close()
=======
            
            self.data= pd.read_hdf(self.store, key)
            
            if 'geometry' in self.data.columns:

                data = store[key]
                data['geometry'] = data['geometry'].apply(loads)
                return gpd.GeoDataFrame(data, geometry='geometry', crs=self.get_default_crs())
            
            elif isinstance(self.data.index, pd.DatetimeIndex):
                return self.data.tz_localize(None)
            
            else:
                
                return self.data
>>>>>>> beb6b426000d0e551bb15eab82f64341cb038acf

