import pandas as pd
import geopandas as gpd
import h5py
from shapely.wkt import loads, dumps
from shapely.geometry.base import BaseGeometry
from linkingtool.AttributesParser import AttributesParser

class DataHandler(AttributesParser):
    def __init__(self,
                 store:str):
        """
        Initialize the DataHandler with the file path.

        :param store_path: Path to the HDF5 file.
        """
        super().__init__() 
        self.store=store
    
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
        
    def to_store(self,
                    
                    data: pd.DataFrame | gpd.GeoDataFrame, 
                    key: str):
        """
        Save the DataFrame or GeoDataFrame to an HDF5 file.

        :param data: The DataFrame or GeoDataFrame to save.
        :param key: Key for saving the DataFrame to the HDF5 file.
        """
        self.data = data
        try:
            if 'geomtery' in self.data.columns:
                
                if isinstance(self.data['geometry'].iloc[0], BaseGeometry):
                    # Convert the geometry to WKT format before saving
                    self.data['geometry'] = self.data['geometry'].apply(dumps)
                else:
                    pass
                # Save to HDF5
                self.data.to_hdf(self.store, key=key) # mode='w', format='table', index=False
                self.log.info(f"Data (GeoDataFrame) saved to {self.store} with key '{key}'")
                
            elif key=='timeseries':
                self.data.to_hdf(self.store, key=key) # , mode='w', format='table', index=False
                self.log.info(f"Data (DataFrame) saved to {self.store} with key '{key}'")
            else :
                self.data.to_hdf(self.store, key=key) # , mode='w', format='table', index=False
                self.log.info(f"Data (DataFrame) saved to {self.store} with key '{key}'")

        except (KeyError, FileNotFoundError) as e:
            self.log.error(f"Error saving data: {e}")
            return None
        
    def from_store(self, key: str):
        with pd.HDFStore(self.store, 'r') as store:
            if key not in store:
                self.log.error(f"Key '{key}' not found in {self.store}")
                return None
            
            columns = store.get_storer('your_data').attrs.data_columns
            
            self.data= pd.read_hdf(self.store, key)
            
            if 'geomtery' in self.data.columns:

                data = store[key]
                data['geometry'] = data['geometry'].apply(loads)
                return gpd.GeoDataFrame(data, geometry='geometry', crs=self.get_default_crs())
            else:
                return 

        # except (KeyError, FileNotFoundError, TypeError) as e:
        #     self.log.error(f"Error loading data: {e}")
        #     return None

