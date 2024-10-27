
# %% [markdown]
# * GLobal

# %%
from dataclasses import dataclass, field
import os
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import logging


from linkingtool.AttributesParser import AttributesParser


@dataclass
class CODERSData(AttributesParser):
    def __post_init__(self):
     # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()

        # Load CODERS data config
        self.coders_data_config = self.config.get('CODERS', {})
        self.url = self.coders_data_config.get('url_1', '')
        self.api_elias = self.coders_data_config.get('api_key', {}).get('Elias', '')
        self.query = f"?key={self.api_elias}"
        self.data_pull = self.coders_data_config.get('data_pull', {})
        self.table_list = list(self.coders_data_config['data_pull'].keys())

    def is_table_name_required(self, table_name: str):
        if table_name in self.table_list:
            return True
        
    def fetch_data(self, 
                   table_name: str) -> pd.DataFrame:
        """
        Fetch data from the CODERS API for the specified table.
        """
        response = requests.get(f"{self.url}/{table_name}{self.query}")
        
        if response.status_code == 200:
            return pd.DataFrame.from_dict(response.json())
        else:
            raise RuntimeError(f">> Error fetching data for {table_name}: {response.status_code}")

    def load_local_data(self, 
                        table_name: str, 
                        province_code: str = None) -> pd.DataFrame:
        
        """
        Load data from a local file if it exists.
        
        ### Args:
            If Province set to NONE, loads Country data.
        """

        file_name = f"{table_name}.pkl" if province_code is None else f"generators_{province_code}.pkl"
        file_path = Path(self.data_pull['root']) / self.data_pull.get(table_name)/file_name
        file_path.mkdir(parents=True, exist_ok=True)  # Creates parent directories if not exists.

        if file_path.is_file():
            self.log.info(f">> Loading data from local file: {file_path}")
            return pd.read_pickle(file_path)
        else:
            self.log.warning(f">> No local file found at: {file_path}")
            return None  # Return None if the file does not exist

    def save_data(self, 
                  data: pd.DataFrame|gpd.GeoDataFrame, 
                  table_name: str, 
                  province_code: str = None):
        
        """Save the fetched data to a pkl file."""
        file_name = f"{table_name}.pkl" if province_code is None else f"generators_{province_code}.pkl"
        file_path = Path(self.data_pull['root']) / self.data_pull.get(table_name)/file_name
        
        data.to_pickle(file_path)
        self.log.info(f"{table_name} data saved to:\n {file_path}")

    def create_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame from the given DataFrame."""
        df = df.copy()
        # Create a geometry column
        df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf

    def get_table_canada(self, table_name: str, force_update: bool = False):
        """Get generator data for all of Canada.
        
        Args:
            table_name (str): The name of the table to fetch data from.
            force_update (bool): If True, force a data fetch from the API, ignoring local data.

        Returns:
            Tuple[pd.DataFrame, gpd.GeoDataFrame]: The generator data as a DataFrame and as a GeoDataFrame.
        """
                   
        file_path = Path(self.data_pull['root']) / self.data_pull.get(table_name, f"{table_name}.pkl")
        file_path.mkdir(parents=True, exist_ok=True)  # Creates parent directories if not exists.
        
        # Check if the data file exists locally and if force_update is not set
        if file_path.is_file() and not force_update:
            data = pd.read_pickle(file_path)  # Load from local CSV
            self.log.info(f"Loaded {table_name} data from local file: {file_path}")
        else:
            # Fetch data from API if not found locally or if force_update is set
            data = self.fetch_data(table_name)
            self.log.info(f">> Data pulled {table_name} from [source checked: CODERS(https://sesit.dev/api/docs)]")
            self.save_data(data, table_name)
            
        df=data
        
        # Check if table_name contains "lines"; if it does, skip creating the GeoDataFrame
        if "lines" not in table_name:
            gdf = self.create_gdf(data)  # Only create GeoDataFrame if "lines" is not in table_name
        else:
            gdf = gpd.GeoDataFrame()  # Or however you wish to handle this case
        return df,gdf


    def get_table_provincial(self, 
                             table_name, 
                             force_update: bool = False):
        """Get generator data for a specific province.
        
        Args:
            province_code (str): The province code to filter the data.
            table_name (str): The name of the table to fetch data from e.g. 'substations','transmission_lines','generators' etc.
            force_update (bool): If True, force a data fetch from the API, ignoring local data.
        """
        if self.is_table_name_required(table_name): #check if the data is required
            if self.province_code_validity:
            
                # Get Canadian data first
                df,gdf= self.get_table_canada(table_name, force_update=force_update)
            

            if "lines" not in table_name:
                # Apply provincial mask
                data=gdf
            else:
                data=df
            
            province_mask = data['province'] == self.province_short_code
            self.province_data = data[province_mask]
            
            if not self.province_data.empty:
                
                return self.province_data  # Return the filtered GeoDataFrame
            else:
                return self.province_code_validity
        else:
                self.log.warning(f"Table: '{table_name}' is not required for this tool and is not configured to work properly.\n Configured/required tables >>>> {self.table_list[1:]}")
                

