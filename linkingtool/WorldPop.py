# Downloads population data from WorldPop

import linkingtool.linking_utility as utils
import pandas as pd
from shapely.geometry import box
import geopandas as gpd
from linkingtool.AttributesParser import AttributesParser
from linkingtool.boundaries import GADMData
from pathlib import Path
from linkingtool.


class WorldPop():
    def __init__(self,config_file_path:Path,province_short_code:str):
        
        self.config_file_path=config_file_path
        self.province_short_code=province_short_code
        
        self.attributes_parser:AttributesParser=AttributesParser(self.config_file_path,None)
        self.gadm=GADMData(self.config_file_path,self.province_short_code)
        
        self.config=self.attributes_parser.config
        self.worldpop_config=self.config['WorldPop']
        self.root=self.worldpop_config['root']
        
        
    def pull_data(self,data_name:str):

        data_names= list(self.worldpop_config['source'].keys())
        
        if data_name in data_names:
            url=Path (self.worldpop_config['source'][data_name])
            # Extract the filename from the URL
            filename = Path(url).name
            # Construct the full save path by combining base path and extracted filename
            file_path = Path(self.root) / filename

            # Download the file using the utils.download_data function
            utils.download_data(url, file_path)
            
            # >>>>> files are downloaded as zip ! create a zip extractor
            self.pop_data:pd.DataFrame=pd.read_csv(file_path)
            
            print(f"File saved to: {file_path}")
        else:
            print(f"{data_name} associated source information not found in user config. \n") 
            print("Please provide required information in user config under 'WorldPop' key.")
            print (f"Available 'data_name' in user config is {data_names}")
            
    def get_provincial_data(self,data_name:str):
        
            province_gadm_gdf=self.gadm.get_province_boundary()
            
            if Path 
            pop_grid= self.pop_data.overlay(province_gadm_gdf, how='intersection', keep_geom_type=True)
            
            pop_grid.to_pickle(f'data/downloaded_data/WorldPop/pop_{self.province_short_code}.pkl')