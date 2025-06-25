"""
# Key Changes and Benefits over v1
    The @dataclass decorator simplifies class creation and automatically generates the __init__, __repr__, and other methods.

## Field Initialization:
    Attributes that require processing during initialization (like reading configurations) are defined with init=False and processed in the __post_init__ method.

## Default Values:
    The resource_type has a default value specified directly in the field declaration, which simplifies the __init__ method.

## Type Annotations:
    Type hints enhance code readability and help with type checking tools.
"""

import pandas as pd
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import logging as log

# Logging Configuration
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class AttributesParser:
    """
    This is the parent class that will extract the core attributes from the User Config file.
    """
    # Attributes that are required as Args.
    region_short_code: str
    config_file_path: Path =field(default='config/config.yaml')
    resource_type: str = field(default='None')
    
    def __post_init__(self):
        self.site_index='cell'

        # Define the path and filename
        self.store = Path(f"data/store/resources_{self.region_short_code}.h5")
        self.store.parent.mkdir(parents=True, exist_ok=True)

        # Convert region_short_code to uppercase to handle user types regarding case-sensitive letter inputs.
        self.region_short_code = self.region_short_code.upper()
        
        # Load the user configuration master file by using the method
        self.config:Dict[str,dict] = self.load_config(self.config_file_path)
        self.disaggregation_config:Dict[str,dict] = self.config['capacity_disaggregation']
        self.region_code_validity=self.is_region_code_valid()
        self.log = log.getLogger(__name__)
        
    def load_config(self,config_file_path):
        """ 
        Loads the yaml file as dictionary and extracts the attributes to pass on child classes. 
        """
        with open(config_file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def is_region_code_valid(self)-> bool:
        """
        Args:
            region_short_code: 2 letter short code of the region.
            
        Description: 
            Checks of the region code is correct. If not, then suggests the available list of codes that are liked to data supply-chain.
        """
        self.region_mapping=self.get_region_mapping()
        
        if self.region_short_code not in self.region_mapping:
            print(f"!!! ERROR !!! \nRecheck the region code.\n{60 * '_'}")
            print("\nPlease provide a Region CODE from the following list: \n ")
            # display(self.region_mapping.keys())
            for key, value in self.region_mapping.items():
                # Assuming you want to show the first item in the value (e.g., the first name or detail)
                name = value.get('name', 'N/A')  # Change 'name' to the actual key you want to display
                print(f"• {key}: {name}")
            return False  # Exit the function if the region code is invalid
        else:
            return True

    def load_snapshot(self)->tuple:
        start_date = self.config.get('cutout', {}).get('snapshots', {}).get('start', [[]])[0]
        end_date = self.config.get('cutout', {}).get('snapshots', {}).get('end', [[]])[0]
        return start_date, end_date

    
    

# Methods for dynamically fetching data from the config

    def get_region_mapping(self) -> Dict[str, dict]:
        return self.config.get('region_mapping', {})
    
    def get_region_name(self)-> str:
        return self.config.get('region_mapping', {}).get(self.region_short_code,{}).get('name',{})

    def get_resource_disaggregation_config(self) -> Dict[str, dict]:

        """
        Returns the capacity disaggregation configuration for the given resource type.
        If the resource type is None or not found, returns an empty dictionary.
        """
       # Access 'capacity_disaggregation' and then the specific resource type (e.g., 'solar' or 'wind')
    
        return self.config.get('capacity_disaggregation', {}).get(self.resource_type, {})

    def get_vis_dir(self) -> Path:
        return (self.config.get('visualization', {}).get('RESource', 'vis')) +"/"+ (self.resource_type if self.resource_type else 'vis/misc')

    def get_linking_data_config(self) -> Dict[str, dict]:
        return self.config.get('processed_data', {}).get('RESource', {})

    def get_gaez_data_config(self) -> Dict[str, dict]:
        return self.config.get('GAEZ', {})

    def get_atb_config(self) -> Dict[str, dict]:
        return self.config.get('NREL', {}).get('ATB', {})
    
    def get_cutout_config(self) -> Dict[str, dict]:
        return self.config.get('cutout', {})
    
    def get_gadm_config(self)-> Dict[str, dict]:
        return self.config.get('GADM', {})
    
    # def get_country(self)-> str:
    #     return self.config.get('country', "Canada") # If NONE, default is Canada
    
    
    def get_default_crs(self)->str:
        return 'EPSG:4326'
    
    def get_custom_land_layers(self):
        return self.config.get('custom_land_layers', {})
    
    def get_osm_config(self):
        return self.config['OSM_data']
    
    def get_region_timezone(self):
        return self.config['region_mapping'][self.region_short_code]['timezone_convert']
    
    def get_cell_resolution(self):
        return self.config.get('grid_cell_resolution',{})
    
    def get_turbines_config(self):
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        return self.resource_disaggregation_config['turbines']
    
    def get_gwa_config(self):
        return self.config.get('GWA',{})
    
    def get_resource_landuse_intensity(self):
        self.resource_disaggregation_config:dict=self.get_resource_disaggregation_config()
        return self.resource_disaggregation_config['landuse_intensity']

    def get_region_timezone(self):
        return self.config['region_mapping'][self.region_short_code]['timezone_convert']