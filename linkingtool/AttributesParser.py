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
    config_file_path: Path =field(default='config/config.yml')
    province_short_code: str=field(default= 'BC')
    resource_type: str = field(default='None')
    
    def __post_init__(self):
        self.site_index='cell'

        # Define the path and filename
        self.store = Path(f"data/store/resources_{self.province_short_code}.h5")
        self.store.parent.mkdir(parents=True, exist_ok=True)

        # Convert province_short_code to uppercase to handle user types regarding case-sensitive letter inputs.
        self.province_short_code = self.province_short_code.upper()
        
        # Load the user configuration master file by using the method
        self.config:Dict[str,dict] = self.load_config(self.config_file_path)
        self.disaggregation_config:Dict[str,dict] = self.config['capacity_disaggregation']
        self.province_code_validity=self.is_province_code_valid()
        self.log = log.getLogger(__name__)
        
    def load_config(self,config_file_path):
        """ 
        Loads the yaml file as dictionary and extracts the attributes to pass on child classes. 
        """
        with open(config_file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def is_province_code_valid(self)-> bool:
        """
        Args:
            Province_short_code: 2 letter short code of the province.
            
        Description: 
            Checks of the province code is correct. If not, then suggests the available list of codes that are liked to data supply-chain.
        """
        self.province_mapping=self.get_province_mapping()
        
        if self.province_short_code not in self.province_mapping:
            print(f"!!! ERROR !!! \nRecheck the province code.\n{60 * '_'}")
            print(f"\nPlease provide a CANADIAN province CODE (2 letters) from the following list: \n ")
            # display(self.province_mapping.keys())
            for key, value in self.province_mapping.items():
                # Assuming you want to show the first item in the value (e.g., the first name or detail)
                name = value.get('name', 'N/A')  # Change 'name' to the actual key you want to display
                print(f"• {key}: {name}")
            return False  # Exit the function if the province code is invalid
        else:
            return True

    def load_snapshot(self)->tuple:
        start_date = self.config.get('cutout', {}).get('snapshots', {}).get('start', [[]])[0]
        end_date = self.config.get('cutout', {}).get('snapshots', {}).get('end', [[]])[0]
        return start_date, end_date

    def load_cost(self):
        grid_connection_cost_per_km = self.disaggregation_config.get('transmission', {}).get('grid_connection_cost_per_Km', 0)
        tx_line_rebuild_cost = self.disaggregation_config.get('transmission', {}).get('tx_line_rebuild_cost', 0)

        self.ATB:Dict[str,dict]=self.get_atb_config()
        atb_file = Path(self.ATB.get('root', ''), self.ATB.get('datafile', {}).get('parquet', ''))
        utility_scale_cost = utility_scale_cost = pd.read_parquet(atb_file) if atb_file.exists() else pd.DataFrame()
        source_column:str= self.ATB.get('column',{})
        cost_params_mapping:Dict[str,str]=self.ATB.get('cost_params',{})
        
        resource_capex:float = (utility_scale_cost[utility_scale_cost[source_column] == cost_params_mapping.get('capex',{})].value.iloc[0] / 1E3 
                          if not utility_scale_cost.empty else 0) # mill. $/ MW
        
        resource_fom : float = (utility_scale_cost[utility_scale_cost[source_column] == cost_params_mapping.get('fom',{})].value.iloc[0] / 1E3 
                        if not utility_scale_cost.empty else 0) # mill. $/ MW
        # Initialize resource_vom based on the availability of 'vom' in cost_params_mapping
        resource_vom: float = 0  # Default value if 'vom' is not found

        if cost_params_mapping.get('vom') is not None:
            # Check if the DataFrame 'utility_scale_cost' is not empty and get the value for 'vom'
            if not utility_scale_cost.empty:
                vom_row = utility_scale_cost[utility_scale_cost[source_column] == cost_params_mapping['vom']]
                if not vom_row.empty:
                    resource_vom = vom_row['value'].iloc[0] / 1E3  # Convert to million $/MW

        return resource_capex, resource_fom, resource_vom, grid_connection_cost_per_km, tx_line_rebuild_cost

# Methods for dynamically fetching data from the config

    def get_province_mapping(self) -> Dict[str, dict]:
        return self.config.get('province_mapping', {})
    
    def get_province_name(self)-> str:
        return self.config.get('province_mapping', {}).get(self.province_short_code,{}).get('name',{})

    def get_resource_disaggregation_config(self) -> Dict[str, dict]:

        """
        Returns the capacity disaggregation configuration for the given resource type.
        If the resource type is None or not found, returns an empty dictionary.
        """
       # Access 'capacity_disaggregation' and then the specific resource type (e.g., 'solar' or 'wind')
    
        return self.config.get('capacity_disaggregation', {}).get(self.resource_type, {})

    def get_vis_dir(self) -> Path:
        return (self.config.get('visualization', {}).get('linking', '')) +"/"+ (self.resource_type if self.resource_type else 'vis/misc')

    def get_linking_data_config(self) -> Dict[str, dict]:
        return self.config.get('processed_data', {}).get('linking', {})

    def get_gaez_data_config(self) -> Dict[str, dict]:
        return self.config.get('GAEZ', {})

    def get_atb_config(self) -> Dict[str, dict]:
        return self.config.get('NREL', {}).get('ATB', {})
    
    def get_cutout_config(self) -> Dict[str, dict]:
        return self.config.get('cutout', {})
    
    def get_gadm_config(self)-> Dict[str, dict]:
        return self.config.get('GADM', {})
    
    def get_country(self)-> str:
        return self.config.get('country', "Canada") # If NONE, default is Canada
    
    def get_default_crs(self)->str:
        return 'EPSG:4326'
    
    def get_custom_land_layers(self):
        return self.config.get('custom_land_layers', {})
    
    def get_osm_config(self):
        return self.config['OSM_data']
    
    def get_province_timezone(self):
        return self.config['province_mapping'][self.province_short_code]['timezone_convert']
    
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