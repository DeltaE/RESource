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

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

import RES.utility as utils

today_str = datetime.now().strftime("%Y%m%d")


@dataclass
class AttributesParser:
    """
    This is the parent class that will extract the core attributes from the User Config file.
    """
    # Attributes that are required as Args.
    
    config_file_path: Path = field(default=None)
    region_short_code: str = field(default=None)
    resource_type: str = field(default=None)
    weather_year: str = field(default=None)  # CLI override; falls back to config key if None
    
    def __post_init__(self):
        self.site_index='cell'

        # Convert region_short_code to uppercase to handle user types regarding case-sensitive letter inputs.
        if self.region_short_code is not None:
            self.region_short_code = self.region_short_code.upper()
        else:
            raise ValueError("region_short_code is required and cannot be None")
        
        # Load the user configuration master file by using the method
        self.config:Dict[str,dict] = self.load_config(self.config_file_path)
        
        # Resolve weather_year: CLI-supplied field takes precedence over config key.
        if self.weather_year is None:
            _yr = self.config.get('weather_year')
            if _yr is None:
                raise ValueError(
                    "weather_year not set. Pass --year YYYY via CLI "
                    "or add 'weather_year: YYYY' to your config YAML."
                )
            self.weather_year = int(_yr)
        else:
            self.weather_year = int(self.weather_year)
            
        ## Process the attributes that are required for the workflow and are extracted from the config file. These attributes will be used by the child classes to perform the data supply-chain steps.
        self.disaggregation_config:Dict[str,dict] = self.config.get('capacity_disaggregation','')
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        self.region_code_validity=self.is_region_code_valid()
        gadm_config = self.get_gadm_config().get('datafield_mapping', {})
        self.sub_national_unit_tag = gadm_config.get('NAME_2') if 'NAME_2' in gadm_config else gadm_config.get('NAME_1')
        self.multi_country_flag = self.get_multi_country_flag # This will set the multi_country_flag based on the config file.
        self.RUN_ID=self.get_RUN_ID() 
        self.country=self.get_country()
        self.country_kwd=self.country.replace(' ','')
        self.results_save_to=self.get_results_save_to_path()
        

            

        
        # Define the store file path and filename
        self.store = Path(f"data/store/{self.country_kwd}/{self.region_short_code}/resources_{self.country_kwd}_{self.region_short_code}_{self.RUN_ID}.h5")
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.default_crs_cfg:dict=self.config.get('default_CRS',None)
        self.crs_d,self.crs_m=self.get_CRS()
        self.vis_root=self.get_vis_dir()
    
    def get_CRS(self):
        """
        Returns the CRS for degrees and meters based on the region code. If not found, defaults to EPSG:4326 for degrees and EPSG:3347 for meters.
        
        Steps:
            1. Loads default CRS from config file.
            2. Checks for region-specific CRS in config file.
            3. If not found, uses default CRS.
            4. If defaults are missing, falls back to EPSG:4326 for degrees and EPSG:3347 for meters.
            5. Returns a tuple of (CRS_degrees, CRS_meters).
        
        Returns:
            tuple: (CRS_degrees, CRS_meters)
        """
        # Load defaults from config
        self.crs_d_default = self.default_crs_cfg.get('degrees') if self.default_crs_cfg else None
        self.crs_m_default = self.default_crs_cfg.get('meters') if self.default_crs_cfg else None

        # If missing, set fallbacks
        if self.crs_d_default is None or self.crs_m_default is None:
            utils.print_warning("Default CRS not fully configured. Using EPSG:4326 for degrees and EPSG:3347 for meters.")
            self.crs_d_default = 'EPSG:4326'
            self.crs_m_default = 'EPSG:3347'
        
        # Check for region-specific CRS in config, else use defaults
        self.crs_d = (self.config.get('region_mapping', {})
                        .get(self.region_short_code, {})
                        .get('CRS_degrees', None))
        if self.crs_d is None:
            self.crs_d = self.crs_d_default
        
        self.crs_m = (self.config.get('region_mapping', {})
                        .get(self.region_short_code, {})
                        .get('CRS_meters', None))
        if self.crs_m is None:
            self.crs_m = self.crs_m_default
        
        return self.crs_d,self.crs_m

        
    def load_config(self,config_file_path):
        """ 
        Loads the yaml file as dictionary and extracts the attributes to pass on child classes. 
        """
        with open(config_file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def get_results_save_to_path(self):
        """
        Returns the path where results will be saved.
        """
        results_save_to=utils.ensure_path(f"results/{self.country_kwd}/{self.region_short_code}/{self.RUN_ID}")
        
        return results_save_to

        
    @property
    def region_timezone(self) -> str:
        region_cfg = (
            self.config
            .get('region_mapping', {})
            .get(self.region_short_code, {})
        )
        if not region_cfg:
            raise KeyError(
                f"Region '{self.region_short_code}' not found in config 'region_mapping'. "
                f"Available codes: {list(self.config.get('region_mapping', {}).keys())}"
            )
        tz = region_cfg.get('timezone_convert')
        
        if tz is None:
            raise KeyError(
                f"'timezone_convert' not defined for region '{self.region_short_code}' "
                f"in config 'region_mapping'. Add e.g. timezone_convert: 'Etc/GMT+8'."
            )
        return tz
   
    def get_snapshot(self,
                year=None) -> tuple:
        """
        Derive UTC-aligned ERA5 snapshot strings for a single calendar year from
        the region's timezone_convert config field.

        Converts local midnight Jan 1 of `year` → UTC (start) and local midnight
        Jan 1 of `year + 1` − 1 h → UTC (end), producing a closed hourly interval
        that covers exactly one local calendar year with no overlap between adjacent
        years.

        Format matches load_snapshot() and atlite.Cutout(time=slice(start, end)).

        Returns
        -------
        tuple[str, str]
            (start_str, end_str) as naive UTC strings 'YYYY-MM-DD HH:MM:SS'.

        Notes
        -----
        POSIX / IANA Etc/GMT±N sign convention (counterintuitive but standard):
            'Etc/GMT+8'  →  UTC-8  (PST, British Columbia)
            'Etc/GMT-2'  →  UTC+2  (CEST, Western Balkans)
        zoneinfo handles this correctly; do not parse the offset manually.

        Leap year awareness is automatic: 2020 yields 8784 h, 2019 yields 8760 h.

        Raises
        ------
        KeyError
            If timezone_convert is absent for this region in the config.
        ZoneInfoNotFoundError
            If the timezone string is not a recognised IANA identifier.
        """
        from datetime import datetime, timedelta, timezone
        from zoneinfo import ZoneInfo

        tz_name = self.region_timezone   # reads region_mapping[region_short_code]['timezone_convert']
        tz      = ZoneInfo(tz_name)

        year = int(year) if year is not None else self.weather_year
        
        utc_start = datetime(year,     1, 1, tzinfo=tz).astimezone(timezone.utc)
        utc_end   = datetime(year + 1, 1, 1, tzinfo=tz).astimezone(timezone.utc) - timedelta(hours=1)

        return (
            utc_start.strftime('%Y-%m-%d %H:%M:%S'),
            utc_end.strftime('%Y-%m-%d %H:%M:%S'),
        )
        
    @property
    def get_multi_country_flag(self) -> bool:
        """
        Returns True if the 'country' represents a multi-country region and 'regions' are nations, 
        False if it is a single country and 'regions' are sub-nations.
        """
        return self.config.get('multi_country_flag', False)
    
    def get_custom_land_layers_config(self):
        return self.config.get('custom_land_layers', {})
    
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
            print("\nPlease provide a CANADIAN region CODE (2 letters) from the following list: \n ")
            # display(self.region_mapping.keys())
            for key, value in self.region_mapping.items():
                # Assuming you want to show the first item in the value (e.g., the first name or detail)
                name = value.get('name', 'N/A')  # Change 'name' to the actual key you want to display
                print(f"• {key}: {name}")
            return False  # Exit the function if the region code is invalid
        else:
            return True

    def get_RUN_ID(self)-> str:
        """
        The RUN_ID is used to identify the specific run of the scenario.
        It is typically used to differentiate between different runs of the same scenario, especially when multiple runs are performed with different parameters or configurations.
        """
        return f"{self.config.get('Scenario').get('run_id')}_{self.weather_year}_{today_str}"

    def get_conserved_lands_CAN_args(self)->dict:
        if self.country=='Canada':
            return {
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
            }
        else:
            print("Conservation Lands data supply chain is configured for Canada only")
            return None

    @property
    def discount_rate(self):
        return self.config.get('economic_parameters', {}).get('discount_rate', 0.03)
    
    @property
    def default_font_size(self):
        return 14

    @property
    def default_font_family(self):
       return 'serif'

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
        vis_path = Path(
            f"vis/{self.country_kwd}/{self.region_short_code}/{self.RUN_ID}/"
            f"{self.resource_type if self.resource_type else f'misc/{self.region_short_code}'}"
        )

        utils.ensure_path(vis_path)
        return vis_path
    
    def get_CLC_raster_config(self) -> Dict[str, dict]:
        return self.config.get('CORINE', {})

    def get_gaez_data_config(self) -> Dict[str, dict]:
        return self.config.get('GAEZ', {})

    def get_atb_config(self) -> Dict[str, dict]:
        return self.config.get('NREL', {}).get('ATB', {})
    
    def get_cutout_config(self) -> Dict[str, dict]:
        return self.config.get('cutout', {})
    
    def get_gadm_config(self)-> Dict[str, dict]:
        return self.config.get('GADM', {})
    
    def get_country(self)-> str:
        return self.config.get('country', None)
    
    def get_custom_land_layers(self):
        return self.config.get('custom_land_layers', {})
    
    def get_osm_config(self):
        return self.config['OSM_data']
    
    # def get_region_timezone(self): # upgraded the method to include error handling and more informative messages.
    #     return self.config['region_mapping'][self.region_short_code]['timezone_convert']

    
    def get_cell_resolution(self):
        return self.config.get('grid_cell_resolution',{})
    
    def get_buses_path(self):
        return Path('data/downloaded_data/CODERS/data-pull/network/substations.csv')
    
    def get_turbines_config(self):
        return self.resource_disaggregation_config['turbines']
    
    def get_gwa_config(self):
        return self.config.get('GWA',{})
    
    def get_resource_landuse_intensity(self):
        self.resource_disaggregation_config:dict=self.get_resource_disaggregation_config()
        return self.resource_disaggregation_config['landuse_intensity']
    
    def get_wcss_tolerance(self):
        self.resource_disaggregation_config:dict=self.get_resource_disaggregation_config()
        return self.resource_disaggregation_config.get('WCSS_tolerance',0.01)
    
    def get_grid_proximity_km(self):
        """
        Returns the grid proximity in kilometers.
        """
        return self.config.get('capacity_disaggregation').get('transmission', {}).get('grid_proximity_km', 100)