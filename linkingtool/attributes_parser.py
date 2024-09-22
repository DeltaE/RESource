import logging as log
import atlite
import os
import geopandas as gpd
import pandas as pd
from atlite.gis import shape_availability
import argparse

# Local Packages
import linkingtool.linking_utility as utils
import linkingtool.linking_vis as vis
import linkingtool.linking_solar as solar

# Logging Configuration
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttributesParser:
    def __init__(self, 
                 config_file_path: str, 
                 resource_type: str = None):  # Default to None if not provided
        
        # Load user configuration
        self.config = utils.load_config(config_file_path)
        self.resource_type = resource_type if resource_type else 'default_resource'  # Use a default if None
        
        # Extract required configurations/Directories
        self.current_region = self.config.get('regional_info', {}).get('region_1', {})
        self.region_code = self.current_region.get('code', 'unknown_code')
        self.disaggregation_config = self.config.get('capacity_disaggregation', {})
        self.vis_dir = os.path.join(self.config.get('visualization', {}).get('linking', ''), self.resource_type)
        self.linking_data = self.config.get('processed_data', {}).get('linking', {})
        self.gaez_data = self.config.get('GAEZ', {})
        self.ATB = self.config.get('NREL', {}).get('ATB', {})
        
    def load_snapshot(self):
        start_date = self.config.get('cutout', {}).get('snapshots', {}).get('start', [[]])[0]
        end_date = self.config.get('cutout', {}).get('snapshots', {}).get('end', [[]])[0]
        return start_date, end_date
    
    def load_geospatial_data(self):
        cutout = atlite.Cutout(self.current_region.get('cutout_datafile', ''))
        
        gadm_file = os.path.join(self.config.get('GADM', {}).get('root', ''), self.config.get('GADM', {}).get('datafile', ''))
        aeroway_file = os.path.join(self.linking_data.get('root', ''), self.resource_type, 
                                      f"aeroway_OSM_{self.region_code}_with_buffer_{self.resource_type}.parquet")
        conservation_lands_file = os.path.join(self.linking_data.get('root', ''), 
                                                self.linking_data.get('CPCAD_org', ''))
        
        gadm_regions_gdf = gpd.read_file(gadm_file) if os.path.exists(gadm_file) else gpd.GeoDataFrame()
        aeroway_with_buffer = gpd.read_parquet(aeroway_file) if os.path.exists(aeroway_file) else gpd.GeoDataFrame()
        conservation_lands_province = gpd.read_parquet(conservation_lands_file) if os.path.exists(conservation_lands_file) else gpd.GeoDataFrame()
        
        buses_gdf = gpd.GeoDataFrame(pd.read_pickle(os.path.join('data/processed_data', 
                                      self.linking_data.get('transmission', {}).get('nodes_datafile', ''))))
        return cutout, gadm_regions_gdf, aeroway_with_buffer, conservation_lands_province, buses_gdf
    
    def load_cost(self):    
        grid_connection_cost_per_km = self.disaggregation_config.get('transmission', {}).get('grid_connection_cost_per_Km', 0)
        tx_line_rebuild_cost = self.disaggregation_config.get('transmission', {}).get('tx_line_rebuild_cost', 0)
        
        atb_file = os.path.join(self.ATB.get('root', ''), self.ATB.get('datafile', {}).get('parquet', ''))
        utility_scale_cost = pd.read_parquet(atb_file) if os.path.exists(atb_file) else pd.DataFrame()
        
        resource_capex = utility_scale_cost[utility_scale_cost['core_metric_parameter'] == 'CAPEX'].value.iloc[0] / 1E3 if not utility_scale_cost.empty else 0
        resource_fom = utility_scale_cost[utility_scale_cost['core_metric_parameter'] == 'Fixed O&M'].value.iloc[0] / 1E3 if not utility_scale_cost.empty else 0
        resource_vom = 0  # Not found in ATB
        
        return resource_capex, resource_fom, resource_vom, grid_connection_cost_per_km, tx_line_rebuild_cost
