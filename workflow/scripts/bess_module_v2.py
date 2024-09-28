# BESS module

import time
import argparse
import logging as log
import numpy as np
import geopandas as gpd
import os
import pandas as pd
import plotly.express as px

#local packages
try:
    # Try importing from the submodule context
    import linkingtool.linking_utility as utils
    import linkingtool.linking_vis as vis
    import linkingtool.linking_solar as solar
    from linkingtool.attributes_parser import AttributesParser
except ImportError:
    # Fallback for when running as a standalone script or outside the submodule
    import Linking_tool.linkingtool.linking_utility as utils
    import Linking_tool.linkingtool.linking_vis as vis
    import Linking_tool.linkingtool.linking_solar as solar
    from Linking_tool.linkingtool.attributes_parser import AttributesParser


class BESSModule:
    def __init__(self, config_file_path: str, resource_type: str):
        self.config_file_path = config_file_path
        self.resource_type = resource_type
        self.script_start_time = time.time()
        
        self.log_path = f'workflow/log/{self.resource_type}_module_log.txt'
        self.log = utils.create_log(self.log_path)
        utils.print_module_title(f'{self.resource_type} module initiated')
        
        self.log.info("Loading Configuration Data and Directories...")
        self.load_config()

    def load_config(self):
        # Load configuration data
        self.config: dict = utils.load_config(self.config_file_path)
        
        # Load necessary files and parameters
        self.gadm_file = os.path.join(self.config['GADM']['root'], self.config['GADM']['datafile'])
        self.resource_disaggregation_config: dict = self.config['capacity_disaggregation'][f'{self.resource_type}']
        self.ATB_NREL_cost_datafile: str = self.resource_disaggregation_config['cost_data']
        self.energy_consumption_data: dict = self.config['Gov']['CEEI']
        self.consumption_year: int = int(self.resource_disaggregation_config['residential_energy_consumption_year'])
        self.unit_storage_discharge_duration: float = self.resource_disaggregation_config['storage_discharge_duration']  # hrs
        self.capacity_estimate_per_energy_unit: dict = self.resource_disaggregation_config['capacity_estimate_per_energy_unit']

        # Load data files
        self.gadm_regions_gdf: gpd.GeoDataFrame = gpd.read_file(self.gadm_file)
        self.building_egy_df: pd.DataFrame = pd.read_excel(os.path.join(self.energy_consumption_data['root'], self.energy_consumption_data['datafile']['buildings']), sheet_name='Combined')
        self.resource_unit_size_MW: int = self.resource_disaggregation_config['unit_size']  # MW
        self.resource_cost: pd.DataFrame = pd.read_csv(self.ATB_NREL_cost_datafile)

    def set_cost_parameters(self):
        # Set COST parameters
        self.resource_fom = self.resource_cost[self.resource_cost['core_metric_parameter'] == 'Fixed O&M'].value.iloc[0] / 1E3
        self.log.info(f"Resource ({self.resource_type}) FOM set to: {round(self.resource_fom, 2)} Mil. USD/ MW.")
        
        self.resource_capex = self.resource_cost[self.resource_cost['core_metric_parameter'] == 'CAPEX'].value.iloc[0] / 1E3
        self.log.info(f"Battery Energy Storage Site CAPEX set to: {round(self.resource_capex, 2)} Mil. USD/ MW.")
        
        if 'Variable O&M' in self.resource_cost.core_metric_parameter.unique():
            self.resource_vom = self.resource_cost[self.resource_cost['core_metric_parameter'] == 'Variable O&M'].value.iloc[0] / 1E3
            self.log.info(f"Battery Energy Storage Site VOM set to: {round(self.resource_vom, 2)} Mil. USD/MW.")
        else:
            self.resource_vom = self.resource_disaggregation_config['vom']
            self.log.info(f"Battery Energy Storage Site VOM not found in NREL ATB dataset. Set to: {round(self.resource_vom, 2)}")

    def prepare_sites_geodataframe(self):
        # Prepare the sites GeoDataFrame
        self.resource_sites = self.gadm_regions_gdf.copy()
        self.resource_sites.columns = ['COUNTRY', 'Province', 'Region', 'Region_ID', 'population', 'geometry']
        
        # Add new columns specific to resource and rearrange columns
        self.resource_sites['cluster_id'] = self.resource_sites['Region'] + '_1'
        self.resource_sites['capex'] = self.resource_capex  
        self.resource_sites['fom'] = self.resource_fom 
        self.resource_sites['vom'] = self.resource_vom 
        self.resource_sites['CF_mean'] = 0 
        self.resource_sites['p_lcoe'] = 0
        self.resource_sites['nearest_station'] = ''
        self.resource_sites['nearest_station_distance_km'] = 0  
        self.resource_sites['potential_capacity'] = 0  

        # Filter the building energy DataFrame
        building_egy_df = (
            self.building_egy_df[self.building_egy_df['ORG_TYPE'] == 'Regional District']
            .loc[:, ['YEAR', 'ORG_NAME', 'ENERGY_TYPE', 'ENERGY_UNIT', 'SUB_SECTOR', 'CONSUMPTION_TOTAL']]
        )

        res_elec_egy_yr_mask = (building_egy_df['ENERGY_TYPE'] == 'ELEC') & (building_egy_df['SUB_SECTOR'] == 'Res') & (building_egy_df['YEAR'] == self.consumption_year)
        rd_res_elec_egy_demand_yr = building_egy_df[res_elec_egy_yr_mask].copy()
        rd_res_elec_egy_demand_yr.rename(columns={'ORG_NAME': 'Region', 'CONSUMPTION_TOTAL': 'CONSUMPTION_TOTAL_kWh'}, inplace=True)

        rgn_name_mapping = self.config['Gov']['Population']['different_name_mapping']
        rd_res_elec_egy_demand_yr.loc[:, 'Region'] = rd_res_elec_egy_demand_yr['Region'].replace(rgn_name_mapping)
        rd_res_elec_egy_demand_yr = rd_res_elec_egy_demand_yr.groupby('Region').sum()

        self.resource_sites = self.resource_sites.merge(rd_res_elec_egy_demand_yr[['CONSUMPTION_TOTAL_kWh']], left_on='Region', right_index=True, how='left')
        self.resource_sites.set_index('cluster_id', inplace=True)

        self.resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'] = self.resource_sites['CONSUMPTION_TOTAL_kWh'] / 1E3
        self.resource_sites.drop(columns=['CONSUMPTION_TOTAL_kWh'], inplace=True)

    def calculate_potential_capacity(self):
        for key in self.capacity_estimate_per_energy_unit.keys():
            # Calculate potential capacity based on consumption and duration
            self.resource_sites['potential_capacity'] = (
                self.resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'] / self.unit_storage_discharge_duration * self.capacity_estimate_per_energy_unit[key]
            ).astype(int)

            # Adjust potential capacity to the nearest multiple of unit size
            self.resource_sites['potential_capacity'] = (
                np.ceil(self.resource_sites['potential_capacity'] / self.resource_unit_size_MW) * self.resource_unit_size_MW
            ).astype('int32')

            self.resource_sites['per_capita_res_elec_energy_consumption'] = self.resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'] / self.resource_sites['population']
            self.resource_sites['discharge_dur_hr'] = self.resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'] / self.resource_sites['potential_capacity']
            self.resource_sites['storage_energy_capacity_MWh'] = self.resource_sites['discharge_dur_hr'] * self.resource_sites['potential_capacity'] * self.capacity_estimate_per_energy_unit[key]

            self.resource_sites['%_res_elec_egy_from_bess'] = (self.resource_sites['storage_energy_capacity_MWh'] / self.resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'])
            
            # Rearrange columns to match the desired DataFrame structure
            self.resource_sites = self.resource_sites[
                ['Region', 'Region_ID', 'potential_capacity', 'discharge_dur_hr', 'storage_energy_capacity_MWh', 'population', 'res_elec_CONSUMPTION_TOTAL_MWh', '%_res_elec_egy_from_bess', 'per_capita_res_elec_energy_consumption', 'CF_mean', 
                'p_lcoe', 'nearest_station', 'nearest_station_distance_km', 'capex', 'fom', 'vom', 'geometry']
            ]

            result_file_path = os.path.join(self.config['results']['linking']['root'], self.config['results']['linking']['clusters_topSites'][f'{self.resource_type}']) + f'{key}.pkl'
            self.resource_sites.to_pickle(result_file_path)

    def run(self):
        self.set_cost_parameters()
        self.prepare_sites_geodataframe()
        self.calculate_potential_capacity()

        # End the script run timer
        script_end_time = time.time()
        runtime = round((script_end_time - self.script_start_time), 2)
        return self.log.info(f"Script runtime: {runtime} seconds")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help="Path to the configuration file 'config_master.yml'")
    parser.add_argument('resource_type', type=str, help="Resource Type")

    # Parse the arguments
    args = parser.parse