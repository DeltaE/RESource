# %%
#BESS module

import time,argparse
# Start the script run timer
script_start_time = time.time()
# %%
import logging as log
import numpy as np

import geopandas as gpd 
import os

import pandas as pd
import linkingtool.linking_vis as vis

# Local Package
import linkingtool.linking_utility as utils
import linkingtool.linking_wind as wind

import plotly.express as px
import pandas as pd


# %%
def main(
    config_file_path:str,
    resource_type: str
    ):
    
    log_path=f'workflow/log/{resource_type}_module_log.txt'
    log=utils.create_log(log_path)
    utils.print_module_title(f'{resource_type} module initiated')
    
    log.info(f"Loading Configuration Data and Directories...")

# User Configuration
## config_file_path='config/config_master.yml' #for notebook run

    config:dict[dict]=utils.load_config(config_file_path)
    
    gadm_file=os.path.join(config['GADM']['root'],config['GADM']['datafile'])
    resource_disaggregation_config:dict=config['capacity_disaggregation'][f'{resource_type}']
    ATB_NREL_cost_datafile:str=resource_disaggregation_config['cost_data']
    energy_consumption_data:dict=config['Gov']['CEEI']
    consumption_year:int=int(resource_disaggregation_config['residential_energy_consumption_year'])
    unit_storage_discharge_duration:float = resource_disaggregation_config['storage_discharge_duration']  # hrs  ; the battery cost should be inline with this data.
    capacity_estimate_per_energy_unit:dict= resource_disaggregation_config['capacity_estimate_per_energy_unit'] # x 100% ; % of residential ELEC energy consumption targeted to be served from BESS

    
## Load Data Files ------------------------------
    gadm_regions_gdf:gpd.GeoDataFrame = gpd.read_file(gadm_file)
    
    building_egy_df:pd.DataFrame =pd.read_excel(os.path.join(energy_consumption_data['root'],energy_consumption_data['datafile']['buildings']),sheet_name='Combined')
    
    resource_unit_size_MW:int= resource_disaggregation_config['unit_size'] #MW
    resource_cost:pd.DataFrame=pd.read_csv(ATB_NREL_cost_datafile)
    
## Set COST parameters
    resource_fom=resource_cost[resource_cost['core_metric_parameter']=='Fixed O&M'].value.iloc[0]/1E3  # mill. $/MW = ($/kw ) /1E3
    print(f"Resource ({resource_type}) FOM set to: {round(resource_fom,2)} Mil. USD/ MW. Sourced from >> Summary data, Utility-Scale Battery Storage (LI) Cost, ATB 2024 , NREL ")

    resource_capex=resource_cost[resource_cost['core_metric_parameter']=='CAPEX'].value.iloc[0]/1E3 # mill. $/MW = ($/kw ) /1E3
    print(f"Battery Energy Storage Site CAPEX set to: {round(resource_capex,2)} Mil. USD/ MW. Sourced from >> Summary data, Utility-Scale Battery Storage (LI) Cost, ATB 2024 , NREL ")

    if 'Variable O&M' in resource_cost.core_metric_parameter.unique():
        resource_vom = resource_cost[resource_cost['core_metric_parameter'] == 'Variable O&M'].value.iloc[0] / 1E3  # Convert to mill. $/MW
        print(f"Battery Energy Storage Site VOM set to: {round(resource_vom, 2)} Mil. USD/MW. Sourced from Summary data, Utility-Scale Battery Storage (LI) Cost, ATB 2024, NREL")
    else:
        resource_vom = resource_disaggregation_config['vom']
        print(f"Battery Energy Storage Site VOM not found in NREL ATB dataset. Set to: {round(resource_vom, 2)}")

# Prepare the sites geodataframe
    ## create an alias gdf from GADM regions.
    resource_sites = gadm_regions_gdf.copy()
    resource_sites.columns = ['COUNTRY', 'Province', 'Region', 'Region_ID', 'population', 'geometry']
    
    ## Add new columns specific to resource and rearrange columns
    resource_sites['cluster_id'] = resource_sites['Region'] + '_1'  # Hardcoded to 1 for simplification, temp.; Later to be replaced by better algorithms
    resource_sites['capex'] = resource_capex  
    resource_sites['fom'] = resource_fom 
    resource_sites['vom'] = resource_vom 
    resource_sites['CF_mean'] = 0 
    resource_sites['p_lcoe'] = 0  # Placeholder values
    resource_sites['nearest_station'] = ''  # Placeholder values, assuming a copperplate within a zone. Later to be replaced by better algorithms
    resource_sites['nearest_station_distance_km'] = 0  # Assuming all the sites will be located at existing grid nodes
    
    resource_sites['potential_capacity'] = 0  # Placeholder values



    ## filter the building_egy_df DataFrame
    building_egy_df = (
        building_egy_df[building_egy_df['ORG_TYPE'] == 'Regional District']
        .loc[:, ['YEAR', 'ORG_NAME', 'ENERGY_TYPE', 'ENERGY_UNIT', 'SUB_SECTOR', 'CONSUMPTION_TOTAL']]
    )

    res_elec_egy_yr_mask=(building_egy_df['ENERGY_TYPE']=='ELEC')&(building_egy_df['SUB_SECTOR']=='Res')&(building_egy_df['YEAR']==consumption_year)
    rd_res_elec_egy_demand_yr=building_egy_df[res_elec_egy_yr_mask].copy()
    rd_res_elec_egy_demand_yr.rename(columns={'ORG_NAME': 'Region', 'CONSUMPTION_TOTAL': 'CONSUMPTION_TOTAL_kWh'}, inplace=True)

    rgn_name_mapping = config['Gov']['Population']['different_name_mapping']
    rd_res_elec_egy_demand_yr.loc[:, 'Region'] = rd_res_elec_egy_demand_yr['Region'].replace(rgn_name_mapping)
    rd_res_elec_egy_demand_yr=rd_res_elec_egy_demand_yr.groupby('Region').sum()

    resource_sites = resource_sites.merge(rd_res_elec_egy_demand_yr[['CONSUMPTION_TOTAL_kWh']], left_on='Region', right_index=True, how='left')
    resource_sites.set_index('cluster_id',inplace=True)

    resource_sites['res_elec_CONSUMPTION_TOTAL_MWh']=resource_sites['CONSUMPTION_TOTAL_kWh']/1E3 # kWh to MWh conversion
    resource_sites.drop(columns=['CONSUMPTION_TOTAL_kWh'], inplace=True)


    for key in capacity_estimate_per_energy_unit.keys():

    # Calculate potential capacity based on consumption and duration
        resource_sites['potential_capacity'] = (
            resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'] / unit_storage_discharge_duration * capacity_estimate_per_energy_unit[key]
            ).astype(int)  # Convert the result to integer

        # Adjust potential capacity to the nearest multiple of unit size
        resource_sites['potential_capacity'] = (
            np.ceil(resource_sites['potential_capacity'] / resource_unit_size_MW) * resource_unit_size_MW
            ).astype('int32')  # Round up to the nearest whole unit and convert to integer
        
        resource_sites['per_capita_res_elec_energy_consumption']= resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'] / resource_sites['population']

        resource_sites['discharge_dur_hr']= resource_sites['res_elec_CONSUMPTION_TOTAL_MWh']/resource_sites['potential_capacity']
        resource_sites['discharge_dur_hr']=resource_sites['discharge_dur_hr']
        
        resource_sites['storage_energy_capacity_MWh']= resource_sites['discharge_dur_hr']*    resource_sites['potential_capacity'] *capacity_estimate_per_energy_unit[key]
        resource_sites['storage_energy_capacity_MWh']=  resource_sites['storage_energy_capacity_MWh']
        
        resource_sites.loc[:,'%_res_elec_egy_from_bess'] =  (resource_sites['storage_energy_capacity_MWh']/resource_sites['res_elec_CONSUMPTION_TOTAL_MWh'])
        resource_sites['%_res_elec_egy_from_bess']=resource_sites['%_res_elec_egy_from_bess']
        
        ## Rearrange columns to match the desired DataFrame structure
        resource_sites = resource_sites[
            ['Region', 'Region_ID', 'potential_capacity','discharge_dur_hr','storage_energy_capacity_MWh','population', 'res_elec_CONSUMPTION_TOTAL_MWh','%_res_elec_egy_from_bess' ,'per_capita_res_elec_energy_consumption','CF_mean', 
            'p_lcoe', 'nearest_station', 'nearest_station_distance_km', 'capex','fom', 'vom','geometry']
            ]

        result_file_path=os.path.join(config['results']['linking']['root'],config['results']['linking']['clusters_topSites'][f'{resource_type}'])+f'{key}.pkl'
        resource_sites.to_pickle(result_file_path)

    # End the script run  timer
    script_end_time = time.time()

    # Calculate runtime in seconds
    runtime = round((script_end_time - script_start_time),2)
    

    # %%
    return log.info (f"Script runtime: {runtime} seconds")  
if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file 'config_master.yml'")
    parser.add_argument('resource_type', type=str, help=f"Resource Type")
    
    # Parse the arguments
    
    # #----------------Main code to be used ----------------
    args = parser.parse_args()
    main(args.config,args.resource_type)
    
    # #----------------------- for notebook run/Debugging------------------------------------
    # config_file_path='config/config_master.yml'
    # main(config_file_path,resource_type)
