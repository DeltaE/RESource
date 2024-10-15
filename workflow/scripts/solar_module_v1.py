#  Solar Module
import time

resource_type:str="solar" #use all small letters
# Start the script run timer
script_start_time = time.time()
# %%
import logging as log

import atlite
import os
from scipy.spatial import cKDTree
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
from atlite.gis import shape_availability, ExclusionContainer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
import xarray as xr
import argparse

# %% [markdown]
# ### Local Packages

# %%
import linkingtool.linking_utility as utils
import linkingtool.visuals as vis
import linkingtool.linking_solar as solar

# %%


# %%
def main(config_file_path:str,
         resource_type:str):
    
    
    log_path=f'workflow/log/{resource_type}_module_log.txt'
    log=utils.create_log(log_path)
    utils.print_module_title(f'{resource_type} module initiated')
    log.info(f"Loading Configuration Data and Directories...")


# User Configuration
    config:dict[dict]=utils.load_config(config_file_path)

    current_region:dict = config['regional_info']['region_1']
    _CRC_=current_region['code'] # Current Region Code i.e. BC
    
    disaggregation_config:dict=config['capacity_disaggregation']
    
    linking_data:dict=config['processed_data']['linking']
    vis_dir:str=os.path.join(config['visualization']['linking'],resource_type)

    resource_landuse_intensity = disaggregation_config[f'{resource_type}']['landuse_intensity'] # (MW/km2) from literature 1.7 MW/km2
    
    linking_data:dict=config['processed_data']['linking']
    
    ATB_NREL_cost_datafile=disaggregation_config[f'{resource_type}']['cost_data']
    utility_PV_cost=pd.read_csv(ATB_NREL_cost_datafile)
    # Set COST parameters (global for PV)
    solar_fom=utility_PV_cost[utility_PV_cost['core_metric_parameter']=='Fixed O&M'].value.iloc[0]/1E3  # mill. $/MW = ($/kw ) /1E3
    print(f"Solar PV FOM set to: {round(solar_fom,2)} Mil. USD/ MW. Sourced from >> Summary data, Utility PV Cost, ATB 2024 , NREL ")

    solar_capex=utility_PV_cost[utility_PV_cost['core_metric_parameter']=='CAPEX'].value.iloc[0]/1E3 # mill. $/MW = ($/kw ) /1E3
    print(f"Solar PV CAPEX set to: {round(solar_capex,2)} Mil. USD/ MW. Sourced from >> Summary data, Utility PV Cost, ATB 2024 , NREL ")
    
    solar_vom=0
    
    # capex = disaggregation_config[f'{resource_type}']['capex'] # Mil. USD/MW  *** Later to be linked to NREL ATB Spreadsheet (via data pipeline automation) for different TECH class, different years.
    grid_connection_cost_per_Km = disaggregation_config['transmission']['grid_connection_cost_per_Km'] #M$/km   # from MISO , may try NREL method for different range of distance
    tx_line_rebuild_cost = disaggregation_config['transmission']['tx_line_rebuild_cost']  #M$/km # from MISO

    grid_node_proximity_filter = disaggregation_config['transmission']['proximity_filter']
    wcss_tolerance=disaggregation_config[f'{resource_type}']['WCSS_tolerance']

    gadm_file=os.path.join(config['GADM']['root'],config['GADM']['datafile'])
    
    # Raster Configurations
    GAEZ_data:dict= config['GAEZ']
    rasters_in_use_direct =GAEZ_data['Rasters_in_use_direct']

    land_cover_config = GAEZ_data['land_cover']
    terrain_resources_config = GAEZ_data['terrain_resources']
    # exclusion_areas_config =GAEZ_data['exclusion_areas']

    # Raster Files
    gaez_landcover_raster = os.path.join(GAEZ_data['root'], rasters_in_use_direct, land_cover_config['zip_extract_direct'], land_cover_config['raster'])
    gaez_terrain_raster = os.path.join(GAEZ_data['root'], rasters_in_use_direct, terrain_resources_config['zip_extract_direct'], terrain_resources_config['raster'])
    # gaez_exclusionAreas_raster = os.path.join(parent_direct, rasters_in_use_direct, exclusion_areas_config['zip_extract_direct'], exclusion_areas_config['raster'])

    # Raster class and buffer information 
    land_class_inclusion = land_cover_config['class_inclusion'][f'{resource_type}']  #inclusion layer
    terrain_class_exclusion= terrain_resources_config['class_exclusion'][f'{resource_type}'] #exclusion layer
    # terrain_class_exclusion_buffer=terrain_resources_config['class_exclusion']['buffer'][f'{resource_type}'] #m
    
    aeroway_file_path=os.path.join(linking_data['root'],resource_type,f"aeroway_OSM_{_CRC_}_with_buffer_{resource_type}.parquet")
    # exclusionAreas__class_exclusion= exclusion_areas_config['class_exclusion'][f'{resource_type}']  #exclusion layer
    # exclusionAreas__class_exclusion_buffer=exclusion_areas_config['class_exclusion']['buffer'][f'{resource_type}'] #m
    
# %%
# Load Data Files ------------------------------
    gadm_regions_gdf = gpd.read_file(gadm_file)
    aeroway_with_buffer=gpd.read_parquet(aeroway_file_path)
    conservation_lands_province_datafile:str= os.path.join(linking_data['root'],linking_data['CPCAD_org'])
    # gov_conservation_lands_consideration=config['Gov']['conservation_lands']['consideration'][f'{resource_type}']
    conservation_lands_province=gpd.read_parquet(conservation_lands_province_datafile)

    log.info(F"Loading Transmission Nodes' data")
    buses_gdf = gpd.GeoDataFrame(pd.read_pickle(os.path.join('data/processed_data',linking_data['transmission']['nodes_datafile'])))

# %%
    ## load ERA5 Cutout 
    log.info(f"Loading ERA5 Cutout...")
    # start_date = str(cutout_year)+"-01-01"
    start_date = "2021-01-01 07:00:00" #sync with pypsa config
    # end_date = str(cutout_year)+"-12-31"
    end_date = '2022-01-01 06:00:00' #sync with pypsa config
    # cutout=atlite.Cutout(os.path.join(config['cutout']['directory'],f"{_CRC_}_{cutout_year}.nc"))
    cutout=atlite.Cutout('data/downloaded_data/cutout/BC_2021_2022.nc')

# %%
# 1 Extract BC Grid Cells from Cutout using Regional Boundaries from GADM
    
    province_grid_cells = cutout.grid.overlay(gadm_regions_gdf, how='intersection',keep_geom_type=True)
    log.info(f"Extracted {len(province_grid_cells)} ERA5 Grid Cells for {current_region['code']} from Cutout")

# %%
# 2 Calculate Potential Capacity for Cells
    ## 2.1-2.2
    # ERA5_Land_cells_noExclusion=calculate_land_availability_raster(cutout,province_grid_cells,' Excluding exclusion areas',gaez_exclusionAreas_raster,'1_land_avail_exclAreas',exclusionAreas__class_exclusion,actual_area_ROI, buffer=exclusionAreas__class_exclusion_buffer,exclusion=True)
    ERA5_Land_cells_low_slop=utils.calculate_land_availability_raster(cutout,province_grid_cells,' Excluding terrain  with >30% slope',gaez_terrain_raster,'2_land_avail_low_slop',terrain_class_exclusion,current_region,buffer=0,exclusion=True)
    ERA5_Land_cells_eligible_landclasses=utils.calculate_land_availability_raster(cutout,ERA5_Land_cells_low_slop,' Eligible Land Classes',gaez_landcover_raster,'3_land_avail_eligible',land_class_inclusion,current_region,buffer=0,exclusion=False)

    ERA5_Land_cells_nonprotected=utils.calculate_land_availability_vector_data(cutout,ERA5_Land_cells_eligible_landclasses,conservation_lands_province,'Excluding Conservation and Protected lands by Canadian Gov.','4_land_avail_excl_protectedLands',current_region)
    ERA5_Land_cells_final=utils.calculate_land_availability_vector_data(cutout,ERA5_Land_cells_nonprotected,aeroway_with_buffer,'Excluding Aeroway with buffer','5_land_avail_excl_aeroway',current_region)
    # ERA5_Land_cells_nonaeroway=utils.calculate_land_availability_vector_data(cutout,ERA5_Land_cells_nonprotected,aeroway_with_buffer,'Excluding Aeroway with buffer','5_land_avail_excl_aeroway',current_region)
   
    # ERA5_Land_cells_final=ERA5_Land_cells_eligible_landclasses
    ERA5_Land_cells_final['land_availablity']=ERA5_Land_cells_final['eligible_land_area']/ERA5_Land_cells_final['land_area_sq_km']
    
    ## vis.plot_data_in_GADM_regions(dataframe,data_column_df,gadm_regions_gdf,color_map,dpi,plt_title,plt_file_name,vis_directory
    vis.plot_data_in_GADM_regions(ERA5_Land_cells_final,'land_availablity',
                                    gadm_regions_gdf,
                                    "Greens",600,f"Land Availability ({resource_type})",
                                    f'Land Availability for Potential {resource_type} Plants.png',vis_dir)

    ## 2.3
    province_grid_cells_cap=utils.calculate_potential_capacity(ERA5_Land_cells_final,resource_landuse_intensity,'cell')

    vis.plot_data_in_GADM_regions(ERA5_Land_cells_final,'potential_capacity',
                                    gadm_regions_gdf,
                                    "Blues",600,f"Potential Capacity ({resource_type})",
                                    f'Land Availability for Potential {resource_type} Plants.png',vis_dir)

# %%
# 3 Find Nearest Grid Nodes and Proximity to these Nodes
    province_grid_cells_cap=utils.find_grid_nodes_ERA5_cells(current_region,buses_gdf,province_grid_cells_cap)
    province_grid_cells_cap=province_grid_cells_cap.loc[:,['x', 'y', 'COUNTRY', 'Province', 'Region',
        'Region_ID', 'land_area_sq_km','land_availablity', 'eligible_land_area', 'potential_capacity', 'nearest_station',
        'nearest_station_distance_km','geometry']]

# %%
# 4 Create CF timeseries
    panel_config = disaggregation_config['solar']['atlite_panel']
    tracking_config = disaggregation_config['solar']['tracking']
    
    log.info(f">> Calculating Generation in Grid Cells as per the Layout Capacity (MW)...")
    province_grid_CF_cells,province_grid_CF_ts_df= solar.create_CF_timeseries_df(cutout,start_date,end_date,province_grid_cells_cap,panel_config,tracking_config,Site_index='cell')

    zero_CF_mask=province_grid_CF_cells.CF_mean>0
    province_grid_CF_cells = province_grid_CF_cells[zero_CF_mask]
# %%

    province_grid_CF_ts_df.to_pickle(os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['ERA5_CF_ts']))
    vis.plot_data_in_GADM_regions(province_grid_CF_cells,'CF_mean',
                                gadm_regions_gdf,
                                'Oranges',600,f'CF_mean ({resource_type})',
                                f'CF_mean of Potential {resource_type} Plants.png',vis_dir)
# %%
    
    province_grid_CF_cells['capex']=solar_capex

# 5 Calculate Scores for Cells
    province_grid_cells_scored=utils.calculate_cell_score(province_grid_CF_cells,grid_connection_cost_per_Km,tx_line_rebuild_cost,'CF_mean',solar_capex)
    scored_cells_save_to=os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['scored_cells'])
    

    province_grid_cells_scored.to_pickle(scored_cells_save_to)
# %%
# 6 Find optimal number of Clusters from K-means clustering 
    ## > k-means clustering based on the Scores of cells in each region.
    ## 6.1-6.3
    ERA5_cells_cluster_map,region_solar_optimal_k_df = utils.cells_to_cluster_mapping(province_grid_cells_scored,vis_dir,wcss_tolerance)

# %%
# 7 Create Clusters
    ## 7.1
    cell_cluster_gdf, dissolved_indices = utils.create_cells_Union_in_clusters(ERA5_cells_cluster_map, region_solar_optimal_k_df)
    
    dissolved_indices_save_to=os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['dissolved_indices'])
    utils.dict_to_pickle(dissolved_indices,dissolved_indices_save_to)

# %%
## 7.2
    cell_cluster_gdf_cropped=utils.clip_cluster_boundaries_upto_regions(cell_cluster_gdf,gadm_regions_gdf)
        
    cell_cluster_gdf_cropped['fom']=solar_fom
    cell_cluster_gdf_cropped['vom']=solar_vom
    
    save_to=os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['cell_clusters'])
    cell_cluster_gdf_cropped.to_pickle(save_to)
    
    print(f"{len(cell_cluster_gdf)} {resource_type} Sites' Clusters Generated.\n Total Capacity : {cell_cluster_gdf.potential_capacity.sum()/1E3} GW")
    
    log.info (f" {resource_type} Module Execution Completed !")

    # End the script run  timer
    script_end_time = time.time()

    # Calculate runtime in seconds
    runtime = round((script_end_time - script_start_time),2)
    
    return log.info (f"Script runtime: {runtime} seconds")  
# %%
if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file '*.yml'")
    parser.add_argument('resource_type', type=str, help=f"Resource Type'")

    # Parse the arguments
    
    #----------------Main code to be used ----------------
    args = parser.parse_args()
    main(args.config,args.resource_type)
    
    #----------------------- for notebook run/Debugging------------------------------------
    # config_file_path='config/config_master.yml'
    # main(config_file_path,resource_type)