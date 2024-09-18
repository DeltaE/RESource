# %%
# Wind Module
import time
# Start the script run timer
script_start_time = time.time()

# %%
resource_type:str="wind" #use all small letters

# %%
import logging as log
import os
import argparse
import atlite
import geopandas as gpd
import pandas as pd

# %%
# import Local Packages
from linkingtool import linking_utility as utils
from linkingtool import linking_vis as vis
from linkingtool import linking_wind as wind

log_path=f'workflow/log/{resource_type}_module_log.txt'
log=utils.create_log(log_path)

# %%
utils.print_module_title(f'{resource_type} Module Initiated')
log.info(f"Loading Configuration Data and Directories...")

# %%
'''--------- for notebook run purposes 
config_file_path='config/config_master.yml'
'''
def main(config_file_path:str,
         resource_type:str):

# %%
    # User Configuration

    config:dict[dict]=utils.load_config(config_file_path)

    current_region:dict = config['regional_info']['region_1']
    _CRC_=current_region['code'] # Current Region Code i.e. BC

    disaggregation_config:dict=config['capacity_disaggregation']

    linking_data:dict=config['processed_data']['linking']
    vis_dir:str=os.path.join(config['visualization']['linking'],resource_type)

    resource_landuse_intensity = disaggregation_config[f'{resource_type}']['landuse_intensity'] # (MW/km2) from literature 1.7 MW/km2

    ATB_NREL_cost_datafile=disaggregation_config[f'{resource_type}']['cost_data']
    land_based_wind_cost=pd.read_csv(ATB_NREL_cost_datafile)

    # Set COST parameters (global for PV)
    wind_fom=land_based_wind_cost[land_based_wind_cost['core_metric_parameter']=='Fixed O&M'].value.iloc[0]/1E3  # mill. $/MW = ($/kw ) /1E3
    print(f"Wind Site FOM set to: {round(wind_fom,2)} Mil. USD/ MW. Sourced from >> Summary data, Land-based Wind Turbine Cost, ATB 2024 , NREL ")

    wind_capex=land_based_wind_cost[land_based_wind_cost['core_metric_parameter']=='CAPEX'].value.iloc[0]/1E3 # mill. $/MW = ($/kw ) /1E3
    print(f"Wind Site CAPEX set to: {round(wind_capex,2)} Mil. USD/ MW. Sourced from >> Summary data, Land-based Wind Turbine Cost, ATB 2024 , NREL ")

    wind_vom=0

    grid_connection_cost_per_Km = disaggregation_config['transmission']['grid_connection_cost_per_Km'] #M$/km   # from MISO , may try NREL method for different range of distance
    tx_line_rebuild_cost = disaggregation_config['transmission']['tx_line_rebuild_cost']  #M$/km # from MISO

    grid_node_proximity_filter = disaggregation_config['transmission']['proximity_filter']
    wcss_tolerance=disaggregation_config[f'{resource_type}']['WCSS_tolerance']
    province_gwa_cells = os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['gwa_cells_raw'])
    turbine_model= disaggregation_config[f'{resource_type}']['turbines']['OEDB']['model_2'] 
    turbine_config_file=disaggregation_config[f'{resource_type}']['turbines']['OEDB']['model_2']['config'] # to be automated
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
    # Load data-files
    ## Regional Boundary Data --------------------------------------------------------
    log.info(F"Loading GADM's regional boundary data")
    gadm_regions_gdf = gpd.read_file(gadm_file)
    aeroway_with_buffer=gpd.read_parquet(aeroway_file_path)
    conservation_lands_province_datafile:str= os.path.join(linking_data['root'],linking_data['CPCAD_org'])
    ## gov_conservation_lands_consideration=config['Gov']['conservation_lands']['consideration'][f'{resource_type}']
    conservation_lands_province=gpd.read_parquet(conservation_lands_province_datafile)

    log.info(F"Loading Transmission Nodes' data")
    buses_gdf = gpd.GeoDataFrame(pd.read_pickle(os.path.join('data/processed_data',linking_data['transmission']['nodes_datafile'])))

    # %%
    ## load ERA5 Cutout

    log.info(f"Loading ERA5 Cutout...")

    ''' replaced with different strategy, to sync with pypsa
    start_date = str(cutout_year)+"-01-01"
    end_date = str(cutout_year)+"-12-31"
    cutout=atlite.Cutout(os.path.join(config['cutout']['directory'],f"{_CRC_}_{cutout_year}.nc"))
    start_date = str(cutout_year)+"-01-01"
    end_date = str(cutout_year)+"-12-31"
    '''
    start_date = config['cutout']['snapshots']['start'][0] # 2021-01-01 07:00:00
    end_date = config['cutout']['snapshots']['end'][0] # 2022-01-01 06:00:00

    # cutout=atlite.Cutout(os.path.join(config['cutout']['directory'],f"{_CRC_}_{cutout_year}.nc"))
    cutout=atlite.Cutout('data/downloaded_data/cutout/BC_2021_2022.nc')

    # %%
    # 1 Extract BC Grid Cells from Cutout using Regional Boundaries from GADM
    province_grid_cells = cutout.grid.overlay(gadm_regions_gdf, how='intersection',keep_geom_type=True)
    log.info(f"Extracted {len(province_grid_cells)} ERA5 Grid Cells for BC from Cutout")

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
    province_grid_cells_2=utils.calculate_potential_capacity(ERA5_Land_cells_final,resource_landuse_intensity,'cell')

    vis.plot_data_in_GADM_regions(ERA5_Land_cells_final,'potential_capacity',
                                    gadm_regions_gdf,
                                    "Blues",600,f"Potential Capacity ({resource_type})",
                                    f'Land Availability for Potential {resource_type} Plants.png',vis_dir)

    # %%
    # --------------------------Extra steps in WIND MODULE starts here-------------------------------
    ## 2.4
    minimum_site_capacity_mask=province_grid_cells_2['potential_capacity']>=disaggregation_config[f'{resource_type}']['turbines']['OEDB']['model_2']['P'] #nominal power of the turbine
    province_grid_cells_3=province_grid_cells_2[minimum_site_capacity_mask]
    print(f'FIltered Site : Total {resource_type} Potential (based on available land): {round(province_grid_cells_3.potential_capacity.sum()/1000,2)} GW \n')

    # %%
    # 3 Extract Windspeed data from Cutout  
    province_grid_cells_4 = wind.impute_ERA5_windspeed_to_Cells(cutout, province_grid_cells_3)

    # %%
    # 4 GWA Cell Processing
    ## 4.1 Load GWA Cells

    gwa_cells_df:pd.DataFrame=pd.read_pickle(province_gwa_cells)
    log.info(f"Global Wind Atlas (GWA) Cells loaded. Size: {len(gwa_cells_df)}")

    gwa_cells_gdf:gpd.GeoDataFrame = gpd.GeoDataFrame(gwa_cells_df, geometry=gpd.points_from_xy(gwa_cells_df['x'], gwa_cells_df['y']))
    log.info(f"Point geometries created for {len(gwa_cells_df)} Cells\n")

    gwa_cells_gdf.crs = province_grid_cells.crs
    gwa_cells_gdf=gwa_cells_gdf.clip(gadm_regions_gdf,keep_geom_type=False)

    gwa_cells_gdf,_=wind.calculate_common_parameters_GWA_cells(gwa_cells_gdf,resource_landuse_intensity)

    # %%
    ## 4.2 Map GWA cells to ERA5
    gwa_cells_mapped_gdf,era5_cells_gdf_mapped=wind.map_GWAcells_to_ERA5cells(gwa_cells_gdf,province_grid_cells_4,resource_landuse_intensity)

    # %%
    ## 4.3 Update ERA5 windspeed and CF with mapped GWA Cells
    era5_cells_gdf_updated=wind.update_ERA5_params_from_mapped_GWA_cells(era5_cells_gdf_mapped,gwa_cells_mapped_gdf)

    # %%
    # 5 Rescale ERA5 Cutout Windspeed 
    ### > Each Windspeed Datapoint scaled with Scalar values from GWA Cells
    cutout=wind.rescale_ERA5_cutout_windspeed_with_mapped_GWA_cells(cutout,era5_cells_gdf_updated)

    # %%
    # 6 Find Nearest Grid Node
    Gwa_cells_df_GridNode_filtered=wind.find_grid_nodes_GWA_cells(buses_gdf,gwa_cells_mapped_gdf,grid_node_proximity_filter)

    # %%
    # 7 Create CF timeseries
    # arguments: cutout,start_date,end_date,geodataframe_sites,turbine_model,turbine_config_file,Site_index,config='OEDB'):
    province_grid_CF_cells,province_grid_CF_ts_df= wind.create_CF_timeseries_df(cutout,start_date,end_date,era5_cells_gdf_updated,turbine_model,turbine_config_file,'cell','OEDB')

    zero_CF_mask=province_grid_CF_cells.CF_mean_atlite>0
    province_grid_CF_cells= province_grid_CF_cells[zero_CF_mask]

    province_grid_CF_cells.loc[:,'capex']=wind_capex

    province_grid_CF_cells=utils.assign_regional_cell_ids(province_grid_CF_cells,'Region','cell')
    province_grid_CF_cells.to_pickle(os.path.join(linking_data['root'],resource_type,'province_grid_CF_cells.pkl'))

    province_grid_CF_ts_df.to_pickle(os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['ERA5_CF_ts']))
    # Visualize
    vis.plot_data_in_GADM_regions(province_grid_CF_cells,'CF_mean_atlite',
                                gadm_regions_gdf,
                                'coolwarm',600,f'CF_mean ({resource_type})',
                                f'CF_mean of Potential {resource_type} Plants.png',vis_dir)

    # %%
    # 8 Calculate Scores for Cells
    gwa_cells_scored=utils.calculate_cell_score(Gwa_cells_df_GridNode_filtered,grid_connection_cost_per_Km,tx_line_rebuild_cost,'CF_IEC3',wind_capex)
    # gwa_cells_scored['CF_mean'] = gwa_cells_scored['ERA5_cell_index'].map(province_grid_CF_cells['CF_mean_atlite'])

    #  Save Local File
    pickle_file_name = linking_data[f'{resource_type}']['scored_cells']
    gwa_cells_scored.to_pickle(os.path.join(linking_data['root'],resource_type,pickle_file_name))

    # %%
    # 9 Find optimal number of Clusters from K-means clustering 
    ###  > k-means clustering based on the Scores of cells in each region.

    ## 9.1-9.3
    GWA_cells_cluster_map,region_wind_optimal_k_df = utils.cells_to_cluster_mapping(gwa_cells_scored,vis_dir,wcss_tolerance)

    # %%
    # 10 Create Clusters
    cell_cluster_gdf, dissolved_indices = utils.create_cells_Union_in_clusters(GWA_cells_cluster_map, region_wind_optimal_k_df)

    dissolved_indices_save_to=os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['dissolved_indices'])
    utils.dict_to_pickle(dissolved_indices,dissolved_indices_save_to)

    # %%
    ## 10.1
    minimum_site_capacity_mask=cell_cluster_gdf['potential_capacity']>=disaggregation_config[f'{resource_type}']['turbines']['OEDB']['model_2']['P'] #nominal power of the turbine
    cell_cluster_gdf_filtered=cell_cluster_gdf[minimum_site_capacity_mask]

    log.info(f'FIltered Clusters : Total {resource_type} Potential : {round(cell_cluster_gdf_filtered.potential_capacity.sum()/1000,2)} GW \n')

    ## 10.2
    cell_cluster_gdf_filtered_c=utils.clip_cluster_boundaries_upto_regions(cell_cluster_gdf_filtered,gadm_regions_gdf)

    cell_cluster_gdf_filtered_c['fom']=wind_fom
    cell_cluster_gdf_filtered_c['vom']=wind_vom

    save_to=os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['cell_clusters'])
    cell_cluster_gdf_filtered_c.to_pickle(save_to)

    log.info(f"{len(cell_cluster_gdf)} {resource_type} Sites' Clusters Generated.\n Total Capacity : {cell_cluster_gdf.potential_capacity.sum()/1E3} GW")

    log.info (f" {resource_type} Module Execution Completed !")


# ### >>>>>>>>>>>> ----------- Notebook run ends here

# %% [markdown]
# # skip this for notebook run

# %%
     
    # End the script run  timer
    script_end_time = time.time()

    # Calculate runtime in seconds
    runtime = round((script_end_time - script_start_time),2)
    
    return log.info (f"Script runtime: {runtime} seconds")

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file 'config_master.yml'")
    parser.add_argument('resource_type', type=str, help=f"Resource Type'")

    # Parse the arguments
    
    #----------------Main code to be used ----------------
    args = parser.parse_args()
    main(args.config,args.resource_type)
    
    # ----------------------- for notebook run/Debugging------------------------------------
    # config_file_path='config/config_master.yml'
    # main(config_file_path)


