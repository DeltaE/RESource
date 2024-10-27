# %% [markdown]
import time
# Start the script run timer
script_start_time = time.time()

# Top Site Selection Module

# %%
import argparse
import logging as log
import os
import pandas as pd
import geopandas as gpd

# local packages
import linkingtool.utility as utils
import linkingtool.linking_wind as wind
import linkingtool.visuals as vis
import linkingtool.linking_solar as solar

# %%
# # Top Site Selection for the Resource Sites' Pool
def main(
    config_file_path:str,
    resource_type: str,
    resource_max_total_capacity: float):
    


# %%
# for notebook run
    # config_file_path='config/config_master.yml'
    # resource_type='wind'

    # %% [markdown]
    # 
    # log_path=f'workflow/log/{resource_type}_top_sites_module_log.txt'
    # utils.create_log(log_path)
    # utils.print_module_title(f'{resource_type} Top Sites Selection Module Initiated')

    # %%
    config=utils.load_config(config_file_path)
    current_region=config['regional_info']['region_1']
    _CRC_=current_region['code']
    disaggregation_config=config['capacity_disaggregation'][f'{resource_type}']

    log.info(f"Loading Configuration and Directories set-up...")
    
    
    print(f"Resource Type: {resource_type}")
    if resource_max_total_capacity is not None:
        print(f"Resource Max Total Capacity: {resource_max_total_capacity} GW")
        resource_max_capacity = resource_max_total_capacity # GW
    else:
        print("Resource Max Total Capacity: Not provided. \n Setting the value from CONFIG.")
        resource_max_capacity = disaggregation_config['max_capacity']  # GW

    vis_dir=os.path.join(config['visualization']['linking'],resource_type)
    linking_data:dict=config['processed_data']['linking']
    result_files:dict=config['results']['linking']


    log.info(f"Loading linking data files..")
    cell_clusters=gpd.GeoDataFrame(pd.read_pickle(os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['cell_clusters'])))
    # dissolved_indices=utils.load_dict_datafile(os.path.join(wind_processed_data_directory,'dissolved_indices.json'))
    if resource_type!='bess':
        dissolved_indices=pd.read_pickle(os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['dissolved_indices']))
        province_grid_CF_ts_df=gpd.GeoDataFrame(pd.read_pickle(os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['ERA5_CF_ts'])))
        scored_cells=gpd.GeoDataFrame(pd.read_pickle(os.path.join(os.path.join(linking_data['root'],resource_type,linking_data[f'{resource_type}']['scored_cells']))))

    # %%
    # 1. Select the Top Sites 
    selected_sites = utils.select_top_sites(cell_clusters, resource_max_capacity)

    ## Check if any sites were selected
    if not selected_sites.empty:
        print(f"> {len(selected_sites)} Top Sites Selected. \n"
            f">> Total Capacity: {round(selected_sites['potential_capacity'].sum() / 1000, 2)} GW")
    else:
        print("No sites selected.")

    #just organizing the columns sequence and keeping the required columns only
    selected_sites.loc[:,['Region', 'Cluster_No','Region_ID', 'potential_capacity',
        'nearest_station', 'nearest_station_distance_km','p_lcoe', 'geometry','capex', 'fom', 'vom']] #'Site_ID','CF_mean',

    # # skipping geom data for nexus data prep
    # columns_to_save = [col for col in selected_sites.columns if col != 'geometry']
    # selected_sites_nexus_datafile = selected_sites[columns_to_save]

    # selected_sites_nexus_datafile.to_csv((os.path.join(result_files['root'],result_files['clusters_topSites'][f'{resource_type}'])))

    # %%
    # 2. Create/Load the Representative Timeseries for the Selected Sites
    if resource_type=="bess":
        log.info(f"Time-slice creation and visuals not required for BESS.")
    else:
        if resource_type=="wind":
            CF_ts_clusters_df,within_cluster_cells_ts_df= wind.create_timeseries_for_Cluster(selected_sites,dissolved_indices,scored_cells,province_grid_CF_ts_df)
            # CF_ts_clusters_df -> Clusters' representative timeseries
            # within_cluster_cells_ts_df -> individual GWA cell's timeseries
        elif resource_type=="solar": #kept it dedicated to solar as future expansion version will include other resources ~
            CF_ts_clusters_df=solar.create_timeseries_for_Cluster(selected_sites,dissolved_indices,province_grid_CF_ts_df)
            # within_cluster_cells_ts_df=province_grid_CF_ts_df
        
        
        ### Save Sites' Timeseries data file Locally
        # CF_ts_clusters_df_save_to=os.path.join(os.path.join(os.path.join(linking_data['root'],linking_data[f'{resource_type}']['cell_cluster_ts'])))
        # CF_ts_clusters_df.to_pickle(CF_ts_clusters_df_save_to)
        # log.info (f"Timeseries for {resource_type} - ALL sites created and saved locally at - {CF_ts_clusters_df_save_to}.")
        
    # 3. Prepare top sites (clusters) representative timeseries

        CF_ts_df_Top_sites = CF_ts_clusters_df[selected_sites.index]
        CF_ts_df_Top_sites=utils.fix_df_ts_index(CF_ts_df_Top_sites,current_region['snapshots_tz_BC'],snapshot_serial=0)

        ## Save File Locally
        result_files:dict=config['results']['linking']
        log.info (f"Plotting timeseries for {len(CF_ts_clusters_df)} top sites for {resource_type}...")
        CF_ts_df_Top_sites.to_pickle(os.path.join(result_files['root'],result_files['clusters_CFts_topSites'][f'{resource_type}']))
        log.info (f"Timeseries for {resource_type} top sites created and saved locally.")

    # 3. Visualization of the Site - Timeseries
        
        plots_save_to=os.path.join(vis_dir,'Site_timeseries')
        # resampling_span:str=result_files['visual_resampling'][f'{resource_type}']
        # vis.create_timeseries_plots(selected_sites, within_cluster_cells_ts_df,resource_max_capacity,dissolved_indices,resampling_span,'blue','skyblue',plots_save_to) 
        # log.info(f"Static plots for {resource_type} site's timeseries created and saved locally at : '{plots_save_to}'") 
        
        vis.create_timeseries_interactive_plots(CF_ts_df_Top_sites,plots_save_to)
        log.info(f"Interactive plots for {resource_type} site's timeseries created and saved locally at : '{plots_save_to}'")
        
        vis.create_sites_ts_plots_all_sites(resource_type,CF_ts_df_Top_sites,plots_save_to)
        vis.create_sites_ts_plots_all_sites(resource_type,CF_ts_df_Top_sites,f'results/linking')


    selected_sites['CF_mean']=selected_sites.index.map(CF_ts_df_Top_sites.mean())

    ## Save Site's Spatial, Technical data file Locally
    selected_sites.to_pickle(os.path.join(result_files['root'],result_files['clusters_topSites'][f'{resource_type}']))


    log.info (f"Top Sites Selection for {resource_type} - Execution Completed !")



    # %% [markdown]
    # # skip this part for notebook run

    # %%
    # End the script run  timer
    script_end_time = time.time()

    # Calculate runtime in seconds
    runtime = round((script_end_time - script_start_time),2)

    return log.info (f"Script runtime: {runtime} seconds")
if __name__ == "__main__":      

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file 'config_master.yml'. Ideally in this directory: 'config/config_master.yml' ")
    parser.add_argument('resource_type', choices=['wind', 'solar','bess'], help="Type of resource: 'wind' or 'solar' or 'bess' ")
    # Make resource_max_total_capacity optional with nargs='?' and provide a default value
    parser.add_argument('resource_max_total_capacity', nargs='?', type=float, default=None, help="Maximum total capacity for the resource (optional)")
    # Parse the arguments
    
    #----------------Main code to be used ----------------
    args = parser.parse_args()
    # Run the main function with arguments
    main(args.config, args.resource_type, args.resource_max_total_capacity)
    
    #----------------------- for notebook run/Debugging------------------------------------
    # config_file_path='config/config_master.yml'
    # main(config_file_path,resource_type='wind')




if __name__ == "__main__":      
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file 'config_master.yml'. Ideally in this directory: 'config/config_master.yml'")
    parser.add_argument('resource_type', choices=['wind', 'solar', 'bess'], help="Type of resource: 'wind', 'solar', or 'bess'")
    
    # Make resource_max_total_capacity optional with nargs='?' and provide a default value
    parser.add_argument('resource_max_total_capacity', nargs='?', type=float, default=None, help="Maximum total capacity (GW) for the resource (optional)")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with arguments
    main(args.config, args.resource_type, args.resource_max_total_capacity)

    #----------------------- for notebook run/Debugging------------------------------------
    # config_file_path = 'config/config_master.yml'
    # main(config_file_path, resource_type='wind', resource_max_total_capacity=None)
