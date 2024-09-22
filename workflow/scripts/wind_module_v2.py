import time,os, argparse
import logging as log
import atlite
import geopandas as gpd
import pandas as pd
from linkingtool import linking_utility as utils
from linkingtool import linking_vis as vis
from linkingtool import linking_wind as wind
from linkingtool.cell_capacity_processor import cell_capacity_processor
from linkingtool.attributes_parser import AttributesParser

class WindModuleProcessor:
    def __init__(self, 
                 config_file_path: str, 
                 resource_type: str):
        
        self.resource_type = resource_type.lower()
        
        # Initialize Attributes Parser Class
        self.attributes_parser: AttributesParser = AttributesParser(config_file_path, self.resource_type)
        
        # Initialize era5_cell_capacity_processor
        self.cell_processor:cell_capacity_processor = cell_capacity_processor(config_file_path, self.resource_type)
        
        # Extract Attributes
        self.current_region:dict = self.attributes_parser.current_region
        self.region_code:str = self.attributes_parser.region_code
        self.disaggregation_config=self.attributes_parser.disaggregation_config
        self.resource_disaggregation_config :dict= self.attributes_parser.disaggregation_config[self.resource_type]
        self.vis_dir :str= self.attributes_parser.vis_dir
        self.linking_data:dict = self.attributes_parser.linking_data
        self.gaez_data:dict = self.attributes_parser.gaez_data
        self.wcss_tolerance:float= self.resource_disaggregation_config['WCSS_tolerance']
        self.turbine_model= self.resource_disaggregation_config['turbines']['OEDB']['model_2'] 
        self.turbine_config_file=self.resource_disaggregation_config['turbines']['OEDB']['model_2']['config'] # to be automated
        self.grid_node_proximity_filter=self.attributes_parser.disaggregation_config['transmission']['proximity_filter']  #may need to create diff for solar and wind
        self.resource_landuse_intensity = self.resource_disaggregation_config['landuse_intensity']
        # Snapshot (range of of the temporal data)
        (
            self.start_date,
            self.end_date,
        ) = self.attributes_parser.load_snapshot()
        
        # Static cost params (static: fixed for entire snapshot)
        (
            self.resource_capex, 
            self.resource_fom, 
            self.resource_vom,
            self.grid_connection_cost_per_km,
            self.tx_line_rebuild_cost
        ) = self.attributes_parser.load_cost()
        
        # Load geospatial data (gdf) and unpack
        (
            self.cutout,
            self.gadm_regions_gdf,
            self.aeroway_with_buffer,
            self.conservation_lands_province,
            self.buses_gdf
        ) = self.attributes_parser.load_geospatial_data()
        
        self.province_grid_cells_capacity:gpd.GeoDataFrame=self.cell_processor.run()

    def filter_site_capacity(self):
        minimum_site_capacity_mask = self.province_grid_cells_capacity['potential_capacity'] >= self.disaggregation_config[
            f'{self.resource_type}']['turbines']['OEDB']['model_2']['P']
        province_grid_cells_filtered:gpd.GeoDataFrame = self.province_grid_cells_capacity[minimum_site_capacity_mask]
        print(f'Filtered Site : Total {self.resource_type} Potential: {round(province_grid_cells_filtered.potential_capacity.sum()/1000, 2)} GW \n')
        return province_grid_cells_filtered

    def extract_windspeed_data(self, province_grid_cells_filtered):
        province_grid_cells_filtered_updated:gpd.GeoDataFrame=wind.impute_ERA5_windspeed_to_Cells(self.cutout, province_grid_cells_filtered)
        return province_grid_cells_filtered_updated

    def load_gwa_cells(self):
        self.province_gwa_cells = os.path.join(self.linking_data['root'],self.resource_type,self.linking_data[self.resource_type]['gwa_cells_raw'])
        gwa_cells_df = pd.read_pickle(self.province_gwa_cells)
        log.info(f"Global Wind Atlas (GWA) Cells loaded. Size: {len(gwa_cells_df)}")

        gwa_cells_gdf = gpd.GeoDataFrame(gwa_cells_df, geometry=gpd.points_from_xy(gwa_cells_df['x'], gwa_cells_df['y']))
        log.info(f"Point geometries created for {len(gwa_cells_df)} Cells\n")

        gwa_cells_gdf.crs = self.gadm_regions_gdf.crs
        gwa_cells_gdf = gwa_cells_gdf.clip(self.gadm_regions_gdf, keep_geom_type=False)
        
        gwa_cells_gdf, _ = wind.calculate_common_parameters_GWA_cells(gwa_cells_gdf, self.resource_landuse_intensity)
        
        return gwa_cells_gdf

    def map_gwa_cells_to_era5(self, gwa_cells_gdf, province_grid_cells_filtered_updated):
        return wind.map_GWAcells_to_ERA5cells(gwa_cells_gdf, province_grid_cells_filtered_updated, self.resource_landuse_intensity)

    def update_era5_params(self, era5_cells_gdf_mapped, gwa_cells_mapped_gdf):
        return wind.update_ERA5_params_from_mapped_GWA_cells(era5_cells_gdf_mapped, gwa_cells_mapped_gdf)

    def rescale_cutout_windspeed(self, cutout, era5_cells_gdf_updated):
        return wind.rescale_ERA5_cutout_windspeed_with_mapped_GWA_cells(cutout, era5_cells_gdf_updated)

    def find_grid_nodes(self, gwa_cells_mapped_gdf):
        return wind.find_grid_nodes_GWA_cells(self.buses_gdf, gwa_cells_mapped_gdf, self.grid_node_proximity_filter)

    def create_cf_timeseries(self,cutout_updated, era5_cells_gdf_updated):
        province_grid_CF_cells, province_grid_CF_ts_df = wind.create_CF_timeseries_df(
            cutout_updated, 
            self.start_date,
            self.end_date, 
            era5_cells_gdf_updated, 
            self.turbine_model, 
            self.turbine_config_file, 
            'cell', 
            'OEDB')

        zero_CF_mask = province_grid_CF_cells.CF_mean_atlite > 0
        province_grid_CF_cells = province_grid_CF_cells[zero_CF_mask]

        province_grid_CF_cells = utils.assign_regional_cell_ids(province_grid_CF_cells, 'Region', 'cell')
        province_grid_CF_cells.to_pickle(os.path.join(self.linking_data['root'], self.resource_type, 'province_grid_CF_cells.pkl'))
        province_grid_CF_ts_df.to_pickle(os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[self.resource_type]['ERA5_CF_ts']))

        # Visualize
        vis.plot_data_in_GADM_regions(province_grid_CF_cells, 'CF_mean_atlite', self.gadm_regions_gdf,
                                           'coolwarm', 600, f'CF_mean ({self.resource_type})', f'CF_mean of Potential {self.resource_type} Plants.png', self.vis_dir)

        return province_grid_CF_cells, province_grid_CF_ts_df

    def calculate_cell_scores(self, gwa_cells_df_GridNode_filtered):
        gwa_cells_scored = utils.calculate_cell_score(gwa_cells_df_GridNode_filtered, self.grid_connection_cost_per_km, self.tx_line_rebuild_cost, 'CF_IEC3', self.resource_capex)
        pickle_file_name = self.linking_data[f'{self.resource_type}']['scored_cells']
        gwa_cells_scored.to_pickle(os.path.join(self.linking_data['root'], self.resource_type, pickle_file_name))
        return gwa_cells_scored

    def create_clusters(self, GWA_cells_cluster_map, region_wind_optimal_k_df):
        cell_cluster_gdf, dissolved_indices = utils.create_cells_Union_in_clusters(GWA_cells_cluster_map, region_wind_optimal_k_df)
        dissolved_indices_save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['dissolved_indices'])
        utils.dict_to_pickle(dissolved_indices, dissolved_indices_save_to)
        return cell_cluster_gdf

    def filter_clusters_by_capacity(self, cell_cluster_gdf):
        minimum_site_capacity_mask = cell_cluster_gdf['potential_capacity'] >= self.resource_disaggregation_config['turbines']['OEDB']['model_2']['P']
        cell_cluster_gdf_filtered = cell_cluster_gdf[minimum_site_capacity_mask]

        log.info(f'Filtered Clusters : Total {self.resource_type} Potential : {round(cell_cluster_gdf_filtered.potential_capacity.sum()/1000,2)} GW \n')

        cell_cluster_gdf_filtered_c = utils.clip_cluster_boundaries_upto_regions(cell_cluster_gdf_filtered, self.gadm_regions_gdf)
        cell_cluster_gdf_filtered_c['fom'] = self.resource_fom
        cell_cluster_gdf_filtered_c['vom'] = self.resource_vom

        save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['cell_clusters'])
        cell_cluster_gdf_filtered_c.to_pickle(save_to)

        

        return cell_cluster_gdf_filtered_c

    def run(self):
        province_grid_cells_3 = self.filter_site_capacity()
        province_grid_cells_4 = self.extract_windspeed_data(province_grid_cells_3)
        gwa_cells_gdf = self.load_gwa_cells()
        gwa_cells_mapped_gdf, era5_cells_gdf_mapped = self.map_gwa_cells_to_era5(gwa_cells_gdf, province_grid_cells_4)
        era5_cells_gdf_updated = self.update_era5_params(era5_cells_gdf_mapped, gwa_cells_mapped_gdf)
        cutout_updated = self.rescale_cutout_windspeed(self.cutout, era5_cells_gdf_updated)
        gwa_cells_df_GridNode_filtered = self.find_grid_nodes(gwa_cells_mapped_gdf)
        province_grid_CF_cells, province_grid_CF_ts_df = self.create_cf_timeseries(cutout_updated,era5_cells_gdf_updated)
        gwa_cells_scored = self.calculate_cell_scores(gwa_cells_df_GridNode_filtered)
        GWA_cells_cluster_map, region_wind_optimal_k_df = utils.cells_to_cluster_mapping(gwa_cells_scored, self.vis_dir, self.wcss_tolerance)
        cell_cluster_gdf = self.create_clusters(GWA_cells_cluster_map, region_wind_optimal_k_df)
        final_clusters_gdf=self.filter_clusters_by_capacity(cell_cluster_gdf)
        log.info(f"{len(final_clusters_gdf)} {self.resource_type} Sites' Clusters Generated.\n Total Capacity : {final_clusters_gdf.potential_capacity.sum()/1E3} GW")
        return 

def main(config_file_path: str, 
         resource_type: str='wind'):
    
    script_start_time = time.time()
    
    _module = WindModuleProcessor(config_file_path, resource_type)
    _module.run()
    
    script_runtime = round((time.time() - script_start_time), 2)
    
    log.info(f"Script runtime: {script_runtime} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run solar module script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file '*.yml'")
    parser.add_argument('resource_type', type=str, help='Specify resource type e.g. solar')
    args = parser.parse_args()
    main(args.config, args.resource_type)
    # main('config/config_linking_tool.yml','wind')