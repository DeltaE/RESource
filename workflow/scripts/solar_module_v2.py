import logging as log
import os,sys,time,argparse
import geopandas as gpd


# Local Packages
try:
    # Try importing from the submodule context
    import linkingtool.linking_utility as utils
    import linkingtool.linking_vis as vis
    import linkingtool.linking_solar as solar
    from linkingtool.attributes_parser import AttributesParser
    from linkingtool.cell_capacity_processor import cell_capacity_processor
except ImportError:
    # Fallback for when running as a standalone script or outside the submodule
    import Linking_tool.linkingtool.linking_utility as utils
    import Linking_tool.linkingtool.linking_vis as vis
    import Linking_tool.linkingtool.linking_solar as solar
    from Linking_tool.linkingtool.attributes_parser import AttributesParser
    from Linking_tool.linkingtool.cell_capacity_processor import cell_capacity_processor

class SolarModuleProcessor:
    def __init__(self, 
                 config_file_path: str, 
                 resource_type: str):
        
        self.resource_type = resource_type.lower()
        
        # Initialize Attributes Parser Class
        self.attributes_parser: attributes_parser = AttributesParser(config_file_path, self.resource_type)
        
        # Initialize era5_cell_capacity_processor
        self.cell_processor:cell_capacity_processor = cell_capacity_processor(config_file_path, self.resource_type)
        
        # Extract Attributes
        self.current_region:dict = self.attributes_parser.current_region
        self.region_code:str = self.attributes_parser.region_code
        self.resource_disaggregation_config :dict= self.attributes_parser.disaggregation_config[self.resource_type]
        self.vis_dir :str= self.attributes_parser.vis_dir
        self.linking_data:dict = self.attributes_parser.linking_data
        self.gaez_data:dict = self.attributes_parser.gaez_data
        self.wcss_tolerance:float= self.resource_disaggregation_config['WCSS_tolerance']
        self.grid_node_proximity_filter=self.attributes_parser.disaggregation_config['transmission']['proximity_filter'] 
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
        
    def find_grid_nodes(self):
        self.province_grid_cells_cap_with_nodes = utils.find_grid_nodes_ERA5_cells(self.current_region, self.buses_gdf, self.province_grid_cells_capacity,self.grid_node_proximity_filter)

    def create_CF_timeseries(self):
        panel_config = self.resource_disaggregation_config['atlite_panel']
        tracking_config = self.resource_disaggregation_config['tracking']
        self.province_grid_CF_cells, self.province_grid_CF_ts_df = solar.create_CF_timeseries_df(self.cutout, self.start_date, self.end_date, self.province_grid_cells_cap_with_nodes, panel_config, tracking_config, Site_index='cell')
        zero_CF_mask = self.province_grid_CF_cells.CF_mean > 0
        self.province_grid_CF_cells = self.province_grid_CF_cells[zero_CF_mask]

    def save_results(self):
        save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['ERA5_CF_ts'])
        self.province_grid_CF_ts_df.to_pickle(save_to)

    def score_cells(self):
        self.province_grid_cells_scored = utils.calculate_cell_score(self.province_grid_CF_cells, self.grid_connection_cost_per_km, self.tx_line_rebuild_cost, 'CF_mean', self.resource_capex)
        scored_cells_save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['scored_cells'])
        self.province_grid_cells_scored.to_pickle(scored_cells_save_to)

    def cluster_cells(self):
        self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df = utils.cells_to_cluster_mapping(self.province_grid_cells_scored, self.vis_dir, self.wcss_tolerance)
        self.cell_cluster_gdf, self.dissolved_indices = utils.create_cells_Union_in_clusters(self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df)
        dissolved_indices_save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['dissolved_indices'])
        utils.dict_to_pickle(self.dissolved_indices, dissolved_indices_save_to)

        
    def progress_bar(self, iteration, total, bar_length=40):
            percent = (iteration / total)
            arrow = '█' * int(round(percent * bar_length) - 1)
            spaces = ' ' * (bar_length - len(arrow))
            sys.stdout.write(f'\rProgress: |{arrow}{spaces}| {percent:.1%}')
            sys.stdout.flush()

    def run(self):
            log.info(f"{self.resource_type} module initiated")
            
            tasks = [
                self.find_grid_nodes,
                self.create_CF_timeseries,
                self.save_results,
                self.score_cells,
                self.cluster_cells
            ]
            
            total_tasks = len(tasks)
            
            for i, task in enumerate(tasks):
                task()  # Execute the task
                self.progress_bar(i + 1, total_tasks)  # Update the progress bar
                
            print()  # For a newline after progress completion
            print(f"{len(self.cell_cluster_gdf)} {self.resource_type} Sites' Clusters Generated.\n Total Capacity: {self.cell_cluster_gdf.potential_capacity.sum() / 1E3} GW")
            log.info(f"{self.resource_type} Module Execution Completed!")


def main(config_file_path: str, 
         resource_type: str='solar'):
    
    script_start_time = time.time()
    
    solar_module = SolarModuleProcessor(config_file_path, resource_type)
    solar_module.run()
    
    script_runtime = round((time.time() - script_start_time), 2)
    
    log.info(f"Script runtime: {script_runtime} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run solar module script')
    parser.add_argument('config', type=str, help=f"Path to the configuration file '*.yml'")
    parser.add_argument('resource_type', type=str, help='Specify resource type e.g. solar')
    args = parser.parse_args()
    main(args.config, args.resource_type)