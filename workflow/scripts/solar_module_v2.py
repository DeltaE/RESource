import logging as log
import os,sys,time,argparse
import geopandas as gpd
import pandas as pd
import atlite
from collections import namedtuple
# Local Packages

from linkingtool.era5_cutout import ERA5Cutout
import linkingtool.linking_utility as utils
import linkingtool.visuals as vis
# import linkingtool.linking_solar as solar

from linkingtool.CellCapacityProcessor import CellCapacityProcessor

from linkingtool.coders import CODERSData
# from workflow.scripts.prepare_data_v2 import LinkingToolData
from linkingtool.find import GridNodeLocator
from linkingtool.timeseries import Timeseries
from linkingtool.hdf5_handler import DataHandler
from linkingtool.AttributesParser import AttributesParser
from linkingtool.score import CellScorer

class SolarResources(AttributesParser
                    #  Timeseries,
                    #  CellCapacityProcessor,
                    #  CODERSData
                     ):
    
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        # This dictionary will be used to pass arguments to external classes
        self.required_args = {   #order doesn't matter
            "config_file_path" : self.config_file_path,
            "province_short_code": self.province_short_code,
            "resource_type": 'solar'
        }
        
        
        # Initiate Class
        self.timeseries=Timeseries(**self.required_args)
        self.datahandler=DataHandler(self.store)
        self.cell_processor=CellCapacityProcessor(**self.required_args)
        self.coders=CODERSData(**self.required_args)
        self.era5_cutout=ERA5Cutout(**self.required_args)
        self.scorer=CellScorer()

        # Snapshot (range of of the temporal data)
        (
            self.start_date,
            self.end_date,
        ) = self.load_snapshot()
    
    def get_CF_timeseries(self,
                          force_update=False)->tuple:
        
        self.cells_with_ts_nt:tuple= self.timeseries.get_cells_timeseries()
        
        return self.cells_with_ts_nt
    
    def get_cell_capacity(self, 
                          force_update=False):
        
        self.cells_with_cap_nt:tuple=self.cell_processor.get_capacity()
        
        return self.cells_with_cap_nt

    def find_grid_nodes(self):

        self.grid=GridNodeLocator(**self.required_args)
        self.grid_ss=self.coders.get_table_provincial('substations')
        # self.datahandler.to_store(self.grid_ss,'substations')
        
        self.cutout,self.province_boundary=self.era5_cutout.get_era5_cutout()
        _grid_cells_=self.cutout.grid.overlay(self.province_boundary, how='intersection',keep_geom_type=True)
        self.province_grid_cells_cap_with_nodes = self.grid.find_grid_nodes_ERA5_cells(self.grid_ss,
                                                                                       _grid_cells_)
        
        # Set the index to sync with stored cells
        self.cells_node_data=utils.assign_cell_id(self.province_grid_cells_cap_with_nodes)
        
        self.store_grid_cells=self.datahandler.from_store('cells')
        # Add new columns to the existing DataFrame
        for column in self.cells_node_data.columns:
            # if column not in self.store_grid_cells.columns:
                # Assign new column with index alignment to avoid NaN values
                self.store_grid_cells[column] = self.cells_node_data[column].reindex(self.store_grid_cells.index)
        
        self.datahandler.to_store(self.store_grid_cells,'cells')
        
        self.store_grid_cells=self.datahandler.from_store('cells')
        return self.store_grid_cells
    
    # def filter_sites(self):
    #     cells=self.province_grid_cells_cap_with_nodes
        
    #     CF_mask=cells['CF_mean']>=cells['cell_static_CF_tolerance']
    #     land_mask=cells['potential_capacity']> cells['cell_capacity_tolerance']#MW
        
    #     CF_filter_df=cells[CF_mask]
    #     land_filter_df=cells[land_mask]
        
    #     matching_indices = CF_filter_df.index.intersection(land_filter_df.index)
    #     matched_cells = cells.loc[matching_indices].combine_first(cells.loc[matching_indices])
    #     self.matched_cells=matched_cells.set_crs(CF_filter_df.crs, inplace=True)
        
    #     vis.get_matched_vs_missed_visuals(cells,
    #                                       self.matched_cells)
    #     return self.matched_cells
        
    def score_cells(self,
                    # cells:pd.DataFrame
                    ):
                
        self.not_scored_cells=self.datahandler.from_store('cells')
        self.scored_cells = self.scorer.get_cell_score(self.not_scored_cells,'CF_mean')
        
        # Add new columns to the existing DataFrame
        for column in self.scored_cells.columns:
                self.not_scored_cells[column] = self.scored_cells[column].reindex(self.not_scored_cells.index)
        
        self.datahandler.to_store(self.not_scored_cells,'cells')
        self.store_grid_cells=self.datahandler.from_store('cells')
        
        return self.scored_cells
    
#     def cluster_cells(self):
#         self.wcss_tolerance:float= self.resource_disaggregation_config['WCSS_tolerance']
#         self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df = utils.cells_to_cluster_mapping(self.province_grid_cells_scored, self.vis_dir, self.wcss_tolerance)
#         self.cell_cluster_gdf, self.dissolved_indices = utils.create_cells_Union_in_clusters(self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df)
#         dissolved_indices_save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['dissolved_indices'])
#         utils.dict_to_pickle(self.dissolved_indices, dissolved_indices_save_to)

        
#     def progress_bar(self, iteration, total, bar_length=40):
#             percent = (iteration / total)
#             arrow = '█' * int(round(percent * bar_length) - 1)
#             spaces = ' ' * (bar_length - len(arrow))
#             sys.stdout.write(f'\rProgress: |{arrow}{spaces}| {percent:.1%}')
#             sys.stdout.flush()

#     def run(self):
#             log.info(f"{self.resource_type} module initiated")
            
#             tasks = [
#                 self.find_grid_nodes,
#                 self.create_CF_timeseries,
#                 self.save_results,
#                 self.score_cells,
#                 self.cluster_cells
#             ]
            
#             total_tasks = len(tasks)
            
#             for i, task in enumerate(tasks):
#                 task()  # Execute the task
#                 self.progress_bar(i + 1, total_tasks)  # Update the progress bar
                
#             print()  # For a newline after progress completion
#             print(f"{len(self.cell_cluster_gdf)} {self.resource_type} Sites' Clusters Generated.\n Total Capacity: {self.cell_cluster_gdf.potential_capacity.sum() / 1E3} GW")
#             log.info(f"{self.resource_type} Module Execution Completed!")


# def main(config_file_path: str, 
#          resource_type: str='solar'):
    
#     script_start_time = time.time()
    
#     solar_module = SolarResources(config_file_path, resource_type)
#     solar_module.run()
    
#     script_runtime = round((time.time() - script_start_time), 2)
    
#     log.info(f"Script runtime: {script_runtime} seconds")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run solar module script')
#     parser.add_argument('config', type=str, help=f"Path to the configuration file '*.yml'")
#     parser.add_argument('resource_type', type=str, help='Specify resource type e.g. solar')
#     args = parser.parse_args()
#     main(args.config, args.resource_type)