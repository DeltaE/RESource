import logging as log
import os,sys,time,argparse
import geopandas as gpd
import pandas as pd
import atlite
from collections import namedtuple
# Local Packages

from linkingtool.era5_cutout import ERA5Cutout
import linkingtool.utility as utils
import linkingtool.visuals as vis
import linkingtool.cluster as cluster
# import linkingtool.linking_solar as solar

from linkingtool.CellCapacityProcessor import CellCapacityProcessor

from linkingtool.coders import CODERSData
# from workflow.scripts.prepare_data_v2 import LinkingToolData
from linkingtool.find import GridNodeLocator
from linkingtool.timeseries import Timeseries
from linkingtool.hdf5_handler import DataHandler
from linkingtool.AttributesParser import AttributesParser
from linkingtool.score import CellScorer
from linkingtool.cell import GridCells

class SolarResources(AttributesParser):
    
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
        self.gridcell=GridCells(**self.required_args)
        self.scorer=CellScorer()


        # Snapshot (range of of the temporal data)
        (
            self.start_date,
            self.end_date,
        ) = self.load_snapshot()
    
<<<<<<< HEAD
    '''
     _________________________________________________________________________________________________________________________
    *** Future Scope to give user flexibility to make their own grid resolution
    
     Set-up the Grid Cells and their Unique Indices to populate incremental datafields and to easy navigation to cells.
    ___________________________________________________________________________________________________________________________
    - Step to create the Cells with unique indices generated from their x,y (centroids). 
    - We fill the incremental datafields for cells as we progress with the methods (functions).
    '''
    # def get_grid_cells(self):
    #     self.log.info("Preparing Grid Cells...")
    #     return self.gridcell.grid()
    
=======
    def get_CF_timeseries(self,
                          force_update=False)->tuple:
        "returns cells geodataframe and timeseries dataframes"
        
        self.cells_with_ts_nt:tuple= self.timeseries.get_cells_timeseries()
        
        return self.cells_with_ts_nt
>>>>>>> beb6b426000d0e551bb15eab82f64341cb038acf
    
    '''
    _______________________________________________________________________________________________
    Step 1: Get Potential Capacity (MW), %CF (static/dynamic), and Grid Node information for Cells 
    _______________________________________________________________________________________________
    
    - Step 1A: 
    - Extract capacity information for the Cells.
    - Potential capacity (MW) = available land (%) x land-use intensity (MW/sq.km) x Area of a cell (sq. km)
    * Remarks:  Could be parallelized with Step 2A/2C.
    '''
    def get_cell_capacity(self, 
                          force_update=False):
        "returns cells geodataframe, capacity matrix data array and cutout "
        
        self.cells_with_cap_nt:tuple=self.cell_processor.get_capacity()
        
        return self.cells_with_cap_nt
    
    '''
    - Step 1B: 
    - Extract timeseries information for the Cells' e.g. static CF (yearly mean) and timeseries (hourly). 
    * Remarks:  Could be parallelized with Step 2B/2C
    '''
    
    def get_CF_timeseries(self,
                          force_update=False)->tuple:
        "returns cells geodataframe and timeseries dataframes"
        self.log.info("Preparing Timeseries for the Cells...")
        self.cells_with_ts_nt:tuple= self.timeseries.get_timeseries()
        
        return self.cells_with_ts_nt
    
    '''
    - Step 1C: 
    - Extract Substation information for the Cells e.g. Nearest Node Id and distance to the Node.
    * Remarks:  Could be parallelized with Step 1B.
    '''
    def find_grid_nodes(self):

        self.grid=GridNodeLocator(**self.required_args)
        self.grid_ss=self.coders.get_table_provincial('substations')
        
        
        self.cutout,self.province_boundary=self.era5_cutout.get_era5_cutout()
        self.store_grid_cells=self.datahandler.from_store('cells')
        # _grid_cells_=self.cutout.grid.overlay(self.province_boundary, how='intersection',keep_geom_type=True)
        self.province_grid_cells_cap_with_nodes = self.grid.find_grid_nodes_ERA5_cells(self.grid_ss,
                                                                                       self.store_grid_cells)
        
        self.datahandler.to_store(self.store_grid_cells,'cells')
        self.datahandler.to_store(self.grid_ss,'substations')
        
<<<<<<< HEAD
        return self.province_grid_cells_cap_with_nodes
    
    '''
    ____________________________________________________________________________________________________________________________________________
    Step 2: Set Scoring Matrix for the Cells. Typical metric includes but not limited to LCOE (Levelized Cost of Electricity in $/MWh) of cells.
    ____________________________________________________________________________________________________________________________________________
    - Calculate Total Cost ($) and Total Energy Yield (MWh) and for each Cell and calculate LCOE ($/MWh).
    * Remarks:  Sequential Step after Step-2 (A+B+C).
    
    * Future Scope(s): 
        1. Apply MCDA (Multi Criteria Decision Analysis) as Scoring Metric of the Cells.
        2. Introduce proxy of the Local Regulations regarding site accessibility/placements.
        3. Introduce proxy of the Local/Govt. incentives for Sites (based on Load Center based placement, land ownership, proximity to transport network etc.)
        4. Introduce proxy of Weather Drought parameters for cells 
            e.g. i. [standardized energy indices in future climate scenarios](https://www.sciencedirect.com/science/article/pii/S0960148123011217?via%3Dihub)
                 ii. [Compound energy droughts](https://www.sciencedirect.com/science/article/pii/S0960148123014659?via%3Dihub#d1e724)
    '''
    def score_cells(self ):
                
        self.not_scored_cells=self.datahandler.from_store('cells')
        self.scored_cells = self.scorer.get_cell_score(self.not_scored_cells,f'{self.resource_type}_CF_mean')
        
        # # Add new columns to the existing DataFrame
        # for column in self.scored_cells.columns:
        #         self.not_scored_cells[column] = self.scored_cells[column].reindex(self.not_scored_cells.index)
        
        self.datahandler.to_store(self.scored_cells,'cells')
        # self.store_grid_cells=self.datahandler.from_store('cells')
        
        return self.scored_cells
    
    '''
    ____________________________________________________________________________________________________________________________________________
    Step 3: Clusterize the Cells to minimize the representative technologies in downstream models.
    ____________________________________________________________________________________________________________________________________________
    - As a starter, we apply simplified spatial clustering by using k-means  based on LCOE of the cells.
    * Remarks:  Sequential Step after Step-3.
    
    * Future Scope(s): 
        1. Apply Spatio-temporal clustering to extract hybrid RE profile (solar + wind) for regions/clusters.
        2. Use ML approaches for comparative results with classical/heuristics based approach.
    '''
    
    '''
    - Step 3A: 
    - As a starter, we apply simplified spatial clustering by using k-means  based on LCOE of the cells.
    * Remarks:  Sequential Step after Step-3.
    '''
    def get_clusters(self,
                     wcss_tolerance=0.05):
        """
        ### Args:
         -Within-cluster Sum of Square. Higher tolerance gives , more simplification and less number of clusters. Default set to 0.05.
        """
        
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        self.wcss_tolerance=wcss_tolerance
        
        # self.wcss_tolerance:float= self.resource_disaggregation_config['WCSS_tolerance']
            
        self.scored_cells=self.score_cells()
        self.log.info(f">> Preparing spatial clusters for {len(self.scored_cells)} Cells")
        
        self.vis_dir=self.get_vis_dir()
        
        self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df = cluster.cells_to_cluster_mapping(self.scored_cells, self.vis_dir, self.wcss_tolerance)
        self.cell_cluster_gdf, self.dissolved_indices = cluster.create_cells_Union_in_clusters(self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df)
=======
        self.store_grid_cells=self.datahandler.from_store('cells')
        return self.store_grid_cells
        
    def score_cells(self ):
                
        self.not_scored_cells=self.datahandler.from_store('cells')
        self.scored_cells = self.scorer.get_cell_score(self.not_scored_cells,'CF_mean')
>>>>>>> beb6b426000d0e551bb15eab82f64341cb038acf
        
        # dissolved_indices_save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['dissolved_indices'])
        # utils.dict_to_pickle(self.dissolved_indices, dissolved_indices_save_to)
        
        # Define a namedtuple
        cluster_data = namedtuple('cluster_data', ['clusters','dissolved_indices'])
        
        self.solar_clusters_nt:tuple=cluster_data(self.cell_cluster_gdf,self.dissolved_indices)
        self.datahandler.to_store(self.cell_cluster_gdf,'clusters')
        self.dissolved_cell_indices_df=pd.DataFrame(self.dissolved_indices).T
        self.datahandler.to_store(self.dissolved_cell_indices_df,'dissolved_indices')
        
        return self.solar_clusters_nt
    
    '''
    - Step 3B: 
    - As a starter, we apply simplified approach by calculating stepwise mean from the associated cells and set it as a representative profile of a cluster.
    * Remarks:  Sequential Step after Step-4A.
    
<<<<<<< HEAD
    * Future Scope(s): 
        1. Apply temporal clustering methods for representative profile. 
        2. Collect hybrid RE profile (solar + wind) for regions/clusters show comparative analysis.
        2. Use ML approaches for comparative results with aforementioned classical/heuristics based approach.
    '''
=======
    def get_clusters(self,
                     wcss_tolerance=0.05):
        """
        ### Args:
         -Within-cluster Sum of Square. Higher tolerance gives , more simplification and less number of clusters. Default set to 0.05.
        """
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        self.wcss_tolerance=wcss_tolerance
        
        # self.wcss_tolerance:float= self.resource_disaggregation_config['WCSS_tolerance']
            
        self.scored_cells=self.score_cells()
        self.vis_dir=self.get_vis_dir()
        
        self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df = cluster.cells_to_cluster_mapping(self.scored_cells, self.vis_dir, self.wcss_tolerance)
        self.cell_cluster_gdf, self.dissolved_indices = cluster.create_cells_Union_in_clusters(self.ERA5_cells_cluster_map, self.region_solar_optimal_k_df)
        
        # dissolved_indices_save_to = os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['dissolved_indices'])
        # utils.dict_to_pickle(self.dissolved_indices, dissolved_indices_save_to)
        
        # Define a namedtuple
        cluster_data = namedtuple('cluster_data', ['clusters','dissolved_indices'])
        
        self.solar_clusters_nt:tuple=cluster_data(self.cell_cluster_gdf,self.dissolved_indices)
        
        return self.solar_clusters_nt
        
>>>>>>> beb6b426000d0e551bb15eab82f64341cb038acf

    def get_cluster_timeseries(self):
        self.log.info(f">> Preparing representative profiles for {len(self.cell_cluster_gdf)} clusters")
        self.cells_timeseries=self.datahandler.from_store('timeseries')
        self.cluster_df=self.timeseries.get_cluster_timeseries(self.cell_cluster_gdf,
                                self.cells_timeseries.PV,
                               self.dissolved_cell_indices_df)
        return self.cluster_df

    def run(self):
            log.info(f"{self.resource_type} module initiated")
            # self.get_grid_cells()
            self.get_cell_capacity(),
            self.get_CF_timeseries(),
            self.find_grid_nodes(),
            self.score_cells(),
            self.get_clusters(),
            self.get_cluster_timeseries()


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