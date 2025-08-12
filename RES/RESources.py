import geopandas as gpd
import pandas as pd
from collections import namedtuple
import warnings
from typing import Optional,Union, Tuple
from pathlib import Path
from datetime import datetime
from itertools import product

from shapely import Point

# RESource's Local
from RES.era5_cutout import ERA5Cutout
from RES import cluster
from RES import windspeed as wind
from RES.CellCapacityProcessor import CellCapacityProcessor
from RES.coders import CODERSData
from RES.power_nodes import GridNodeLocator
from RES.timeseries import Timeseries
from RES.hdf5_handler import DataHandler
from RES.AttributesParser import AttributesParser
from RES.score import CellScorer
from RES.cell import GridCells
from RES.gwa import GWACells
from RES.boundaries import GADMBoundaries
from RES.units import Units
from RES import utility as utils



    
# Get the current local time
current_local_time = datetime.now()
warnings.filterwarnings("ignore")

PRINT_LEVEL_BASE:int=1

class RESources_builder(AttributesParser):  
    """
    Main orchestrator class for renewable energy resource assessment workflows.

    __RESources_builder__ coordinates the complete workflow for assessing solar and wind potential at sub-national scales. It integrates spatial grid cell preparation, land availability analysis, weather data processing, economic evaluation, and site clustering into a unified framework.

    This class implements a modular architecture where each assessment step is handled by specialized components, enabling reproducible, scalable, and transparent renewable energy assessments.

    Parameters
    ----------
    config_file_path : str or Path
        Path to the YAML configuration file containing project settings
    region_short_code : str
        ISO or custom short code for the target region (e.g., 'BC' for British Columbia)
    resource_type : {'solar', 'wind'}
        Type of renewable energy resource to assess
        
    Attributes
    ----------
    store : Path
        Root directory for data storage (HDF5 file) and caching.
    units : Units
        Handler for unit conversions and standardization
    gridcells : GridCells
        Spatial grid generation and management
    timeseries : Timeseries
        Climate data processing and capacity factor calculations
    datahandler : DataHandler
        HDF5-based data storage and retrieval interface
    cell_processor : CellCapacityProcessor
        Land availability and capacity potential calculations
    coders : CODERSData
        Canadian power system data integration (substations, transmission lines).
    era5_cutout : ERA5Cutout
        ERA5 climate data cutout management
    scorer : CellScorer
        Economic scoring and LCOE calculations
    gwa_cells : GWACells
        Global Wind Atlas data integration (wind resources only)
    results_save_to : Path
        Output directory for assessment results
    region_name : str
        Full name of the assessed region
        
    Methods
    -------
    get_grid_cells()
        Generate spatial grid cells covering the region boundary
    get_cell_capacity(force_update=False)
        Calculate potential capacity based on land availability constraints
    extract_weather_data()
        Process climate data for capacity factor calculations
    update_gwa_scaled_params(memory_resource_limitation=False)
        Integrate Global Wind Atlas wind speed corrections (wind only)
    get_CF_timeseries(cells=None, force_update=False)
        Generate hourly capacity factor time series
    find_grid_nodes(cells=None, use_pypsa_buses=False)
        Identify nearest electrical grid connection points
    score_cells(cells=None)
        Calculate economic scores based on LCOE methodology
    get_clusters(scored_cells=None, wcss_tolerance=0.05)
        Perform spatial clustering of viable sites
    get_cluster_timeseries(clusters=None, dissolved_indices=None, cells_timeseries=None)
        Generate representative time series for each cluster
    build(select_top_sites=True, use_pypsa_buses=True, memory_resource_limitation=True)
        Execute complete assessment workflow
    export_results(resource_type, resource_clusters, cluster_timeseries, save_to=Path('results'))
        Export results in standardized format for downstream models
    select_top_sites(sites, timeseries, resource_max_capacity=10)
        Filter results to highest-potential sites within capacity constraints
        
    Examples
    --------
    Basic wind assessment workflow:
    
    >>> from RES.RESources import RESources_builder
    >>> builder = RESources_builder(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC", 
    ...     resource_type="wind"
    ... )
    >>> results = builder.build()
    >>> builder.export_results(*results)
    
    Step-by-step workflow with intermediate inspection:
    
    >>> builder = RESources_builder("config/config.yaml", "AB", "solar")
    >>> cells = builder.get_grid_cells()
    >>> cells_with_capacity = builder.get_cell_capacity()
    >>> scored_cells = builder.score_cells(cells_with_capacity)
    >>> clusters = builder.get_clusters(scored_cells)
    
    Notes
    -----
    - Inherits configuration parsing capabilities from AttributesParser
    - Uses HDF5 storage for efficient handling of large geospatial datasets
    - Implements caching mechanisms to avoid redundant computations
    - Supports both solar PV and onshore wind technologies
    - Economic calculations follow NREL LCOE methodology
    - Clustering uses k-means with automatic cluster number optimization
    """
    def __post_init__(self):
        """
        Initialize inherited attributes and orchestrate the complete RESources workflow setup.
        
        This comprehensive initialization method:
        1. Calls parent __post_init__ to inherit configuration and regional attributes
        2. Sets up required arguments dictionary for consistent parameter passing
        3. Initializes all specialized component classes for the assessment workflow
        4. Configures data storage and persistence infrastructure
        5. Establishes temporal boundaries for the analysis period
        6. Prepares regional and project-specific settings
        
        Component Initialization:
        - Units: Standardized unit management and conversion
        - GridCells: Spatial grid generation and boundary management
        - Timeseries: Climate data processing and capacity factor calculations
        - GADMBoundaries: Administrative boundary handling
        - DataHandler: HDF5-based data storage and retrieval
        - CellCapacityProcessor: Land availability and capacity calculations
        - CODERSData: Canadian power system data (if applicable)
        - ERA5Cutout: Climate data cutout management
        - CellScorer: Economic evaluation and LCOE calculations
        - GWACells: Global Wind Atlas integration
        
        Inherited Capabilities:
        - Configuration file parsing and validation
        - Regional identification and validation
        - Logging infrastructure and operation tracking
        - Data storage path management
        - Temporal analysis period configuration
        - Coordinate reference system management
        
        Raises:
            ConfigurationError: If required configuration parameters are missing
            RegionValidationError: If region_short_code is invalid
            StorageError: If HDF5 store cannot be initialized
            ComponentInitializationError: If any component class fails to initialize
            
        Notes:
            - Canadian power system integration requires valid region code
            - Temporal snapshot automatically loaded from configuration
            - All components share consistent parameter dictionary
            - HDF5 storage enables efficient large dataset management
        """
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        utils.print_module_title(f'Initiating RESource Builder | {__name__}')
        # This dictionary will be used to pass arguments to external classes
        self.required_args = {   #order doesn't matter
            "config_file_path" : self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
        }

        self.country_name = self.get_country()
        if self.country_name is None:
            utils.print_warning("Country name is not set in the configuration file.")
        
        # Initiate Classes
        self.units=Units(**self.required_args)
        self.gridcells=GridCells(**self.required_args)
        self.timeseries=Timeseries(**self.required_args)
        self.gadmBoundary=GADMBoundaries(**self.required_args)
        self.datahandler=DataHandler(self.store)
        self.cell_processor=CellCapacityProcessor(**self.required_args)
        if self.country_name == 'Canada':
            self.coders=CODERSData(**self.required_args) # only for Canada
        self.era5_cutout=ERA5Cutout(**self.required_args)
        self.scorer=CellScorer(**self.required_args)
        self.gwa_cells=GWACells(**self.required_args)
        
        self.results_save_to=Path('results/RESource')
        self.region_name=self.get_region_name()
        
        
        # Initiate the attributes for the RESources_builder
        self.store_grid_cells:gpd.GeoDataFrame=None
        self.region_grid_cells:gpd.GeoDataFrame=None
        
        # Snapshot (range of of the temporal data)
        (
            self.start_date,
            self.end_date,
        ) = self.load_snapshot()
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"Snapshot for Resources: {self.start_date} to {self.end_date}")
        

    def get_grid_cells(self)->gpd.GeoDataFrame:
        """
        Retrieves the default grid cells for the region.
    
        Args:
            None
    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the grid cells with their coordinates, geometry, and unique cell ids.
    
        Notes:
            - The `get_default_grid()` method creates several attributes, such as the atlite `cutout` object and the `region_boundary`.
            - Uses the `cutout.grid` attribute to create the analysis grid cells (GeoDataFrame).
        _________________________________________________________________________________________________________________________
        - Future Scopes:
            To give user flexibility to make their own grid resolution
            Step 0: Set-up the Grid Cells and their Unique Indices to populate incremental datafields and to easy navigation to cells
                - Step to create the Cells with unique indices generated from their x,y (centroids).
        """

        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Preparing Grid Cells...")
        
        self.region_grid_cells:gpd.GeoDataFrame=self.gridcells.get_default_grid()
        
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Grid Cells updated.")
        
        return self.region_grid_cells

    def get_cell_capacity(self):
        """
        Retrieves the potential capacity of the cells based on land availability and land-use intensity.
        Args:   

            force_update (bool): If True, forces the update of the cell capacity data.
        Returns:
            tuple: A namedtuple containing the cells GeoDataFrame and a capacity matrix.
        Notes:
            - The capacity matrix is a 2D array where each row corresponds to a cell and each column corresponds to a time step.
            - The potential capacity is calculated as:
                - Potential capacity (MW) = available land (%) x land-use intensity (MW/sq.km) x Area of a cell (sq. km)
                - The method uses the `CellCapacityProcessor` class to process the capacity data.
            - The method returns a namedtuple with two attributes: `data` (the cells GeoDataFrame) and `matrix` (the capacity matrix).
            - Could be parallelized with Step 2A/2C.

        """       
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Preparing Cells' capacity...")
        
        self.cells_with_cap_nt:tuple=self.cell_processor.get_capacity()
        
                
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Cells' capacity updated.")
        
        return self.cells_with_cap_nt # returns a namedtuple with `data` and `matrix attributes

    def extract_weather_data(self):
        """Extracts weather data for the cells (e.g. windspeed, solar influx).
        This method retrieves the ERA5 cutout and extracts windspeed data for the cells.
        If the windspeed data is already present in the stored dataset, it skips the extraction from the source.
        If the resource type is 'wind', it extracts the 'windspeed_ERA5' from the cutout and updates the cells GeoDataFrame.
        If the resource type is 'solar', it currently does not support extraction from the Global Solar Atlas data.
        Args:
            None
    
        Returns:
            None
        Notes:
            - Currently active for windspeed only due to significant contrast with high resolution data.
        """
        
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Extracting ERA5 windspeed from cutout...")
        
        self.store_grid_cells=self.datahandler.from_store('cells')
        self.cutout,_=self.era5_cutout.get_era5_cutout()
            
        if self.resource_type=='wind': 
            if all(column in self.store_grid_cells.columns for column in ['windspeed_ERA5']):
                utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}|'windspeed_ERA5' already present in the stored dataset, skipping the data extraction from source.")
                pass
            else:
                utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Extracting 'windspeed_ERA5' from cutout.")
                _store_grid_cells_updated_:gpd.GeoDataFrame=wind.impute_ERA5_windspeed_to_Cells(self.cutout,
                                               self.store_grid_cells)
                self.store_grid_cells_updated=utils.assign_cell_id(_store_grid_cells_updated_,self.gadmBoundary.boundary_datafields['NAME_2'])
                self.datahandler.to_store(self.store_grid_cells_updated,'cells')
                utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| 'windspeed_ERA5' extraction completed and stored.")
                return self.store_grid_cells_updated
            
        elif self.resource_type=='solar': 
            # utils.print_update(level=PRINT_LEVEL_BASE+2,
            #                message="Extracting 'solar_influx'  from source.")
            # self.store_grid_cells_updated:gpd.GeoDataFrame= xxx
            utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__}| Global Solar Atlas data is not yet supported for Solar Resources")
            pass
    
    #---------------------------
    def update_gwa_scaled_params(self,
                                memory_resource_limitation:Optional[bool]=False):
        if self.resource_type=='wind': 
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Preparing high resolution windspeed data from Global Wind Atlas")
            
            if all(column in self.store_grid_cells.columns for column in ['CF_IEC2', 'CF_IEC3', 'windspeed_gwa','windspeed_ERA5']):
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| 'CF_IEC2', 'CF_IEC3', 'windspeed_gwa' are already present in the store information, skipping data extraction from source")
                pass
            else:
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Data extracting from source: 'CF_IEC2', 'CF_IEC3', 'windspeed_gwa' ")
                self.gwa_cells.map_GWA_cells_to_ERA5(aggregation_level=self.gadmBoundary.boundary_datafields['NAME_2'],
                                                     memory_resource_limitation=memory_resource_limitation)

        elif self.resource_type=='solar': 
            # Not activated for solar resources yet as the high resolution data processing is computationally expensive and the data contrast for solar doesn't provide satisfactory incentive for that.
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Global Solar Atlas data not yet supported for solar.")
            pass
        
        return self.store_grid_cells 
    
    def get_CF_timeseries(self,
                          cells:gpd.GeoDataFrame=None,
                          force_update=False)->tuple:
        """
        Extract timeseries information for the Cells' e.g. static CF (yearly mean) and timeseries (hourly). 
        
        Args:
            cells (gpd.GeoDataFrame): Cells with their coordinates, geometry, and unique cell ids.
            force_update (bool): If True, forces the update of the CF timeseries data.  
        Returns:
            tuple: A namedtuple containing the cells with their timeseries data.
        Notes:
            - The method uses the `Timeseries` class to retrieve the timeseries data for the cells.
            - The timeseries data is retrieved based on the resource type (e.g., 'solar' or 'wind').
            - If the `cells` argument is not provided, it retrieves the cells from the data handler.
            - Could be parallelized with Step 2B/2C
        """
        utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Preparing Timeseries for the Cells...")
        if cells is None:
            self.datahandler.refresh()
            cells=self.datahandler.from_store('cells')
        cells_withCF,cells_timeseries= self.timeseries.get_timeseries(cells=cells)
        return cells_withCF,cells_timeseries

    def find_grid_nodes(self,
                        cells:gpd.GeoDataFrame=None,
                        use_pypsa_buses:bool=False,
                        apply_proximity_filter:bool=False) -> gpd.GeoDataFrame:
        """
        Find the grid nodes for the given cells.                            

        Args:
            cells (gpd.GeoDataFrame, optional): Cells with their coordinates, geometry, and unique cell ids. Defaults to None.
            use_pypsa_buses (bool, optional): Whether to use PyPSA buses as preferred nodes for resource connection. Defaults to False.
        Returns:
            gpd.GeoDataFrame: Updated grid cells with nearest grid node information
        Notes:
            Could be parallelized with Step 1B/C.
        """

        self.gridNodesProcessor=GridNodeLocator(**self.required_args)
        self.cutout,self.region_boundary=self.era5_cutout.get_era5_cutout()
        
        # Initialize cells data
        if cells is None:
            self.store_grid_cells=self.datahandler.from_store('cells')
        else:
            self.store_grid_cells = cells.copy()
            
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Grid Node Location initiated...")
        
        if use_pypsa_buses:
            
            utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Using PyPSA nodes as preferred nodes for resource connection.")
            utils.print_info(f"{__name__}| PyPSA buses are configured in the config file under 'capacity\
                             -disaggregation/transmission/buses'")
            
            buses_data_path=self.get_buses_path()
            utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| PyPSA buses loading from: {buses_data_path}")

            grid_ss_df=pd.read_csv(buses_data_path)
            assert 'x' in grid_ss_df.columns and 'y' in grid_ss_df.columns, \
                "The buses data must contain 'x' and 'y' columns for coordinates."
            
            self.grid_ss = gpd.GeoDataFrame(
                grid_ss_df,
                geometry=gpd.points_from_xy(grid_ss_df['x'], grid_ss_df['y']),
                crs=self.get_default_crs(),  # Set the coordinate reference system (e.g., WGS84)
                ) 
            
            utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Searching for nearest grid nodes for each cell...")
            self.region_grid_cells_cap_with_nodes = self.gridNodesProcessor.find_grid_nodes_ERA5_cells(self.grid_ss,
                                                                                            self.store_grid_cells,
                                                                                            apply_proximity_filter=apply_proximity_filter)
            utils.print_update(level=PRINT_LEVEL_BASE+3,
                            message=f"{__name__}| ✔ Closest grid nodes and distance calculation completed.")
            
            self.datahandler.to_store(self.store_grid_cells,'cells')
            self.datahandler.to_store(self.grid_ss,'buses')
        else:
            utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Using Substations (sourced from CODERS) preferred nodes for resource connection.")
            
            if self.country_name == 'Canada':
                self.grid_ss:gpd.GeoDataFrame=self.coders.get_table_provincial('substations')
                
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Searching for nearest grid nodes for each cell...")
                self.region_grid_cells_cap_with_nodes = self.gridNodesProcessor.find_grid_nodes_ERA5_cells(self.grid_ss,
                                                                                                self.store_grid_cells,
                                                                                                apply_proximity_filter=apply_proximity_filter)
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                                message=f"{__name__}| ✔ Closest grid nodes and distance calculation completed.")
                
                self.datahandler.to_store(self.store_grid_cells,'cells')
                self.datahandler.to_store(self.grid_ss,'substations')
            else:
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Using OSM grid lines for connection point analysis...")
                self.grid_lines:gpd.GeoDataFrame=self.gridNodesProcessor.get_OSM_grid_lines()
                
                if self.grid_lines is None or len(self.grid_lines) == 0:
                    utils.print_update(level=PRINT_LEVEL_BASE+3,
                               message=f"{__name__}| Warning: No OSM grid lines found for region {self.region_short_code}")
                    return self.store_grid_cells
                
                # self.grid_lines=self.grid_lines[self.grid_lines['max_voltage']>=11000]
                self.store_grid_cells["centroid"] = self.store_grid_cells.apply(lambda row: Point(row["x"], row["y"]), axis=1)

                # Apply to each row to compute the single connection point
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| Calculating nearest connection points to transmission lines...")
                
                connection_results = self.store_grid_cells.apply(
                    lambda row: self.gridNodesProcessor.find_nearest_connection_point(
                        row["centroid"], row["geometry"], self.store_grid_cells, self.grid_lines
                    ),
                    axis=1, result_type="expand"
                )
                
                self.store_grid_cells[["nearest_connection_point", "nearest_distance"]] = connection_results
                
                # self.store_grid_cells.drop(columns=["nearest_connection_point","centroid"], inplace=True) #temp
                
                utils.print_update(level=PRINT_LEVEL_BASE+3,
                                message=f"{__name__}| ✔ Connection point analysis completed.")
                
                self.datahandler.to_store(self.store_grid_cells,'cells')
                self.datahandler.to_store(self.grid_lines,'lines')
        
        # Refresh and return updated cells
        self.datahandler.refresh()
        self.store_grid_cells=self.datahandler.from_store('cells')
        return self.store_grid_cells
    
    '''
    ____________________________________________________________________________________________________________________________________________
    Step 2: Set Scoring Matrix for the Cells. 
    ____________________________________________________________________________________________________________________________________________
    - We populated necessary parameters to evaluate the cells. We can set the scoring metric using the parameters.
    - Typical metric includes but not limited to LCOE (Levelized Cost of Electricity in $/MWh) of cells.
    - As a starter and simplified metric, we calculate Total Cost ($) and Total Energy Yield (MWh) and for each Cell and calculate LCOE ($/MWh).
    * Remarks:  Sequential Step after Step-1
    
    * Future Scope(s): 
        1. Apply MCDA (Multi Criteria Decision Analysis) as Scoring Metric of the Cells.
        2. Introduce proxy of the Local Regulations regarding site accessibility/placements.
        3. Introduce proxy of the Local/Govt. incentives for Sites (based on Load Center based placement, land ownership, proximity to transport network etc.)
        4. Introduce proxy of Weather Drought parameters for cells 
            e.g. i. [standardized energy indices in future climate scenarios](https://www.sciencedirect.com/science/article/pii/S0960148123011217?via%3Dihub)
                 ii. [Compound energy droughts](https://www.sciencedirect.com/science/article/pii/S0960148123014659?via%3Dihub#d1e724)
    '''
    def score_cells(self,
                    cells:gpd.GeoDataFrame=None):
        """
        Scores the Cells based on calculated LCOE ($/MWh). </br>
        Wrapper of the _.get_cell_score()_ method of **_CellScorer_** object.
        """
        self.not_scored_cells=cells
        
        if self.not_scored_cells is None:    
            self.datahandler.refresh()
            self.not_scored_cells=self.datahandler.from_store('cells')
            
        self.scored_cells = self.scorer.get_cell_score(self.not_scored_cells,f'{self.resource_type}_CF_mean') # 
        
        # # Add new columns to the existing DataFrame
        # for column in self.scored_cells.columns:
        #         self.not_scored_cells[column] = self.scored_cells[column].reindex(self.not_scored_cells.index)
        
        self.datahandler.to_store(self.scored_cells,'cells',force_update=True)
        # self.store_grid_cells=self.datahandler.from_store('cells')
        
        return self.scored_cells

    # def rescale_cutout_windspeed(self, cutout, era5_cells_gdf_updated):
    #     return wind.rescale_ERA5_cutout_windspeed_with_mapped_GWA_cells(cutout, era5_cells_gdf_updated)

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
    ___________________
    - Step 3A: 
    - As a starter, we apply simplified spatial clustering by using k-means  based on LCOE of the cells.
    * Remarks:  Sequential Step after Step-2.
    ___________________
    '''
    def get_clusters(self,
                     scored_cells:gpd.GeoDataFrame=None,
                     wcss_tolerance=0.05):
        """
        ### Args:
         - **WCSS (Within-cluster Sum of Square) tolerance**. Higher tolerance gives , more simplification and less number of clusters. 
         - **Default set to 0.05**.
        """
 
        
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        self.wcss_tolerance=wcss_tolerance
        self.scored_cells=scored_cells
        
        # self.wcss_tolerance:float= self.resource_disaggregation_config['WCSS_tolerance']
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           
                           message=f"{__name__}| Preparing cluster of resources...")
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Clustering requires scored cells. The default scoring method is set to 'lcoe'. Checking for 'lcoe' in datafields...")
        
        if not hasattr(self, f'lcoe_{self.resource_type}') or self.scored_cells is None:
            utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message=f"{__name__}| 'lcoe_{self.resource_type}' not found in available datafields...") 
            self.scored_cells = self.score_cells()
            

        
        self.vis_dir=self.get_vis_dir()
        
        self.ERA5_cells_cluster_map, self.region_optimal_k_df = cluster.cells_to_cluster_mapping(self.scored_cells, 
                                                                                                 self.vis_dir, 
                                                                                                 self.wcss_tolerance,
                                                                                                 self.resource_type,
                                                                                                #  [f'LCOE_{self.resource_type}', f'potential_capacity_{self.resource_type}']
                                                                                                 [f'lcoe_{self.resource_type}', f'potential_capacity_{self.resource_type}']
                                                                                                 )
        
        self.cell_cluster_gdf, self.dissolved_indices = cluster.create_cells_Union_in_clusters(self.ERA5_cells_cluster_map, 
                                                                                               self.region_optimal_k_df,
                                                                                               self.resource_type)
        
        self.cell_cluster_gdf['Operational_life'] = self.resource_disaggregation_config.get('Operational_life', 20)
        self.cell_cluster_gdf.loc[:, 'resource_type'] = self.resource_type.lower()
        # Define a namedtuple
        cluster_data = namedtuple('cluster_data', ['clusters','dissolved_indices'])
        
        self.clusters_nt:tuple=cluster_data(self.cell_cluster_gdf,self.dissolved_indices)
        # Corrected version of the code
        self.datahandler.to_store(self.cell_cluster_gdf,f'clusters/{self.resource_type}',force_update=True)
        self.dissolved_cell_indices_df=pd.DataFrame(self.dissolved_indices).T
        self.dissolved_cell_indices_df.index.name='Region'
        self.datahandler.to_store(self.dissolved_cell_indices_df,f'dissolved_indices/{self.resource_type}',force_update=True)
        
        return self.clusters_nt
    
    '''
    ___________________
    - Step 3B: 
    - As a starter, we apply simplified approach by calculating stepwise mean from the associated cells and set it as a representative profile of a cluster.
    * Remarks:  Sequential Step after Step-4A.
    ___________________
    
    * Future Scope(s): 
        1. Apply temporal clustering methods for representative profile. 
        2. Collect hybrid RE profile (solar + wind) for regions/clusters show comparative analysis.
        2. Use ML approaches for comparative results with aforementioned classical/heuristics based approach.
    '''

    def get_cluster_timeseries(self,
                               clusters:gpd.GeoDataFrame=None,
                               dissolved_indices:pd.DataFrame=None,
                               cells_timeseries:pd.DataFrame=None,
                               ):

        self.cells_timeseries=cells_timeseries
        self.cell_cluster_gdf=clusters
        self.dissolved_cell_indices_df=dissolved_indices
        

        if self.cells_timeseries is None:
            self.cells_timeseries=self.datahandler.from_store(F'timeseries/{self.resource_type}')
        if self.cell_cluster_gdf is None:
            self.cell_cluster_gdf=self.datahandler.from_store(f'clusters/{self.resource_type}')
            utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Preparing representative profiles for {len(self.cell_cluster_gdf)} clusters")
        if self.dissolved_cell_indices_df is None:
            self.dissolved_cell_indices_df=self.datahandler.from_store(f'dissolved_indices/{self.resource_type}')
        

        self.cluster_ts_df=self.timeseries.get_cluster_timeseries(self.cell_cluster_gdf,
                                                                self.cells_timeseries,
                                                                self.dissolved_cell_indices_df)
        return self.cluster_ts_df

    # _________________________________________________________________________________

    def build(self,
            select_top_sites:Optional[bool]=True,
            use_pypsa_buses:Optional[bool]=False,
            memory_resource_limitation:Optional[bool]=True):
        """
        Execute the specific module logic for the given resource type ('solar' or 'wind').
        """
        utils.print_module_title(f"Initiating {self.resource_type} module for {self.get_region_name()}...")
        self.memory_resource_limitation=memory_resource_limitation
        
        utils.print_banner("Step 1 : Prepare Cutout and Grid Cells")
        self.get_grid_cells()
        
        utils.print_banner("Step 2 : Calculate Land availability and process capacity matrix")
        self.get_cell_capacity()
        
        utils.print_banner("Step 3 : [if Wind Resources] Collect and rescale Global Wind Atlas Data and calibrate ERA5's windspeed. ")
        # Store CF data for validation purposes
        self.extract_weather_data()
        self.update_gwa_scaled_params(self.memory_resource_limitation) # testing, 2025 04 21
        
        utils.print_banner("Step 4 : Create timeseries for Resources CF")
        self.get_CF_timeseries()
        
        utils.print_banner("Step 5 : Find closed grid connection nodes")
        self.find_grid_nodes(use_pypsa_buses=use_pypsa_buses)
        
        utils.print_banner("Step 6 : Use capacity, energy yield and cost attributes to score each cell")
        self.score_cells()
        
        utils.print_banner("Step 7.1 : Use score similarities to find clusterized representation (sites) of cells")
        self.get_clusters()
        
        utils.print_banner("Step 7.2 : Prepare representative timeseries of the clusterized sites")
        self.get_cluster_timeseries()
        
        utils.print_info("To avoid confusion, Units dictionary method should be updated if any units are changed across modules. However, units dictionary is for documentation purposes only. It doesn't have any calculation impacts on any of the methods.")
        self.units.create_units_dictionary()
        
        if select_top_sites:
            utils.print_banner("Step 7 : Top Site Selections for Targeted Capacity Investments/Plans")
            resource_max_capacity=self.resource_disaggregation_config.get('max_capacity',10) # Collects max_capacity from resource_disaggregation_config (if set), otherwise defaults to 10 GW
            
            resource_clusters,cluster_timeseries=self.select_top_sites(self.get_clusters().clusters,
                                                                        self.get_cluster_timeseries(),
                                                                        resource_max_capacity=resource_max_capacity)
               
            utils.print_module_title(f"Top Sites(clusters) from {self.resource_type} module saved to {self.store} for {self.get_region_name()}...")
            
        else: # When user wants all of the sites
            resource_clusters=self.get_clusters().clusters,
            cluster_timeseries=self.get_cluster_timeseries(),
    
            utils.print_module_title(f"All Sites (clusters) from {self.resource_type} module saved to {self.store} for {self.get_region_name()}...")
   
        
        self.export_results(self.resource_type,
                            self.region_name,
                            resource_clusters,
                            cluster_timeseries,
                            self.results_save_to)
        
        sites_summary:str=self.create_summary_info(self.resource_type,
                                                   self.region_name,
                                                   resource_clusters,
                                                   cluster_timeseries)
        self.dump_export_metadata(sites_summary,
                                  self.results_save_to)



       
    @staticmethod
    def export_results(resource_type:str,
                       region:str,
                    resource_clusters:pd.DataFrame,
                    cluster_timeseries:pd.DataFrame,
                    save_to : Optional[Path]=Path('results')):
        """
        Export processed resource cluster results (geodataframe) to standard datafield csvs as input for downstream models.
        ### Args
        - **resource_type**: The type of resource ('solar' or 'wind').
        - **resource_clusters**: A DataFrame containing resource cluster information.
        - **output_dir** [optional]: The directory to save the output files. Default to : 'results/*.csv'
        
        > Currently supports: CLEWs, PyPSA
        """
        # Check if resource_clusters is a DataFrame or GeoDataFrame
        if not isinstance(resource_clusters, (pd.DataFrame, gpd.GeoDataFrame)):
            raise TypeError(
                f"Invalid input: resource_clusters must be a Pandas DataFrame or GeoDataFrame, "
                f"but got {type(resource_clusters).__name__}."
            )
        
        if not isinstance(cluster_timeseries, (pd.DataFrame)):
            raise TypeError(
                f"Invalid input: resource_clusters must be a Pandas DataFrame or GeoDataFrame, "
                f"but got {type(resource_clusters).__name__}."
            )
        # Exclude all columns containing geometry-related data as these are not required for downstream models in consideration i.e. CLEWs, PyPSA
        resource_clusters_excld_geom = resource_clusters[[col for col in resource_clusters.columns if col != 'geometry']]

        # CSV -> Save to 
        save_to=utils.ensure_path(save_to)
        save_to.mkdir(parents=True,exist_ok=True)
        
        resource_clusters_excld_geom.to_csv(save_to/f'resource_options_{resource_type}_{region}.csv', index=True)
        cluster_timeseries.to_csv(save_to/f'resource_options_{resource_type}_{region}_timeseries.csv', index=True)

        utils.print_update(level=2, message=f"{resource_type} clusters exported to :{save_to}")

    @staticmethod
    def create_summary_info(resource_type:str,
                            region:str,
                            sites:pd.DataFrame,
                            timeseries:pd.DataFrame)->str:
        
        """
        Creates summary information to be exported alongside results data.
        """
        
        formatted_time = current_local_time.strftime("%H:%M:%S")
        
        info = (
            f"{'_'*25} Top Block Represents the latest results' summary <{'_'*25}\n"
            f"{'-'*100}\n"
            f"* {resource_type.upper()} for {region.upper()}*\n"
            f"Total Capacity of the Sites: {sites['potential_capacity'].sum() / 1e3} GW\n"
            f">> No. of Sites (Clusters): {len(sites)}\n"
            f" >> Snapshot Points: {len(timeseries)}"
            f"\n Results Generated on Local Time (hh:mm:ss): {formatted_time}\n"
            f"{'-'*100}\n"
        )
        return info
    
    @staticmethod
    def dump_export_metadata(info: str, save_to: Optional[Path] = 'results/linking'):
        """
        Dumps the metadata summary information to a file. If the file already exists,
        it prepends the new info at the top of the file.
        """
        save_to = utils.ensure_path(save_to)  # Ensures that the provided save path is a Path object
        file_name = "Resource_options_summary.txt"
        # File path
        file_path = save_to / file_name

        # Check if the file exists and read the existing content
        if file_path.exists():
            with open(file_path, "r") as file:
                existing_content = file.read()
        else:
            existing_content = ""

        # Prepend the new info to the existing content
        updated_content = info + "\n" + existing_content

        # Save the updated content to the file
        with open(file_path, "w") as file:
            file.write(updated_content)

    @staticmethod    
    def select_top_sites(sites:Union[gpd.GeoDataFrame, pd.DataFrame],
                        sites_timeseries:pd.DataFrame,
                        resource_max_capacity:float,
                        )-> Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.DataFrame]:
        print(f">>> Selecting TOP Sites to for {resource_max_capacity} GW Capacity Investment in BC...")
        """
        Select the top sites based on potential capacity and a maximum resource capacity limit.

        Args:
            sites_gdf: GeoDataFrame containing  cell and bucket information.
            resource_max_capacity (float) : Maximum allowable  capacity in GW.

        Returns:
        - selected_sites: GeoDataFrame with the selected top sites.
        """
        print(f"{'_'*100}")
        print(f"Selecting the Top Ranked Sites to invest in {resource_max_capacity} GW PV in BC")
        print(f"{'_'*100}")
     
        # Initialize variables
        selected_rows:list = []
        total_capacity:float = 0.0

        top_sites:gpd.GeoDataFrame = sites.copy()

        if top_sites['potential_capacity'].iloc[0] < resource_max_capacity * 1000:
            # Iterate through the sorted GeoDataFrame
            for index, row in top_sites.iterrows():
                # Check if adding the current row's capacity exceeds resource capacity
                if total_capacity + row['potential_capacity'] <= resource_max_capacity * 1000:
                    selected_rows.append(index)  # Add the row to the selection
                    # Update the total capacity
                    total_capacity += row['potential_capacity']
                # If adding the current row's capacity would exceed max resource capacity, stop the loop
                else:
                    break

            # Create a new GeoDataFrame with the selected rows
            top_sites:gpd.GeoDataFrame = top_sites.loc[selected_rows]

            # Apply the additional logic
            # mask = sites['cluster_id'] > top_sites['cluster_id'].max()
            mask = sites.index > top_sites.index.max()
            selected_additional_sites:gpd.GeoDataFrame = sites[mask].head(1)
            
            remaining_capacity:float = resource_max_capacity * 1000 - top_sites['potential_capacity'].sum()

            if remaining_capacity > 0:
                
                if len(selected_additional_sites) > 0:
                    print(f"\n!! Note: The Last cluster ({selected_additional_sites.index[-1]}) originally had {round(selected_additional_sites['potential_capacity'].iloc[0] / 1000,2)} GW potential capacity."
                        f"To fit the maximum capacity investment of {resource_max_capacity} GW, it has been adjusted to {round(remaining_capacity / 1000,2)} GW\n")
                else:
                    print(f"\n!! Note: No additional sites selected. Remaining capacity: {round(remaining_capacity / 1000,2)} GW\n")
                selected_additional_sites['potential_capacity'] = remaining_capacity
            # Concatenate the DataFrames
            top_sites = pd.concat([top_sites, selected_additional_sites])
        else:
            original_capacity = sites['potential_capacity'].iloc[0]

            print(f"!!Note: The first cluster originally had {round(original_capacity / 1000,2)} GW potential capacity.\n"
                f"To fit the maximum capacity investment of {resource_max_capacity} GW, it has been adjusted. \n")

            top_sites = top_sites.iloc[:1]  # Keep only the first row
            # Adjust the potential_capacity of the first row
            top_sites.at[top_sites.index[0], 'potential_capacity'] = resource_max_capacity * 1000
        
        # top_sites_ts = sites_timeseries[top_sites.index.astype(str)]
        # sites_timeseries.columns = sites_timeseries.columns.str.strip()
        # top_sites.index = top_sites.index.str.strip()
        top_sites_ts = sites_timeseries[top_sites.index]

        return top_sites ,top_sites_ts  # gdf

def build_resources(regions:list,
                    resource_types: list, 
                    config_path: str | Path = 'config/config.yaml'):
    """
    Builds resources for specified regions and resource types using the RESources_builder module.
    Args:
        regions (list): A list of region short codes to process.
        resource_types (list): A list of resource types to build for each region.
        config_path (str | Path, optional): Path to the configuration file. Defaults to 'config/config.yaml'.
    Returns:
        None
    """
    
    for region, resource in product(regions, resource_types):
        RES_module = RESources_builder(
            config_file_path=config_path,
            region_short_code=region,
            resource_type=resource
        )
        RES_module.build(select_top_sites=True, 
                         use_pypsa_buses=False)