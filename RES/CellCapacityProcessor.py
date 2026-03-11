import inspect
from typing import Dict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from atlite import ExclusionContainer
from shapely.geometry import Polygon

# Local Packages
import RES.utility as utils
from RES.atb import NREL_ATBProcessor
from RES.AttributesParser import AttributesParser
from RES.era5_cutout import ERA5Cutout
from RES.hdf5_handler import DataHandler

# from RES.AttributesParser import AttributesParser
from RES.lands import LandContainer

PRINT_LEVEL_BASE=2

class CellCapacityProcessor(AttributesParser):
    """
    Renewable energy capacity processor for grid cell-based resource assessment.
    
    This class processes renewable energy potential capacity at the grid cell level by
    integrating climate data, land availability constraints, and techno-economic parameters.
    It calculates potential capacity matrices for solar and wind resources, applies land-use
    exclusions, and generates cost-attributed capacity datasets for energy system modeling.
    
    The class serves as the core processing engine for renewable energy resource assessment,
    combining spatial analysis, climate data processing, and economic modeling to produce
    grid cell-level capacity estimates suitable for energy planning and optimization models.
    
    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_resource_disaggregation_config() -> Dict[str, dict]: Get resource-specific config
    - get_cutout_config() -> Dict[str, dict]: Get ERA5 cutout configuration
    - get_gadm_config() -> Dict[str, dict]: Get GADM boundary configuration
    - get_region_name() -> str: Get region name from config
    - get_atb_config() -> Dict[str, dict]: Get NREL ATB cost configuration
    - get_default_crs() -> str: Get default coordinate reference system
    
    INHERITED ATTRIBUTES FROM AttributesParser:
    -------------------------------------------
    - config (property): Full configuration dictionary
    - store (property): HDF5 store path for data persistence
    - config_file_path: Path to configuration file
    - region_short_code: Region identifier code
    - resource_type: Resource type identifier
    - Plus other configuration access methods
    
    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - load_cost(resource_atb): Extract and process cost parameters from ATB data
    - __get_unified_region_shape__(): Create unified regional boundary geometry (private)
    - __create_cell_geom__(x, y): Create grid cell geometry from coordinates (private)
    - get_capacity(): Main method to process and calculate renewable energy capacity
    - plot_ERA5_grid_land_availability(): Visualize land availability on ERA5 grid
    - plot_excluder_land_availability(): Visualize land availability at excluder resolution
    
    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing processing settings
    region_short_code : str
        Region identifier for boundary and data processing
    resource_type : str
        Resource type ('solar', 'wind', or 'bess')
        
    Attributes
    ----------
    ERA5Cutout : ERA5Cutout
        ERA5 climate data cutout processor instance
    LandContainer : LandContainer
        Land exclusion and constraint processor instance
    resource_disaggregation_config : dict
        Resource-specific disaggregation configuration
    resource_landuse_intensity : float
        Land-use intensity for capacity calculation (MW/km²)
    atb : NREL_ATBProcessor
        NREL Annual Technology Baseline cost data processor
    datahandler : DataHandler
        HDF5 data storage interface
    cutout_config : dict
        ERA5 cutout configuration parameters
    gadm_config : dict
        GADM boundary configuration parameters
    disaggregation_config : dict
        General disaggregation configuration
    region_name : str
        Full region name from configuration
    utility_pv_cost : pd.DataFrame
        Utility-scale PV cost data from NREL ATB
    land_based_wind_cost : pd.DataFrame
        Land-based wind cost data from NREL ATB
    composite_excluder : ExclusionContainer
        Combined land exclusion container from atlite
    cell_resolution : float
        Grid cell resolution in degrees
    cutout : atlite.Cutout
        ERA5 cutout object with climate data
    region_boundary : gpd.GeoDataFrame
        Regional boundary geometry
    region_shape : gpd.GeoDataFrame
        Unified regional shape for availability calculations
    Availability_matrix : xr.DataArray
        Land availability matrix from atlite
    capacity_matrix : xr.DataArray
        Potential capacity matrix with resource and land constraints
    provincial_cells : gpd.GeoDataFrame
        Final processed grid cells with capacity and cost attributes
    resource_capex : float
        Capital expenditure cost (million $/MW)
    resource_fom : float
        Fixed operation and maintenance cost (million $/MW)
    resource_vom : float
        Variable operation and maintenance cost (million $/MW)
    grid_connection_cost_per_km : float
        Grid connection cost per kilometer (million $)
    tx_line_rebuild_cost : float
        Transmission line rebuild cost (million $)
        
    Methods
    -------
    load_cost(resource_atb: pd.DataFrame) -> tuple
        Extract cost parameters from NREL ATB data and convert units
    __get_unified_region_shape__() -> gpd.GeoDataFrame
        Create unified regional boundary by dissolving sub-regional boundaries
    __create_cell_geom__(x: float, y: float) -> Polygon
        Create square grid cell geometry from center coordinates
    get_capacity() -> tuple[gpd.GeoDataFrame, xr.DataArray]
        Main processing method to calculate renewable energy capacity with constraints
    plot_ERA5_grid_land_availability(...) -> matplotlib.figure.Figure
        Create visualization of land availability on ERA5 grid resolution
    plot_excluder_land_availability(...) -> matplotlib.figure.Figure
        Create visualization of land availability at excluder resolution
        
    Examples
    --------
    Process solar capacity for British Columbia:
    
    >>> from RES.CellCapacityProcessor import CellCapacityProcessor
    >>> processor = CellCapacityProcessor(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="solar"
    ... )
    >>> cells_gdf, capacity_matrix = processor.get_capacity()
    >>> print(f"Processed {len(cells_gdf)} cells with total capacity: "
    ...       f"{cells_gdf['potential_capacity_solar'].sum():.1f} MW")
    
    Process wind capacity with visualization:
    
    >>> processor = CellCapacityProcessor(
    ...     config_file_path="config/config_AB.yaml",
    ...     region_short_code="AB",
    ...     resource_type="wind"
    ... )
    >>> cells_gdf, capacity_matrix = processor.get_capacity()
    >>> # Visualizations are automatically generated and saved
    
    Extract cost parameters:
    
    >>> capex, vom, fom, grid_cost, tx_cost = processor.load_cost(
    ...     processor.utility_pv_cost
    ... )
    >>> print(f"Solar CAPEX: {capex:.3f} million $/MW")
    
    Notes
    -----
    - Integrates climate data from ERA5 via atlite cutouts
    - Applies land-use constraints via ExclusionContainer
    - Converts NREL ATB costs from $/kW to million $/MW
    - Creates square grid cells based on ERA5 resolution (~30km at 0.25°)
    - Supports solar, wind, and battery energy storage systems (BESS)
    - Automatically generates land availability visualizations
    - Uses HDF5 storage for efficient data persistence
    - Grid cells are trimmed to exact regional boundaries
    - Assigns unique cell IDs for downstream processing
    - Cost parameters include CAPEX, FOM, VOM, and transmission costs
    
    Processing Workflow
    -------------------
    1. Load ERA5 cutout and regional boundaries
    2. Set up land exclusion constraints
    3. Extract cost parameters from NREL ATB
    4. Calculate availability matrix with land constraints
    5. Apply land-use intensity to compute capacity matrix
    6. Convert to GeoDataFrame with cell geometries
    7. Assign static cost parameters to each cell
    8. Trim cells to precise regional boundaries
    9. Generate visualizations and store results
    
    Cost Parameter Processing
    ------------------------
    - CAPEX: Capital expenditure (converted from $/kW to million $/MW)
    - FOM: Fixed operation & maintenance (million $/MW annually)
    - VOM: Variable operation & maintenance (million $/MWh, if applicable)
    - Grid connection: Cost per kilometer for grid connection
    - Transmission rebuild: Cost for transmission line upgrades
    - Operational life: Asset lifetime (25 years solar, 20 years wind)
    
    Dependencies
    ------------
    - geopandas: Spatial data manipulation
    - xarray: Multi-dimensional array operations
    - pandas: Data frame operations
    - shapely.geometry: Geometric operations
    - matplotlib.pyplot: Visualization
    - atlite: Climate data processing and exclusions
    - RES.AttributesParser: Parent class for configuration management
    - RES.lands.LandContainer: Land constraint processing
    - RES.era5_cutout.ERA5Cutout: Climate data cutout handling
    - RES.hdf5_handler.DataHandler: HDF5 data storage
    - RES.atb.NREL_ATBProcessor: Cost data processing
    - RES.utility: Utility functions for cell operations
    
    Raises
    ------
    KeyError
        If required configuration parameters are missing
    ValueError
        If resource type is not supported or data processing fails
    FileNotFoundError
        If configuration files or data dependencies are not found
    """
    
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
    
        # This dictionary will be used to pass arguments to external classes
        self.required_args = { #order doesn't matter
                "config_file_path": self.config_file_path,  # INHERITED ATTRIBUTE from AttributesParser
                "region_short_code": self.region_short_code,  # INHERITED ATTRIBUTE from AttributesParser
                "resource_type": self.resource_type  # INHERITED ATTRIBUTE from AttributesParser
            }
        self.ERA5Cutout=ERA5Cutout(**self.required_args)
        self.LandContainer=LandContainer(**self.required_args)
        self.resource_disaggregation_config=super().get_resource_disaggregation_config()
        self.resource_landuse_intensity = self.resource_disaggregation_config['landuse_intensity']
        self.atb=NREL_ATBProcessor(**self.required_args)
        
        # Load configuration attributes that were previously inherited
        self.cutout_config = super().get_cutout_config()  # INHERITED METHOD from AttributesParser
        self.gadm_config = super().get_gadm_config()  # INHERITED METHOD from AttributesParser
        self.region_name = super().get_region_name()  # INHERITED METHOD from AttributesParser
        
        (self.utility_pv_cost, 
        self.land_based_wind_cost, 
        # self.bess_cost
        )= self.atb.pull_data()
        
        ## Initiate the Store and Datahandler (interfacing with the Store)
        self.datahandler=DataHandler(self.store)  # INHERITED ATTRIBUTE from AttributesParser
        
        ### Exclusion Layer Container
        self.composite_excluder:ExclusionContainer=None
        ### ERA5 Cutout Resolution
        self.cell_resolution=self.cutout_config['dx']
    
    
    def __get_unified_region_shape__(self):
        # if self.sub_national_unit_tag in self.LandContainer.region_shape.columns:
        self.region_shape=self.LandContainer.region_shape.dissolve(by=self.gadm_config.get('datafield_mapping').get('NAME_0')) # .drop(columns =['Region'])
        self.region_shape = self.region_shape[['geometry']]
        return self.region_shape


#     def load_cost(self,
#                   resource_atb:pd.DataFrame=None):
#         """
#         Extracts cost parameters from the NREL ATB DataFrame and converts them to million $/MW.

#         Args:
#             resource_atb (pd.DataFrame): DataFrame containing NREL ATB cost data for the resource type.

#         Returns:
#             dict: A dictionary containing the following cost parameters:
#                 - resource_capex: Capital expenditure in million $/MW
#                 - resource_vom: Variable operation and maintenance cost in million $/MW
#                 - resource_fom: Fixed operation and maintenance cost in million $/MW
#                 - grid_connection_cost_per_km: Grid connection cost per kilometer in million $
#                 - tx_line_rebuild_cost: Transmission line rebuild cost in million $
#         """
        
#         utils.print_update(level=PRINT_LEVEL_BASE+1,
#                            message=f"{__name__}| Extracting cost attributes...")
#         self.transmission_config = self.config.get('capacity_disaggregation', {}).get('transmission', {}) # INHERITED ATTRIBUTE from AttributesParser
#         grid_connection_cost_per_km = self.transmission_config.get('grid_connection_cost_per_Km', 0)
#         utils.print_info(f"{__name__}| @ Line: {inspect.currentframe().f_lineno-1} | `grid_connection_cost_per_km` is set to {grid_connection_cost_per_km} million $/km. If this is not set in the config, it will be set to 0.")
        
#         tx_line_rebuild_cost = self.disaggregation_config.get('transmission', {}).get('tx_line_rebuild_cost', 0)
#         utils.print_info(f"{__name__}| @ Line: {inspect.currentframe().f_lineno-1} | `tx_line_rebuild_cost` is set to {tx_line_rebuild_cost} million $. If this is not set in the config, it will be set to 0.")
        
#         if resource_atb is None:
#             utils.print_info((f"{__name__}| No ATB cost data set for {self.resource_type}. Attempting to extract from resource disaggregation config..."))
#             resource_capex:float=float(self.resource_disaggregation_config.get('capex', None))
#             if resource_capex is None:
#                 raise ValueError(f"CAPEX value is missing for {self.resource_type} in the resource disaggregation config.")
#             resource_fom:float=float(self.resource_disaggregation_config.get('fom', None))
#             if resource_fom is None:
#                 raise ValueError(f"FOM value is missing for {self.resource_type} in the resource disaggregation config.")
#             resource_vom:float=float(self.resource_disaggregation_config.get('vom', None))
#             if resource_vom is None:
#                 raise ValueError(f"VOM value is missing for {self.resource_type} in the resource disaggregation config.")
            
#         else:
#             self.ATB:Dict[str,dict]=super().get_atb_config()
#             source_column:str= self.ATB.get('column',{})
#             cost_params_mapping:Dict[str,str]=self.ATB.get('cost_params',{})
            
#             # capex,fom,vom in NREL is given in US$/kw and we need to convert it to million $/MW
#             resource_capex:float=resource_atb[resource_atb[source_column]==cost_params_mapping.get('capex',{})].value.iloc[0]/ 1E3  # Convert to million $/MW
#             resource_fom:float=resource_atb[resource_atb[source_column]==cost_params_mapping.get('fom',{})].value.iloc[0] /1E3  # Convert to million $/MW
            
#             # Initialize resource_vom based on the availability of 'vom' in cost_params_mapping
#             resource_vom = 0.0

#             if cost_params_mapping.get('vom') is not None:
#                 # Check if the DataFrame 'utility_scale_cost' is not empty and get the value for 'vom'
#                 if not resource_atb.empty:
#                     vom_row = resource_atb[resource_atb[source_column] == cost_params_mapping['vom']]
#                     if not vom_row.empty:
#                         resource_vom = vom_row['value'].iloc[0] / 1E3  # Convert to million $/MW
        
#         cost_components:dict={
#             'resource_capex': resource_capex,                    # million $/MW
#             'resource_vom': resource_vom,                        # million $/MW
#             'resource_fom': resource_fom,                        # million $/MW
#             'grid_connection_cost_per_km': grid_connection_cost_per_km,  # million $
#             'tx_line_rebuild_cost': tx_line_rebuild_cost         # million $
#         }
        
#         # Validation
#         expected_keys = ['resource_capex', 'resource_vom', 'resource_fom', 
#                         'grid_connection_cost_per_km', 'tx_line_rebuild_cost']
        
#         assert all(key in cost_components for key in expected_keys), "Missing cost components"
#         assert all(isinstance(value, (int, float)) for value in cost_components.values()), "All values must be numeric"
        
#         # Return as ordered dictionary
#         return cost_components
# """ 
    def load_cost(self, resource_atb: pd.DataFrame = None) -> dict:
        """
        Extract cost parameters for the resource and return them in million $/MW.

        Precedence for each resource cost parameter (capex, fom, vom):
        1. resource_disaggregation_config
        2. NREL ATB (if provided)
        3. default / error handling

        Rules:
        - capex: required
        - fom: required
        - vom: optional, defaults to 0.0 if missing in both config and ATB

        Args:
            resource_atb (pd.DataFrame, optional):
                DataFrame containing NREL ATB cost data for the resource type.

        Returns:
            dict:
                {
                    'resource_capex': float,               # million $/MW
                    'resource_vom': float,                 # million $/MW
                    'resource_fom': float,                 # million $/MW
                    'grid_connection_cost_per_km': float, # million $/km
                    'tx_line_rebuild_cost': float         # million $
                }
        """

        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__}| Extracting cost attributes..."
        )

        self.transmission_config = self.config.get("capacity_disaggregation", {}).get("transmission", {})
        grid_connection_cost_per_km = float(
            self.transmission_config.get("grid_connection_cost_per_Km", 0)
        )
        utils.print_info(
            f"{__name__}| `grid_connection_cost_per_km` = {grid_connection_cost_per_km} million $/km."
        )

        tx_line_rebuild_cost = float(
            self.disaggregation_config.get("transmission", {}).get("tx_line_rebuild_cost", 0)
        )
        utils.print_info(
            f"{__name__}| `tx_line_rebuild_cost` = {tx_line_rebuild_cost} million $."
        )

        resource_cfg = self.resource_disaggregation_config or {}

        source_column = None
        cost_params_mapping = {}

        if resource_atb is not None:
            self.ATB: Dict[str, dict] = super().get_atb_config()
            source_column = self.ATB.get("column")
            cost_params_mapping = self.ATB.get("cost_params", {})

            if source_column is None:
                raise ValueError("ATB config is missing the 'column' definition.")

        def _get_cfg_value(param_name: str):
            """
            Return float(config[param_name]) if present and not None, else None.
            """
            value = resource_cfg.get(param_name)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid {param_name} value in resource_disaggregation_config "
                    f"for {self.resource_type}: {value}"
                ) from e

        def _get_atb_value(param_name: str):
            """
            Return float value from ATB (converted from US$/kW to million $/MW),
            or None if unavailable.
            """
            if resource_atb is None or resource_atb.empty:
                return None

            atb_label = cost_params_mapping.get(param_name)
            if atb_label is None:
                return None

            row = resource_atb[resource_atb[source_column] == atb_label]
            if row.empty:
                return None

            try:
                return float(row["value"].iloc[0]) / 1e3
            except (TypeError, ValueError, KeyError, IndexError) as e:
                raise ValueError(
                    f"Could not extract ATB value for '{param_name}' "
                    f"for {self.resource_type}."
                ) from e

        def _resolve_cost(param_name: str, required: bool = True, default=None) -> float:
            """
            Resolve cost parameter using priority:
            config -> ATB -> default/error
            """
            cfg_value = _get_cfg_value(param_name)
            if cfg_value is not None:
                utils.print_info(
                    f"{__name__}| Using '{param_name}' from resource_disaggregation_config "
                    f"for {self.resource_type}: {cfg_value}"
                )
                return cfg_value

            atb_value = _get_atb_value(param_name)
            if atb_value is not None:
                utils.print_info(
                    f"{__name__}| Using '{param_name}' from ATB for {self.resource_type}: "
                    f"{atb_value}"
                )
                return atb_value

            if default is not None:
                utils.print_info(
                    f"{__name__}| '{param_name}' missing in both config and ATB for "
                    f"{self.resource_type}. Using default = {default}"
                )
                return float(default)

            if required:
                raise ValueError(
                    f"Missing required cost parameter '{param_name}' for {self.resource_type}. "
                    f"Not found in resource_disaggregation_config or ATB."
                )

            return None

        resource_capex = _resolve_cost("capex", required=True)
        resource_fom = _resolve_cost("fom", required=True)
        resource_vom = _resolve_cost("vom", required=False, default=0.0)

        cost_components = {
            "resource_capex": resource_capex,                      # million $/MW
            "resource_vom": resource_vom,                          # million $/MW
            "resource_fom": resource_fom,                          # million $/MW
            "grid_connection_cost_per_km": grid_connection_cost_per_km,  # million $/km
            "tx_line_rebuild_cost": tx_line_rebuild_cost          # million $
        }

        expected_keys = [
            "resource_capex",
            "resource_vom",
            "resource_fom",
            "grid_connection_cost_per_km",
            "tx_line_rebuild_cost",
        ]

        assert all(key in cost_components for key in expected_keys), "Missing cost components"
        assert all(isinstance(value, (int, float)) for value in cost_components.values()), \
            "All values must be numeric"

        return cost_components
    
    # Define a function to create bounding boxes (of cell) directly from coordinates (x, y) and resolution
    def __create_cell_geom__(self,x, y):
        half_res = self.cell_resolution / 2
        return Polygon([
            (x - half_res, y - half_res),  # Bottom-left
            (x + half_res, y - half_res),  # Bottom-right
            (x + half_res, y + half_res),  # Top-right
            (x - half_res, y + half_res)   # Top-left
        ])
        
    def get_capacity(self)->tuple:
        """
        This method processes the capacity of the resources based on the availability matrix and other parameters.
        It calculates the potential capacity for each cell in the region and returns a named tuple containing the
        processed data and the capacity matrix.
        Returns:
            namedtuple: A named tuple containing the processed `data` and the capacity `matrix`. Can be accessed as:
            `<self.resources_nt>.data` and `<self.resources_nt>.matrix`
        """
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Cell capacity processor initiated...")
        
    #a. load cutout and region boundary for which the cutout has been created.
        self.cutout,self.region_boundary=self.ERA5Cutout.get_era5_cutout()
        
    #b. load excluder
        self.composite_excluder=self.LandContainer.set_excluder()
        
        
    #d. Load costs (float)
    
    # Load cost data as dictionary
        cost_components:dict = self.load_cost(
                                    resource_atb=(
                                        self.utility_pv_cost if self.resource_type == 'solar' else
                                        self.land_based_wind_cost if self.resource_type == 'wind' else
                                        self.bess_cost if self.resource_type == 'bess' else
                                        None
                                    )
                                )


        # Assign to instance variables
        self.resource_capex = cost_components['resource_capex']
        self.resource_vom = cost_components['resource_vom'] 
        self.resource_fom = cost_components['resource_fom']
        self.grid_connection_cost_per_km = cost_components['grid_connection_cost_per_km']
        self.tx_line_rebuild_cost = cost_components['tx_line_rebuild_cost']
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| ✓ Cost parameters loaded for {self.resource_type} resources.")
        
    ## 2.1 Compute availability Matrix
        self.region_shape= self.__get_unified_region_shape__() # we need to pass the unified region shape to the availability matrix calculation.
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                   message=f"{__name__}| Processing Availability Matrix... ")
        self.Availability_matrix:xr = self.cutout.availabilitymatrix(self.region_shape, self.composite_excluder)
        
        utils.print_info(f"{__name__}| @ Line: {inspect.currentframe().f_lineno-1} | We need to pass the unified `region_shape` to the cutout to calculate availability for the entire region as in one of the dimensions e.g. here 'Province'. If we pass multipolygons/geoms of each Regional district (sub-provincial) we will get availability for each regional district as a dimension; which adds additional step to produce our intended data. For this analysis, one unified shape for entire region is sufficient")
        
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                   message=f"{__name__}| ✓ Availability Matrix processed for {self.region_name}. ")
        
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Creating visuals for land-availability")
        self.availability_gdf=self.plot_ERA5_grid_land_availability()
        self.plot_excluder_land_availability(excluder=self.composite_excluder)
        
        area = self.cutout.grid.set_index(["y", "x"]).to_crs(self.crs_m).area / 1e6 # in Sq. km
        area = xr.DataArray(area, dims=("spatial"))
        
        utils.print_update(level=PRINT_LEVEL_BASE+1,
               message=f"{__name__}| Calculating capacity matrix, using land-use intensity for {self.resource_type} resources: {self.resource_landuse_intensity} MW/km²")
        capacity_matrix:xr.DataArray = self.Availability_matrix.stack(spatial=["y", "x"]) * area * self.resource_landuse_intensity
        
        self.capacity_matrix=capacity_matrix.rename(f'potential_capacity_{self.resource_type}')
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                   message=f"{__name__}| ✓ Capacity Matrix processed for {self.region_name}. ")
        
    ## ## 2.1 convert the Availability Matrix to dataframe.
        # Keep x/y coordinates! self.capacity_matrix has a stacked 'spatial' MultiIndex (y,x).
        # Using reset_index(drop=True) discards those coordinates, so DON'T drop=True.

        # First try the straightforward path:
        _df_flat = self.capacity_matrix.to_dataframe()#.reset_index()

        # If for any reason x/y didn’t appear (older xarray/pandas edge cases), fall back to Series:
        if ("x" not in _df_flat.columns) or ("y" not in _df_flat.columns):
            _df_flat   =_df_flat.reset_index()
            s = self.capacity_matrix.to_series()  # expands MultiIndex levels to columns on reset_index()
            _df_flat = s.reset_index()
            # ensure the value column is correctly named
            value_col = self.capacity_matrix.name or f"potential_capacity_{self.resource_type}"
            if 0 in _df_flat.columns:
                _df_flat.rename(columns={0: value_col}, inplace=True)

        # Optional cleanups: remove obvious empties
        # _df_flat = _df_flat.drop_duplicates(subset=["y", "x"], keep="first")
        # _df_flat = _df_flat[_df_flat[value_col] > 0]

    ## 2.1 convert the Availability Matrix to dataframe.

        _df_flat:pd.DataFrame=self.capacity_matrix.to_dataframe()

        if not {'x', 'y'}.issubset(_df_flat.columns):
            _df_flat = _df_flat.reset_index()

        self.provincial_cell_capacity_gdf:gpd.GeoDataFrame = gpd.GeoDataFrame(
            _df_flat,
            geometry=[self.__create_cell_geom__(x, y) for x, y in zip(_df_flat['x'], _df_flat['y'])],
            crs=self.crs_d  # INHERITED METHOD from AttributesParser
        )
        
    ## 3 Assign Static exogenous Costs after potential capacity calculation
        parameters_to_add = {
            'capex': self.resource_capex, # m$/MW
            'fom': self.resource_fom, # m$/MW
            'vom': round(self.resource_vom, 4), # m$/MW
            'grid_connection_cost_per_km': self.grid_connection_cost_per_km, # m$/km
            'tx_line_rebuild_cost': self.tx_line_rebuild_cost,  # m$/km
            'Operational_life': int(25) if self.resource_type == 'solar' else int(20) if self.resource_type == 'wind' else 0   # years
        }

        # Create a new dictionary with stylized keys
        stylized_columns = {f'{key}_{self.resource_type}': value for key, value in parameters_to_add.items()}

        # Assign the new stylized columns to the DataFrame
        self.provincial_cell_capacity_gdf =  self.provincial_cell_capacity_gdf.assign(**stylized_columns)

    ## 4 Trim the cells to sub-provincial boundaries instead of overlapping cell (boxes) in the regional boundaries.
        self.provincial_cell_capacity_gdf= self.provincial_cell_capacity_gdf.overlay(self.region_boundary)
        
        # print(_provincial_cell_capacity_gdf.columns) # debugging purpose
        
        self.provincial_cells=utils.assign_cell_id(cells= self.provincial_cell_capacity_gdf,
            source_column=self.sub_national_unit_tag)
        self.provincial_cells_with_LandAvailability = self.provincial_cells.join(
            self.availability_gdf[[f"LandAvailability_{self.resource_type}"]],
            how="left"
        )
        
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                   message=f"{__name__}| ✓ Capacity dataframe cleaned and processed")
        
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                   message=f"{__name__}| ERA5 cells' capacity loaded for : {len(self.provincial_cell_capacity_gdf)} Cells [each with .025 deg. (~30km) resolution ]")
       
        self.datahandler.to_store(self.provincial_cells_with_LandAvailability,'cells')
     
        return  self.provincial_cells_with_LandAvailability,self.capacity_matrix
    
    
    
## Visuals 
    def plot_ERA5_grid_land_availability(self,
                                          region_boundary:gpd.GeoDataFrame=None,
                                          Availability_matrix:xr.DataArray=None,
                                          figsize=(8, 6),
                                          legend_box_x_y:tuple=(1.2, 1)):
        
        """
        Plots the land availability based on the ERA5 grid cells.
        Args:
            region_boundary (gpd.GeoDataFrame, optional): The region boundary to plot. If
                not provided, the default region boundary will be used.
            Availability_matrix (xr.DataArray, optional): The availability matrix to plot.
                If not provided, the default Availability matrix will be used.
            figsize (tuple, optional): The size of the figure to create. Defaults to (8, 6).
            legend_box_x_y (tuple, optional): The position of the legend box in the plot.
                Defaults to (1.2, 1).
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
        """
        if region_boundary is None:
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                               message=f"{__name__}| No region boundary provided, using the default region boundary.")
            # Use the default region boundary if not provided   
            region_boundary = self.region_boundary
        else:
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                               message=f"{__name__}| Using provided region boundary for plotting.")
        # Ensure the region boundary is in the correct CRS
        if region_boundary.crs is None or region_boundary.crs.to_string() != 'EPSG:4326'    :
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                               message=f"{__name__}| Region boundary CRS is None, setting to EPSG:4326.")
            region_boundary = region_boundary.set_crs('EPSG:4326')
        
        # Load availability data
        if Availability_matrix is None:
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                               message=f"{__name__}| No Availability matrix provided, using the default Availability matrix.")
            # Use the default Availability matrix if not provided
            Availability_matrix:xr.DataArray = self.Availability_matrix
        else:
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                               message=f"{__name__}| Using provided Availability matrix for plotting.")
            Availability_matrix:xr.DataArray = Availability_matrix
            
        Availability_df=Availability_matrix.to_dataframe(name="availability").reset_index()
        
        # Define bins and labels
        bins = [x / 100 for x in [1, 10, 30, 60, 90, 100]]  # Define bin edges
        labels = ["1-10%", "10-30%", "30-60%", "60-90%", ">90%"]
        
        
        # Categorize availability into bins
        # Availability_df["availability_category"] = pd.cut(Availability_df["availability"], bins=bins, labels=labels, include_lowest=True)

        # Convert to GeoDataFrame
        A_gdf:gpd.GeoDataFrame = gpd.GeoDataFrame(
                    Availability_df,
                    geometry=[self.__create_cell_geom__(x, y) for x, y in zip(Availability_df['x'], Availability_df['y'])],
                    crs='EPSG:4326'
                )
                
        A_gdf=A_gdf.overlay(region_boundary)
        A_gdf = A_gdf.rename(columns={"availability": f"LandAvailability_{self.resource_type}"})
        A_gdf=utils.assign_cell_id(cells=A_gdf, source_column=self.sub_national_unit_tag)
        self.datahandler.to_store(A_gdf,"LandAvailability")

        # Categorize availability into bins
        A_gdf_categorized=A_gdf.copy()
        A_gdf_categorized[f"LandAvailability_{self.resource_type}_category"] = pd.cut(A_gdf_categorized[f"LandAvailability_{self.resource_type}"], bins=bins, labels=labels, include_lowest=True)
        # Create figure and axes for side-by-side plotting
        fig, ax = plt.subplots(figsize=figsize,constrained_layout=True)

        # Set axis off for both subplots
        ax.set_axis_off()

        # Shadow effect offset
        # shadow_offset = 0.004

        if region_boundary.crs is None or region_boundary.crs.to_string() != self.crs_m:
            region_boundary_proj = region_boundary.to_crs(self.crs_m)
            A_gdf_proj = A_gdf_categorized.to_crs(self.crs_m)
            utils.print_update(level=PRINT_LEVEL_BASE+2,
                               message=f"{__name__}| Converted region boundary and availability GeoDataFrame to {self.crs_m} for plotting.")
        # Plot solar map on ax1
        # Add shadow effect for solar map
        # region_boundary_proj.geometry = region_boundary_proj.geometry.translate(xoff=shadow_offset, yoff=-shadow_offset)
        # region_boundary_proj.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=2, alpha=0.3)  # Shadow layer

        # Plot solar cells
        A_gdf_proj.plot(column=f'LandAvailability_{self.resource_type}_category', ax=ax, cmap='Greens', legend=True, 
                legend_kwds={'title': "Land Availability", 'loc': 'upper right', 'bbox_to_anchor': legend_box_x_y,'borderpad': 1,'frameon': False})

        # Plot actual boundary for solar map
        region_boundary_proj.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.9)
        plt.subplots_adjust(right=0.85)  # Increase space on the right
        ax.set_title(f"Land Availability for {self.resource_type} resources ({self.region_name})", fontsize=14)
        # Adjust layout for cleaner appearance
        plt.tight_layout()
        vis_save_to_root=self.get_vis_dir()
        plot_save_to=Path(vis_save_to_root)/'lands'
        utils.ensure_path(plot_save_to)
        plt.savefig(f'{plot_save_to}/land_availability_ERA5grid_{self.region_short_code}_{self.resource_type}.png',dpi=500)
        utils.print_update(level=PRINT_LEVEL_BASE+3,message=f"{__name__}|Land availability (grid cells) map saved at {vis_save_to_root}")
        return A_gdf
        

    def plot_excluder_land_availability(self,
                                        excluder:ExclusionContainer=None):
        """
        Plots the land availability based on the excluder resolution.
        Args:
            excluder (ExclusionContainer, optional): The excluder to use for plotting
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot
        """
        
        if excluder is None:
            excluder = self.composite_excluder
            
        fig, ax = plt.subplots(figsize=(9, 6),constrained_layout=True)
        excluder.plot_shape_availability(geometry=self.region_shape,
                                        plot_kwargs={'facecolor':'none','edgecolor':'black'},
                                        ax=ax)
        ax.axis("off")
        vis_save_to_root=self.get_vis_dir()
        plot_save_to=Path(vis_save_to_root)/'lands'
        utils.ensure_path(plot_save_to)
        plt.savefig(f'{plot_save_to}/land_availability_excluderResolution_{self.region_name}.png',dpi=500)
        utils.print_update(level=PRINT_LEVEL_BASE+3,message=f"{__name__}|Land availability map (excluder resolution) saved to {plot_save_to}/land_availability_excluderResolution_{self.region_name}.png")
        return fig
    
@staticmethod
def get_sub_nationally_aggregated_capacity(cells_with_capacity:gpd.GeoDataFrame=None,
                                           sub_national_unit_tag:str=None):
    
    if cells_with_capacity is None or not isinstance(cells_with_capacity, gpd.GeoDataFrame):
        raise ValueError("The input must be a valid GeoDataFrame with capacity data.")
    if 'potential_capacity_solar' not in cells_with_capacity.columns or 'potential_capacity_wind' not in cells_with_capacity.columns:
        raise ValueError("The input GeoDataFrame must contain 'potential_capacity_solar' and 'potential_capacity_wind' columns.")
    if sub_national_unit_tag is None or sub_national_unit_tag not in cells_with_capacity.columns:
        raise ValueError("The input GeoDataFrame must contain 'sub_national_unit_tag' column for aggregation.")
    
        
    cells_aggr=cells_with_capacity.groupby(sub_national_unit_tag).aggregate({'potential_capacity_solar':'sum','potential_capacity_wind':'sum'})

    cells_aggr[['potential_capacity_solar', 'potential_capacity_wind']] = cells_aggr[['potential_capacity_solar', 'potential_capacity_wind']].round().astype(int)

    return cells_aggr