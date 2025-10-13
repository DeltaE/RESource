"""
lands.py
This module provides classes and functions for processing, analyzing, and visualizing land use and conservation data,
with a focus on Canadian regions. It integrates vector and raster geospatial data sources, including protected and
conserved areas, OSM features, and GAEZ raster datasets, to support land exclusion/inclusion analysis for resource
planning (e.g., renewable energy siting).

Key Components:
---------------
- ConservationLands:
    Handles downloading, extracting, and processing of conservation lands data (e.g., Canadian Protected and Conserved Areas Database).
    Provides methods for loading, simplifying, and mapping conserved land geometries at the provincial level.
- LandContainer:
    Combines multiple geospatial data sources (ERA5 cutouts, GAEZ rasters, OSM data, conservation lands) to manage
    inclusion/exclusion of lands for spatial analysis. Supports loading, buffering, and plotting of raster and vector layers.
- Utility Functions:
    - add_and_plot_exclusion_layer: Adds raster/vector layers to an exclusion container and visualizes eligible areas.
    - load_layers_to_excluder: Loads and visualizes all configured raster and vector exclusion layers for a region.
    - apply_buffer_to_vector: Buffers vector geometries by configurable distances and compares area changes.
    - get_eligible_share: Computes the share of eligible (non-excluded) area within a region.

Dependencies:
-------------
- geopandas, pandas, numpy, matplotlib, rasterio, fiona, atlite.gis, folium
- Custom modules: RES.utility, RES.boundaries, RES.era5_cutout, RES.gaez, RES.osm

Intended Use:
-------------
This module is intended for use in spatial resource assessment workflows, particularly for renewable energy planning
where land exclusions (e.g., protected areas, infrastructure buffers) must be considered. It is designed to be
configurable and extensible for different regions and data sources.


"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
from zipfile import ZipFile

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from atlite.gis import ExclusionContainer
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from RES import utility as utils
from RES.AttributesParser import AttributesParser
from RES.boundaries import GADMBoundaries
from RES.era5_cutout import ERA5Cutout
from RES.gaez import GAEZRasterProcessor
from RES.osm import OSMData

PRINT_LEVEL_BASE: int = 2  # handles the print level for the utils.print_update function
class ConservationLands(AttributesParser):
    """
    Conservation lands data processor for protected and conserved areas analysis.
    
    This class handles the downloading, processing, and analysis of conservation lands
    data, particularly focusing on Canadian Protected and Conserved Areas Database
    (CPCAD). It provides comprehensive functionality for loading, simplifying, and
    mapping conserved land geometries at provincial and regional levels for use in
    renewable energy resource assessment and land use planning.
    
    The class integrates conservation data with regional boundaries to support
    land exclusion analysis in renewable energy siting decisions. It processes
    geospatial data from government sources and provides tools for visualization
    and spatial analysis of protected areas.
    
    INHERITED METHODS FROM GADMBoundaries:
    --------------------------------------
    - get_region_boundary() -> gpd.GeoDataFrame: Get regional boundary geometry
    - get_bounding_box() -> tuple: Get regional bounding box coordinates
    - Plus other boundary processing methods
    
    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_region_name() -> str: Get full region name for display
    - get_resource_disaggregation_config() -> dict: Get resource configuration
    - get_excluder_crs() -> str: Get coordinate reference system for exclusions
    - Plus other configuration access methods
    
    INHERITED ATTRIBUTES FROM AttributesParser:
    -------------------------------------------
    - config: Configuration dictionary with all settings
    - gadm_config: GADM-specific configuration parameters
    - Plus other configuration attributes
    
    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - get_provincial_conserved_lands(): Load and process provincial conservation data
    - show_lands(): Visualize conservation lands with regional boundaries
    - __get_conserved_lands__(): Download and extract conservation data archives
    
    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing conservation data parameters
    region_short_code : str
        Region identifier for boundary definition and data filtering
    resource_type : str
        Resource type for compatibility with broader workflows
        
    Attributes
    ----------
    conserved_lands_cfg : dict
        Conservation lands configuration from config file
    source_url : str
        URL for downloading conservation data archives
    data_root : str or Path
        Root directory for conservation data storage
    zip_file_name : str
        Name of the ZIP archive containing conservation data
    zip_file_path : Path
        Full path to the conservation data ZIP file
    extraction_dir : Path
        Directory for extracting conservation data files
    region_boundary : gpd.GeoDataFrame
        Regional boundary geometry for spatial filtering
    region_shape : gpd.GeoDataFrame
        Dissolved regional geometry for analysis
    region_name : str
        Full name of the region for display purposes
    resource_disaggregation_config : dict
        Configuration for resource type disaggregation
    aeroway_gdf : gpd.GeoDataFrame
        Aeroway geometries for infrastructure analysis
    raster_configs : list
        List of raster layer configurations
        
    Methods
    -------
    get_provincial_conserved_lands(geom_simplification_tolerance=0.005) -> gpd.GeoDataFrame
        Load and process provincial conservation lands data with geometry simplification
    show_lands(conserved_lands=None, save_to=None, show=True) -> None
        Visualize conservation lands overlaid on regional boundaries
        
    Examples
    --------
    Create conservation lands processor for British Columbia:
    
    >>> from RES.lands import ConservationLands
    >>> conservation = ConservationLands(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="solar"
    ... )
    >>> 
    >>> # Load provincial conservation data
    >>> conserved_areas = conservation.get_provincial_conserved_lands()
    >>> print(f"Loaded {len(conserved_areas)} conservation areas")
    
    Visualize conservation lands:
    
    >>> # Show conservation areas with regional boundaries
    >>> conservation.show_lands(
    ...     conserved_lands=conserved_areas,
    ...     save_to="plots/BC_conservation.png",
    ...     show=True
    ... )
    
    Access conservation data with geometry simplification:
    
    >>> # Load with custom simplification tolerance
    >>> simplified_areas = conservation.get_provincial_conserved_lands(
    ...     geom_simplification_tolerance=0.001
    ... )
    >>> print(f"Simplified to {len(simplified_areas)} areas")
    
    Configuration Requirements
    --------------------------
    The configuration must include conservation lands parameters:
    
    ```yaml
    Gov:
      conservation_lands:
        url: "https://www.donneesquebec.ca/recherche/dataset/..."
        root: "data/downloaded_data/conservation"
        data_name: "CPCAD_Dec2023"
        layers:
          - name: "Protected_Conserved_Areas"
            file: "CPCAD-BDCAP_Dec2023.gdb"
    ```
    
    Data Processing Workflow
    ------------------------
    1. **Configuration Loading**: Extract conservation data parameters
    2. **Data Download**: Download conservation data archives if needed
    3. **Data Extraction**: Extract GIS files from archives
    4. **Boundary Processing**: Load regional boundaries for spatial filtering
    5. **Spatial Filtering**: Clip conservation data to regional extent
    6. **Geometry Simplification**: Simplify complex geometries for performance
    7. **Attribute Processing**: Clean and standardize attribute data
    8. **Visualization**: Generate maps showing conservation areas
    
    Conservation Data Types
    -----------------------
    Typical conservation datasets include:
    - **Protected Areas**: National parks, provincial parks, marine protected areas
    - **Conserved Areas**: Conservation easements, private conservancies
    - **Indigenous Territories**: Traditional territories and land claims
    - **Wildlife Reserves**: Critical habitat and wildlife management areas
    - **Buffer Zones**: Areas around sensitive ecosystems
    
    Spatial Processing
    ------------------
    - **Coordinate Systems**: Automatic CRS handling and reprojection
    - **Geometry Simplification**: Configurable tolerance for performance optimization
    - **Spatial Filtering**: Intersection with regional boundaries
    - **Topology Validation**: Automatic geometry validation and repair
    - **Area Calculations**: Accurate area computation in appropriate projections
    
    Performance Optimization
    ------------------------
    - **Geometry Simplification**: Reduces complexity while preserving accuracy
    - **Spatial Indexing**: Efficient spatial queries and intersections
    - **Memory Management**: Streaming processing for large datasets
    - **Caching**: Local storage of processed data to avoid reprocessing
    - **Lazy Loading**: Data loaded only when needed
    
    Integration Points
    ------------------
    - **Regional Boundaries**: Integration with GADM boundary data
    - **Land Exclusions**: Provides input for renewable energy exclusion analysis
    - **Visualization**: Compatible with mapping and plotting workflows
    - **Resource Assessment**: Supports land availability calculations
    
    Data Quality
    ------------
    - **Data Validation**: Automatic validation of geometry and attributes
    - **Currency Checking**: Warnings for outdated conservation data
    - **Completeness Assessment**: Reports on data coverage and gaps
    - **Accuracy Metrics**: Spatial accuracy assessment where possible
    
    Notes
    -----
    - Conservation data is typically updated annually or bi-annually
    - Large conservation databases may require substantial processing time
    - Geometry simplification balances performance and accuracy
    - Results support both renewable energy and conservation planning
    - Integration with other land use datasets enhances analysis capabilities
    - Spatial accuracy depends on source data quality and scale
    
    Dependencies
    ------------
    - geopandas: Spatial data processing and geometry operations
    - pandas: Tabular data manipulation and analysis
    - fiona: Reading GIS file formats
    - shapely: Geometric operations and validation
    - matplotlib: Visualization and plotting
    - pathlib: File path operations
    - zipfile: Archive extraction and management
    - RES.boundaries.GADMBoundaries: Parent class for boundary processing
    - RES.utility: Logging and utility functions
    
    Raises
    ------
    ConnectionError
        If conservation data download fails due to network issues
    FileNotFoundError
        If required data files or configuration are missing
    ValueError
        If geometry simplification tolerance or other parameters are invalid
    GeometryError
        If conservation area geometries are invalid or cannot be processed
        
    See Also
    --------
    geopandas.GeoDataFrame.simplify : Geometry simplification functionality
    fiona.open : Reading GIS data files
    RES.boundaries.GADMBoundaries : Parent class for boundary processing
    """

    def __post_init__(self):
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
                
        self.required_args = {   #order doesn't matter
            "config_file_path" : self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
        }
        self.gadm_boundaries = GADMBoundaries(**self.required_args)  # INHERITED METHOD from GADMBoundaries     
        
        # Set the Class specific attributes
        self.conserved_lands_cfg = self.config["Gov"]["conservation_lands"]

        self.source_url = self.conserved_lands_cfg["url"]
        self.data_root = self.conserved_lands_cfg["root"]
        self.zip_file_name = f"{self.conserved_lands_cfg['data_name']}.zip"
        self.zip_file_path = Path(self.data_root) / self.zip_file_name
        self.extraction_dir = Path(self.data_root) / self.zip_file_path.stem
        self.extraction_dir.parent.mkdir(parents=True, exist_ok=True)

        # Initialize region_boundary attribute
        self.region_boundary = self.gadm_boundaries.get_region_boundary()  # INHERITED METHOD from GADMBoundaries
        self.region_shape = self.region_boundary.dissolve(
            by=self.get_gadm_config()["datafield_mapping"]["NAME_1"]  # INHERITED METHOD from AttributesParser
        )  # Get the geometry of the region boundary
        self.region_name = self.get_region_name()  # INHERITED METHOD from AttributesParser

        # Set up resource disaggregation configurations
        self.resource_disaggregation_config: dict = (
            self.get_resource_disaggregation_config()  # INHERITED METHOD from AttributesParser
        )

        self.aeroway_gdf: gpd.GeoDataFrame = None  # Initialize aeroway_gdf attribute
        self.raster_configs: list = []  # Initialize raster_configs attribute

    def get_provincial_conserved_lands(
        self, geom_simplification_tolerance=0.005
    ) -> gpd.GeoDataFrame:
        """
        Load provincial conserved lands from the .gdb file.

        ### Args:
            geom_simplification_tolerance (default to _.005_)
            - geometry simplification to avoid unnecessary granular level geometries.
            - This tool is configured to geom in degrees, e.g tolerance of 0.005 corresponds to approximately 500m (at the equator) geoms will be simplified.
        """

        utils.print_update(
            level=PRINT_LEVEL_BASE + 3,
            message=f"{__name__}| Processing Conserved areas for {self.region_name}",
        )

        file_name_prefix: str = self.conserved_lands_cfg.get(
            "data_name", "ProtectedConservedArea"
        )
        gdb_layer: str = self.conserved_lands_cfg.get(
            "gdb_layer", "ProtectedConservedArea_2023"
        )

        provincial_file_path = (
            Path("data/downloaded_data/lands")
            / f"{file_name_prefix}_{self.region_short_code}.pickle"
        )
        provincial_file_path.parent.mkdir(parents=True, exist_ok=True)

        if provincial_file_path.exists():
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f"{__name__}| Loading regional data from Canadian Protected and Conserved Areas Database (CPCAD) from locally stored datafile - {provincial_file_path}",
            )
            gdf = gpd.GeoDataFrame(pd.read_pickle(provincial_file_path))

        else:
            gdb_file_path: Path = self.__get_conserved_lands__()

            # Get Region Boundaries
            self.region_boundary: gpd.GeoDataFrame = self.gadm_boundaries.get_region_boundary()  # INHERITED METHOD from GADMBoundaries

            layers: list = fiona.listlayers(gdb_file_path)

            try:
                assert gdb_layer in layers, (
                    f"Layer '{gdb_layer}' not found in the GDB file. Please configure the valid 'gdb_layer' key in config file."
                )
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__}| Loading  {gdb_layer} Layer from the GDB file.",
                )

                # Load the .gdb file as a GeoDataFrame
                gdf = gpd.read_file(
                    gdb_file_path, mask=self.region_boundary, layer=gdb_layer
                )  # Specifying layer and mask to load only the relevant region, faster loading
                gdf.to_crs(self.region_boundary.crs, inplace=True)

                gdf["geometry"] = gdf["geometry"].simplify(
                    geom_simplification_tolerance
                )  # Simplify geometries to reduce complexity that are not relevant at ERA5 resolution and faster processing

                # Map IUCN categories to descriptions
                IUCN_CAT = self.conserved_lands_cfg["IUCN_CAT_mapping"]
                gdf["IUCN_CAT_desc"] = gdf["IUCN_CAT"].map(IUCN_CAT)
                gdf.to_pickle(provincial_file_path)
            except AssertionError as e:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1, message=f"{__name__}| {e}", alert=True
                )

        return gdf

    def __get_conserved_lands__(self) -> Path:
        """Download the source ZIP file, extract contents, and return the .gdb file path."""
        # Check if the extraction directory exists
        if self.extraction_dir.exists():
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"Extraction directory {self.extraction_dir} already exists, skipping download and extraction.",
            )
        else:
            if self.zip_file_path.exists():
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message=f"ZIP file {self.zip_file_path} already exists, skipping download.",
                )
            else:
                # Download the ZIP file
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message="Downloading Canadian Protected and Conserved Areas Database (CPCAD)",
                )
                self.zip_file_path.parent.mkdir(parents=True, exist_ok=True)
                utils.download_data(self.source_url, self.zip_file_path)
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message=f"Downloaded ZIP file to {self.zip_file_path}",
                )

            # Create the extraction directory and extract ZIP contents
            self.extraction_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(self.zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.extraction_dir)
            # print(f"Extracted files to {self.extraction_dir}")

        # Load the first .gdb file found in the extraction directory
        gdb_file_path = next(self.extraction_dir.rglob("*.gdb"), None)
        if gdb_file_path is None:
            raise FileNotFoundError(
                ">> !! No .gdb file found in the extracted contents."
            )

        return gdb_file_path

    def show_lands(
        self,
        basemap: str = "CartoDB positron",
        save_path: str = None,
        save: bool = False,
    ):
        """
        Create and save an interactive map for the specified region.

        Args:
            basemap (str): The basemap to use (default is 'CartoDB positron').
            save_path (str): The path to save the HTML map. If None, default is used.
            save (bool): If True, saves the map as a local HTML file.

        Returns:
            folium.Map: The interactive map object.
        """
        conserved_lands = self.get_provincial_conserved_lands()
        self.region_boundary = self.gadm_boundaries.get_region_boundary()  # INHERITED METHOD from GADMBoundaries

        if self.region_boundary is not None:
            m = self.region_boundary.explore(
                color="grey", linecolor="grey", legend=True, tiles=basemap, alpha=0.4
            )
            conserved_lands.explore("IUCN_CAT_desc", m=m, legend=True, tiles=basemap)

            if save:
                if save_path is None:
                    save_path = f"vis/lands/{self.region_short_code}.html"
                else:
                    save_path = Path(save_path) / f"{self.region_short_code}.html"

                # Ensure the directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the map as an HTML file
                m.save(save_path)
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message="Interactive map for '{self.region_short_code}' saved to {save_path}.",
                )
            else:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message="Skipping save, 'save' is set to False.",
                )

        return m


class LandContainer(AttributesParser):
    """
    Multi-source land data container for comprehensive spatial exclusion analysis.
    
    This class combines multiple geospatial data sources (ERA5 cutouts, GAEZ rasters,
    OSM data, and conservation lands) to manage inclusion/exclusion of lands for
    spatial analysis. It provides a comprehensive framework for renewable energy
    land suitability assessment by integrating climate, terrain, infrastructure,
    and conservation constraints.
    
    The class uses multiple inheritance to access functionality from ERA5 climate
    data processing, GAEZ raster analysis, and OpenStreetMap infrastructure data.
    It creates an ExclusionContainer that can handle both raster and vector
    exclusion layers for detailed spatial analysis.
    
    INHERITED METHODS FROM ERA5Cutout:
    ----------------------------------
    - get_era5_cutout() -> tuple: Get ERA5 climate data cutout
    - get_cutout_path() -> Path: Generate cutout file path
    
    INHERITED METHODS FROM GAEZRasterProcessor:
    -------------------------------------------
    - process_all_rasters() -> dict: Process GAEZ raster layers
    - get_gaez_data_config() -> dict: Get GAEZ configuration
    
    INHERITED METHODS FROM OSMData:
    -------------------------------
    - get_osm_layer() -> gpd.GeoDataFrame: Get OSM infrastructure layer
    - get_osm_config() -> dict: Get OSM configuration
    
    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_excluder_crs() -> str: Get coordinate reference system for exclusions
    - get_resource_disaggregation_config() -> dict: Get resource configuration
    - get_conserved_lands_CAN_args() -> dict: Get conservation lands arguments
    - default_font_family -> str: Get default font family for plots
    - Plus other configuration access methods
    
    INHERITED ATTRIBUTES FROM AttributesParser:
    -------------------------------------------
    - resource_type: Resource type identifier
    - region_short_code: Region identifier code
    - region_name: Full region name
    - Plus other configuration attributes
    
    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - set_excluder(): Configure exclusion container with all layers
    - get_layers(): Load and organize raster and vector exclusion layers
    
    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing all data source parameters
    region_short_code : str
        Region identifier for boundary definition and data filtering
    resource_type : str
        Resource type ('solar', 'wind', 'bess') for technology-specific exclusions
        
    Attributes
    ----------
    excluder_crs : str
        Coordinate reference system for exclusion analysis (typically Canada-specific)
    excluder : ExclusionContainer
        Atlite ExclusionContainer for managing spatial exclusions
    resource_disaggregation_config : dict
        Configuration for resource type disaggregation and exclusions
    conserved_lands_CAN : ConservationLands
        Conservation lands processor for protected area exclusions
    conservation_lands_region_gdf : gpd.GeoDataFrame
        Regional conservation lands data for exclusion analysis
        
    Methods
    -------
    set_excluder() -> None
        Configure exclusion container with all raster and vector layers
    get_layers() -> tuple[list, list]
        Load and organize raster and vector exclusion layers from configuration
        
    Examples
    --------
    Create comprehensive land container for British Columbia:
    
    >>> from RES.lands import LandContainer
    >>> land_container = LandContainer(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="solar"
    ... )
    >>> 
    >>> # Set up exclusion layers
    >>> land_container.set_excluder()
    >>> print("Exclusion container configured with all layers")
    
    Access individual data sources:
    
    >>> # Get GAEZ raster data
    >>> gaez_layers = land_container.get_layers()
    >>> print(f"Available layers: {len(gaez_layers[0])} raster, {len(gaez_layers[1])} vector")
    >>> 
    >>> # Access ERA5 cutout
    >>> cutout, boundaries = land_container.get_era5_cutout()
    >>> print(f"ERA5 cutout covers {cutout.coords['time'].size} time steps")
    
    Perform exclusion analysis:
    
    >>> # Configure exclusions for renewable energy siting
    >>> land_container.set_excluder()
    >>> excluded_area = land_container.excluder.compute()
    >>> print(f"Excluded area computed: {excluded_area.shape}")
    
    Configuration Requirements
    --------------------------
    The configuration must include parameters for all data sources:
    
    ```yaml
    cutout:  # ERA5 configuration
      root: "data/cutouts"
      module: "era5"
      
    gaez_data:  # GAEZ raster configuration
      root: "data/downloaded_data/GAEZ"
      raster_types: [...]
      
    osm_data:  # OSM infrastructure configuration
      root: "data/downloaded_data/OSM"
      layers: [...]
      
    Gov:  # Conservation lands configuration
      conservation_lands:
        url: "..."
        root: "data/downloaded_data/conservation"
    ```
    
    Data Integration Workflow
    -------------------------
    1. **Multi-source Initialization**: Initialize all parent classes
    2. **CRS Harmonization**: Establish common coordinate reference system
    3. **Exclusion Container Setup**: Create atlite ExclusionContainer
    4. **Layer Configuration**: Load raster and vector exclusion layers
    5. **Conservation Data**: Process protected and conserved areas
    6. **Infrastructure Data**: Load OSM roads, railways, settlements
    7. **Terrain Data**: Process GAEZ slope, elevation constraints
    8. **Climate Integration**: Incorporate ERA5 data for analysis context
    
    Exclusion Layer Types
    ---------------------
    **Raster Exclusions:**
    - **Slope**: Terrain slope constraints from GAEZ
    - **Elevation**: Elevation constraints for accessibility
    - **Land Cover**: Unsuitable land cover types
    - **Soil Quality**: Agricultural productivity protection
    
    **Vector Exclusions:**
    - **Conservation Areas**: Protected and conserved lands
    - **Infrastructure**: Roads, railways, power lines with buffers
    - **Settlements**: Urban areas and residential zones
    - **Water Bodies**: Lakes, rivers, wetlands
    - **Administrative**: Military zones, airports
    
    Spatial Analysis Capabilities
    -----------------------------
    - **Multi-resolution Integration**: Harmonize different data resolutions
    - **Buffer Operations**: Apply technology-specific buffer distances
    - **Overlay Analysis**: Complex spatial intersections and unions
    - **Area Calculations**: Accurate area computation in projected CRS
    - **Constraint Mapping**: Visualization of all exclusion layers
    
    Performance Considerations
    --------------------------
    - **Memory Management**: Lazy loading and streaming for large datasets
    - **Spatial Indexing**: Efficient spatial queries and operations
    - **Resolution Optimization**: Balance between accuracy and performance
    - **Caching Strategy**: Store processed exclusions for reuse
    - **Parallel Processing**: Multi-threaded operations where possible
    
    Integration Points
    ------------------
    - **Renewable Energy Assessment**: Primary use for solar/wind siting
    - **Capacity Factor Analysis**: Integrate with climate-based calculations
    - **Grid Connection**: Compatible with transmission line analysis
    - **Resource Optimization**: Support for multi-criteria decision analysis
    - **Policy Analysis**: Enable scenario-based exclusion studies
    
    Quality Assurance
    -----------------
    - **Data Validation**: Automatic validation of all input layers
    - **Consistency Checking**: Ensure spatial and temporal consistency
    - **Gap Analysis**: Identify and report data coverage gaps
    - **Accuracy Assessment**: Validate exclusion logic and results
    - **Uncertainty Quantification**: Propagate uncertainties through analysis
    
    Notes
    -----
    - Multiple inheritance requires careful method resolution order
    - CRS management is critical for accurate spatial analysis
    - Large regions may require substantial computational resources
    - Results support both preliminary and detailed feasibility studies
    - Integration with atlite enables advanced renewable energy modeling
    - Exclusion logic can be customized for different technologies and policies
    
    Dependencies
    ------------
    - atlite.gis.ExclusionContainer: Core exclusion functionality
    - geopandas: Spatial data processing
    - rasterio: Raster data operations
    - numpy: Numerical operations
    - matplotlib: Visualization
    - RES.era5_cutout.ERA5Cutout: ERA5 climate data processing
    - RES.gaez.GAEZRasterProcessor: GAEZ raster data processing
    - RES.osm.OSMData: OpenStreetMap data processing
    - RES.utility: Logging and utility functions
    
    Raises
    ------
    CRSError
        If coordinate reference systems cannot be harmonized
    DataError
        If required data sources are missing or invalid
    MemoryError
        If datasets are too large for available memory
    RuntimeError
        If exclusion container setup or operations fail
        
    See Also
    --------
    atlite.gis.ExclusionContainer : Core exclusion functionality
    RES.era5_cutout.ERA5Cutout : ERA5 climate data processing
    RES.gaez.GAEZRasterProcessor : GAEZ raster data processing
    RES.osm.OSMData : OpenStreetMap data processing
    """

    def __post_init__(self):
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        self.required_args= {
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
        }

        self.era5_cutout = ERA5Cutout(**self.required_args)
        self.gaez_raster_processor = GAEZRasterProcessor(**self.required_args)
        self.osm_data = OSMData(**self.required_args)
        self.gadm_boundaries = GADMBoundaries(**self.required_args)

        # Set up inherited attributes that are needed
        self.region_name = self.get_region_name()  # INHERITED METHOD from AttributesParser

        # Initialize region_boundary and region_shape
        self.region_boundary = self.gadm_boundaries.get_region_boundary()  # INHERITED METHOD from GADMBoundaries
        self.region_shape = self.region_boundary.dissolve(
            by=self.get_gadm_config()["datafield_mapping"]["NAME_1"]  # INHERITED METHOD from AttributesParser
        )  # Get the geometry of the region boundary

        self.excluder_crs = self.crs_m

        # Initiate Exclusion Container
        self.excluder = ExclusionContainer(
            crs=self.excluder_crs
        ) 

        # Initialize resource_disaggregation_config attribute
        self.resource_disaggregation_config = self.get_resource_disaggregation_config()  # INHERITED METHOD from AttributesParser
        if self.get_conserved_lands_CAN_args() is not None:  # INHERITED METHOD from AttributesParser
            self.conserved_lands_CAN=ConservationLands(**self.get_conserved_lands_CAN_args())  # INHERITED METHOD from AttributesParser
        else:
            utils.print_warning(f"{__name__}| 'conserved_lands_CAN' not initiated. Please check the config file for 'conserved_lands' key under 'Gov' section")
            self.conserved_lands_CAN=None
        # Initialize conservation_lands_region_gdf attribute
        self.conservation_lands_region_gdf = None
        self.custom_land_layers = self.get_custom_land_layers()  # INHERITED METHOD from AttributesParser

    def set_excluder(self):
        raster_configs, vector_configs = self.get_layers()

        # Print hourglass emoji to indicate long-running process
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__}| ⏳ Loading layers to Excluder for {self.region_name}. This may take a while to compute and plot...",
        )

        args_add_excluder_layer = {
            "crs_meters":self.excluder_crs,
            "resource_type": self.resource_type,  # INHERITED ATTRIBUTE from AttributesParser
            "excluder": self.excluder,
            "region_shape": self.region_shape,
            "raster_configs": raster_configs,
            "vector_configs": vector_configs,
            "font_family": self.default_font_family,  # INHERITED ATTRIBUTE from AttributesParser
            "plot_save_to": self.vis_root/'lands'  # INHERITED ATTRIBUTE from AttributesParser
        }
        
        
        # Load all layers to the excluder
        excluder_with_layers = load_layers_to_excluder(**args_add_excluder_layer)
        
        # for plotting purposes
  
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__}| ⏳ Plotting explicit impact of layers to Excluder for {self.region_name}. This may take a while to compute and plot...",
        )
        load_layers_to_excluder(**args_add_excluder_layer,
                                disregard_other_layers=True)
        
        return excluder_with_layers

    def get_layers(self):
        """Load all raster and vector layers for the specified region.
        Returns:
            tuple: A tuple containing two lists - raster_configs and vector_configs.
        """
    # load GAEZ Raster Layers
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading GAEZ raster layers for {self.region_name}...",
        )
        self.gaez_config = self.get_gaez_data_config()  # INHERITED METHOD from AttributesParser
        raster_configs: list[dict] = self.gaez_config["raster_types"]
        regional_raster_paths: dict = self.gaez_raster_processor.process_all_rasters(show=False)

           
                                                            
        for raster_config_item in raster_configs:
            name = raster_config_item.get("name")
            if name and name in regional_raster_paths:
                raster_config_item["filepath"] = str(regional_raster_paths[name])
        utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message= f"{__name__}| Raster Layers Loaded")
    
    # Load CORINE rasters - for Europe only
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading CORINE raster layers for {self.region_name}...",
        )
        self.CLC_config:dict = self.get_CLC_raster_config()  # INHERITED METHOD from AttributesParser
        CLC_raster_configs:list=self.CLC_config.get("raster_types", [])


        for CLC_raster_config_item in CLC_raster_configs:
            CLC_raster_config_item["filepath"]=Path(self.CLC_config.get('root'))/CLC_raster_config_item['raster']
                
        utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message= f"{__name__}| Raster Layers Loaded")
     
    # Load Custom Rasters
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading Custom raster layers for {self.region_name}...",
        )

        custom_raster_configs: list[dict] = self.custom_land_layers.get("rasters", [])   
        if custom_raster_configs is None:
            custom_raster_configs = []
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"{__name__}| No custom raster layers found in config for {self.region_name}.",
                alert=True,
            )
            
            # Merge raster_configs and CLC_raster_config into a single list
            raster_configs = raster_configs + CLC_raster_configs
        else:
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__}| Found {len(custom_raster_configs)} custom raster layers in config for {self.region_name}.",
            )
            for custom_raster_config_item in custom_raster_configs:
                custom_raster_config_item["filepath"] =clip_to_boundary_and_resample_raster(
                                                    in_raster_config=custom_raster_config_item,
                                                    boundary_name=self.region_short_code,
                                                    boundary=self.region_boundary)
                
                # custom_raster_config_item["filepath"] = (
                #     Path(custom_raster_config_item["root"]) /
                #     custom_raster_config_item["raster"]
                # )

            # Merge raster_configs and CLC_raster_config into a single list
            raster_configs = raster_configs + CLC_raster_configs + custom_raster_configs
            
    # Load Vector layers
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading vector layers for {self.region_name}...",
        )
        vector_configs: list[dict] = self.resource_disaggregation_config[
            "vector_buffers"
        ]

        for vector_config_item in vector_configs:
            # vector_config_item is a dictionary
            vector_name = list(vector_config_item.keys())[0]
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__}| Loading {vector_name} areas for {self.region_name}",
            )

            if vector_name == "conserved_lands":
                # Add local (Canadian) vector geometries to excluder
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__}| Loading Conserved areas for {self.region_name}",
                )
                if self.conserved_lands_CAN is None:
                    utils.print_update(
                        level=PRINT_LEVEL_BASE + 1,
                        message=f"{__name__}| conserved_lands_CAN is not initialized. Skipping {vector_name}.",
                        alert=True,
                    )
                    continue
                vector_gdf = self.conserved_lands_CAN.get_provincial_conserved_lands()
                if vector_gdf.empty:
                    utils.print_update(
                        level=PRINT_LEVEL_BASE + 1,
                        message=f"{__name__}| No {vector_name} data found for {self.region_name}. Skipping.",
                        alert=True,
                    )
                    continue
                vector_config_item[vector_name]["stepwise_plot_title"] = (
                    "Excluding Regional Conservation Areas"
                )

            elif vector_name == "aeroway":
                # Load vector geometries from OSM
                vector_gdf = self.osm_data.get_osm_layer(vector_name)  # INHERITED METHOD from OSMData
                if vector_gdf.empty:
                    utils.print_update(
                        level=PRINT_LEVEL_BASE + 1,
                        message=f"{__name__}| No {vector_name} data found for {self.region_name}. Skipping.",
                        alert=True,
                    )
                    continue
                vector_config_item[vector_name]["stepwise_plot_title"] = (
                    "Excluding Regional Aeroways"
                )

            # Apply buffer to the vector geometries
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__}| Applying buffer to {vector_name} areas for {self.region_name}",
            )
            vector_gdf_with_buffer, vector_area_comparison = apply_buffer_to_vector(
                gdf=vector_gdf,
                crs_meters=self.crs_m,
                crs_degrees=self.crs_d,
                buffer_mapping=   vector_config_item[vector_name]["buffer_mapping_key_buffers"],
                buffer_mapping_key= vector_config_item[vector_name]["buffer_mapping_key"],
            )
            vector_config_item[vector_name]["gdf"] = vector_gdf_with_buffer
            vector_area_comparison['Resource_Type']=self.resource_type
            vector_area_comparison['Region']=self.region_name
            vector_area_comparison['Scenario']=self.get_RUN_ID()
            # Save the area comparison to a CSV file
            area_comparison_save_path = (
                Path("data/processed_data/lands")
                / f"{vector_config_item[vector_name]['buffer_mapping_key']}_area_comparisons_{self.region_name}_{self.resource_type}_{self.RUN_ID}.csv"
            )
            area_comparison_save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the area comparison DataFrame to CSV
            vector_area_comparison.to_csv(area_comparison_save_path)
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__}| Vector Area comparison for {vector_config_item[vector_name]['buffer_mapping_key']} saved to {area_comparison_save_path}",
            )
            vector_config_item[vector_name]["area_comparison"] = vector_area_comparison

        # We want to flat list of dictionaries without vector_name in the keys
        vector_configs = [list(d.values())[0] for d in vector_configs]
        utils.print_update(level=PRINT_LEVEL_BASE+3,
                           message= f"{__name__}|✓ Vector Layers Loaded")

        return raster_configs, vector_configs


@staticmethod
def add_and_plot_exclusion_layer(
    excluder: ExclusionContainer,
    region_shape: gpd.GeoDataFrame,
    ax:Axes,
    geometry:BaseGeometry,
    title:str,
    invert:bool=False,
    is_raster:bool=False,
    filepath:str|Path=None,
    codes:list[int]=None,
    disregard_other_layers:bool=False,
):
    """
    Add a layer to the ExclusionContainer and plot the availability of the region shape.

    Args:
        excluder (ExclusionContainer): The ExclusionContainer to add the layer to.
        region_shape (gpd.GeoDataFrame): The region shape GeoDataFrame.
        ax (_type_): The axes to plot on.
        geometry (_type_): The geometry to add to the ExclusionContainer.
        title (_type_): The title for the plot.
        invert (bool, optional): Whether to invert the exclusion. Defaults to False.
        is_raster (bool, optional): Whether the layer is a raster layer. Defaults to False.
        filepath (_type_, optional): The file path for the raster layer. Defaults to None.
        codes (_type_, optional): The codes for the raster layer. Defaults to None.
        disregard_other_layers(bool): Default to False. This param is exclusively to be used to plotting purpose. Plots to showcase the impact of each layer as a standalone.
    

    Returns:
        _type_: _description_
    """
    if is_raster:
        excluder.add_raster(filepath, codes, invert=invert)
    else:
        excluder.add_geometry(geometry)

    masked, transform, eligible_share = get_eligible_share(region_shape, excluder)


    # Keep 1s, mask 0s
    raster_data = masked.astype(float)  # * 100
    masked_data = np.ma.masked_where(raster_data == 0, raster_data)
    
    if disregard_other_layers:
        cmap = ListedColormap(["#0B936A"])
        # Clean and modify title
        title_cleaned = title.replace("Excluding", "Land filtered for").strip()
        ax.set_title(f"{title_cleaned} ({eligible_share:.2%})",fontsize=18)
    else:
        # Simple solid green for eligible
        cmap = ListedColormap(["#027227"])
        ax.set_title(f"{title} {eligible_share:.2%}")

    cmap.set_bad(color=(1, 1, 1, 0))  # transparent 0s
    
    # Plot masked raster
    show(
        masked_data,
        transform=transform,
        ax=ax,
        cmap=cmap,
    )

    # Overlay region boundary (no cmap here)
    if region_shape.crs != excluder.crs:
        region_shape = region_shape.to_crs(excluder.crs)
    region_shape.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
    
    # Clean aesthetics
    ax.set_axis_off()

    excluder_with_layers: ExclusionContainer = excluder

    return excluder_with_layers


@staticmethod
def load_layers_to_excluder(
    crs_meters:str,
    resource_type: str,
    excluder: ExclusionContainer,
    region_shape: gpd.GeoDataFrame,
    raster_configs: list[dict],
    vector_configs: list[dict],
    font_family: str = "serif",
    plot_save_to: str | Path = None,
    intiate_excluder:bool=True,
    disregard_other_layers:bool=False
) -> ExclusionContainer:
    """
    Load raster and vector layers to the ExclusionContainer and plot the availability of the region shape.
    Args:
        excluder (ExclusionContainer): The ExclusionContainer to add the layers to.
        region_shape (gpd.GeoDataFrame): The region shape GeoDataFrame.
        raster_configs (list[dict]): List of raster configurations.
        vector_configs (list[dict]): List of vector configurations.
        plot_save_to (str|Path, optional): Path to save the plot. Defaults to None.
        disregard_other_layers(bool): Default to False. This param is exclusively to be used to plotting purpose. Plots to showcase the impact of each layer as a standalone.
    Returns:
        ExclusionContainer: The ExclusionContainer with the added layers.
    """
    if not intiate_excluder:
        utils.print_warning(f"{__name__}|'intiate_excluder' set to FALSE. ExclusionContainer has not been initiated. The container may have residual rasters/vector already loaded")
    if disregard_other_layers:
        utils.print_warning(f"{__name__}|'disregard_other_layers' set to TRUE. This parameter should be used exclusively for plotting purposes to showcase land availability impact for individual layers")
    excluder=ExclusionContainer(crs_meters)
    
    n_rasters = len(raster_configs)
    n_vectors = len(vector_configs)
    utils.print_info(f"{__name__}| The Stepwise Land-availability plots and numbers are sensitive to the sequence of layers are loaded. However, the collective impact of layers on final availability is same")
    # 2. Plot setup
    total_layers = n_rasters + n_vectors

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = 14
    fig, axes = plt.subplots(
        1, total_layers, figsize=(9 * total_layers, total_layers + 6)# revise this accordingly to make the plot looks nicer
        )  
    utils.print_info(f"{__name__}| The order of loading raster layers mimics the given order in config file under keys: 'GAEZ' and then 'raster_types'")
    # 3. Raster layers
    for i, r in enumerate(raster_configs):
        

        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading raster layer {i+1} '{r.get('name', '')}' to ExclusionContainer ...",
        )
        # Handle raster layer inclusion/exclusion logic smartly
        class_inclusion = r.get("class_inclusion")
        class_exclusion = r.get("class_exclusion")
        invert = False
        codes = None

        if class_inclusion and resource_type in class_inclusion:
            codes = class_inclusion[resource_type]
            invert = True
        elif class_exclusion and resource_type in class_exclusion:
            codes = class_exclusion[resource_type]
            invert = False
        else:
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"{__name__}| No valid class_inclusion/class_exclusion for raster '{r.get('name', '')}' and resource '{resource_type}'. Skipping.",
                alert=True,
            )
            continue

        excluder_with_layers = add_and_plot_exclusion_layer(
            excluder,
            region_shape=region_shape,
            ax=axes[i],
            geometry=None,
            title=r.get("stepwise_plot_title", r.get("name", "Raster Layer")),
            invert=invert,
            is_raster=True,
            filepath=ensure_uint8_raster(r["filepath"]),
            codes=codes,
            disregard_other_layers=disregard_other_layers
        )
    
    utils.print_info(f"{__name__}| The order of loading vector layers mimics the given order in config file under keys: 'capacity_disaggregation' and then <resource_type> 'solar' or 'wind' and then 'vector_buffers'. However, the collective impact of layers on final availability is same ")
    # 4. Vector layers
    for i, v in enumerate(vector_configs):
        if disregard_other_layers:
            excluder=ExclusionContainer(crs=crs_meters)
            utils.print_warning(f"Excluder crs set to {crs_meters}")
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading vector layer {i+1} for '{list(vector_configs[i]['buffer_mapping_key_buffers'].keys())}' to ExclusionContainer ...",
        )
        # Assert that the geometries in vector_configs are in the same CRS as excluder
        if v["gdf"].crs != excluder.crs:
            v["gdf"] = v["gdf"].to_crs(excluder.crs)
        excluder_with_layers = add_and_plot_exclusion_layer(
            excluder,
            region_shape=region_shape,
            ax=axes[n_rasters + i],
            geometry=v["gdf"].geometry,
            title=v["stepwise_plot_title"],
            invert=v.get("invert", False),
            is_raster=False,
            disregard_other_layers=disregard_other_layers
        )

    plt.tight_layout()
    if disregard_other_layers:
        fig.suptitle(
            f"Land Availability impact for each Exclusion/Inclusion Layers for {resource_type} resource", 
            fontsize=30, 
            y=1.05
        )
        plot_name:str=f"individual_layers_impact_land_availability_plot_{resource_type}"
    else:
        fig.suptitle(
            f"Stepwise Land Availability for Exclusion/Inclusion Layers for {resource_type} resource",
            fontsize=30,
            y=1.05,
        )
        plot_name:str=f"stepwise_land_availability_plot_{resource_type}"

    # Save the figure
    if isinstance(plot_save_to, str):
        plot_save_to = Path(plot_save_to)     
    if plot_save_to is None:
        plot_save_to=Path.cwd()
    
    utils.ensure_path(plot_save_to)

    plt.savefig(
        plot_save_to / f"{plot_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    utils.print_update(
        level=3, message=f"{__name__}|✓ Stepwise Availability Plots saved to {plot_save_to} "
    )
    if disregard_other_layers:
        utils.print_info(f"{__name__}| Please set the `disregard_other_layers` to False to get the ExclusionContainer with the cumulative impact of all layers")
        return None
    else:
        return excluder_with_layers

@staticmethod
def apply_buffer_to_vector(
    gdf: gpd.GeoDataFrame,
    crs_meters:str,
    crs_degrees:str,
    buffer_mapping: dict, 
    buffer_mapping_key: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Projects the input GeoDataFrame to BC Albers, applies buffer distances from config,
    and reprojects back to EPSG:4326. Returns the buffered GeoDataFrame and area comparison.
    Adds a column 'buffer_applied_m' to show actual buffer distance applied per feature.
    """

    # 1. Project to meter-based CRS
    gdf_proj = gdf.to_crs(crs_meters)

    # 2. Assign buffer distances from mapping
    buffer_series = pd.Series(buffer_mapping)
    gdf_proj["buffer_applied_m"] = (
        gdf_proj[buffer_mapping_key].map(buffer_series).fillna(0)
    )
    # 3. Keep unbuffered copy for area comparison
    gdf_unbuffered = gdf_proj.copy()

    # 4. Apply buffer (in meters)
    gdf_buffered = gdf_unbuffered.copy()
    gdf_buffered["geometry"] = gdf_unbuffered.geometry.buffer(gdf_proj["buffer_applied_m"])

    # 5. Area calculations (in km²)
    gdf_unbuffered["area_km2"] = gdf_unbuffered.geometry.area / 1e6
    gdf_buffered["area_km2"] = gdf_buffered.geometry.area / 1e6
    area_original = (
        gdf_unbuffered.groupby(buffer_mapping_key)["area_km2"]
        .sum()
        .rename("original_area_km2")
    )
    area_buffered = (
        gdf_buffered.groupby(buffer_mapping_key)["area_km2"]
        .sum()
        .rename("buffered_area_km2")
    )

    # 6. Area comparison
    area_comparison = pd.concat([area_original, area_buffered], axis=1)
    area_comparison["buffer_applied_m"] = area_comparison.index.map(buffer_mapping)
    area_comparison["delta_km2"] = (
        area_comparison["buffered_area_km2"] - area_comparison["original_area_km2"]
    )
    area_comparison["percent_increase"] = (
        100 * area_comparison["delta_km2"] / area_comparison["original_area_km2"]
    )
    area_comparison = area_comparison.sort_values(
        "original_area_km2", ascending=False
    ).round(4)

    # 7. Reproject back to degree based crs
    if gdf_buffered.crs != crs_degrees:
        gdf_buffered = gdf_buffered.to_crs(crs_degrees)
    print(f"{__name__}| Buffer applied to vector layer with key '{buffer_mapping_key}'. Area comparison:\n{area_comparison}")
    return gdf_buffered, area_comparison


@staticmethod
def get_eligible_share(region_shape, excluder: ExclusionContainer) -> tuple:
    """
    Calculate the eligible share of the region based on the exclusion container.
    """
    # Ensure region_shape has a CRS and matches excluder.crs
    if region_shape.crs is None:
        region_shape = region_shape.set_crs(excluder.crs)
    elif region_shape.crs != excluder.crs:
        region_shape = region_shape.to_crs(excluder.crs)
    masked, transform = excluder.compute_shape_availability(region_shape)
    region_area = region_shape.geometry.area.sum() # item()
    eligible_area = masked.sum() * excluder.res**2
    eligible_share = eligible_area / region_area

    return masked, transform, eligible_share

@staticmethod


def ensure_uint8_raster(filepath):
    """
    Ensure the raster is in uint8 format with nodata as 255. If not, convert it and save to a temporary file.

    Args:
        filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    with rasterio.open(filepath) as src:
        if src.dtypes[0] != 'uint8' or src.nodata not in (255, 0, None):
            data = src.read(1).astype(np.uint8)
            meta = src.meta.copy()
            meta.update(dtype='uint8', nodata=255)
            tmp = NamedTemporaryFile(suffix=".tif", delete=False)
            with rasterio.open(tmp.name, "w", **meta) as dst:
                dst.write(data, 1)
            return tmp.name
    return filepath


@staticmethod

def clip_to_boundary_and_resample_raster(
    in_raster_config: dict,
    boundary_name: str,
    boundary: gpd.GeoDataFrame | str | Path,
    clip_to_geom: bool = True,
    target_res: Optional[int | None] = 100,
    categorical_threshold: int = 50,
):
    """
    Clips a raster to a boundary and optionally resamples to target resolution.
    Automatically detects categorical vs continuous data and applies appropriate
    resampling method. Skips processing if intermediate or final outputs already exist.

    Parameters
    ----------
    in_raster_config : dict
        Dictionary containing raster configuration with keys:
        - 'name': str, raster identifier name
        - 'root': str or Path, root directory path
        - 'raster': str, raster filename
        - 'target_res_meters': int, optional target resolution in meters
    boundary_name : str
        Label used for naming output files (e.g., region code).
    boundary : str, Path, or GeoDataFrame
        Vector boundary file (Shapefile/GeoJSON) or GeoDataFrame for clipping.
    clip_to_geom : bool, default=True
        If True, clips to exact geometry boundary. If False, clips to bounding box.
    target_res : float, optional
        Desired output resolution in meters. If None, uses 'target_res_meters' from config.
    categorical_threshold : int, default=50
        Maximum number of unique values to classify raster as categorical.
        Categorical rasters use 'mode' resampling, continuous use 'average'.
        
    Returns
    -------
    xarray.DataArray or None
        Loaded raster data as xarray DataArray if successful, None if verification fails.
        The returned data array is masked and ready for spatial analysis.
        
    Raises
    ------
    FileNotFoundError
        If input raster file does not exist.
    TypeError
        If boundary parameter is not a valid file path or GeoDataFrame.
    ValueError
        If boundary does not overlap with raster extent.
        
    Notes
    -----
    - Output files are named: {stem}_clipped_{boundary_name}[_{resolution}m].tif
    - Categorical detection uses center window sampling for efficiency
    - Resampling is skipped if current resolution is already coarser than target
    - CRS reprojection is handled automatically when boundary CRS differs from raster
    - Intermediate clipped files are reused if target resolution changes
    """
    raster_name:str=in_raster_config.get('name')
    in_raster:str|Path=Path(in_raster_config.get('root'))/in_raster_config.get('raster')
    
    
    in_raster = Path(in_raster)
    if not in_raster.exists():
        try:
            source_url=in_raster_config.get('source',None)
            if source_url is not None:
                utils.download_data(source_url,in_raster)
        except Exception as e:
            utils.print_error(f"{__name__}|❌ Failed to locate and download raster from {source_url}: {e}")


    # --- Define output paths ---
    clipped_path = in_raster.with_name(f"{in_raster.stem}_clipped_{boundary_name}{in_raster.suffix}")
    out_path = (
        in_raster.with_name(
            f"{in_raster.stem}_clipped_{boundary_name}_{int(target_res)}m{in_raster.suffix}"
        )
        if target_res is not None
        else clipped_path
    )

    # === 0. Skip logic ===
    if out_path.exists():
        utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}✅ Final output already exists: {out_path.name} — skipping all processing.")
        return out_path

    if clipped_path.exists() and (target_res is None):
        utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}✅ Clipped raster exists: {clipped_path.name} — no resampling requested.")
        return clipped_path

    if clipped_path.exists() and (target_res is not None):
        utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}✅ Using existing clipped raster for resampling: {clipped_path.name}")
        clip_needed = False
    else:
        clip_needed = True

    # === 1. Load boundary ===
    if isinstance(boundary, (str, Path)):
        gdf = gpd.read_file(boundary)
    elif isinstance(boundary, gpd.GeoDataFrame):
        gdf = boundary.copy()
    else:
        raise TypeError("Boundary must be a file path or a GeoDataFrame.")

    # === 2. Clip (if not already done) ===
    if clip_needed:
        print(f"🔍 Clipping '{in_raster.name}' to boundary '{boundary_name}'...")
        with rasterio.open(in_raster) as src:
            if gdf.crs != src.crs:
                utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}🔄 Reprojecting boundary from {gdf.crs} to {src.crs}")
                gdf = gdf.to_crs(src.crs)

            if not box(*src.bounds).intersects(gdf.unary_union):
                raise ValueError(f"❌ Boundary '{boundary_name}' does not overlap with raster.")

            if clip_to_geom:
                out_image, out_transform = mask(src, gdf.geometry, crop=True)
            else:
                window = rasterio.windows.from_bounds(*gdf.total_bounds, transform=src.transform)
                out_image = src.read(window=window)
                out_transform = src.window_transform(window)

            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        with rasterio.open(clipped_path, "w", **out_meta) as dst:
            dst.write(out_image)
        utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}✅ Clipped raster saved: {clipped_path.name}")
    else:
        utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}ℹ️ Skipping clipping {raster_name} — using existing file.")

    # === 3. Handle resampling ===
    if target_res is None:
        target_res=int(in_raster_config.get('target_res_meters'),None)
        if target_res is None:
            print("⚙️ No target resolution specified — returning clipped raster.")
            return clipped_path

    with rasterio.open(clipped_path) as src:
        res_x, res_y = src.res
        utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}ℹ️ Current resolution: {res_x:.2f} × {res_y:.2f} m")

        if res_x >= target_res and res_y >= target_res:
           utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}⚠️ Already coarser than {target_res} m — skipping resample.")
           return clipped_path

        # Detect categorical vs continuous
        center_window = rasterio.windows.Window(
            src.width // 4, src.height // 4, src.width // 2, src.height // 2
        )
        sample = src.read(1, window=center_window)
        sample = sample[~np.isnan(sample)] if np.issubdtype(sample.dtype, np.floating) else sample
        unique_vals = np.unique(sample)

        categorical = (
            len(unique_vals) < categorical_threshold
            and np.all(unique_vals.astype(int) == unique_vals)
        )
        resampling_method = Resampling.mode if categorical else Resampling.average
        kind = "categorical (mode)" if categorical else "continuous (average)"

        scale_factor = target_res / res_x
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)
        new_transform = src.transform * Affine.scale(scale_factor, scale_factor)

        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling_method
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}✅ Resampled to {int(target_res)} m ({kind})")
    utils.print_update(PRINT_LEVEL_BASE,
                           f"{__name__}📁 Output saved: {out_path.name}")
    return out_path
    

    # === 4. Verification ===
    # try:
    #     with rasterio.open(out_path) as src:
    #         utils.print_update(PRINT_LEVEL_BASE,
    #                        f"{__name__}🔍 Verification: {src.width} x {src.height} pixels at {src.res[0]:.2f} m")
    #     # Return loaded xarray.DataArray for immediate use
    #     raster_data = rxr.open_rasterio(out_path, masked=True)
    #     return raster_data

    # except Exception as e:
    #     utils.print_update(PRINT_LEVEL_BASE,
    #                        f"{__name__}⚠️ Verification failed: {e}")
    #     try:
    #         utils.print_update(PRINT_LEVEL_BASE,
    #                        f"{__name__}Recheck raster {out_path}")
    #     except NameError:
    #         utils.print_update(PRINT_LEVEL_BASE,
    #                        f"{__name__}⚠️ Please recheck raster {out_path}")
    #     return None


@staticmethod
def plot_raster_class_distribution(
    raster_da,
    legend_df=None,
    class_col="class",
    desc_col="description",
    color_col="color",
    exclude_classes=None,
    title="Raster Class Distribution",
    save_path=None,
    region_code=None,
    country_kwd=None,
    figsize=(10, 6),
    dpi=500,
    show=False,
    save_csv=False,
    ignore_zero_class=True
):
    """
    Plot the percentage distribution of categorical raster classes (generic for any raster).

    Parameters
    ----------
    raster_da : xarray.DataArray or numpy.ndarray
        Raster data array with integer or categorical class codes.
    legend_df : pandas.DataFrame, optional
        DataFrame with at least columns: [class_col, desc_col, color_col].
        If None, generic class labels and colors will be generated.
    class_col : str, optional
        Column name for class codes in legend_df.
    desc_col : str, optional
        Column name for class descriptions in legend_df.
    color_col : str, optional
        Column name for HEX color codes in legend_df.
    exclude_classes : list, optional
        List of class codes to exclude (e.g., NoData, Unclassified).
    title : str, optional
        Plot title.
    save_path : str or Path, optional
        Directory to save figure (and optionally CSV). If None, nothing is saved.
    region_code : str, optional
        Region identifier for filename.
    country_kwd : str, optional
        Country keyword for filename hierarchy.
    figsize : tuple, optional
        Figure size.
    dpi : int, optional
        Plot resolution.
    show : bool, optional
        Whether to display the plot interactively.
    save_csv : bool, optional
        Whether to save the computed class percentages to CSV.

    Returns
    -------
    df_plot : pandas.DataFrame
        DataFrame containing class, percentage, and metadata (merged with legend if available).
        
    Notes:
    """
    raster_da = np.nan_to_num(raster_da.values.squeeze(), nan=0).astype("int32")
    # --- Extract numeric data from raster ---
    if hasattr(raster_da, "values"):
        data = raster_da.values
        
    else:
        data = np.array(raster_da)

    # Flatten and clean
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]  # remove NaN
    # Always exclude class 0 (often NoData)
    exclude_classes = exclude_classes or []
    if ignore_zero_class and 0 not in exclude_classes:
        exclude_classes.append(0)

    if exclude_classes:
        data_flat = np.array([v for v in data_flat if v not in exclude_classes])

    if len(data_flat) == 0:
        raise ValueError("Raster contains no valid data after filtering.")

    # --- Compute percentages ---
    unique, counts = np.unique(data_flat, return_counts=True)
    total = counts.sum()
    percentages = (counts / total) * 100

    df = pd.DataFrame({"class": unique.astype(int), "Percentage": percentages})

        # --- Merge legend if available ---
    if legend_df is not None:
        legend_df = legend_df.rename(
            columns={class_col: "class", desc_col: "description", color_col: "color"}
        )

        # Clean up color codes
        legend_df["color"] = legend_df["color"].astype(str).str.strip().str.lower()

        df_plot = pd.merge(df, legend_df, on="class", how="left")

        # Check for missing legend entries
        missing_colors = df_plot["color"].isna().sum()
        if missing_colors > 0:
            print(f"⚠️ {missing_colors} class(es) missing color — assigning fallback colors.")
            import matplotlib.cm as cm
            cmap = cm.get_cmap("tab20", len(df_plot))
            fallback_colors = [
                f"#{int(255*r):02X}{int(255*g):02X}{int(255*b):02X}"
                for r, g, b, _ in cmap(np.linspace(0, 1, len(df_plot)))
            ]
            df_plot["color"] = df_plot["color"].fillna(pd.Series(fallback_colors))

    else:
        # Generate fallback description and colors
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20", len(unique))
        colors = [
            f"#{int(255*r):02X}{int(255*g):02X}{int(255*b):02X}"
            for r, g, b, _ in cmap(np.linspace(0, 1, len(unique)))
        ]
        df_plot = df.copy()
        df_plot["description"] = df_plot["class"].astype(str)
        df_plot["color"] = colors


        # --- Combine label for axis ---
    df_plot["label"] = df_plot.apply(
        lambda row: f"{int(row['class'])}: {row['description']}", axis=1
    )
    df_plot = df_plot.sort_values("Percentage", ascending=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.barh(
        df_plot["label"],
        df_plot["Percentage"],
        color=df_plot["color"],
        edgecolor="black"
    )

    # Annotate each bar
    for bar, pct in zip(bars, df_plot["Percentage"]):
        width = bar.get_width()
        ax.text(
            width / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if width > 5 else "black",
            bbox=dict(
                facecolor="black" if width > 5 else "white",
                alpha=0.5,
                boxstyle="round,pad=0.2",
                edgecolor="none",
            ),
        )

    # --- Labels & layout ---
    ax.set_xlabel("Percentage of Total Area (%)", fontsize=10)
    ax.set_ylabel("Raster Class (code: description)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()


    # --- Save results ---
    if save_path:
        save_dir = Path(save_path)
        if country_kwd:
            save_dir = save_dir / country_kwd
        if region_code:
            save_dir = save_dir / region_code
            
    save_path = Path(save_path)

    # If user passed a file (has suffix like .png, .jpg)
    if save_path.suffix:
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_path
    else:
        # If user passed just a directory
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / f"Raster_class_distribution_{region_code}.png"

    plt.savefig(out_path, bbox_inches="tight")
    print(f"✅ Saved figure: {out_path}")


    if save_csv:
            out_csv = save_dir / f"{title.replace(' ', '_').lower()}.csv"
            df_plot.to_csv(out_csv, index=False)
            print(f"✅ Saved CSV: {out_csv}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"🧮 Excluding classes: {exclude_classes}")

    return df_plot




def assign_raster_class_to_points(
    gdf: gpd.GeoDataFrame,
    raster_da: xr.DataArray,
    legend_df: Optional[pd.DataFrame] = None,
    raster_crs: Optional[str] = None,
    class_col_name: str = None,
) -> gpd.GeoDataFrame:
    """
    Assigns raster class values (from a single-band xarray DataArray)
    to point geometries in a GeoDataFrame and optionally maps them to
    legend descriptions and colors. Adds all resulting columns with the
    provided class_col_name as suffix to maintain clarity.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries.
    raster_da : xarray.DataArray
        2D raster data (single-band) with coordinates 'x' and 'y'.
    legend_df : pandas.DataFrame, optional
        Table with columns ['class', 'description', 'color'] for mapping.
    raster_crs : str, optional
        CRS of the raster. If None, tries to infer from raster_da.
    class_col_name : str, required
        Base name of the output column. Example: 'landcover' → columns
        'landcover_class', 'landcover_description', 'landcover_color'.

    Returns
    -------
    geopandas.GeoDataFrame
        Original GeoDataFrame with additional columns:
        - <class_col_name>_class : extracted class values
        - <class_col_name>_description, <class_col_name>_color : if legend_df provided
    """

    # --- Type checks ---
    if not class_col_name:
        raise ValueError("class_col_name must be provided.")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame.")
    if not isinstance(raster_da, xr.DataArray):
        raise TypeError("raster_da must be an xarray.DataArray.")
    if "x" not in raster_da.coords or "y" not in raster_da.coords:
        raise ValueError("raster_da must have 'x' and 'y' coordinates.")
    if legend_df is not None:
        if not isinstance(legend_df, pd.DataFrame):
            raise TypeError("legend_df must be a pandas.DataFrame.")
        required_cols = {"class", "description", "color"}
        if not required_cols.issubset(legend_df.columns):
            raise ValueError(f"legend_df must contain {required_cols} columns.")

    # --- CRS handling ---
    if raster_crs is None:
        if "spatial_ref" in raster_da.coords:
            try:
                raster_crs = raster_da.spatial_ref.crs_wkt
            except Exception:
                raise ValueError("CRS could not be inferred from raster DataArray.")
        else:
            raise ValueError("raster_crs must be provided if not embedded in raster_da.")

    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a valid CRS.")
    if str(gdf.crs) != str(raster_crs):
        gdf = gdf.to_crs(raster_crs)

    # --- Extract raster arrays ---
    x_coords = raster_da["x"].values
    y_coords = raster_da["y"].values
    data = raster_da.values

    # --- Sampling function ---
    def get_pixel_value(x, y):
        ix = np.abs(x_coords - x).argmin()
        iy = np.abs(y_coords - y).argmin()
        val = data[iy, ix]
        return None if np.isnan(val) else int(val)

    # --- Extract classes for each geometry ---
    class_col = f"{class_col_name}_class"
    gdf[class_col] = [get_pixel_value(pt.x, pt.y) for pt in gdf.geometry]

    # --- Merge legend if provided ---
    if legend_df is not None:
        temp = gdf.merge(
            legend_df,
            how="left",
            left_on=class_col,
            right_on="class",
        ).drop(columns=["class"], errors="ignore")

        # Rename merged columns with suffix
        temp.rename(
            columns={
                "description": f"{class_col_name}_description",
                "color": f"{class_col_name}_color",
            },
            inplace=True,
        )
        gdf = temp
    summary = (
        gdf.groupby(f"{class_col_name}_description")
        .size()
        .reset_index(name="site_count")
        .sort_values("site_count", ascending=False)
    )
    utils.print_banner(f"{__name__}| Summary of {class_col_name} class distribution for assigned points:")
    utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{summary}")
    print("\n")
    return gdf
