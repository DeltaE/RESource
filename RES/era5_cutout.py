import os
from pathlib import Path

import atlite

import RES.utility as utils
from RES.AttributesParser import AttributesParser
from RES.boundaries import GADMBoundaries

os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'

print_level_base=3

class ERA5Cutout(AttributesParser):
    """
    ERA5 climate data cutout processor for renewable energy resource assessment.
    
    This class handles the creation and management of ERA5 climate data cutouts
    using the atlite library. It processes climate data within specified regional
    boundaries and temporal ranges to support renewable energy potential analysis.
    ERA5 cutouts provide the foundational climate data for capacity factor
    calculations, resource assessment, and energy yield modeling.
    
    The class integrates with GADM boundaries to ensure that climate data cutouts
    align with administrative regions and provide appropriate spatial coverage
    for renewable energy assessment workflows.
    
    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_cutout_config() -> Dict[str, dict]: Get ERA5 cutout configuration parameters
    - Plus other configuration access methods
    
    INHERITED ATTRIBUTES FROM AttributesParser:
    -------------------------------------------
    - config_file_path: Path to configuration file
    - region_short_code: Region identifier code
    - resource_type: Resource type identifier
    - Plus other configuration attributes
    
    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - get_cutout_path(): Generate unique file path for cutout storage
    - get_era5_cutout(): Create and prepare ERA5 cutout with regional boundaries
    
    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing cutout parameters
    region_short_code : str
        Region identifier for boundary definition and file naming
    resource_type : str
        Resource type ('solar', 'wind', 'bess') - passed through to dependencies
        
    Attributes
    ----------
    gadmBoundary : GADMBoundaries
        GADM boundary processor for regional extent definition
    cutout_config : dict
        ERA5 cutout configuration parameters from config file
    start_year : str
        Starting year extracted from cutout temporal configuration
    end_year : str
        Ending year extracted from cutout temporal configuration
    cutout_path : Path
        File path for storing the ERA5 cutout data
        
    Methods
    -------
    get_cutout_path() -> Path
        Generate unique file path based on region and temporal extent
    get_era5_cutout() -> tuple[atlite.Cutout, gpd.GeoDataFrame]
        Create ERA5 cutout with climate data and return with regional boundaries
        
    Examples
    --------
    Create ERA5 cutout for British Columbia:
    
    >>> from RES.era5_cutout import ERA5Cutout
    >>> era5_processor = ERA5Cutout(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="solar"
    ... )
    >>> cutout, boundaries = era5_processor.get_era5_cutout()
    >>> print(f"Cutout covers {cutout.coords['time'].size} time steps")
    
    Access cutout data:
    
    >>> # Climate data is stored as xarray Dataset
    >>> climate_data = cutout.data
    >>> print(f"Available variables: {list(climate_data.data_vars)}")
    >>> 
    >>> # Data is stored on disk at cutout path
    >>> print(f"Data stored at: {era5_processor.cutout_path}")
    
    Configuration Requirements
    --------------------------
    The cutout configuration must include:
    
    ```yaml
    cutout:
      root: "data/cutouts"  # Storage directory
      module: "era5"        # Data source module
      dx: 0.25             # Longitude resolution (degrees)
      dy: 0.25             # Latitude resolution (degrees)
      snapshots:
        start: ["2020-01-01"]  # Start date
        end: ["2020-12-31"]    # End date
    ```
    
    Data Processing Workflow
    ------------------------
    1. **Configuration Loading**: Extract cutout parameters from config file
    2. **Boundary Definition**: Get regional boundaries from GADM processor
    3. **Path Generation**: Create unique file path for cutout storage
    4. **Cutout Creation**: Initialize atlite Cutout with spatial/temporal bounds
    5. **Data Preparation**: Download and prepare ERA5 climate data
    6. **Return Processing**: Provide cutout and boundary data for analysis
    
    Spatial Configuration
    ---------------------
    - **Boundary Source**: GADM administrative boundaries
    - **Spatial Buffer**: Automatic buffer (dx, dy) around region boundaries
    - **Resolution**: Configurable grid resolution (typically 0.25° for ERA5)
    - **Coordinate System**: WGS84 (EPSG:4326) for global compatibility
    
    Temporal Configuration
    ----------------------
    - **Date Range**: Flexible start/end date specification
    - **Time Resolution**: Hourly ERA5 data (standard)
    - **Year Handling**: Single year or multi-year cutout support
    - **File Naming**: Automatic naming based on region and temporal extent
    
    Data Management
    ---------------
    - **Storage**: Cutouts stored as NetCDF files on disk
    - **Memory Efficiency**: Data loaded as Dask arrays to minimize memory usage
    - **Caching**: Existing cutouts are reused if available
    - **File Organization**: Systematic naming for easy identification
    
    Climate Data Variables
    ----------------------
    ERA5 cutouts typically include:
    - Wind speed components (u, v) at multiple heights
    - Surface solar radiation (downward shortwave)
    - Temperature at 2m height
    - Surface pressure
    - Other meteorological variables as configured
    
    Performance Considerations
    --------------------------
    - Download time scales with spatial extent and temporal range
    - Large regions or long time periods require substantial storage
    - Network connectivity affects download performance
    - Concurrent requests are disabled to respect API limits
    - Monthly request chunking improves download reliability
    
    Integration Points
    ------------------
    - **Boundaries**: Integrates with GADMBoundaries for regional extent
    - **Capacity Calculation**: Provides climate data for capacity factor analysis
    - **Resource Assessment**: Supports wind and solar resource calculations
    - **Grid Cells**: Compatible with grid cell generation workflows
    
    Notes
    -----
    - Requires CDSAPI credentials for ERA5 data access
    - Data is downloaded from Copernicus Climate Data Store
    - Atlite library handles ERA5 data processing and conversion
    - Cutout preparation may take considerable time for large regions
    - Results are compatible with renewable energy analysis workflows
    - Memory usage is optimized through Dask array implementation
    
    Dependencies
    ------------
    - atlite: Climate data processing and cutout management
    - pathlib: File path operations
    - RES.AttributesParser: Parent class for configuration management
    - RES.boundaries.GADMBoundaries: Regional boundary processing
    - RES.utility: Utility functions for logging and updates
    
    Raises
    ------
    ConnectionError
        If ERA5 data download fails or API is unavailable
    ValueError
        If configuration parameters are invalid or missing
    FileNotFoundError
        If configuration files or directories don't exist
        
    See Also
    --------
    atlite.Cutout : Core cutout functionality
    RES.boundaries.GADMBoundaries : Regional boundary processing
    RES.CellCapacityProcessor : Downstream capacity calculation
    """
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        self.required_args = {   #order doesn't matter
            "config_file_path" : self.config_file_path,  # INHERITED ATTRIBUTE from AttributesParser
            "region_short_code": self.region_short_code,  # INHERITED ATTRIBUTE from AttributesParser
            "resource_type": self.resource_type  # INHERITED ATTRIBUTE from AttributesParser
        }

        self.gadmBoundary= GADMBoundaries(**self.required_args)
        # Set the targeted data specific attributes
        self.cutout_config:dict = super().get_cutout_config()  # INHERITED METHOD from AttributesParser
        
        # Extract start and end years
        self.start_year = self.cutout_config["snapshots"]["start"][0][:4]
        self.end_year = self.cutout_config["snapshots"]["end"][0][:4]
        self.cutout_path:Path = self.get_cutout_path()

        
    def get_cutout_path(self)->Path:
        '''
        ### takes:
        cutout configuration dictionary. Specifically the snapshot information.
        
        ### does:
        creates an unique name based on the region and start/end year
        for a cutout.
        
        ### returns: 
        file path + unique name for the cutout described by selections in the
        cutout configuration.
        '''
        
        # Get the base directory and region name
        base_dir = Path(self.cutout_config['root'])
        
        # Construct the file name based on whether it's a single year or multi-year file
        if self.start_year == self.end_year:
            suffix = self.start_year
        else:
            suffix = "_".join([self.start_year, self.end_year])
        
        # Combine region and year(s) to form the file name
        file_name = f"{self.region_short_code}_{suffix}.nc"
        
        # Join the base directory and file name to form the full path
        file_path:Path = base_dir / self.country_kwd / file_name
        
        return file_path

        
    def get_era5_cutout(self) -> atlite.Cutout:
        """
        This method creates a cutout based on data for ERA5.

        Args:
            bounding_box (dict): A dictionary containing the bounding box with 'min_x', 'max_x', 'min_y', 'max_y'.
            region_code (str, optional): Optional string representing the code of the region for which the cutout is created.

        Returns:
            the 'cutout' object from atlite.

        Raises:
            ValueError: If the bounding box is not valid.
            ConnectionError: If there is an issue connecting to the data source.

        Note:
            After execution, all downloaded data is stored at cutout.path. By default, it is not loaded into memory but into Dask arrays to keep memory consumption low. The data is accessible via cutout.data, which is an xarray.Dataset.
        """
        utils.print_update(level=print_level_base+1,
                           message=f"{__name__}|  Processing ERA5's cutout...")

        MBR,region_boundary=self.gadmBoundary.get_bounding_box()
        utils.print_update(level=print_level_base+2,
                   message=f"{__name__}| ✓ MBR and regional boundary created. ")
        
        # Extract parameters from the configuration file
        dx, dy = self.cutout_config["dx"], self.cutout_config['dy']
        time_horizon = slice(self.cutout_config["snapshots"]['start'][0], self.cutout_config["snapshots"]['end'][0])
        min_x, max_x, min_y, max_y = MBR.values()

        # Create the cutout based on bounds found from above
        cutout = atlite.Cutout(
            path=self.cutout_path,
            module=self.cutout_config["module"],
            x=slice(min_x - dx, max_x + dx),  # Longitude
            y=slice(min_y - dy, max_y + dy),  # Latitude
            dx=dx,
            dy=dy,
            time=time_horizon
        )

        cutout.prepare(monthly_requests=True, 
                       concurrent_requests=False)  # Prepare the cutout data
        utils.print_info(info=""" Memory management remarks:
    * After execution, all downloaded data is stored at cutout.path. By default, it is not loaded into memory, but into dask arrays. This keeps the memory consumption extremely low.
    * The data is accessible in cutout.data, which is an xarray.Dataset. Querying the cutout gives us some basic information on which data is contained in it.
    * For more operations related to cutout, check the tool docs @ https://atlite.readthedocs.io/en/master/examples/create_cutout.html#
        """)

    
        utils.print_update(level=print_level_base+1,
                   message=f"{__name__}| ✓ Cutout and regional boundary processed. ")
        return cutout,region_boundary
    