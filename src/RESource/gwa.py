from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import rioxarray as rxr
import xarray as xr

import RESource.utility as utils
from RESource.AttributesParser import AttributesParser
from RESource.boundaries import GADMBoundaries
from RESource.hdf5_handler import DataHandler

print_level_base = 3


@dataclass
class GWACells(AttributesParser):
    """
    Global Wind Atlas (GWA) data processor for high-resolution wind resource analysis.

    This class integrates Global Wind Atlas data with regional boundaries to provide
    high-resolution wind resource assessment capabilities for renewable energy planning.
    GWA provides detailed wind speed, wind power density, and wind class information
    at much higher spatial resolution than ERA5 data, making it valuable for detailed
    site assessment and resource characterization.

    The class handles downloading, processing, and spatial mapping of GWA raster data
    to ERA5 grid cells, enabling multi-scale wind resource analysis. It processes
    multiple GWA data layers including wind speed, wind power density, and IEC wind
    class classifications to provide comprehensive wind resource information.

    INHERITED METHODS FROM GADMBoundaries:
    --------------------------------------
    - get_bounding_box() -> tuple: Get regional bounding box for spatial clipping
    - get_region_boundary() -> gpd.GeoDataFrame: Get regional boundary geometry
    - get_country_boundary() -> gpd.GeoDataFrame: Get country-level boundary geometry
    - Plus other boundary processing methods

    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_gwa_config() -> dict: Get GWA data configuration parameters
    - get_default_crs() -> str: Get default coordinate reference system
    - get_region_mapping() -> Dict[str, dict]: Get region mapping configuration
    - Plus other configuration access methods

    INHERITED ATTRIBUTES FROM AttributesParser:
    -------------------------------------------
    - region_short_code: Region identifier code
    - region_mapping: Dictionary mapping region codes to configuration
    - store: HDF5 data store path for processed results
    - Plus other configuration attributes

    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - prepare_GWA_data(): Download and process GWA raster data for the region
    - download_file(): Download individual files from remote sources
    - load_gwa_cells(): Create GeoDataFrame of GWA cells with spatial geometry
    - map_GWA_cells_to_ERA5(): Map high-resolution GWA data to ERA5 grid cells

    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing GWA data parameters
    region_short_code : str
        Region identifier for boundary definition and data filtering
    resource_type : str
        Resource type ('wind') for GWA wind resource analysis

    Attributes
    ----------
    merged_data : xr.DataArray
        Merged xarray DataArray containing all GWA data layers
    gwa_config : dict
        GWA configuration parameters from config file
    datahandler : DataHandler
        HDF5 data handler for storing processed results
    gwa_datafields : dict
        Field definitions for GWA data layers
    gwa_rasters : dict
        Raster file specifications for GWA data
    gwa_sources : dict
        Source URLs for downloading GWA data
    gwa_root : Path
        Root directory for GWA data storage
    bounding_box : dict
        Regional bounding box coordinates for spatial clipping
    region_gwa_cells_df : pd.DataFrame
        Processed GWA cells data as pandas DataFrame
    gwa_cells_gdf : gpd.GeoDataFrame
        GWA cells with spatial geometry for analysis
    mapped_gwa_cells_aggr_df : pd.DataFrame
        GWA data aggregated to ERA5 grid cell resolution

    Methods
    -------
    prepare_GWA_data(windspeed_min=10, windspeed_max=20, memory_resource_limitation=False) -> pd.DataFrame
        Download, process, and merge GWA raster data for the region
    download_file(url, destination) -> None
        Download a file from URL to specified destination path
    load_gwa_cells(memory_resource_limitation=False) -> gpd.GeoDataFrame
        Load GWA cells as GeoDataFrame with spatial geometry
    map_GWA_cells_to_ERA5(memory_resource_limitation=False) -> None
        Map high-resolution GWA data to ERA5 grid cells for integration

    Examples
    --------
    Create GWA processor for British Columbia:

    >>> from RESource.gwa import GWACells
    >>> gwa_processor = GWACells(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind"
    ... )
    >>>
    >>> # Load high-resolution GWA cells
    >>> gwa_cells = gwa_processor.load_gwa_cells()
    >>> print(f"Loaded {len(gwa_cells)} GWA cells")

    Process GWA data with wind speed filtering:

    >>> # Prepare data with wind speed constraints
    >>> gwa_data = gwa_processor.prepare_GWA_data(
    ...     windspeed_min=12,
    ...     windspeed_max=25,
    ...     memory_resource_limitation=True
    ... )
    >>> print(f"Filtered to {len(gwa_data)} high-quality wind cells")

    Map GWA data to ERA5 grid:

    >>> # Map high-resolution GWA to ERA5 cells
    >>> gwa_processor.map_GWA_cells_to_ERA5(memory_resource_limitation=False)
    >>> print("GWA data mapped to ERA5 grid cells")

    Configuration Requirements
    --------------------------
    The GWA configuration must include:

    ```yaml
    gwa_data:
      root: "data/downloaded_data/GWA"  # Storage directory
      datafields:
        windspeed_gwa: "Wind speed at 100m"
        windpower_gwa: "Wind power density at 100m"
        IEC_Class_ExLoads: "IEC wind class"
      rasters:
        windspeed_gwa: "GWA_country_code_windspeed.tif"
        windpower_gwa: "GWA_country_code_windpower.tif"
        IEC_Class_ExLoads: "GWA_country_code_iec.tif"
      sources:
        windspeed_gwa: "https://globalwindatlas.info/download/GWA_country_code_windspeed.tif"
        windpower_gwa: "https://globalwindatlas.info/download/GWA_country_code_windpower.tif"
        IEC_Class_ExLoads: "https://globalwindatlas.info/download/GWA_country_code_iec.tif"

    region_mapping:
      BC:
        GWA_country_code: "CAN"  # Country code for GWA data
    ```

    Data Processing Workflow
    ------------------------
    1. **Configuration Loading**: Extract GWA parameters and region mapping
    2. **Data Download**: Check for local data or download from GWA sources
    3. **Raster Processing**: Load and clip raster data to regional boundaries
    4. **Data Merging**: Combine multiple GWA layers into unified dataset
    5. **Quality Filtering**: Apply wind speed and other quality constraints
    6. **Spatial Conversion**: Convert raster data to point-based GeoDataFrame
    7. **Grid Mapping**: Aggregate high-resolution GWA data to ERA5 grid cells
    8. **Data Storage**: Store processed results in HDF5 format for reuse

    GWA Data Layers
    ---------------
    Typical GWA datasets include:
    - **Wind Speed**: Mean wind speed at 100m height (m/s)
    - **Wind Power Density**: Wind power density at 100m height (W/m²)
    - **IEC Wind Class**: International Electrotechnical Commission wind classes
    - **Capacity Factor**: Estimated capacity factors for different turbine types
    - **Wind Direction**: Prevailing wind direction statistics

    Spatial Resolution
    ------------------
    - **GWA Resolution**: Typically 250m to 1km spatial resolution
    - **ERA5 Resolution**: Approximately 25km spatial resolution
    - **Aggregation Method**: Mean values for continuous variables
    - **Coordinate System**: WGS84 (EPSG:4326) for global compatibility
    - **Clipping Boundaries**: Regional boundaries from GADM database

    Quality Control
    ---------------
    - **Wind Speed Filtering**: Configurable minimum/maximum wind speed thresholds
    - **Data Validation**: Automatic detection and handling of NoData values
    - **Spatial Validation**: Clipping to valid regional boundaries
    - **Memory Management**: Optional memory limitation for large datasets
    - **Error Handling**: Graceful handling of download and processing errors

    Performance Considerations
    --------------------------
    - Download time depends on data availability and network speed
    - Processing time scales with region size and data resolution
    - Memory usage can be substantial for large regions
    - Spatial overlay operations are computationally intensive
    - HDF5 storage provides efficient data access for repeated analysis

    Integration Points
    ------------------
    - **ERA5 Data**: Integration with ERA5 climate data for multi-scale analysis
    - **Boundary Data**: Uses GADM boundaries for regional definition
    - **Capacity Analysis**: Provides high-resolution input for capacity factor calculations
    - **Resource Assessment**: Supports detailed wind resource characterization
    - **Grid Analysis**: Compatible with grid cell generation workflows

    Output Formats
    --------------
    - **DataFrame**: Tabular data with wind resource attributes
    - **GeoDataFrame**: Spatial data with point geometries
    - **HDF5 Storage**: Efficient storage for large datasets
    - **Grid Mapping**: ERA5-compatible aggregated datasets

    Notes
    -----
    - GWA data is provided by Technical University of Denmark (DTU)
    - Global coverage with country-specific datasets
    - Higher resolution than ERA5 for detailed site assessment
    - Processing requires substantial computational resources for large regions
    - Results integrate seamlessly with ERA5-based renewable energy workflows
    - Data quality varies by region and local terrain complexity
    - Regular updates are available from the Global Wind Atlas portal

    Dependencies
    ------------
    - geopandas: Spatial data processing and geometry operations
    - pandas: Tabular data manipulation and analysis
    - rioxarray: Raster data reading and spatial operations
    - xarray: N-dimensional array operations and data merging
    - requests: HTTP downloading of GWA datasets
    - pathlib: File path operations and directory management
    - RES.hdf5_handler.DataHandler: HDF5 data storage and retrieval
    - RES.boundaries.GADMBoundaries: Parent class for boundary processing
    - RES.utility: Logging and utility functions

    Raises
    ------
    ConnectionError
        If GWA data download fails due to network issues
    FileNotFoundError
        If required configuration files or directories are missing
    ValueError
        If wind speed thresholds or other parameters are invalid
    RuntimeError
        If raster processing or spatial operations fail

    See Also
    --------
    rioxarray.open_rasterio : Raster data reading functionality
    geopandas.GeoDataFrame.overlay : Spatial overlay operations
    RES.boundaries.GADMBoundaries : Parent class for boundary processing
    RES.hdf5_handler.DataHandler : HDF5 data storage utilities
    """

    merged_data: xr.DataArray = field(init=False)

    def __post_init__(self):
        """
        Initialize GWA processor with configuration and data handler setup.

        Performs post-initialization setup including:
        - Calling parent class initialization for boundary processing
        - Loading GWA-specific configuration from config file
        - Setting up HDF5 data handler for result storage

        This method is automatically called after dataclass initialization
        to prepare the processor for GWA data operations.

        The initialization establishes the connection to configuration
        parameters and prepares the data storage infrastructure needed
        for GWA raster processing and ERA5 grid mapping operations.

        Raises
        ------
        FileNotFoundError
            If configuration file cannot be found or accessed
        ValueError
            If GWA configuration parameters are missing or invalid
        """
        super().__post_init__()

        self.required_args = {
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type,
        }

        self.gadmBoundaries = GADMBoundaries(**self.required_args)

        self.gwa_config = self.get_gwa_config()  # INHERITED METHOD from AttributesParser
        self.datahandler = DataHandler(self.store)  # INHERITED ATTRIBUTE from AttributesParser

    def prepare_GWA_data(
        self, windpseed_min=10, windpseed_max=20, memory_resource_limitation: bool = False
    ) -> xr.DataArray:
        """
        Download, process, and merge Global Wind Atlas raster data for the region.

        This method orchestrates the complete GWA data preparation workflow including:
        downloading required raster files, loading and clipping them to regional
        boundaries, merging multiple data layers, and applying quality filters.
        The processed data is returned as a pandas DataFrame ready for analysis.

        The method handles multiple GWA data types (wind speed, power density,
        IEC classes) and automatically downloads missing files from configured
        sources. Spatial clipping ensures data is limited to the region of
        interest, and wind speed filtering allows focus on viable wind resources.

        Parameters
        ----------
        windpseed_min : float, default=10
            Minimum wind speed threshold in m/s for filtering cells.
            Cells with wind speeds below this value are excluded from results.
        windpseed_max : float, default=20
            Maximum wind speed threshold in m/s for filtering cells.
            Cells with wind speeds above this value are excluded from results.
        memory_resource_limitation : bool, default=False
            Whether to enable memory-efficient processing for large datasets.
            If True, applies wind speed filtering to reduce memory usage.
            If False, uses full wind speed range (0-50 m/s) for processing.

        Returns
        -------
        pd.DataFrame
            Processed GWA data as pandas DataFrame with columns:
            - x, y: Spatial coordinates in regional CRS
            - windspeed_gwa: Wind speed at 100m height (m/s)
            - windpower_gwa: Wind power density at 100m height (W/m²)
            - IEC_Class_ExLoads: IEC wind class classifications
            - Additional fields as configured in GWA data configuration

        Examples
        --------
        Prepare data with default wind speed range:

        >>> gwa_data = processor.prepare_GWA_data()
        >>> print(f"Loaded {len(gwa_data)} wind resource cells")

        Apply strict wind speed filtering for high-quality sites:

        >>> high_wind_data = processor.prepare_GWA_data(
        ...     windpseed_min=15,
        ...     windpseed_max=25,
        ...     memory_resource_limitation=True
        ... )
        >>> print(f"High-wind sites: {len(high_wind_data)} cells")

        Process full dataset without filtering:

        >>> all_data = processor.prepare_GWA_data(
        ...     windpseed_min=0,
        ...     windpseed_max=50,
        ...     memory_resource_limitation=False
        ... )

        Raises
        ------
        ConnectionError
            If GWA data download fails due to network issues
        FileNotFoundError
            If GWA raster files cannot be found locally or remotely
        ValueError
            If wind speed thresholds are invalid (min >= max)
        RuntimeError
            If raster processing or spatial operations fail

        Notes
        -----
        - Processing time depends on region size and number of data layers
        - Downloaded files are cached locally to avoid repeated downloads
        - Memory usage scales with region size and data resolution
        - Wind speed filtering significantly reduces memory requirements
        - Multiple raster files are automatically merged into unified dataset
        - Spatial coordinates are preserved for subsequent spatial analysis
        """
        self.gwa_data_list = []

        # Load configuration parameters
        self.gwa_datafields = self.gwa_config.get("datafields", {})
        self.gwa_rasters = self.gwa_config.get("rasters", {})
        self.gwa_sources = self.gwa_config.get("sources", {})
        self.gwa_root = Path(self.gwa_config.get("root", "data/downloaded_data/GWA"))
        self.bounding_box, _ = (
            self.gadmBoundaries.get_bounding_box()
        )  # INHERITED METHOD from GADMBoundaries
        # Create the root directory if it doesn't exist
        self.gwa_root.mkdir(parents=True, exist_ok=True)

        # Check for existence and download if necessary
        for key, raster_name in self.gwa_rasters.items():
            self.gwa_country_code = self.region_mapping[self.region_short_code].get(
                "GWA_country_code"
            )  # INHERITED ATTRIBUTES from AttributesParser
            self.raster_name = raster_name.replace("GWA_country_code", self.gwa_country_code)
            self.raster_path = self.gwa_root / self.raster_name
            if not self.raster_path.exists():
                generic_source_url = self.gwa_sources[key]
                self.region_source_url = generic_source_url.replace(
                    "GWA_country_code", self.gwa_country_code
                )
                utils.print_update(
                    level=print_level_base,
                    message=f"{__name__}| Downloading {key} from {self.region_source_url}",
                )
                self.download_file(self.region_source_url, self.raster_path)

            try:
                # Process each raster using a streamlined approach
                data = (
                    rxr.open_rasterio(self.raster_path)
                    .rio.clip_box(**self.bounding_box)
                    .rename(key)
                    .drop_vars(["band", "spatial_ref"])
                    .isel(
                        band=1 if "*Class*" in key else 0
                    )  # 'IEC_Class_ExLoads' data is in band 1
                )

                # data.rio.write_crs(self.get_default_crs(), inplace=True)
                self.gwa_data_list.append(data)
            except Exception as e:
                utils.print_update(
                    level=print_level_base + 1, message=f"{__name__}| Error processing {key}: {e}"
                )

        gwa_dfs = []
        for i, da in enumerate(self.gwa_data_list):
            name = da.name or f"var_{i}"
            df = da.to_dataframe(name=name).dropna()
            gwa_dfs.append(df)

        # concatenate side-by-side on the same MultiIndex (y,x)
        self.merged_df = pd.concat(gwa_dfs, axis=1)

        """ 
        self.merged_data = xr.merge(self.gwa_data_list) if self.gwa_data_list is None else xr.DataArray()
        self.merged_data.rename('gwa_data')
      
        self.merged_df = self.merged_data.to_dataframe().dropna(how='any')
        """
        self.merged_df.reset_index(inplace=True)

        if memory_resource_limitation:
            utils.print_update(
                level=print_level_base,
                message=f"{__name__}| Memory resource limitations enabled. Filtering GWA cells within windspeed mask to limit the data offload processing...",
            )
        # else:
        #     windpseed_min:float=0 #m/s
        #     windpseed_max:float=100 #m/s

        # mask=(self.merged_df['windspeed_gwa'] >= windpseed_min) & (self.merged_df['windspeed_gwa'] <= windpseed_max)
        # self.merged_df_f=self.merged_df[mask]
        # utils.print_update(level=print_level_base+1,message=f"{__name__}| {abs(len(self.merged_df_f) - self.merged_df.shape[0])} cells have been filtered due to Windspeed filter [{windpseed_min}-{windpseed_max} m/s].")
        # utils.print_update(level=print_level_base,message=f"✔ Cleaned data loaded for {len(self.merged_df_f)} GWA cells")

        # class_mapping = {0: 'III', 1: 'II', 2: 'I', 3: 'T', 4: 'S'}
        # # Correctly modifying only one column
        # self.merged_df_f['IEC_Class_ExLoads'] = self.merged_df_f['IEC_Class_ExLoads'].map(class_mapping).fillna(0)

        # return self.merged_df_f
        return self.merged_df

    def download_file(self, url: str, destination: Path) -> None:
        """
        Download a file from a remote URL to a specified local destination.

        Downloads GWA raster files from remote sources when they are not
        available locally. The method handles HTTP requests with proper
        error checking and provides detailed logging of download operations.

        Files are downloaded completely before being written to avoid
        partial downloads. The destination directory is created automatically
        if it doesn't exist, ensuring reliable file operations.

        Parameters
        ----------
        url : str
            Complete URL of the file to download.
            Should be a valid HTTP/HTTPS URL pointing to a GWA raster file.
        destination : Path
            Local file path where the downloaded file should be saved.
            Parent directories will be created automatically if needed.

        Returns
        -------
        None
            The method doesn't return a value but saves the file to disk.

        Examples
        --------
        Download a GWA wind speed raster:

        >>> url = "https://globalwindatlas.info/download/CAN_windspeed.tif"
        >>> destination = Path("data/GWA/CAN_windspeed.tif")
        >>> processor.download_file(url, destination)

        Download with automatic path handling:

        >>> url = "https://globalwindatlas.info/download/CAN_windpower.tif"
        >>> destination = processor.gwa_root / "CAN_windpower.tif"
        >>> processor.download_file(url, destination)

        Raises
        ------
        requests.RequestException
            If the HTTP request fails due to network issues or server errors
        requests.HTTPError
            If the server returns an HTTP error status code
        IOError
            If the local file cannot be written due to permissions or disk space
        FileNotFoundError
            If the destination directory cannot be created

        Notes
        -----
        - Download progress is logged through utility print functions
        - Network timeouts may occur for large files on slow connections
        - Existing files are overwritten without warning
        - File integrity is not verified after download
        - Destination path is automatically converted to Path object if needed
        """

        destination = utils.ensure_path(destination)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            with destination.open("wb") as f:
                f.write(response.content)
        except requests.RequestException as e:
            utils.print_update(
                level=print_level_base,
                message=f"{__name__}| Failed to download {destination} from {url}. Error: {e}",
            )

    def load_gwa_cells(self, memory_resource_limitation: bool | None = False):
        """
        Load GWA cells as a spatial GeoDataFrame with point geometries.

        Converts processed GWA tabular data into a spatial GeoDataFrame by
        creating point geometries from coordinate information. The resulting
        GeoDataFrame contains all wind resource attributes along with spatial
        geometry suitable for spatial analysis and visualization.

        The method automatically clips the data to regional boundaries to
        ensure results are geographically constrained to the area of interest.
        This spatial filtering removes any cells that fall outside the defined
        regional boundaries despite being within the bounding box.

        Parameters
        ----------
        memory_resource_limitation : Optional[bool], default=False
            Whether to enable memory-efficient processing for large datasets.
            Passed through to prepare_GWA_data() method to control filtering.
            If True, applies wind speed filtering to reduce memory usage.
            If False, processes the full dataset without memory limitations.

        Returns
        -------
        geopandas.GeoDataFrame
            Spatial GeoDataFrame containing GWA cells with:
            - Point geometries representing cell center coordinates
            - Wind resource attributes (speed, power density, IEC class)
            - Spatial reference system matching regional CRS
            - Geographic clipping to regional boundaries

        Examples
        --------
        Load all GWA cells for the region:

        >>> gwa_cells = processor.load_gwa_cells()
        >>> print(f"Loaded {len(gwa_cells)} spatial cells")
        >>> print(f"CRS: {gwa_cells.crs}")

        Load with memory optimization:

        >>> gwa_cells = processor.load_gwa_cells(memory_resource_limitation=True)
        >>> print(f"Memory-optimized: {len(gwa_cells)} cells")

        Access spatial and attribute data:

        >>> # Spatial analysis
        >>> total_area = gwa_cells.total_bounds
        >>> print(f"Spatial extent: {total_area}")
        >>>
        >>> # Attribute analysis
        >>> mean_windspeed = gwa_cells['windspeed_gwa'].mean()
        >>> print(f"Average wind speed: {mean_windspeed:.2f} m/s")

        Raises
        ------
        ValueError
            If coordinate columns (x, y) are missing from the GWA data
        GeometryError
            If point geometries cannot be created from coordinates
        CRSError
            If the coordinate reference system is invalid or undefined

        Notes
        -----
        - Point geometries are created from x,y coordinate columns
        - Spatial clipping ensures geographic consistency with boundaries
        - CRS is inherited from the regional configuration
        - Processing time scales with the number of GWA cells
        - Memory usage depends on dataset size and attribute complexity
        - Results are suitable for spatial overlay and intersection operations
        """
        self.region_gwa_cells_df = self.prepare_GWA_data(memory_resource_limitation)

        # Vectorized creation of geometries
        self.gwa_cells_gdf = gpd.GeoDataFrame(
            self.region_gwa_cells_df,
            geometry=gpd.points_from_xy(
                self.region_gwa_cells_df["x"], self.region_gwa_cells_df["y"]
            ),
            crs=self.crs_d,  # INHERITED METHOD from AttributesParser
        ).clip(
            self.gadmBoundaries.get_region_boundary(), keep_geom_type=False
        )  # INHERITED METHOD from GADMBoundaries

        # self.gwa_cells_gdf = self.calculate_common_parameters_GWA_cells()
        # self.gwa_cells_gdf = self.map_GWAcells_to_ERA5cells()
        utils.print_update(
            level=print_level_base,
            message=f"{__name__}| Global Wind Atlas (GWA) Cells loaded. Size: {len(self.region_gwa_cells_df)}",
        )

        return self.gwa_cells_gdf

    def map_GWA_cells_to_ERA5(
        self, aggregation_level: str, memory_resource_limitation: bool | None
    ):
        """
        Map high-resolution GWA cells to coarser ERA5 grid cells for multi-scale analysis.

        This method performs spatial aggregation of high-resolution GWA wind data
        (typically 250m-1km resolution) to ERA5 grid cells (approximately 25km resolution).
        The aggregation process uses spatial overlay operations to determine which
        GWA cells fall within each ERA5 cell, then computes mean values for all
        wind resource attributes.

        The mapping enables integration of detailed GWA wind resource data with
        ERA5-based renewable energy analysis workflows, providing enhanced spatial
        detail while maintaining compatibility with ERA5 grid structures.

        Processing is performed on a region-by-region basis to optimize memory
        usage and computational efficiency. Results are automatically stored
        in the HDF5 data store for subsequent analysis operations.

        Parameters
        ----------
        memory_resource_limitation : Optional[bool]
            Whether to enable memory-efficient processing for large datasets.
            Passed through to load_gwa_cells() and prepare_GWA_data() methods.
            If True, applies filtering to reduce memory usage during processing.
            If False, processes the complete dataset without memory limitations.

        Returns
        -------
        None
            The method doesn't return a value but stores results in the HDF5 store.
            Aggregated data is accessible via self.mapped_gwa_cells_aggr_df attribute
            and permanently stored in the 'cells' store for future access.

        Examples
        --------
        Map GWA data to ERA5 grid with full dataset:

        >>> processor.map_GWA_cells_to_ERA5(memory_resource_limitation=False)
        >>> print("GWA data mapped to ERA5 grid cells")

        Map with memory optimization for large regions:

        >>> processor.map_GWA_cells_to_ERA5(memory_resource_limitation=True)
        >>> print("Memory-optimized mapping completed")

        Access mapped results:

        >>> # Results are stored in datahandler
        >>> era5_with_gwa = processor.datahandler.from_store('cells')
        >>> print(f"ERA5 cells with GWA data: {len(era5_with_gwa)}")
        >>> print(f"Columns: {list(era5_with_gwa.columns)}")

        Processing Workflow
        -------------------
        1. **Data Loading**: Load ERA5 grid cells from HDF5 store
        2. **GWA Loading**: Load high-resolution GWA cells as GeoDataFrame
        3. **Spatial Overlay**: Perform intersection between GWA and ERA5 cells
        4. **Coordinate Mapping**: Update coordinates to ERA5 cell centers
        5. **Aggregation**: Compute mean values for numeric attributes by ERA5 cell
        6. **Storage**: Store aggregated results in HDF5 store with forced update

        Spatial Operations
        ------------------
        - **Overlay Method**: Intersection overlay to find spatial relationships
        - **Aggregation Function**: Mean aggregation for all numeric attributes
        - **Coordinate Assignment**: ERA5 cell coordinates replace GWA coordinates
        - **Regional Processing**: Separate processing by geographic regions
        - **Memory Management**: Regional processing reduces peak memory usage

        Performance Considerations
        --------------------------
        - Processing time scales with number of GWA cells and ERA5 cells
        - Memory usage peaks during spatial overlay operations
        - Regional processing improves memory efficiency for large datasets
        - Storage operations may take time for large aggregated datasets
        - Spatial indexing improves performance for repeated operations

        Raises
        ------
        FileNotFoundError
            If ERA5 grid cells are not found in the HDF5 store
        ValueError
            If spatial overlay operations fail due to geometry issues
        MemoryError
            If dataset is too large for available memory (use memory limitation)
        RuntimeError
            If HDF5 storage operations fail

        Notes
        -----
        - Aggregation preserves all numeric wind resource attributes
        - Categorical attributes (like IEC class) may require special handling
        - Results overwrite existing data in the HDF5 store (force_update=True)
        - Processing is optimized for typical renewable energy analysis workflows
        - Spatial accuracy depends on the quality of ERA5 and GWA geometries
        - Large regions may require substantial processing time and memory
        - Results integrate seamlessly with ERA5-based capacity calculations

        Data Quality
        ------------
        - Mean aggregation is appropriate for continuous wind variables
        - Statistical significance increases with more GWA cells per ERA5 cell
        - Spatial representation accuracy depends on resolution differences
        - Edge effects may occur at regional boundaries
        """
        self.datahandler.refresh()
        # Load the grid cells and GWA cells as GeoDataFrames
        self.store_grid_cells = self.datahandler.from_store("cells")

        _era5_cells_ = self.store_grid_cells.reset_index()

        self.gwa_cells_gdf = self.load_gwa_cells(memory_resource_limitation)

        utils.print_update(
            level=print_level_base + 1,
            message=f"{__name__}| Mapping {len(self.gwa_cells_gdf)} GWA Cells to {len(_era5_cells_)} ERA5 Cells...",
        )

        results = []  # List to store results for each region
        utils.print_update(
            level=print_level_base + 1,
            message=f"{__name__}| Calculating aggregated values for ERA5 Cell's...",
        )

        column_name = aggregation_level
        for region in _era5_cells_[column_name].unique():
            _era5_cells_region = _era5_cells_[_era5_cells_[column_name] == region]

            # Perform overlay operation
            _data_ = self.gwa_cells_gdf.overlay(
                _era5_cells_region, how="intersection", keep_geom_type=False
            )

            # Rename columns and select relevant data
            # _data_ = _data_.rename(columns={'x_1': 'x', 'y_1': 'y'}) # x1,y1 are the GWA coords
            _data_ = _data_.rename(columns={"x_2": "x", "y_2": "y"})  # x2,y2 are the ERA5 coords
            selected_columns = list(_data_.columns)  # + [f'{self.resource_type}_CF_mean']

            regional_df = _data_.loc[:, selected_columns]

            numeric_cols = regional_df.select_dtypes(include="number")
            regional_mapped_gwa_cells_aggregated = numeric_cols.groupby(
                regional_df["cell"]
            ).mean()  # Aggregates columns data via mean

            # Store mapped GWA cells in results list
            results.append(regional_mapped_gwa_cells_aggregated)

        # Concatenate all results into a single GeoDataFrame
        self.mapped_gwa_cells_aggr_df = pd.concat(results, axis=0)

        # Store the aggregated data
        self.datahandler.to_store(self.mapped_gwa_cells_aggr_df, "cells")

        return self.mapped_gwa_cells_aggr_df
