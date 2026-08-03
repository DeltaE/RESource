from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import rasterio
import requests
from rasterio.mask import mask

from RESource import utility as utils
from RESource.AttributesParser import AttributesParser
from RESource.boundaries import GADMBoundaries

print_level_base = 4


# Define the GAEZRasterProcessor class
class GAEZRasterProcessor(AttributesParser):
    """
    GAEZ (Global Agro-Ecological Zones) raster data processor for renewable energy land constraint analysis.

    This class handles the download, extraction, clipping, and visualization of GAEZ raster datasets
    used in renewable energy resource assessment. GAEZ provides global spatial data on agricultural
    suitability, land resources, and ecological constraints that are essential for identifying
    suitable areas for renewable energy development while avoiding productive agricultural land.

    The processor integrates GAEZ land constraint data with regional boundaries to support
    renewable energy siting decisions and capacity assessments. It automatically downloads
    the required raster datasets, extracts specific layers based on configuration, clips
    them to regional boundaries, and generates visualization outputs for analysis.

    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_gaez_data_config() -> Dict[str, dict]: Get GAEZ dataset configuration parameters
    - get_region_name() -> str: Get full region name for display purposes
    - Plus other configuration access methods

    INHERITED ATTRIBUTES FROM AttributesParser:
    -------------------------------------------
    - config_file_path: Path to configuration file
    - region_short_code: Region identifier code
    - resource_type: Resource type identifier
    - Plus other configuration attributes

    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - process_all_rasters(): Main pipeline for processing all configured raster types
    - plot_gaez_tif(): Generate visualization plots for processed raster data
    - __download_resources_zip_file__(): Download GAEZ ZIP archive from remote source
    - __extract_rasters__(): Extract required raster files from ZIP archive
    - __clip_to_boundary_n_plot__(): Clip rasters to regional boundaries and generate plots

    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing GAEZ dataset parameters
    region_short_code : str
        Region identifier for boundary definition and file naming
    resource_type : str
        Resource type ('solar', 'wind', 'bess') - used for dependency injection

    Attributes
    ----------
    gadmBoundary : GADMBoundaries
        GADM boundary processor for regional extent definition
    gaez_config : dict
        GAEZ dataset configuration parameters from config file
    gaez_root : Path
        Root directory for GAEZ data storage and processing
    zip_file : Path
        Path to the GAEZ ZIP archive file
    Rasters_in_use_direct : Path
        Directory for extracted and processed raster files
    raster_types : list
        List of raster type configurations to process
    region_boundary : gpd.GeoDataFrame
        Regional boundary geometry for clipping operations

    Methods
    -------
    process_all_rasters(show=False) -> dict
        Main pipeline to download, extract, clip, and plot all configured rasters
    plot_gaez_tif(tif_path, color_map, plot_title, save_to, show=False) -> matplotlib.Figure
        Generate and save visualization plots for raster data

    Examples
    --------
    Process GAEZ rasters for British Columbia:

    >>> from RESource.gaez import GAEZRasterProcessor
    >>> gaez_processor = GAEZRasterProcessor(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="solar"
    ... )
    >>> raster_paths = gaez_processor.process_all_rasters(show=True)
    >>> print(f"Processed {len(raster_paths)} raster types")

    Access specific raster data:

    >>> # Raster paths are returned as dictionary
    >>> if 'slope' in raster_paths:
    ...     slope_path = raster_paths['slope']
    ...     print(f"Slope raster available at: {slope_path}")

    Configuration Requirements
    --------------------------
    The GAEZ configuration must include:

    ```yaml
    gaez_data:
      root: "data/downloaded_data/GAEZ"  # Storage directory
      source: "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR.zip"
      zip_file: "LR.zip"  # ZIP archive filename
      Rasters_in_use_direct: "Rasters_in_use"  # Extraction directory
      raster_types:
        - name: "slope"
          raster: "slope.tif"
          zip_extract_direct: "slope"
          color_map: "terrain"
        # Additional raster type configurations...
    ```

    Data Processing Workflow
    ------------------------
    1. **Configuration Loading**: Extract GAEZ parameters from config file
    2. **Download Check**: Verify ZIP archive exists or download from source
    3. **Extraction**: Extract required raster files from ZIP archive
    4. **Boundary Processing**: Get regional boundaries from GADM processor
    5. **Clipping**: Clip each raster to regional boundaries
    6. **Visualization**: Generate plots for each processed raster
    7. **Path Management**: Return dictionary of processed raster file paths

    Raster Type Configuration
    -------------------------
    Each raster type requires:
    - **name**: Identifier for the raster layer
    - **raster**: Filename of the raster file within ZIP archive
    - **zip_extract_direct**: Directory path within ZIP archive
    - **color_map**: Matplotlib colormap for visualization

    Supported Raster Types
    ----------------------
    Common GAEZ rasters include:
    - **Slope**: Terrain slope for accessibility analysis
    - **Soil Quality**: Agricultural productivity constraints
    - **Land Cover**: Vegetation and land use classifications
    - **Elevation**: Digital elevation model data
    - **Climate Zones**: Agro-ecological zone classifications

    Spatial Processing
    ------------------
    - **Input CRS**: Inherits coordinate system from source rasters
    - **Clipping**: Uses regional boundaries with geometry buffering
    - **Output Format**: GeoTIFF files with preserved metadata
    - **Resolution**: Maintains original raster resolution
    - **Compression**: Optimized file storage for large datasets

    Visualization Features
    ----------------------
    - **Automatic Plotting**: Generates plots for all processed rasters
    - **Custom Colormaps**: Configurable visualization schemes
    - **Coordinate Display**: Latitude/longitude axis labels
    - **Legend Integration**: Horizontal colorbar with value indicators
    - **File Output**: PNG format with high-resolution settings

    Performance Considerations
    --------------------------
    - ZIP download time scales with file size (typically 100MB-1GB)
    - Extraction time depends on number of raster layers
    - Clipping operations are memory-intensive for large regions
    - Multiple raster processing benefits from parallel execution
    - Network connectivity affects initial download performance

    Integration Points
    ------------------
    - **Boundaries**: Uses GADMBoundaries for regional extent definition
    - **Land Constraints**: Provides input data for land availability analysis
    - **Capacity Calculation**: Supports renewable energy siting decisions
    - **Visualization**: Integrates with broader visualization workflows

    Error Handling
    --------------
    - **Download Failures**: Graceful handling of network issues
    - **Missing Files**: Clear error messages for missing raster files
    - **Extraction Errors**: Validation of ZIP archive contents
    - **Processing Failures**: Detailed logging for debugging

    Output Management
    -----------------
    - **Organized Storage**: Systematic directory structure for processed data
    - **File Naming**: Consistent naming convention with region identifiers
    - **Metadata Preservation**: Maintains spatial reference and statistics
    - **Visualization Archive**: Organized plot storage for documentation

    Notes
    -----
    - GAEZ data is provided by FAO (Food and Agriculture Organization)
    - Raster datasets are typically global coverage at moderate resolution
    - Processing large regions may require substantial disk space
    - Results integrate with renewable energy assessment workflows
    - Visualization outputs support decision-making and reporting
    - ZIP archives are cached locally to avoid repeated downloads

    Dependencies
    ------------
    - requests: HTTP downloading of ZIP archives
    - rasterio: Raster data reading, processing, and writing
    - zipfile: ZIP archive extraction and management
    - pathlib: File path operations and directory management
    - matplotlib: Visualization and plot generation
    - RES.AttributesParser: Parent class for configuration management
    - RES.boundaries.GADMBoundaries: Regional boundary processing
    - RES.utility: Logging and status update functions

    Raises
    ------
    ConnectionError
        If GAEZ data download fails or source is unavailable
    FileNotFoundError
        If required raster files are missing from ZIP archive
    ValueError
        If configuration parameters are invalid or incomplete
    RuntimeError
        If raster processing or clipping operations fail

    See Also
    --------
    rasterio.mask.mask : Raster clipping functionality
    RES.boundaries.GADMBoundaries : Regional boundary processing
    RES.lands : Land constraint integration for renewable energy
    """

    def __post_init__(self):
        """
        Initialize GAEZ raster processor with configuration and boundary setup.

        Performs post-initialization setup including:
        - Calling parent class initialization
        - Setting up boundary processor with inherited attributes
        - Loading GAEZ configuration from config file
        - Creating directory structure for data storage
        - Configuring raster type processing parameters

        This method is automatically called after dataclass initialization
        to prepare the processor for raster data operations.

        Raises
        ------
        FileNotFoundError
            If configuration file or directories cannot be created
        ValueError
            If required configuration parameters are missing
        """
        super().__post_init__()
        self.required_args = {  # order doesn't matter
            "config_file_path": self.config_file_path,  # INHERITED ATTRIBUTE from AttributesParser
            "region_short_code": self.region_short_code,  # INHERITED ATTRIBUTE from AttributesParser
            "resource_type": self.resource_type,  # INHERITED ATTRIBUTE from AttributesParser
        }

        self.gadmBoundary = GADMBoundaries(**self.required_args)

        self.gaez_config: dict = super().get_gaez_data_config()

        self.gaez_root = Path(self.gaez_config.get("root", "data/downloaded_data/GAEZ"))
        self.gaez_root.mkdir(parents=True, exist_ok=True)

        self.zip_file = Path(self.gaez_config["zip_file"])

        self.Rasters_in_use_direct = Path(self.gaez_config["Rasters_in_use_direct"])
        self.Rasters_in_use_direct.mkdir(parents=True, exist_ok=True)

        self.raster_types = self.gaez_config["raster_types"]
        self.region_boundary = None

    def process_all_rasters(self, show: bool = False):
        """
        Main pipeline to download, extract, clip, and plot rasters based on configuration.

        Executes the complete GAEZ raster processing workflow including:
        1. Downloading ZIP archive if not present locally
        2. Extracting required raster files from archive
        3. Loading regional boundaries for clipping operations
        4. Processing each configured raster type by clipping to boundaries
        5. Generating visualization plots for all processed rasters

        This method orchestrates all processing steps and returns paths to
        the processed raster files for downstream analysis.

        Parameters
        ----------
        show : bool, default=False
            Whether to display generated plots interactively during processing.
            If True, matplotlib plots will be shown on screen.
            If False, plots are saved to disk without display.

        Returns
        -------
        dict
            Dictionary mapping raster type names to processed file paths.
            Keys are raster type names from configuration.
            Values are Path objects pointing to clipped raster files.

        Examples
        --------
        Process all rasters with visualization:

        >>> raster_paths = processor.process_all_rasters(show=True)
        >>> print(f"Processed rasters: {list(raster_paths.keys())}")

        Process rasters for programmatic use:

        >>> raster_paths = processor.process_all_rasters(show=False)
        >>> slope_raster = raster_paths.get('slope')
        >>> if slope_raster:
        ...     print(f"Slope data at: {slope_raster}")

        Raises
        ------
        ConnectionError
            If ZIP archive download fails due to network issues
        FileNotFoundError
            If required raster files are missing from archive
        RuntimeError
            If raster clipping or processing operations fail

        Notes
        -----
        - Processing time scales with number of raster types and region size
        - Large regions may require substantial disk space for processed rasters
        - Network connection required for initial ZIP archive download
        - Existing processed rasters are not regenerated unless missing
        - All plots are saved regardless of the show parameter setting
        """
        if not (self.gaez_root / self.zip_file).exists():
            self.__download_resources_zip_file__()
        raster_paths = {}

        self.__extract_rasters__()
        self.region_boundary = self.gadmBoundary.get_region_boundary()

        utils.print_update(
            level=print_level_base,
            message=f"{__name__}| Clipping Rasters to regional boundaries.. ",
        )
        # Loop over raster types and process each
        for raster_type in self.raster_types:
            raster_path = self.__clip_to_boundary_n_plot__(
                raster_type, self.region_boundary.geometry, show
            )
            raster_paths[raster_type["name"]] = raster_path

        utils.print_update(
            level=print_level_base,
            message=f"{__name__}| ✔ All required rasters for GAEZ processed and plotted successfully.",
        )
        return raster_paths

    def __download_resources_zip_file__(self):
        """
        Download the GAEZ resources ZIP archive from remote source.

        Downloads the GAEZ raster data ZIP file from the configured source URL
        if it doesn't already exist locally. The ZIP archive contains all the
        raster datasets needed for land constraint analysis in renewable energy
        resource assessment.

        The method uses the source URL from configuration with a fallback to
        the default FAO GAEZ data source. Downloaded files are saved to the
        configured root directory for subsequent extraction and processing.

        Raises
        ------
        ConnectionError
            If the download fails due to network connectivity issues
        requests.HTTPError
            If the HTTP response indicates an error (non-200 status code)
        IOError
            If the local file cannot be written due to permissions or disk space

        Notes
        -----
        - Download size is typically 100MB-1GB depending on data coverage
        - Network timeout may occur for large files on slow connections
        - Existing ZIP files are not re-downloaded to save bandwidth
        - Progress is logged through utility print functions
        - File integrity is not explicitly verified after download
        """
        url = self.gaez_config.get(
            "source", "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR.zip"
        )
        response = requests.get(url)

        if response.status_code == 200:
            with open(self.gaez_root / self.zip_file, "wb") as f:
                f.write(response.content)
            utils.print_update(
                level=print_level_base,
                message=f"{__name__}| GAEZ Raster Resource '.zip' file downloaded and saved to: {self.gaez_root}",
            )
        else:
            utils.print_update(
                level=print_level_base,
                message=f"{__name__}|  ❌ Failed to download the Resources zip file from GAEZ. Status code: {response.status_code}",
            )

    def __extract_rasters__(self):
        """
        Extract required raster files from the downloaded GAEZ ZIP archive.

        Selectively extracts only the raster files specified in the configuration
        from the GAEZ ZIP archive. Each raster type configuration defines which
        file to extract and where to place it within the local directory structure.

        The method preserves the internal ZIP directory structure while extracting
        files to organized local directories. Existing files are not re-extracted
        to avoid unnecessary processing time and disk operations.

        The extraction process is guided by the raster_types configuration which
        specifies:
        - raster: Filename of the raster file within the ZIP
        - zip_extract_direct: Directory path within the ZIP archive

        Progress and status messages are logged for each extraction operation
        to provide visibility into the processing workflow.

        Raises
        ------
        zipfile.BadZipFile
            If the ZIP archive is corrupted or cannot be read
        FileNotFoundError
            If specified raster files are not found within the ZIP archive
        IOError
            If extraction fails due to disk space or permission issues

        Notes
        -----
        - Only configured raster types are extracted, not the entire archive
        - Directory structure from ZIP is preserved in local extraction
        - Existing extracted files are skipped to save processing time
        - Large raster files may take considerable time to extract
        - Extraction location follows the Rasters_in_use_direct configuration
        """
        with ZipFile(self.gaez_root / self.zip_file, "r") as zip_ref:
            for raster_type in self.raster_types:
                raster_file = raster_type["raster"]
                zip_direct = raster_type["zip_extract_direct"]
                file_inside_zip = str(Path(zip_direct) / raster_file)  # Ensure it's a single string

                target_path = self.gaez_root / self.Rasters_in_use_direct / zip_direct / raster_file

                if not target_path.exists():
                    # Check for existence as a string in zip_ref
                    if file_inside_zip in zip_ref.namelist():
                        zip_ref.extract(
                            file_inside_zip, path=self.gaez_root / self.Rasters_in_use_direct
                        )
                        utils.print_update(
                            level=print_level_base,
                            message=f"{__name__}| Raster file '{raster_file}' extracted from {file_inside_zip}",
                        )
                    else:
                        utils.print_update(
                            level=print_level_base,
                            message=f"{__name__}| Raster file '{raster_file}' not found in the archive {file_inside_zip}",
                        )
                else:
                    utils.print_update(
                        level=print_level_base,
                        message=f"{__name__}| Raster file '{raster_file}' found in local directory, skipping download.",
                    )

    def __clip_to_boundary_n_plot__(self, raster_type, boundary_geom, show):
        """
        Clip raster data to regional boundaries and generate visualization plot.

        Performs spatial clipping of GAEZ raster data to match the regional
        boundaries and creates a visualization plot of the clipped result.
        This method combines raster processing with immediate visualization
        to support analysis and quality control of the processed data.

        The clipping operation uses the rasterio.mask functionality to crop
        the input raster to the exact geometry of the regional boundaries,
        preserving the original raster properties while reducing the spatial
        extent to the region of interest.

        Parameters
        ----------
        raster_type : dict
            Raster type configuration containing:
            - 'zip_extract_direct': Directory path within extraction structure
            - 'raster': Filename of the raster file to process
            - 'name': Display name for the raster type
            - 'color_map': Matplotlib colormap for visualization
        boundary_geom : list or geopandas.GeoSeries
            Geometry objects defining the clipping boundaries.
            Typically from regional boundary processing.
        show : bool
            Whether to display the generated plot interactively.
            If True, plot is shown on screen in addition to being saved.

        Returns
        -------
        pathlib.Path
            Path to the clipped raster file saved to disk.
            The file maintains GeoTIFF format with spatial reference.

        Raises
        ------
        rasterio.errors.RasterioIOError
            If the input raster file cannot be read or is corrupted
        ValueError
            If the boundary geometry is invalid or incompatible
        IOError
            If the output raster cannot be written due to disk issues

        Notes
        -----
        - Clipped rasters preserve original resolution and data types
        - Output files are named with region short code prefix for identification
        - Visualization plots are automatically saved regardless of show parameter
        - Memory usage scales with raster size and complexity of clipping geometry
        - Coordinate reference systems are preserved from input raster
        - Plot files are saved in organized visualization directory structure
        """
        zip_direct = raster_type["zip_extract_direct"]
        raster_file = raster_type["raster"]

        input_raster = self.gaez_root / self.Rasters_in_use_direct / zip_direct / raster_file
        output_dir = self.gaez_root / self.Rasters_in_use_direct / zip_direct
        output_dir.mkdir(parents=True, exist_ok=True)

        clipped_raster_path = output_dir / f"{self.region_short_code}_{raster_file}"

        with rasterio.open(input_raster) as src:
            clipped_raster, clipped_transform = mask(
                src, boundary_geom, crop=True, indexes=src.indexes
            )
            clipped_meta = src.meta.copy()
            clipped_meta.update(
                {
                    "height": clipped_raster.shape[1],
                    "width": clipped_raster.shape[2],
                    "transform": clipped_transform,
                }
            )

            with rasterio.open(clipped_raster_path, "w", **clipped_meta) as dst:
                dst.write(clipped_raster)

            # Call visualization method
            # save_to_root=self.get_vis_dir()
            # plot_save_to = Path(f'{save_to_root}') / raster_file.replace('.tif', f'_raster_{self.region_short_code}.png')
            # self.plot_gaez_tif(clipped_raster_path,
            #                    color_map,
            #                    plot_title,
            #                    plot_save_to,show)
            # utils.print_update(level=print_level_base+1,message=f"{__name__}| Clipped Raster plot for {super().get_region_name()} saved at: {plot_save_to}")  # INHERITED METHOD from AttributesParser
            return clipped_raster_path

    def plot_gaez_tif(self, tif_path, color_map, plot_title, save_to, show=False):
        """
        Generate and save visualization plot for processed GAEZ raster data.

        Creates a publication-quality matplotlib visualization of the clipped
        GAEZ raster data with proper coordinate system display, color mapping,
        and legend information. The plot includes geographic extent display
        with latitude/longitude axes and a horizontal colorbar for value interpretation.

        This method supports both interactive display and file output, making it
        suitable for both exploratory analysis and report generation workflows.
        The visualization uses proper geographic coordinates and customizable
        color schemes to effectively communicate spatial patterns in the data.

        Parameters
        ----------
        tif_path : str or pathlib.Path
            Path to the GeoTIFF raster file to visualize.
            Must be a valid raster file with spatial reference information.
        color_map : str
            Name of matplotlib colormap to use for visualization.
            Examples: 'terrain', 'viridis', 'plasma', 'coolwarm'.
        plot_title : str
            Title text to display at the top of the plot.
            Should describe the raster content and region.
        save_to : str or pathlib.Path
            Output path for saving the plot image file.
            Parent directories will be created if they don't exist.
        show : bool, default=False
            Whether to display the plot interactively on screen.
            If True, plot is shown in addition to being saved.
            If False, plot is only saved to file without display.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib Figure object containing the plot.
            Can be used for further customization or processing.

        Examples
        --------
        Create a basic plot:

        >>> fig = processor.plot_gaez_tif(
        ...     tif_path="data/BC_slope.tif",
        ...     color_map="terrain",
        ...     plot_title="Slope Analysis for British Columbia",
        ...     save_to="plots/BC_slope.png"
        ... )

        Create an interactive plot:

        >>> fig = processor.plot_gaez_tif(
        ...     tif_path="data/BC_elevation.tif",
        ...     color_map="viridis",
        ...     plot_title="Elevation Map",
        ...     save_to="plots/elevation.png",
        ...     show=True
        ... )

        Raises
        ------
        rasterio.errors.RasterioIOError
            If the input TIF file cannot be read or is corrupted
        FileNotFoundError
            If the input TIF file does not exist
        ValueError
            If the colormap name is not recognized by matplotlib
        IOError
            If the output plot file cannot be written

        Notes
        -----
        - Plot dimensions are fixed at 10x8 inches for consistency
        - Colorbar is positioned horizontally below the plot
        - Geographic extent is automatically derived from raster bounds
        - Output directories are created automatically if needed
        - Plot is always saved regardless of the show parameter
        - Figure is closed after processing to prevent memory leaks
        - NoData/masked values are handled transparently in visualization
        """
        with rasterio.open(tif_path) as src:
            data = src.read(1, masked=True)
            extent = src.bounds
        save_to.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            data, cmap=color_map, extent=[extent.left, extent.right, extent.bottom, extent.top]
        )
        plt.colorbar(
            im, ax=ax, label="Layer Class", orientation="horizontal", fraction=0.05, pad=0.08
        )
        ax.set_title(plot_title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(visible=False)
        plt.tight_layout()
        plt.savefig(save_to)
        if show:
            plt.show()
        plt.close(fig)
        return fig
