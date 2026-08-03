from pathlib import Path

import geopandas as gpd
import osmnx as ox

import RESource.utility as utils
from RESource.AttributesParser import AttributesParser

print_level_base = 3

ox.settings.max_query_area_size = 10_000 * 1e6  # 10,000 sq km


class OSMData(AttributesParser):
    """
    OpenStreetMap data processor for extracting and managing geospatial infrastructure data.

    This class inherits from AttributesParser and provides functionality to:
    - Download and cache OSM data for specific geographic regions
    - Process tagged OSM features (e.g., power lines, airports, railways)
    - Store and retrieve geospatial data as GeoDataFrames
    - Save OSM data locally as GeoJSON files for efficient reuse

    Inherits from:
        AttributesParser: Base class providing configuration management and regional attributes

    Attributes:
        osm_data_config (dict): Configuration for OSM data keys and storage paths
        data_keys (dict): Mapping of data keys to their corresponding OSM tags
        root_path (Path): Root directory for storing OSM data files
        area_name (str): Formatted area name for OSM queries (region, country)
        gdfs (dict): Cache of loaded GeoDataFrames by data key

    Key Methods:
        get_osm_layer(): Retrieve or load OSM data for a specific data key
        run(): Process all configured OSM data keys

    Example:
        >>> osm_processor = OSMData(**config)
        >>> power_lines = osm_processor.get_osm_layer('power')
        >>> all_data = osm_processor.run()
    """

    def __post_init__(self):

        super().__post_init__()

        # Load OSM-specific configurations
        self.osm_data_config = self.get_osm_config()

        # Extract data keys and root path from configuration
        self.data_keys: dict = {
            key: value["tags"] for key, value in self.osm_data_config["data_keys"].items()
        }
        self.root_path: Path = Path(self.osm_data_config["root"])

        # Create the directory (and any necessary parent directories) if it doesn't exist
        self.root_path.mkdir(parents=True, exist_ok=True)

        # Format area name for OSM queries
        if self.multi_country_flag:
            self.area_name: str = f"{self.get_region_name()}"
        else:
            self.area_name: str = f"{self.get_region_name()}, {self.get_country()}"

        # Dictionary to store GeoDataFrames by data_key
        self.gdfs: dict = {}

    def get_osm_layer(self, data_key: str) -> gpd.GeoDataFrame:
        """
        Access or load the GeoDataFrame for a specific OSM data key.

        This method implements a caching mechanism for OSM data retrieval:
        1. Checks if data is already loaded in memory (self.gdfs)
        2. If not cached, attempts to load from local storage or download from OSM
        3. Validates the data_key against configured OSM data keys
        4. Returns the GeoDataFrame or None if data_key is invalid

        Args:
            data_key (str): The configuration key identifying the type of OSM data
                          (e.g., 'power', 'aeroway', 'railway', 'highway')

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the requested OSM features,
                            or None if data_key is not configured

        Raises:
            OSMNetworkError: If OSM data download fails
            FileNotFoundError: If local file is corrupted or missing

        Example:
            >>> power_lines = self.get_osm_layer('power')
            >>> airports = self.get_osm_layer('aeroway')
        """
        utils.print_update(
            level=print_level_base + 1,
            message=f"{__name__}| processing Aeroways data for {self.region_short_code}",
        )

        if data_key in self.gdfs:
            utils.print_update(
                level=print_level_base + 1,
                message=f"{__name__}|GeoDataFrame for '{data_key}' already exists, returning it.",
            )
            return self.gdfs[data_key]

        # Load the data if it doesn't exist in memory
        if data_key in self.data_keys:
            gdf = self.__load_tagged_data_from_OSM__(self.data_keys[data_key], data_key)
            self.gdfs[data_key] = gdf  # Cache for later use
            return gdf
        else:
            utils.print_update(
                level=print_level_base + 1,
                message=f"{__name__}|  ❌{data_key}' is not a valid key in the configuration.",
            )
            return None

    def run(self) -> dict:
        """
        Run the OSM data retrieval process for all configured data keys.

        This method processes all data keys defined in the OSM configuration,
        downloading and caching OSM data for each type. It serves as the main
        entry point for bulk OSM data processing.

        Process:
        1. Iterates through all configured data keys
        2. Calls get_osm_layer() for each data key
        3. Stores results in self.gdfs cache
        4. Returns complete dictionary of loaded GeoDataFrames

        Returns:
            dict: Dictionary mapping data keys to their corresponding GeoDataFrames
                 Keys are strings (e.g., 'power', 'aeroway')
                 Values are gpd.GeoDataFrame objects

        Raises:
            OSMNetworkError: If any OSM downloads fail
            ConfigurationError: If data keys configuration is invalid

        Example:
            >>> all_osm_data = osm_processor.run()
            >>> power_gdf = all_osm_data['power']
            >>> aeroway_gdf = all_osm_data['aeroway']
        """
        for data_key in self.data_keys.keys():
            print(f"Processing OSM data for key: {data_key}")
            self.get_osm_layer(data_key)
        return self.gdfs

    def __load_tagged_data_from_OSM__(self, tags: dict, data_key: str) -> gpd.GeoDataFrame:
        """
        Retrieve and cache OSM data for the specified area and tags.

        This private method handles the core OSM data retrieval logic:
        1. Constructs local file path for caching
        2. Checks for existing local data to avoid redundant downloads
        3. Downloads fresh data from OSM API if needed
        4. Saves downloaded data locally for future use
        5. Returns the GeoDataFrame with OSM features

        Args:
            tags (dict): OSM tags dictionary specifying features to extract
                        (e.g., {'power': ['line', 'substation']})
            data_key (str): Configuration key identifying the data type

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing OSM features matching the tags

        Raises:
            OSMNetworkError: If OSM API request fails or times out
            GeometryError: If downloaded OSM data has invalid geometries
            FileSystemError: If local file operations fail

        Note:
            This method uses the osmnx library for OSM data retrieval and
            implements local caching to minimize API calls and improve performance.
        """
        geojson_path = self.root_path / f"{self.region_short_code}_{data_key}.geojson"
        tags_dict = {data_key: tags}

        # Check if data is already stored locally
        if geojson_path.exists():
            utils.print_update(
                level=print_level_base + 1,
                message=f"{__name__}| Loading locally stored OSM data for '{data_key}' from {geojson_path}",
            )
            gdf = gpd.read_file(geojson_path)
            return gdf
        else:
            print(
                f">> Downloading data for {self.area_name} with tags {tags} and saving to {geojson_path}"
            )
            gdf = ox.features_from_place(self.area_name, tags_dict)
            self.__save_local_file__(gdf, geojson_path)
            return gdf

    def __save_local_file__(self, gdf: gpd.GeoDataFrame, geojson_path: Path):
        """
        Save the GeoDataFrame to a local GeoJSON file with collision prevention.

        This private method handles local file storage for OSM data caching:
        1. Checks if file already exists to prevent overwriting
        2. Saves GeoDataFrame in GeoJSON format for portability
        3. Ensures data persistence for future sessions

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing OSM features to save
            geojson_path (Path): Full path where the GeoJSON file should be saved

        Raises:
            PermissionError: If unable to write to the specified path
            DiskSpaceError: If insufficient disk space for file writing
            SerializationError: If GeoDataFrame cannot be converted to GeoJSON

        Note:
            The method skips saving if the file already exists, preventing
            accidental data overwrites and reducing unnecessary disk I/O.
        """

        if not geojson_path.exists():
            utils.print_update(
                level=print_level_base + 1, message=f"{__name__}| Saving OSM data to {geojson_path}"
            )
            gdf.to_file(geojson_path, driver="GeoJSON")
        else:
            utils.print_update(
                level=print_level_base + 1,
                message=f"{__name__}| File {geojson_path} already exists, skipping save.",
            )
