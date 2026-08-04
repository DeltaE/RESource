# for CANADian power system data only.

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from RESource import utility as utils
from RESource.AttributesParser import AttributesParser

PRINT_LEVEL_BASE = 3

default_coders_cfg_file_path = "credentials/coders_api.yaml"
CODERS_CREDENTIALS_SOURCE = (
    "https://github.com/eliasinul/modeling_inventory/blob/main/PyPSA/coders_api.yaml"
)
DEFAULT_CONNECTION_NODE_TYPES = ("Generation", "Terminal")
DEFAULT_EXCLUDED_NODE_SUFFIXES = ("INT", "IPT", "SWS", "JCT")


def load_api_key(file_path=default_coders_cfg_file_path) -> tuple[str | None, str | None]:
    """Load the first usable CODERS API key from a local credential file.

    Args:
        file_path: YAML file containing an ``api_keys`` list. Older mappings are
            accepted for backward compatibility.

    Returns:
        A tuple containing the API key and a non-secret key label, or
        ``(None, None)`` when no usable key is available.
    """
    try:
        api_cfg = utils.load_config(file_path)
        if api_cfg is None:
            utils.print_update(
                level=1,
                message=f"API key file is empty or could not be loaded: {file_path}",
                alert=True,
            )
            utils.print_update(
                level=2,
                message="Please create a YAML file at the above path with the following structure:",
            )
            utils.print_update(
                level=2,
                message="""
        api_keys:
          - your_api_key_here  # optional local note
            """,
            )
            utils.print_update(level=2, message=f"save the file to : {file_path} and try again.")
            utils.print_update(
                level=2, message="Refer to the CODERS API setup guide for more details."
            )
            utils.print_update(
                level=2,
                message=f"Authorized credential source: {CODERS_CREDENTIALS_SOURCE}",
            )
            return None, None
    except FileNotFoundError:
        utils.print_update(level=1, message=f"API key file not found: {file_path}")
        utils.print_update(
            level=2,
            message="Please create a YAML file at the above path with the following structure:",
        )
        utils.print_update(
            level=2,
            message="""
        api_keys:
          - your_api_key_here  # optional local note
            """,
        )
        utils.print_update(level=1, message="Refer to the CODERS API setup guide for more details.")
        utils.print_update(
            level=1,
            message=f"Authorized credential source: {CODERS_CREDENTIALS_SOURCE}",
        )
        return None, None

    api_keys = api_cfg.get("api_keys", [])

    if isinstance(api_keys, list):
        for index, key in enumerate(api_keys, start=1):
            if isinstance(key, str) and key.strip():
                return key.strip(), f"key_{index}"
        return None, None

    if not isinstance(api_keys, dict):
        return None, None

    # Backward compatibility with the former username-to-key mapping schema.
    default_user = api_cfg.get("Default_user")
    if default_user:
        api_key = api_keys.get(default_user)
        if api_key:
            return api_key, default_user

    # Fallback: try any other key in the legacy mapping.
    for user, key in api_keys.items():
        if key:
            return key, user

    return None, None


@dataclass
class CODERSData(AttributesParser):
    """
    Canadian power system data processor using the CODERS API.

    This class provides comprehensive access to Canadian power system infrastructure
    data through the CODERS (Canadian Open Data Exchange for Renewable Energy Systems)
    API. It enables retrieval, caching, and processing of transmission lines,
    substations, generators, and other power system components for renewable energy
    integration analysis.

    Key Functionality:
    - API-based data retrieval from CODERS database
    - Local data caching and persistence for improved performance
    - Provincial and national data filtering capabilities
    - Geographic data processing with GeoDataFrame support
    - Data validation and error handling for API operations

    Data Sources Available:
    - Power generation facilities (generators)
    - Transmission infrastructure (lines, substations)
    - Regional power system characteristics
    - Provincial energy system data

    Inherits from:
        AttributesParser: Base class providing configuration management and regional attributes

    Attributes:
        coders_data_config (dict): CODERS-specific configuration parameters
        url (str): Base URL for CODERS API endpoints
        api_user (str): API authentication key for CODERS access
        query (str): Formatted query string with authentication
        data_pull (dict): Configuration for data retrieval and storage
        table_list (list): Available data tables from configuration
        region_data (pd.DataFrame/gpd.GeoDataFrame): Filtered regional data

    API Requirements:
        - Valid CODERS API key (stored in coders_api.yaml)
        - Network connectivity for data retrieval
        - Proper authentication configuration

    Example:
        >>> coders = CODERSData(
        ...     config_file_path="config/config_BC.yaml",
        ...     region_short_code="BC"
        ... )
        >>>
        >>> # Get provincial transmission data
        >>> bc_substations = coders.get_table_provincial('substations')
        >>>
        >>> # Get national generator data with forced update
        >>> generators_df, generators_gdf = coders.get_table_canada(
        ...     'generators',
        ...     force_update=True
        ... )

    Data Persistence:
        - Automatic local caching reduces API calls
        - Pickle format for efficient data storage
        - Configurable data refresh policies
        - Regional data filtering and storage

    Notes:
        - Requires active internet connection for initial data retrieval
        - API rate limits may apply for excessive requests
        - Local data cache improves performance for repeated analyses
        - Geographic data automatically converted to EPSG:4326 projection

    References:
        - CODERS API: http://api.sesit.ca
        - Canadian power system data standards and formats
    """

    def __post_init__(self):
        """
        Initialize inherited attributes and CODERS API configuration.

        This method:
        1. Calls parent __post_init__ to inherit configuration and regional attributes
        2. Loads CODERS-specific configuration from config files
        3. Sets up API authentication and connection parameters
        4. Initializes data retrieval and storage configuration
        5. Prepares table list and query formatting

        Inherited attributes from AttributesParser:
        - Configuration file parsing and validation
        - Regional identification (region_short_code, region_code_validity)
        - Data storage paths and directory management

        CODERS Configuration:
        - API endpoint URLs and authentication
        - Data table specifications and requirements
        - Local storage paths and file naming conventions
        - Regional filtering and validation parameters

        Raises:
            ConfigurationError: If CODERS configuration is missing or invalid
            AuthenticationError: If API key is not properly configured
            NetworkError: If API connectivity cannot be established
        """

        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()

        self.coders_data_config = self.config.get("infrastructure", {}).get("CODERS", {})
        credentials_path = self.coders_data_config.get(
            "credentials_path", default_coders_cfg_file_path
        )
        api_key, user = load_api_key(credentials_path)
        if api_key is None:
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message="No API key found. Please ensure you have a valid API key in the configuration file.",
                alert=True,
            )
        else:
            utils.print_update(level=2, message=f"CODERS API key loaded from: {credentials_path}")
            utils.print_update(level=3, message=f"CODERS credentials loaded for user: {user}")

        # Load CODERS data config
        self.url = self.coders_data_config.get("url_1", "")
        self.api_user = api_key

        self.query = f"?key={self.api_user}"
        self.data_pull = self.coders_data_config.get("data_pull", {})
        self.table_list = list(self.coders_data_config["data_pull"].keys())

    def is_table_name_required(self, table_name: str):
        """
        Validate if a specified table name is configured and required for analysis.

        This method checks whether a requested data table is included in the
        configured list of required tables for the current analysis. It serves
        as a validation gate to prevent unnecessary API calls and ensure only
        relevant data is processed.

        Args:
            table_name (str): Name of the data table to validate
                            (e.g., 'generators', 'substations', 'transmission_lines')

        Returns:
            bool: True if table is configured and required, False otherwise

        Example:
            >>> coders = CODERSData(**config)
            >>> if coders.is_table_name_required('generators'):
            ...     data = coders.get_table_provincial('generators')
        """
        if table_name in self.table_list:
            return True

    def show_list(self, source: str = "cef") -> list:
        """
        Fetch and display available data tables from the CODERS API for a specified source.

        This method queries the CODERS API to retrieve and display the complete list
        of available data tables for a given data source. It provides users with
        an inventory of accessible datasets and helps identify appropriate table
        names for data retrieval operations.

        Data Sources:
        - 'cef': Canadian Energy Facts data tables
        - 'coders': Core CODERS power system infrastructure tables

        Args:
            source (str, optional): Data source identifier. Defaults to "cef".
                                  Valid options: 'cef', 'coders'

        Returns:
            list: List of available table names for the specified source.
                 Returns empty list if API request fails.

        Raises:
            RuntimeError: If API returns non-200 status code
            requests.RequestException: If network connectivity issues occur

        Example:
            >>> coders = CODERSData(**config)
            >>>
            >>> # List Canadian Energy Facts tables
            >>> cef_tables = coders.show_list('cef')
            >>> print(f"Available CEF tables: {cef_tables}")
            >>>
            >>> # List core CODERS infrastructure tables
            >>> coders_tables = coders.show_list('coders')
            >>> print(f"Available CODERS tables: {coders_tables}")

        Notes:
            - Requires active internet connection and valid API authentication
            - Table availability may vary based on data source updates
            - Use returned table names for subsequent data retrieval calls
        """
        print(f">> Fetching the list of data tables from {source}")
        try:
            response = requests.get(f"{self.url}/tables/{source}{self.query}")
            if response.status_code == 200:
                tables_list = response.json()
                print(f"{source.upper()} data available:\n {tables_list}")
                return tables_list
            else:
                raise RuntimeError(
                    f">> Error fetching tables list for {source}: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            print(f">> Connection error while fetching tables list: {e}")
            return []

    def fetch_data(self, table_name: str) -> pd.DataFrame:
        """
        Retrieve data from the CODERS API for a specified table.

        This method performs direct API calls to fetch power system data from
        the CODERS database. It handles HTTP requests, response validation,
        and data format conversion to return structured pandas DataFrames
        suitable for analysis.

        Args:
            table_name (str): Name of the data table to retrieve from CODERS API
                            (e.g., 'generators', 'substations', 'transmission_lines')

        Returns:
            pd.DataFrame: Structured data from the specified CODERS table

        Raises:
            RuntimeError: If API returns non-200 status code or request fails
            requests.RequestException: If network connectivity issues occur
            JSONDecodeError: If API response cannot be parsed as valid JSON

        Example:
            >>> coders = CODERSData(**config)
            >>> generators_data = coders.fetch_data('generators')
            >>> print(f"Retrieved {len(generators_data)} generator records")

        Notes:
            - Requires valid API authentication and network connectivity
            - Raw data retrieval without local caching or persistence
            - Use get_table_canada() or get_table_provincial() for cached access
            - Response data automatically converted to pandas DataFrame format
        """
        response = requests.get(f"{self.url}/{table_name}{self.query}")

        if response.status_code == 200:
            return pd.DataFrame.from_dict(response.json())
        else:
            raise RuntimeError(f">> Error fetching data for {table_name}: {response.status_code}")

    def load_local_data(self, table_name: str, region_code: str = None) -> pd.DataFrame:
        """
        Load data from a local file if it exists.

        ### Args:
            If region set to NONE, loads Country data.
        """

        file_name = f"{table_name}.pkl" if region_code is None else f"generators_{region_code}.pkl"
        file_path = Path(self.data_pull["root"]) / self.data_pull.get(table_name) / file_name
        file_path.mkdir(parents=True, exist_ok=True)  # Creates parent directories if not exists.

        if file_path.is_file():
            utils.print_update(
                level=PRINT_LEVEL_BASE, message=f">> Loading data from local file: {file_path}"
            )
            return pd.read_pickle(file_path)
        else:
            utils.print_warning(f">> No local file found at: {file_path}")
            return None  # Return None if the file does not exist

    def save_data(
        self, data: pd.DataFrame | gpd.GeoDataFrame, table_name: str, region_code: str = None
    ):
        """Save the fetched data to a pkl file."""
        file_name = f"{table_name}.pkl" if region_code is None else f"generators_{region_code}.pkl"
        file_path = Path(self.data_pull["root"]) / self.data_pull.get(table_name) / file_name

        data.to_pickle(file_path)
        utils.print_update(
            level=PRINT_LEVEL_BASE, message=f"{table_name} data saved to:\n {file_path}"
        )

    def create_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame from the given DataFrame."""
        df = df.copy()
        # Create a geometry column
        df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)

        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        return gdf

    def filter_connection_substations(
        self,
        substations: gpd.GeoDataFrame,
        transmission_lines: pd.DataFrame | None = None,
    ) -> gpd.GeoDataFrame:
        """Filter CODERS substations to configured generation-connection candidates."""
        filter_config = self.coders_data_config.get("connection_filter", {})
        if not filter_config.get("enabled", True):
            return substations.copy()

        eligible_types = filter_config.get(
            "eligible_node_types", list(DEFAULT_CONNECTION_NODE_TYPES)
        )
        if "node_type" not in substations.columns:
            raise ValueError("CODERS substations are missing the required 'node_type' column")
        filtered = substations[substations["node_type"].isin(eligible_types)].copy()

        excluded_suffixes = set(
            filter_config.get("excluded_node_suffixes", list(DEFAULT_EXCLUDED_NODE_SUFFIXES))
        )
        if excluded_suffixes:
            if "node_code" not in filtered.columns:
                raise ValueError("CODERS substations are missing the required 'node_code' column")
            node_suffixes = filtered["node_code"].astype("string").str.rsplit("_", n=1).str[-1]
            filtered = filtered[~node_suffixes.isin(excluded_suffixes)].copy()

        if filter_config.get("require_transmission_endpoint", True):
            if transmission_lines is None:
                raise ValueError("CODERS transmission lines are required by connection_filter")
            endpoint_columns = {
                "network_node_code_starting",
                "network_node_code_ending",
            }
            missing_columns = endpoint_columns.difference(transmission_lines.columns)
            if missing_columns:
                raise ValueError(
                    "CODERS transmission lines are missing endpoint columns: "
                    + ", ".join(sorted(missing_columns))
                )
            endpoint_codes = pd.concat(
                [transmission_lines[column] for column in sorted(endpoint_columns)],
                ignore_index=True,
            ).dropna()
            filtered = filtered[filtered["node_code"].isin(endpoint_codes)].copy()

        utils.print_update(
            level=PRINT_LEVEL_BASE,
            message=(
                f"CODERS connection filter retained {len(filtered)}/{len(substations)} "
                f"substations (node types: {', '.join(eligible_types)}; excluded suffixes: "
                f"{', '.join(sorted(excluded_suffixes)) or 'none'})."
            ),
        )
        return filtered

    def get_table_canada(self, table_name: str, force_update: bool = False):
        """Get generator data for all of Canada.

        Args:
            table_name (str): The name of the table to fetch data from.
            force_update (bool): If True, force a data fetch from the API, ignoring local data.

        Returns:
            Tuple[pd.DataFrame, gpd.GeoDataFrame]: The generator data as a DataFrame and as a GeoDataFrame.
        """

        file_path = Path(self.data_pull["root"]) / self.data_pull.get(
            table_name, f"{table_name}.pkl"
        )
        file_path.mkdir(parents=True, exist_ok=True)  # Creates parent directories if not exists.

        # Check if the data file exists locally and if force_update is not set
        if file_path.is_file() and not force_update:
            data = pd.read_pickle(file_path)  # Load from local CSV
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f"Loaded {table_name} data from local file: {file_path}",
            )
        else:
            # Fetch data from API if not found locally or if force_update is set
            data = self.fetch_data(table_name)
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f">> Data pulled {table_name} from [source checked: CODERS({self.url})]",
            )
            self.save_data(data, table_name)

        df = data

        # Check if table_name contains "lines"; if it does, skip creating the GeoDataFrame
        if "lines" not in table_name:
            gdf = self.create_gdf(data)  # Only create GeoDataFrame if "lines" is not in table_name
        else:
            gdf = gpd.GeoDataFrame()  # Or however you wish to handle this case
        return df, gdf

    def get_table_provincial(self, table_name, force_update: bool = False):
        """Get generator data for a specific province.

        Args:
            table_name (str): The name of the table to fetch data from e.g. 'substations','transmission_lines','generators' etc.
            force_update (bool): If True, force a data fetch from the API, ignoring local data.
        """
        if self.is_table_name_required(table_name):  # check if the data is required
            if self.region_code_validity:
                # Get Canadian data first
                df, gdf = self.get_table_canada(table_name, force_update=force_update)

            if "lines" not in table_name:
                # Apply provincial mask
                data = gdf
            else:
                data = df

            region_mask = data["province"] == self.region_short_code
            self.region_data = data[region_mask]

            if not self.region_data.empty:
                return self.region_data  # Return the filtered GeoDataFrame
            else:
                return self.region_code_validity
        else:
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f"Table: '{table_name}' is not required for this tool and is not configured to work properly.\n Configured/required tables >>>> {self.table_list[1:]}",
            )
