from pathlib import Path

import yaml
from requests import get

from RESource import utility as utils

PRINT_LEVEL_BASE = 3


# * Check 'windpowerlib' package
class OEDBTurbines:
    """Wind turbine configuration manager for OEDB (Open Energy Database) integration.

    Handles fetching, processing, and caching of wind turbine technical specifications
    from the Open Energy Database. Provides standardized turbine configurations for
    renewable energy modeling including power curves, hub heights, and rotor specifications.

    Parameters
    ----------
    OEDB_config : dict
        Configuration dictionary containing OEDB connection parameters,
        turbine model specifications, and data source information

    Attributes
    ----------
    OEDB_config : dict
        Configuration parameters for OEDB connection and turbine selection
    turbine_name : str
        Name of the selected turbine model
    turbine_config_dir : Path
        Directory for storing downloaded turbine configuration files
    turbine_config_file : Path
        Path to the specific turbine configuration YAML file
    """

    def __init__(self, OEDB_config: dict):
        """Initialize OEDB turbine configuration manager.

        Args:
            OEDB_config: Configuration dictionary containing turbine model
                        specifications and OEDB connection parameters
        """
        self.OEDB_config = OEDB_config

    def load_turbine_config(self):
        """Load turbine configuration from cached YAML file.

        Returns:
            dict: Turbine configuration with technical specifications
                  including power curve, hub height, and rotor diameter
        """
        # Read the YAML file into a dictionary
        with open(self.turbine_config_file) as file:
            turbine_config: dict = yaml.safe_load(file)
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f">> selected Wind Turbine  Model : {turbine_config['name']} @ {turbine_config['hub_height']}m Hub Height",
            )
            return turbine_config

    def fetch_turbine_config(self, model):
        """Fetch turbine configuration from OEDB and cache locally.

        Retrieves turbine technical specifications from the Open Energy Database
        and formats them for use in renewable energy modeling. Implements local
        caching to avoid repeated downloads.

        Args:
            model: Turbine model identifier corresponding to OEDB configuration

        Returns:
            dict: Formatted turbine configuration with power curve and specifications,
                  or None if fetch/processing fails
        """
        OEDB_id = self.OEDB_config["models"][model]["ID"]
        OEDB_source = self.OEDB_config["source"]
        self.turbine_name = self.OEDB_config["models"][model]["name"]

        # Define the directory and file path
        self.turbine_config_dir = Path("data/downloaded_data") / "OEDB"
        self.turbine_config_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        self.turbine_config_file = self.turbine_config_dir / f"{self.turbine_name}.yaml"

        utils.print_update(
            level=PRINT_LEVEL_BASE,
            message=f"Fetching Turbine: '{self.turbine_name}' data from OEDB",
        )

        # Check if the turbine config file already exists
        if self.turbine_config_file.exists():
            if self.turbine_config_file.is_file():
                utils.print_update(
                    level=PRINT_LEVEL_BASE,
                    message=f">> Loading turbine config from: {self.turbine_config_file}",
                )
                return self.load_turbine_config()
            else:
                utils.print_update(
                    level=PRINT_LEVEL_BASE,
                    message=f">> Expected a file but found a directory: {self.turbine_config_file}",
                )
                return None

        else:
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f">> Fetching turbine config for: {self.turbine_name} from OEDB",
            )
            try:
                OEDB_data = get(OEDB_source).json()
            except Exception as e:
                utils.print_update(
                    level=PRINT_LEVEL_BASE,
                    message=f">> !! Failed to fetch OEDB data from {OEDB_source}: {e}",
                )
                return None

        # Process fetched data
        turbine_data = self.__get_required_turbines__(OEDB_data, "id", OEDB_id)
        if turbine_data:
            try:
                self.format_and_save_turbine_config(turbine_data, self.turbine_config_file)
                return self.load_turbine_config()
            except Exception as e:
                utils.print_update(
                    level=PRINT_LEVEL_BASE, message=f">> !! Error saving turbine configuration: {e}"
                )
                return None
        else:
            utils.print_update(
                level=PRINT_LEVEL_BASE, message=f">> !! No data found for turbine ID {OEDB_id}"
            )
            return None

    def __get_required_turbines__(self, OEDB_data: dict, key: str, value: str):
        """Search for specific turbine configuration in OEDB dataset.

        Args:
            OEDB_data: Complete turbine database from OEDB
            key: Field name to search by (typically 'id')
            value: Target value to match

        Returns:
            dict: Matching turbine configuration or None if not found
        """
        for entry in OEDB_data:
            if entry.get(key) == value:
                return entry
        return None  # Return None if no match is found

    def format_and_save_turbine_config(self, turbine_data: dict, save_to: str):
        """
        Formats (to sync Atlite's Requirement) and saves the turbine's specification data to a YAML configuration file.

        Args:
            - turbine_data (dict): Turbine specification data.
            - save_to (str): The directory path where the YAML file will be saved.
        """
        # Extracted information
        name = turbine_data["name"]
        manufacturer = turbine_data["manufacturer"]
        source = turbine_data["source"]
        hub_heights = list(map(float, turbine_data["hub_height"].split(";")))
        power_curve_wind_speeds = eval(turbine_data["power_curve_wind_speeds"])
        power_curve_values = eval(turbine_data["power_curve_values"])  # kW
        power_curve_values = [value / 1000 for value in power_curve_values]  # Convert to MW

        nominal_power = turbine_data["nominal_power"] / 1000  # Convert to MW

        # Create a dictionary for YAML output
        formatted_data = {
            "name": name,
            "manufacturer": manufacturer,
            "hub_height": hub_heights[0],
            "V": power_curve_wind_speeds,
            "POW": power_curve_values,
            "source": source,
            "P": nominal_power,
        }

        # Save formatted data to YAML
        self.__create_blank_yaml__(self.turbine_config_file)
        with open(self.turbine_config_file, "a") as file:
            yaml.dump(formatted_data, file, default_flow_style=False)

        utils.print_update(
            level=PRINT_LEVEL_BASE,
            message=f">> {self.turbine_name} turbine config saved to '{self.turbine_config_file}'",
        )

    def __create_blank_yaml__(self, filepath: Path):
        """
        Create a blank YAML file.

        Parameters:
        filepath (str): Path to the file.
        """

        with open(filepath, "w") as file:
            file.write("")
