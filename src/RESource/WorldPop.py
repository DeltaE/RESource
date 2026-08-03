# Downloads population data from WorldPop

from pathlib import Path

import pandas as pd

import RESource.utility as utils
from RESource.AttributesParser import AttributesParser
from RESource.boundaries import GADMBoundaries


class WorldPop:
    """Population data processor using WorldPop global datasets.

    Handles downloading and processing of population density data from WorldPop
    for renewable energy assessment and demographic analysis. Integrates with
    regional boundaries to provide population context for energy infrastructure
    planning and environmental impact assessment.

    Parameters
    ----------
    config_file_path : Path
        Path to configuration file containing WorldPop data sources
    region_short_code : str
        Region identifier for boundary definition

    Attributes
    ----------
    config : dict
        Complete configuration dictionary
    worldpop_config : dict
        WorldPop-specific configuration parameters
    root : str
        Root directory for population data storage
    """

    def __init__(self, config_file_path: Path, region_short_code: str):
        """Initialize WorldPop data processor.

        Args:
            config_file_path: Path to configuration file
            region_short_code: Region identifier code
        """
        self.config_file_path = config_file_path
        self.region_short_code = region_short_code

        self.attributes_parser: AttributesParser = AttributesParser(self.config_file_path, None)
        self.gadm = GADMBoundaries(self.config_file_path, self.region_short_code)

        self.config = self.attributes_parser.config
        self.worldpop_config = self.config["WorldPop"]
        self.root = self.worldpop_config["root"]

    def pull_data(self, data_name: str):
        """Download population data from WorldPop sources.

        Args:
            data_name: Name of dataset to download from configuration sources
        """

        data_names = list(self.worldpop_config["source"].keys())

        if data_name in data_names:
            url = Path(self.worldpop_config["source"][data_name])
            # Extract the filename from the URL
            filename = Path(url).name
            # Construct the full save path by combining base path and extracted filename
            file_path = Path(self.root) / filename

            # Download the file using the utils.download_data function
            utils.download_data(url, file_path)

            # >>>>> files are downloaded as zip ! create a zip extractor
            self.pop_data: pd.DataFrame = pd.read_csv(file_path)

            print(f"File saved to: {file_path}")
        else:
            print(f"{data_name} associated source information not found in user config. \n")
            print("Please provide required information in user config under 'WorldPop' key.")
            print(f"Available 'data_name' in user config is {data_names}")

    def get_provincial_data(self, data_name: str):
        """Extract regional population data within administrative boundaries.

        Args:
            data_name: Name of population dataset to process
        """
        region_gadm_gdf = self.gadm.get_region_boundary()

        pop_grid = self.pop_data.overlay(region_gadm_gdf, how="intersection", keep_geom_type=True)

        pop_grid.to_pickle(f"data/downloaded_data/WorldPop/pop_{self.region_short_code}.pkl")
