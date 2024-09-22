import time
import argparse
import os
import pandas as pd
import geopandas as gpd
from requests import get
from pyrosm import OSM, get_data

# Import local packages
from linkingtool import linking_utility as utils
from linkingtool import linking_vis as vis
from linkingtool import linking_solar as solar
from linkingtool import linking_wind as wind
from linkingtool import linking_data as dataprep
from linkingtool import cell_capacity_processor
from linkingtool.attributes_parser import AttributesParser

class data_preparator:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.log = None
        self.config = None
        self.base_path = os.getcwd()
        self.country = None
        self.current_region = None
        self._CRC_ = None
        
        # Initialize Attributes Parser Class
        self.AttributesParser:AttributesParser = AttributesParser(config_file_path)
    
    def setup_logging(self):
        log_path = f'workflow/log/data_preparation_log.txt'
        self.log = utils.create_log(log_path)

    def load_config(self):
        self.config = self.AttributesParser.config
        self.country = self.config['country']
        self.current_region = self.AttributesParser.current_region
        self._CRC_ = self.AttributesParser.region_code
        self.log.info(f"Processing data files for {self.current_region['name']}[{self._CRC_}]")

    def create_directories(self):
        utils.create_directories(self.base_path, self.config['required_directories'])
        self.log.info("All directories have been created successfully.")

    def prepare_units_dictionary(self):
        units_file_path = self.config['units_dictionary']
        dataprep.create_units_dictionary(units_file_path)

    def prepare_and_update_gadm_data(self):
        # Step 1: Prepare GADM data
        GADM_file_save_to = os.path.join(self.config.get('GADM', {}).get('root', ''), self.config.get('GADM', {}).get('datafile', ''))
        admin_level = 2
        plot_save_to = os.path.join("vis/misc", f"{self._CRC_}_gadm_regions.png")

        utils.check_LocalCopy_and_run_function(
            GADM_file_save_to,
            lambda: dataprep.prepare_GADM_data(self.country, self.current_region, GADM_file_save_to, plot_save_to, admin_level),
            force_update=True)

        # Load the GADM data into a GeoDataFrame
        self.province_gadm_regions_gdf = gpd.read_file(GADM_file_save_to)
        self.log.info("GADM data prepared and saved.")

        # Step 2: Update GADM data with population data
        config_population = self.config['Gov']['Population']
        population_csv_data_path = os.path.join(config_population['root'], config_population['datafile'])
        self.log.info(f"Updating population data")

        # Update the GADM GeoDataFrame with population data
        self.province_gadm_regions_gdf = utils.update_population_data(config_population, self.province_gadm_regions_gdf, population_csv_data_path)
        
        # Save the updated GADM data with population information
        self.province_gadm_regions_gdf.to_file(GADM_file_save_to, driver='GeoJSON')
        self.log.info("Population data imputed to GADM datafile.")

    def process_coders_data(self):
        CODERS_data = self.config['CODERS']
        CODERS_url = CODERS_data['url_1']
        api_key_elias = '?key=' + CODERS_data['api_key']['Elias']
        
        provincial_bus_csv_file_path = self.config['capacity_disaggregation']['transmission']['buses']
        provincial_lines_csv_file_path = self.config['capacity_disaggregation']['transmission']['lines']

        if os.path.exists(provincial_bus_csv_file_path):
            self.log.info(f"Bus data for {self._CRC_} found locally.")
        else:
            tx_lines_df = dataprep.create_dataframe_from_CODERS('transmission_lines', CODERS_url, api_key_elias)
            province_tx_lines = tx_lines_df[tx_lines_df['province'] == self._CRC_]
            province_tx_lines.to_csv(provincial_lines_csv_file_path)
            self.log.info(f"Lines data created for {self._CRC_} and saved to {provincial_lines_csv_file_path}")

    def prepare_turbine_data(self):
        OEDB_config = self.config['capacity_disaggregation']['wind']['turbines']['OEDB']
        ids_to_search = [OEDB_config['model_1']['ID'], OEDB_config['model_2']['ID']]
        OEDB_source = OEDB_config['source']
        OEDB_data = get(OEDB_source).json()

        for turbine_id in ids_to_search:
            turbine_data = dataprep.get_OEDB_dict(OEDB_data, 'id', turbine_id)
            if turbine_data:
                dataprep.format_and_save_turbine_config(turbine_data, os.path.join('data/downloaded_data', "OEDB"))
            else:
                self.log.info(f"No data found for turbine ID {turbine_id}")

    def prepare_era5_cutout(self):
        bounding_box_save_to = os.path.join(self.config['cutout']['root'], f"{self._CRC_}_MBR.geojson")
        bounding_box = dataprep.plot_n_save_bounding_box(self.province_gadm_regions_gdf, self.current_region, bounding_box_save_to)
        self.log.info(f"Minimum Bounding Rectangle (MBR) created and visuals saved to {bounding_box_save_to}")
        dataprep.create_era5_cutout(self._CRC_, bounding_box, self.config['cutout'])

    def prepare_gaez_rasters(self):
        self.log.info(f"Preparing GAEZ raster files")
        dataprep.prepare_GAEZ_raster_files(self.config, self.province_gadm_regions_gdf)

    def prepare_osm_data(self):
        OSM_data = self.config['OSM_data']
        province_osm_data_file_userdefined_name = OSM_data['province_datafile']
        file_path = os.path.join(OSM_data['root'], province_osm_data_file_userdefined_name)

        utils.check_LocalCopy_and_run_function(file_path, lambda: dataprep.prepare_province_OSM_datafile(OSM_data['root'], province_osm_data_file_userdefined_name), force_update=False)

    def run(self):
        # Call all the necessary methods to execute the full workflow
        self.setup_logging()
        self.load_config()
        self.create_directories()
        self.prepare_units_dictionary()
        self.prepare_and_update_gadm_data()
        self.process_coders_data()
        self.prepare_turbine_data()
        self.prepare_era5_cutout()
        self.prepare_gaez_rasters()
        self.prepare_osm_data()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run the linking tool with specified configuration file.")
    parser.add_argument('config_file_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # Create an instance of LinkingTool and run the main workflow
    _module = data_preparator(args.config_file_path)
    _module.run()
