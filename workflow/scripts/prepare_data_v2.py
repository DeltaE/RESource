import argparse
from pathlib import Path
import geopandas as gpd
from requests import get
from dataclasses import dataclass

# Local Packages
import linkingtool.linking_utility as utils
import linkingtool.visuals as vis
import linkingtool.linking_solar as solar
import linkingtool.dataprep as dataprep

from linkingtool.coders import CODERSData
from linkingtool.AttributesParser import AttributesParser
from linkingtool.boundaries import GADMBoundaries
from linkingtool.era5_cutout import ERA5Cutout
from linkingtool.osm import OSMData
from linkingtool.lands import ConservationLands

@dataclass

class LinkingToolData:
    def __init__(self, 
                 config_file_path: Path, 
                 province_short_code:str,
                 resource_type=None):
        
        # Set default attributes
        self.base_path = Path.cwd()
        
        # Set attributes from >> args
        self.config_file_path = config_file_path
        self.province_short_code=province_short_code.upper()
        
        # Set some more attributes >> initially NONE, later to be updated from AttributesParser methods
        self.log = None
        self.config = None
        self.country = None
        self.current_region_config = None # Dictionary related to current region
        self.province_mapping =None
        self.resource_type=None
        self.cutout_config=None
        
        # This dictionary will be used to pass arguments to external classes
        self.required_args = {   #order doesn't matter
            "config_file_path" : config_file_path,
            "province_short_code": province_short_code.upper(),
        }
        
        # Initialize Attributes Parser Class to use it's methods that updates the None parameters.
        self.attributes_parser: AttributesParser = AttributesParser(**self.required_args)
        
        # Initialize Lands data loader class
        self.lands:ConservationLands =ConservationLands(**self.required_args)
        
        # default methods to run to update >> required_args. These methods has dependency on the AttributesParser class.
        self.setup_logging()
        self.load_config()
        self.get_units_dictionary()
        
        self.country = self.config['country']
        self.province_mapping = self.config['province_mapping']
        self.cutout_config = self.config['cutout']

        # Initiate external classes to pull/update data

        ## Initiate class to pull GADM data
        self.gadm: GADMBoundaries = GADMBoundaries(**self.required_args)
        self.province_gadm_gdf:gpd.GeoDataFrame=self.gadm.get_province_boundary()
        
        ## Initialise Initiate class to pull ERA5 Cutout
        self.era5_cutout:ERA5Cutout=ERA5Cutout(**self.required_args)
        
        ## Initiate class to pull CODERS data
        self.coders:CODERSData= CODERSData(**self.required_args)
        
        ## Initiate class to pull OSM data 
        self.osm:OSMData=OSMData(**self.required_args)
        
    def setup_logging(self):
        log_path = Path('workflow/log/data_preparation_log.txt')
        self.log = utils.create_log(log_path)

    def load_config(self):
        self.province_code_validity=self.attributes_parser.province_code_validator
        
        if self.province_code_validity :
            self.config = self.attributes_parser.config
        else:
            self.province_code_validity
            
    def get_units_dictionary(self):
        units_file_path = Path(self.config['units_dictionary'])
        dataprep.create_units_dictionary(units_file_path)

    # def create_directories(self, base_path=None, structure=None):
    #     """Recursively create directories based on the structure."""
    #     if base_path is None:
    #         base_path = self.base_path
    #     if structure is None:
    #         # self.load_config()
    #         structure = self.config['required_directories']
        
    #     for key, value in structure.items():
    #         # Create the main directory
    #         dir_path = base_path / key
    #         if dir_path.exists():
    #             print(f" >> !! '{key}' already exists")
    #         else:
    #             dir_path.mkdir(parents=True, exist_ok=True)
    #             print(f"- '{key}' created")
            
    #         # Recursively create subdirectories
    #         if isinstance(value, dict):
    #             self.create_directories(dir_path, value)
    #         elif value is None:
    #             print(f"'{key}' has no subdirectories")


    def get_grid_ss(self, force_update=False):
        # self.province_ss_gdf=self.coders.get_table_provincial('substations', force_update)
        return self.coders.get_table_provincial('substations', force_update)

    def get_turbine_data(self):
        OEDB_config = self.config['capacity_disaggregation']['wind']['turbines']['OEDB']
        ids_to_search = [OEDB_config['model_1']['ID'], OEDB_config['model_2']['ID']]
        OEDB_source = OEDB_config['source']
        OEDB_data = get(OEDB_source).json()

        for turbine_id in ids_to_search:
            turbine_data = dataprep.get_OEDB_dict(OEDB_data, 'id', turbine_id)
            if turbine_data:
                dataprep.format_and_save_turbine_config(turbine_data, Path('data/downloaded_data') / "OEDB")
            else:
                self.log.info(f"No data found for turbine ID {turbine_id}")
                
    def get_era5_cutout(self):
        # bounding_box_save_to = Path(self.config['cutout']['root']) / f"{self.current_region}_MBR.geojson"
        # bounding_box = dataprep.plot_n_save_bounding_box(self.province_gadm_gdf, self.current_region, bounding_box_save_to)
        # self.log.info(f"Minimum Bounding Rectangle (MBR) created and visuals saved to {bounding_box_save_to}")
        return self.era5_cutout.get_era5_cutout(bounding_box=self.gadm.get_bounding_box())

    def get_gaez_rasters(self):
        province_gadm_gdf=self.gadm.get_province_boundary()
        self.log.info(f"Preparing GAEZ raster files")
        dataprep.prepare_GAEZ_raster_files(self.config, province_gadm_gdf,self.province_short_code)
    
    def get_conservation_lands_data(self)->gpd.GeoDataFrame:
        return self.lands.get_provincial_conserved_lands()

    def run(self):
        # Call all the necessary methods to execute the full workflow

        self.gadm.run()
        self.get_grid_ss(force_update=False)
        self.get_turbine_data()
        self.get_era5_cutout()
        self.get_gaez_rasters()
        self.get_conservation_lands_data()
        self.osm.run()

        print(f"{100*"_"}\nData preparation module completed !!!")
        
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run the linking tool with specified configuration file.")
    parser.add_argument('config_file_path', type=str, help='Path to the configuration file')
    # parser.add_argument('province_name', type=str, help='Province name (e.g., British Columbia)')
    parser.add_argument('province_short_code', type=str, help='Province Short Code (2 letter)')
    args = parser.parse_args()

    # Create an instance of DataPreparator and run the main workflow
    module = LinkingToolData(Path(args.config_file_path), args.province_short_code)
    module.run()
