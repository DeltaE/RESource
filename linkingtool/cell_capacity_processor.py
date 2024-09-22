import logging as log
import os,sys
from atlite.gis import shape_availability

# Local Packages
import linkingtool.linking_utility as utils
from linkingtool.attributes_parser import AttributesParser

# Logging Configuration
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
class cell_capacity_processor:
    def __init__(self, config_file_path: str, resource_type: str):  # Added resource_type argument
        
        self.resource_type = resource_type  # Initialized resource_type
        
        # Initialize Attributes Parser Class
        self.attributes_parser: AttributesParser = AttributesParser(config_file_path, self.resource_type)
        
        # Extract Attributes
        self.current_region:dict = self.attributes_parser.current_region
        self.region_code:str = self.attributes_parser.region_code
        self.disaggregation_config :dict= self.attributes_parser.disaggregation_config
        self.vis_dir :str= self.attributes_parser.vis_dir
        self.linking_data:dict = self.attributes_parser.linking_data
        self.gaez_data:dict = self.attributes_parser.gaez_data
        
        # Static cost params (static: fixed for entire snapshot)
        # Load costs (float) and unpack
        (
            self.resource_capex, 
            self.resource_fom, 
            self.resource_vom,
            self.grid_connection_cost_per_km,
            self.tx_line_rebuild_cost
        ) = self.attributes_parser.load_cost()
        
        # snapshot (range of of the temporal data)
        (
            self.start_date,
            self.end_date,
        ) = self.attributes_parser.load_snapshot()
        
        # Load geospatial data (gdf) and unpack
        (
            self.cutout,
            self.gadm_regions_gdf,
            self.aeroway_with_buffer,
            self.conservation_lands_province,
            self.buses_gdf
        ) = self.attributes_parser.load_geospatial_data()

            
    def extract_grid_cells(self):
        self.province_grid_cells = self.cutout.grid.overlay(self.gadm_regions_gdf, how='intersection', keep_geom_type=True)
        
        log.info(f"Extracted {len(self.province_grid_cells)} ERA5 Grid Cells for {self.current_region['code']} from Cutout")
        return self.province_grid_cells  # Return the extracted grid cells

    def calculate_land_availability(self):
        land_cover_config = self.gaez_data['land_cover']
        terrain_resources_config = self.gaez_data['terrain_resources']
        gaez_landcover_raster = os.path.join(self.gaez_data['root'], self.gaez_data['Rasters_in_use_direct'], land_cover_config['zip_extract_direct'], land_cover_config['raster'])
        gaez_terrain_raster = os.path.join(self.gaez_data['root'], self.gaez_data['Rasters_in_use_direct'], terrain_resources_config['zip_extract_direct'], terrain_resources_config['raster'])
        
        terrain_class_exclusion = terrain_resources_config['class_exclusion'][self.resource_type]
        land_class_inclusion = land_cover_config['class_inclusion'][self.resource_type]

        self.ERA5_Land_cells_low_slope = utils.calculate_land_availability_raster(self.cutout, self.province_grid_cells, 'Excluding terrain with >30% slope', gaez_terrain_raster, '2_land_avail_low_slope', terrain_class_exclusion, self.current_region, buffer=0, exclusion=True)
        self.ERA5_Land_cells_eligible = utils.calculate_land_availability_raster(self.cutout, self.ERA5_Land_cells_low_slope, 'Eligible Land Classes', gaez_landcover_raster, '3_land_avail_eligible', land_class_inclusion, self.current_region, buffer=0, exclusion=False)
        self.ERA5_Land_cells_nonprotected = utils.calculate_land_availability_vector_data(self.cutout, self.ERA5_Land_cells_eligible, self.conservation_lands_province, 'Excluding Conservation Lands', '4_land_avail_excl_protectedLands', self.current_region)
        self.ERA5_Land_cells_final = utils.calculate_land_availability_vector_data(self.cutout, self.ERA5_Land_cells_nonprotected, self.aeroway_with_buffer, 'Excluding Aeroway with buffer', '5_land_avail_excl_aeroway', self.current_region)
        
        self.ERA5_Land_cells_final['land_availablity'] = self.ERA5_Land_cells_final['eligible_land_area'] / self.ERA5_Land_cells_final['land_area_sq_km']


    def calculate_potential_capacity(self):
        resource_landuse_intensity = self.disaggregation_config[f'{self.resource_type}']['landuse_intensity']
        self.province_grid_cells_capacity = utils.calculate_potential_capacity(self.ERA5_Land_cells_final, resource_landuse_intensity, 'cell')
        return self.province_grid_cells_capacity  # Return processed data
    
    def progress_bar(self, iteration, total, bar_length=40):
        percent = (iteration / total)
        arrow = '█' * int(round(percent * bar_length) - 1)
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write(f'\rProgress: |{arrow}{spaces}| {percent:.2%}\n')
        sys.stdout.flush()

    def run(self):
        log.info(f"ERA5 cell capacity processor module initiated")
        
        tasks = [
            self.extract_grid_cells,
            self.calculate_land_availability,
            self.calculate_potential_capacity
        ]
        
        total_tasks = len(tasks)
        
        era5_cell_capacity = None
        
        for i, task in enumerate(tasks):
            if task == self.calculate_potential_capacity:
                era5_cell_capacity = task()  # Call and capture the output of this specific task
            else:
                task()  # Execute the other tasks
            self.progress_bar(i + 1, total_tasks)  # Update the progress bar
        
        print('\n')  # For a newline after progress completion
        
        # Assign values after potential capacity calculation
        era5_cell_capacity = era5_cell_capacity.assign(
            capex=self.resource_capex,
            fom=self.resource_fom,
            vom=self.resource_vom,
            grid_connection_cost_per_km=self.grid_connection_cost_per_km,
            tx_line_rebuild_cost=self.tx_line_rebuild_cost
        )

        return era5_cell_capacity  # Return processed data

# def main(config_file_path: str, 
#          resource_type: str):  # Added resource_type argument
#     script_start_time = time.time()
    
#     # Create an instance of the processor class
#     processor = era5_cell_capacity_processor(config_file_path, resource_type)  # Pass resource_type
#     era5_cell_capacity = processor.run()  # Capture the returned capacity
    
#     output_file_path = os.path.join(f'era5_cell_capacity_{resource_type}.pkl')  # Change path as needed
#     # era5_cell_capacity.to_pickle(output_file_path)
    
#     script_runtime = round((time.time() - script_start_time), 2)
#     log.info(f"Script runtime: {script_runtime} seconds")
    
#     # Optionally, do something with the result
#     log.info(f"Processed capacity data: {era5_cell_capacity}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run solar module script')
#     parser.add_argument('config', type=str, help=f"Path to the configuration file '*.yml'")
#     parser.add_argument('resource_type', type=str, help=f"Resource type (e.g., solar, wind)")
#     args = parser.parse_args()
#     main(args.config, args.resource_type)
