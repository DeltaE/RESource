import geopandas as gpd
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from RES import utility as utils

from atlite.gis import ExclusionContainer,shape_availability

import RES.utility as utils  # Custom module for handling data operations
from RES.boundaries import GADMBoundaries
from RES.era5_cutout import ERA5Cutout
from RES.gaez import GAEZRasterProcessor
from RES.osm import OSMData

print_base_level=2
class ConservationLands(GADMBoundaries):

    """
    ConservationLands class
    """
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        # Set the Class specific attributes
        self.conserved_lands_cfg = self.config['Gov']['conservation_lands']
        
        self.source_url = self.conserved_lands_cfg['url']
        self.data_root = self.conserved_lands_cfg['root']
        self.zip_file_name = f"{self.conserved_lands_cfg['data_name']}.zip"
        self.zip_file_path = Path(self.data_root) / self.zip_file_name
        self.extraction_dir = Path (self.data_root) / self.zip_file_path.stem
        self.extraction_dir.parent.mkdir(parents=True, exist_ok=True)
        
    def get_provincial_conserved_lands(self,
                                       geom_simplification_tolerance=0.005) -> gpd.GeoDataFrame:
        
        """
        Load provincial conserved lands from the .gdb file.
        
        ### Args:
            geom_simplification_tolerance (default to _.005_)
            - geometry simplification to avoid unnecessary granular level geometries. 
            - This tool is configured to geom in degrees, e.g tolerance of 0.005 corresponds to approximately 500m (at the equator) geoms will be simplified.
        """

        
        file_name_prefix = self.conserved_lands_cfg['data_name']
        provincial_file_path = Path('data/downloaded_data/lands') / f"{file_name_prefix}_{self.province_short_code}.pickle"
        provincial_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if provincial_file_path.exists():
            utils.print_update(level=print_base_level,message="Loading Canadian Protected and Conserved Areas Database (CPCAD) from locally stored datafile - {provincial_file_path}")
            gdf=gpd.GeoDataFrame(pd.read_pickle(provincial_file_path))
        else:
            gdb_file_path = self.__get_conserved_lands__()
            
            
            # Get Region Boundaries
            self.province_boundary=self.get_province_boundary()
            
            # Load the .gdb file as a GeoDataFrame
            gdf = gpd.read_file(gdb_file_path, mask=self.province_boundary)
            gdf.to_crs(self.province_boundary.crs, inplace=True)
            
            gdf['geometry'] = gdf['geometry'].simplify(geom_simplification_tolerance)

            # Map IUCN categories to descriptions
            IUCN_CAT = self.conserved_lands_cfg['IUCN_CAT_mapping']
            gdf['IUCN_CAT_desc'] = gdf['IUCN_CAT'].map(IUCN_CAT)
            
            gdf[['IUCN_CAT_desc','NAME_E', 'ZONEDESC_E',
            'BIOME', 'PA_OECM_DF', 'IUCN_CAT', 'IPCA', 'IND_TERM',
            'O_AREA_HA', 'LOC', 'MECH_E', 'TYPE_E', 'OWNER_TYPE', 'OWNER_E',
            'GOV_TYPE', 'MGMT_E', 'STATUS', 'ESTYEAR', 'QUALYEAR', 'DELISTYEAR',
            'SUBSRIGHT', 'CONS_OBJ', 'NO_TAKE', 'NO_TAKE_HA', 'MPLAN', 'MPLAN_REF',
            'M_EFF', 'AUDITYEAR', 'AUDITRES', 'PROVIDER', 'Shape_Length',
            'Shape_Area', 'geometry']]
        
            gdf.to_pickle(provincial_file_path)
            
        
        return gdf

    def __get_conserved_lands__(self) -> Path:
        """Download the source ZIP file, extract contents, and return the .gdb file path."""
        # Check if the extraction directory exists
        if self.extraction_dir.exists():
           utils.print_update(level=print_base_level+1,message=f"Extraction directory {self.extraction_dir} already exists, skipping download and extraction.")
        else:
            if self.zip_file_path.exists():
                utils.print_update(level=print_base_level+1,message=f"ZIP file {self.zip_file_path} already exists, skipping download.")
            else:
                # Download the ZIP file
                utils.print_update(level=print_base_level+1,message="Downloading Canadian Protected and Conserved Areas Database (CPCAD)")
                self.zip_file_path.parent.mkdir(parents=True, exist_ok=True)
                utils.download_data(self.source_url, self.zip_file_path)
                utils.print_update(level=print_base_level+1,message=f"Downloaded ZIP file to {self.zip_file_path}")

            # Create the extraction directory and extract ZIP contents
            self.extraction_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(self.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extraction_dir)
            # print(f"Extracted files to {self.extraction_dir}")

        # Load the first .gdb file found in the extraction directory
        gdb_file_path = next(self.extraction_dir.rglob("*.gdb"), None)
        if gdb_file_path is None:
            raise FileNotFoundError(">> !! No .gdb file found in the extracted contents.")
        
        return gdb_file_path
    

    def show_lands(self, 
               basemap: str = 'CartoDB positron', 
               save_path: str = None, 
                save: bool = False):
        """
        Create and save an interactive map for the specified region.

        Args:
            basemap (str): The basemap to use (default is 'CartoDB positron').
            save_path (str): The path to save the HTML map. If None, default is used.
            save (bool): If True, saves the map as a local HTML file.
            
        Returns:
            folium.Map: The interactive map object.
        """
        conserved_lands = self.get_provincial_conserved_lands()
        self.province_boundary = self.get_province_boundary()

        if self.province_boundary is not None:
            m = self.province_boundary.explore(color='grey',linecolor='grey', legend=True, tiles=basemap,alpha=0.4)
            conserved_lands.explore('IUCN_CAT_desc', m=m, legend=True, tiles=basemap)

            if save:
                if save_path is None:
                    save_path = f'vis/lands/{self.province_short_code}.html'
                else:
                    save_path = Path(save_path) / f"{self.province_short_code}.html"
                
                # Ensure the directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the map as an HTML file
                m.save(save_path)
                utils.print_update(level=print_base_level+1,message="Interactive map for '{self.province_short_code}' saved to {save_path}.")
            else:
                utils.print_update(level=print_base_level+1,message="Skipping save, 'save' is set to False.")
        
        return m
    
class LandContainer(ERA5Cutout,
                    GAEZRasterProcessor,
                    ConservationLands,
                    OSMData
                    ):
    """
    Handles the inclusion/exclusion of lands from raster/vector data.
    
    """
    
    def __post_init__(self):
    # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()

        self.excluder_crs=3347
        # Initiate Exclusion Container
        self.excluder = ExclusionContainer(crs=self.excluder_crs)  # CRS 3347 fit for Canada
    
    def set_excluder(self):
        # GAEZ configs
        self.gaez_config = self.get_gaez_data_config()
        
        # Initialize raster configurations
        raster_configs = {}
        
        # Retrieve custom land configuration if available
        custom_land_config = self.get_custom_land_layers()
        utils.print_update(level=print_base_level+1,message="Loading global filters' rasters from GAEZ, trimmed to {self.province_name}")
        self.process_all_rasters(show=False) # Donwloads and processes all GAEZ rasters
        # Loop over each raster type in GAEZ config and set up each raster
        for raster_type in self.gaez_config['raster_types']:
            raster_name = raster_type['name']
            raster_file = str(self.province_short_code+"_"+raster_type['raster'])
            zip_direct = raster_type['zip_extract_direct']
            
            # Determine the class inclusion/exclusion based on resource type
            inclusion_key = 'class_inclusion' if 'class_inclusion' in raster_type else 'class_exclusion'
            
            raster_configs[f"gaez_{raster_name}"] = {
                # 'raster': self.__get_raster_path__(raster_type, self.gaez_root, self.Rasters_in_use_direct),
                'raster': self.gaez_root/ self.Rasters_in_use_direct/raster_type['zip_extract_direct']/raster_file,
                inclusion_key: raster_type[inclusion_key][self.resource_type],
                'buffer': raster_type.get('buffer', {}).get(self.resource_type, 0),
                'invert': inclusion_key == 'class_inclusion'  # invert if it's an inclusion class
            }
            
            utils.print_update(level=print_base_level+1,message="Loading {raster_name.capitalize()} layers from {raster_configs[f'gaez_{raster_name}']['raster']}")
        
        # Load additional custom raster configurations from YAML if specified
        # for raster_name, config in custom_land_config.get('rasters', {}).items():
        #     raster_path = config['raster']
            
        #     # Skip if raster path is missing or empty
        #     if raster_path is None:
        #         self.log.warning(f"Skipping {raster_name}: Raster file does not exist or is empty.")
        #         continue
            
        #     raster_configs[raster_name] = {
        #         'raster': raster_path,
        #         'class_exclusion': config.get('class_exclusion', []),
        #         'buffer': config.get('buffer', 0),
        #         'invert': config.get('invert', False)
        #     }

        # Add all raster configurations to the excluder
        for key, config in raster_configs.items():
            inclusion_or_exclusion = config.get('class_inclusion', config.get('class_exclusion'))
            self.excluder.add_raster(
                config['raster'], 
                inclusion_or_exclusion, 
                buffer=config['buffer'], 
                invert=config['invert']
            )

        # Load additional layers
        self.conservation_lands_province_gdf = self.get_provincial_conserved_lands()
        self.aeroway_gdf = self.get_osm_layer('aeroway')
        
        # Set up resource disaggregation configurations
        self.resource_disaggregation_config = self.get_resource_disaggregation_config()

        # Add local (Canadian) vector geometries to excluder
        self.excluder.add_geometry(
            self.conservation_lands_province_gdf.geometry, 
            buffer=self.resource_disaggregation_config['buffer']['conserved_lands']
        )
        self.excluder.add_geometry(
            self.aeroway_gdf.geometry, 
            buffer=self.resource_disaggregation_config['buffer']['aeroway']
        )
        
        self.log.info(f"{self.excluder}")
        
        return self.excluder