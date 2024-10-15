# Import Global Packages
from pathlib import Path
import geopandas as gpd
import pygadm
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
# import logging

# Import local scripts
from linkingtool.AttributesParser import AttributesParser

from dataclasses import dataclass
@dataclass

class GADMBoundaries(AttributesParser):
    """
    A class to process GADM country files and extract specific regional boundaries at a given administrative level.
    
    :Dependency for core data:
        pygadm
    :crs:
        EPSG:4326
    """
    
    def __post_init__(self):
        
     # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        self.country = self.get_country()
        self.province_name =self.get_province_name()
        self.admin_level:int= 2 # hardcoded to keep the workflow intact. The workflow has dependency on Regional District name i.e. level 2 boundaries.

        # Setup paths and ensure directories exist
        self.gadm_config=self.get_gadm_config()
        
        self.gadm_root = Path(self.gadm_config['root'])
        self.gadm_root.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.gadm_processed = Path(self.gadm_config['processed'])
        self.gadm_processed.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.crs=self.get_default_crs()
        self.country_file:Path=Path (self.gadm_root) /  f'gadm41_{self.country}_L{self.admin_level}.geojson'
        self.province_file:Path = Path(self.gadm_processed) / f'gadm41_{self.country}_L{self.admin_level}_{self.province_short_code}.geojson'

    def get_country_boundary(self, 
                             force_update: bool = False) -> gpd.GeoDataFrame:
        """
        Retrieves and prepares the GADM boundaries dataset for the specified country.

        Args:
            force_update (bool): If True, re-fetch the GADM data even if a local file exists.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the country's GADM regions in crs '4326'
        """
        # store the user input (via method args)
        
                      
        try:
            # country_gadm_regions_file_path = Path (self.gadm_root)/ f'gadm41_{self.country}_L{self.admin_level}.geojson'

            # Load or fetch data
            if self.country_file.exists() and not force_update: # load the country gdf from local file
                self.log.info(f"Loading GADM data for {self.country} from local datafile {self.country_file}.")
                self.boundary_country=gpd.read_file(self.country_file)

            else:
                # Fetch and save data if file does not exist or force_update is True
                self.log.info(f"Fetching GADM data for {self.country} at Administrative Level {self.admin_level}....from source: https://gadm.org/data.html")
                
                _country_gdf_:gpd.GeoDataFrame = pygadm.AdmItems(name=self.country, content_level=self.admin_level)
                _country_gdf_.set_crs(self.crs)
                self.boundary_country=_country_gdf_
                # save to local file
                self.boundary_country.to_file(self.country_file, driver='GeoJSON')
                self.log.info(f"GADM data saved to {self.country_file}.")
                
            return self.boundary_country

        except Exception as e:
            self.log.error(f"Error fetching or loading GADM data: {e}")
            raise

    def get_province_boundary(self,
                              force_update: bool = False) -> gpd.GeoDataFrame:
        """
        Prepares the boundaries for the specified region within the country.

        Args:
            force_update (bool): To force update the data and formatting.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the province boundaries.
        """
        
        if self.province_code_validity:
    
            # It should be saved in processed because the column names have been modified from source.
            
            if self.province_file.exists() and not force_update: # There is a local file and no update required
                self.log.info(f"Loading GADM boundaries (Sub-provincial | level =2) for {self.province_name}  from local file {self.province_file}.")
                self.boundary_province:gpd.GeoDataFrame=gpd.read_file(self.province_file)
            
            else: # When the local file for province doesn't exist, Filter province data from country file and save locally
    
                _boundary_country = self.get_country_boundary(force_update)
                _boundary_province_ = _boundary_country.loc[self.boundary_country['NAME_1'] == self.province_name]

                if _boundary_province_.empty : 
                    self.log.error(f"No data found for province '{self.province_name}'.")
                    exit
                else:
                    _boundary_province_ = _boundary_province_[['NAME_0', 'NAME_1', 'NAME_2', 'geometry']].rename(columns={
                        'NAME_0': 'Country', 'NAME_1': 'Province', 'NAME_2': 'Region'
                    })
                    self.boundary_province:gpd.GeoDataFrame=_boundary_province_
      
                
                self.boundary_province.to_file(self.province_file, driver='GeoJSON')
                self.log.info(f"GADM data for {self.province_name} saved to {self.province_file}.")
            return self.boundary_province
        else:
            self.log.error(f"Invalid province code: {self.province_short_code}.")
            exit
    
    def get_bounding_box(self)->tuple:
        
        """
        This method returns the Minimum Bounding Rectangle (MBR) to extract the 
        """
        self.actual_boundary=self.get_province_boundary()
        # self.log.info(f"Setting up the Minimum Bounding Region (MBR) for {self.province_short_code}...")
        min_x, min_y, max_x, max_y=self.actual_boundary.geometry.total_bounds
        
        """
        Alternate:
        self.province_gadm_gdf.unary_union.buffer(1).bounds # 
        Key Differences:
            Performance: .total_bounds is much faster because it doesn’t require merging or buffering geometries.
            Output: Both return bounding coordinates, but .unary_union.buffer(1).bounds includes a buffer, whereas .total_bounds is the simplest, direct bounding box of the original geometries.
            Complexity: Use .unary_union.buffer(1).bounds when you need more advanced spatial transformations (e.g., merging or buffering), otherwise use .total_bounds for basic bounding box calculations.
        """
        
        # MBR=box(min_x, min_y, max_x, max_y)
        self.bounding_box:dict={
            'minx': min_x,
            'maxx': max_x,
            'miny': min_y,
            'maxy': max_y
            }
        # plot_info='(Minimum Bounding Rectangle)'
        # bounding_box_gdf = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs=province_gadm_regions_gdf.crs)
        return self.bounding_box,self.actual_boundary
        

    def show_regions(self, 
                    basemap: str = 'CartoDB positron', 
                    save_path: str = f'vis/regions',
                    save:bool=False):
        """
        Create and save an interactive map for the specified region.

        Args:
            basemap (str): The basemap to use (default is 'CartoDB positron').
            save_path (str): The path to save the HTML map. The default is given.
            save(bool): If the user want's to skip saving as local file.
            
        """
        boundary_province = self.get_province_boundary()

        if boundary_province is not None:
            m = boundary_province.explore('Region', legend=True, tiles=basemap)
            
            if save:
                file_path = Path(save_path) / f"{self.province_short_code}.html"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                m.save(file_path)
                self.log.info(f"Interactive map for '{self.province_short_code}' saved to {file_path}.")
            else:
                self.log.info(f"Skipping the save to local directories as 'save' is set to False.")
        
        return m

    def run(self):
        """
        Executes the process of extracting boundaries and creating an interactive map.
        """
        if self.province_code_validity:
            province_gadm_gdf=self.get_province_boundary()
            self.get_bounding_box()
            self.create_interactive_map()
            return province_gadm_gdf
        else:
            self.log.error("Province code is not valid.")
            self.province_code_validity
            return None
