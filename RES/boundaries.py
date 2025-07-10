# Import Global Packages
from pathlib import Path
import geopandas as gpd
import pygadm
from dataclasses import dataclass
import inspect

# Import local packages
from RES.AttributesParser import AttributesParser
import RES.utility as utils
print_level_base=4
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
        
        self.admin_level:int= 2 # hardcoded to keep the workflow intact. The workflow has dependency on Regional District name i.e. level 2 boundaries.

        # Setup paths and ensure directories exist
        self.gadm_config = super().get_gadm_config()
        
        self.gadm_root = Path(self.gadm_config['root'])
        self.gadm_root.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.gadm_processed = Path(self.gadm_config['processed'])
        self.gadm_processed.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.crs=self.get_default_crs()
        self.country=self.get_country()

        self.region_file:Path = Path(self.gadm_processed) / f'gadm41_{self.country}_L{self.admin_level}_{self.region_short_code}.geojson'
        
        self.boundary_datafields = self.gadm_config.get('datafield_mapping')
        
    # Define a function to create bounding boxes (of cell) directly from coordinates (x, y) and resolution
    
    def get_country_boundary(self,
                             country: str = None,
                             force_update: bool = False) -> gpd.GeoDataFrame:
        """
        Retrieves and prepares the GADM boundaries dataset for the specified country (Administrative Level 2).

        Args:
            country(str): The name of the country to fetch GADM data for. If None, extracts the country from the user config file.
            force_update (bool): If True, re-fetch the GADM data even if a local file exists.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the country's GADM regions in crs '4326'
        
        Dependency:
            Depends on pygadm package to fetch the GADM data.
            
        :raises ValueError: If the country is not found in the GADM dataset.
        
        :raises Exception: If there is an error fetching or loading the GADM data.
        
        """
        utils.print_update(level=print_level_base+1,message=f"{__name__} | Country Selected: {self.country}.")
        
        # store the user input (via method args)
        if country is not None:
            self.country = country.capitalize()
            
        utils.print_update(level=print_level_base+1,message=f"{__name__} | Country Selected: {self.country}.")
        self.country_file:Path=Path (self.gadm_root) /  f'gadm41_{self.country}_L{self.admin_level}.geojson'  
        
        try:
            # country_gadm_regions_file_path = Path (self.gadm_root)/ f'gadm41_{self.country}_L{self.admin_level}.geojson'

            # Load or fetch data
            if self.country_file.exists() and not force_update: # load the country gdf from local file
                utils.print_update(level=print_level_base+1,message=f"{__name__} | Loading GADM data for {self.country} from local datafile {self.country_file}.")
                self.boundary_country=gpd.read_file(self.country_file)

            else:
                # Fetch and save data if file does not exist or force_update is True
                utils.print_update(level=print_level_base+1,message=f"{__name__} | Fetching GADM data for {self.country} at Administrative Level {self.admin_level}....from source: https://gadm.org/data.html")
                
                _country_gdf_:gpd.GeoDataFrame = pygadm.AdmItems(name=self.country, content_level=self.admin_level)
                _country_gdf_.set_crs(self.crs)
                self.boundary_country=_country_gdf_
                # save to local file
                self.boundary_country.to_file(self.country_file, driver='GeoJSON')
                utils.print_update(level=print_level_base+1,message=f"{__name__} | GADM data saved to {self.country_file}.")
                
            return self.boundary_country

        except Exception as e:
            utils.print_update(level=print_level_base+1,message=f"{__name__} | Error fetching or loading GADM data: {e}")
            raise

    def get_region_boundary(self,
                            region_name: str = None,
                            force_update: bool = False) -> gpd.GeoDataFrame:
        """
        Prepares the boundaries for the specified region within the country. The defaults datafields (e.g. NAME_0, NAME_1, NAME_2) gets modified to match the user config file.

        Args:
            force_update (bool): To force update the data and formatting.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the region boundaries.
            
        Raises:
            ValueError: If the region code is invalid or no data is found for the specified region

        """
        if region_name is not None:
            self.region_short_code = region_name.upper()
        else:
            self.region_name =self.get_region_name()
        
        utils.print_update(level=print_level_base+2,
                           message=f"{__name__}| Region Set to >> Short Code : {self.region_short_code}, Name: {self.region_name}).")
        utils.print_update(level=print_level_base+2,
                           message=f"{__name__}| Collecting regional boundary...")

        if self.region_code_validity:
    
            # It should be saved in processed because the column names have been modified from source.
            
            if self.region_file.exists() and not force_update: # There is a local file and no update required
                utils.print_update(level=print_level_base+1,message=f"{__name__}| Loading GADM boundaries (Sub-provincial | level =2) for {self.region_name} from local file {self.region_file}.")
                
                self.boundary_region:gpd.GeoDataFrame=gpd.read_file(self.region_file)
            
            else: # When the local file for region doesn't exist, Filter region data from country file and save locally
    
                _boundary_country = self.get_country_boundary(force_update)
                _boundary_region_ = _boundary_country.loc[self.boundary_country['NAME_1'] == self.region_name]

                if _boundary_region_.empty : 
                    utils.print_update(level=print_level_base+1,message=f"{__name__}|  No data found for region '{self.region_name}'.")
                    utils.print_update(level=print_level_base+1,message=f"{__name__}| | @ LINE | Consider revising '{self.region_name}' to match source (e.g. https://gadm.org/maps.html); Select 'show sub-divisions' to get the list of Supported Regional Names")
                    raise ValueError(f"{__name__} | @ LINE {inspect.currentframe().f_lineno} | No data found for region '{self.region_name}'.")
                else:
                    _boundary_region_ = _boundary_region_[['NAME_0', 'NAME_1', 'NAME_2', 'geometry']].rename(columns={
                        'NAME_0': self.boundary_datafields['NAME_0'], 'NAME_1': self.boundary_datafields['NAME_1'], 'NAME_2': self.boundary_datafields['NAME_2']
                    })
                    self.boundary_region:gpd.GeoDataFrame=_boundary_region_
      
                
                self.boundary_region.to_file(self.region_file, driver='GeoJSON')
                utils.print_update(level=print_level_base+1,message=f"{__name__}| GADM data for {self.region_name} saved to {self.region_file}.")
            return self.boundary_region
        else:
            raise ValueError(f"{__name__} | @ LINE {inspect.currentframe().f_lineno} Invalid region code: {self.region_short_code}.")

    def get_bounding_box(self)->tuple:
        
        """
        This method loads the region boundary using `get_region_boundary()` method and gets Minimum Bounding Rectangle (MBR).
        
        Returns:
            tuple: A tuple containing the dictionary of bounding box coordinates, and the actual boundary GeoDataFrame for the specified region.
            
        Purpose:
            To be used internally to get the bounding box of the region to set ERA5 cutout boundaries.
            
        """
        utils.print_update(level=print_level_base+1,
                           message=f"{__name__}| Processing regional bounding box...")
        
        self.actual_boundary=self.get_region_boundary()
        
        utils.print_update(level=print_level_base+1,message=f"{__name__}| Setting up the Minimum Bounding Region (MBR) for {self.region_short_code}...")
        min_x, min_y, max_x, max_y=self.actual_boundary.geometry.total_bounds
        
        """
        Alternate:
        self.region_gadm_gdf.unary_union.buffer(1).bounds # 
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
        # bounding_box_gdf = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs=region_gadm_regions_gdf.crs)
        
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
        
        Returns:
            folium.Map: An interactive map object showing the region boundaries.
            
        """
        boundary_region = self.get_region_boundary()

        if boundary_region is not None:
            m = boundary_region.explore('Region', legend=True, tiles=basemap)
            
            if save:
                file_path = Path(save_path) / f"{self.region_short_code}.html"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                m.save(file_path)
                utils.print_update(level=print_level_base+1,
                           message=f"{__name__}| Interactive map for '{self.region_short_code}' saved to {file_path}.")
            else:
                utils.print_update(level=print_level_base+1,
                                  message=f"{__name__}| Skipping the save to local directories as 'save' is set to False.")
        
        return m

    def run(self):
        """
        Executes the process of extracting boundaries and creating an interactive map. 
        To be used as a main method to run the class's sequential tasks.
        """
        if self.region_code_validity:
            region_gadm_gdf=self.get_region_boundary()
            self.get_bounding_box()
            self.show_regions()
            return region_gadm_gdf
        else:
            utils.print_update(level=print_level_base+1,
                           message=f"{__name__}| Region code is not valid.")
            self.region_code_validity
            return None
