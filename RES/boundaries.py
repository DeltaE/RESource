# Import Global Packages
from pathlib import Path
import geopandas as gpd
import pygadm
from dataclasses import dataclass

# Import local packages
from RES.AttributesParser import AttributesParser

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
        
        self.country = self.get_region_name() # self.get_country()
        self.region_name =self.get_region_name()
        self.admin_level:int= 2 # hardcoded to keep the workflow intact. The workflow has dependency on Regional District name i.e. level 2 boundaries.

        # Setup paths and ensure directories exist
        self.gadm_config=self.get_gadm_config()
        
        self.gadm_root = Path(self.gadm_config['root'])
        self.gadm_root.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.gadm_processed = Path(self.gadm_config['processed'])
        self.gadm_processed.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.crs=self.get_default_crs()
        self.country_file:Path=Path (self.gadm_root) /  f'gadm41_{self.country}_L{self.admin_level}.geojson'
        self.region_file:Path = Path(self.gadm_processed) / f'gadm41_{self.country}_L{self.admin_level}_{self.region_short_code}.geojson'
        
    # Define a function to create bounding boxes (of cell) directly from coordinates (x, y) and resolution
    
    def get_country_boundary(self, 
                             country_level:bool=True,# for WB6 analysis, this is the default
                             force_update: bool = False) -> gpd.GeoDataFrame:
        """
        Retrieves and prepares the GADM boundaries dataset for the specified country.

        Args:
            force_update (bool): If True, re-fetch the GADM data even if a local file exists.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the country's GADM regions in crs '4326'
        """
        # store the user input (via method args)
        
        self.admin_level=1 if country_level else self.admin_level
        
        try:
            # country_gadm_regions_file_path = Path (self.gadm_root)/ f'gadm41_{self.country}_L{self.admin_level}.geojson'

            # Load or fetch data
            if self.country_file.exists() and not force_update: # load the country gdf from local file
                self.log.info(f">> Loading GADM data for {self.country} from local datafile {self.country_file}.")
                self.boundary_country=gpd.read_file(self.country_file)
                
            else:
                # Fetch and save data if file does not exist or force_update is True
                self.log.info(f">> Fetching GADM data for {self.country} at Administrative Level {self.admin_level}....from source: https://gadm.org/data.html")
                
                _country_gdf_:gpd.GeoDataFrame = pygadm.AdmItems(name=self.country, content_level=self.admin_level)
                _country_gdf_.set_crs(self.crs)
                self.boundary_country=_country_gdf_
                # save to local file
                self.boundary_country.to_file(self.country_file, driver='GeoJSON')
                self.log.info(f">> GADM data saved to {self.country_file}.")
        
                self.boundary_country = self.boundary_country[['NAME_0', 'NAME_1', 'NAME_2', 'geometry']].rename(columns={
                        'NAME_0': 'Country', 'NAME_1': 'region', 'NAME_2': 'Region'
                    })
            
            # if self.admin_level==1:
            #     self.boundary_country_aggr = self.boundary_country.dissolve(by="Country")  
            #     self.boundary_country_aggr.reset_index(inplace=True)
            #     self.boundary_country = self.boundary_country_aggr[['Country', 'geometry']]
            
            return self.boundary_country

        except Exception as e:
            self.log.error(f">> Error fetching or loading GADM data: {e}")
            raise

    def get_region_boundary(self,
                            country_level:bool=False, # for WB6 analysis, this is the default
                            force_update: bool = False) -> gpd.GeoDataFrame:
        """
        Prepares the boundaries for the specified region within the country.

        Args:
            force_update (bool): To force update the data and formatting.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the region boundaries.
        """
        
        if self.region_code_validity:
    
            # It should be saved in processed because the column names have been modified from source.
            
            if self.region_file.exists() and not force_update: # There is a local file and no update required
                self.log.info(f">> Loading GADM boundaries (Sub-provincial | level =2) for {self.region_name}  from local file {self.region_file}.")
                self.boundary_region:gpd.GeoDataFrame=gpd.read_file(self.region_file)
            
            else: # When the local file for region doesn't exist, Filter region data from country file and save locally
                _boundary_country = self.get_country_boundary(force_update)
                if country_level:
                    
                #     self.admin_level=1
                    
                #     self.boundary_country_aggr = _boundary_country.dissolve(by="Country")  # or "region" if that's the right column
                #     self.boundary_country_aggr.reset_index(inplace=True)
                    
                #     self.region_file:Path = Path(self.gadm_processed) / f'gadm41_{self.country}_L{self.admin_level}_{self.region_short_code}.geojson'
                #     self.boundary_country_aggr.to_file(self.region_file, driver='GeoJSON')
                #     self.log.info(f"GADM data for {self.region_name} saved to {self.region_file}.")
                    
                #     return self.boundary_country_aggr
                
                # _boundary_region_ = _boundary_country.loc[self.boundary_country['NAME_1'] == self.region_name]
                    _boundary_region_ = _boundary_country

                if _boundary_region_.empty : 
                    self.log.error(f">> No data found for region '{self.region_name}'.")
                    exit(123)
                else:
                    if not country_level:
                        # Check which columns are present and select/rename accordingly
                        columns_to_keep = ['NAME_0', 'geometry']
                        rename_dict = {'NAME_0': 'Country'}
                        if 'NAME_1' in _boundary_region_.columns:
                            columns_to_keep.append('NAME_1')
                            rename_dict['NAME_1'] = 'region'
                        if 'NAME_2' in _boundary_region_.columns:
                            columns_to_keep.append('NAME_2')
                            rename_dict['NAME_2'] = 'Region'
                        _boundary_region_ = _boundary_region_[columns_to_keep].rename(columns=rename_dict)
                    
                    self.boundary_region: gpd.GeoDataFrame = _boundary_region_
                    self.boundary_region.to_file(self.region_file, driver='GeoJSON')
                    self.log.info(f"GADM data for {self.region_name} saved to {self.region_file}.")
                    
                return self.boundary_region
        else:
            self.log.error(f">> Invalid region code: {self.region_short_code}.")
            exit
    
    def get_bounding_box(self,
                         country_level:bool=True # for WB6 analysis, this is the default
                         )->tuple:
        
        """
        This method returns the Minimum Bounding Rectangle (MBR) to extract the 
        """
        if country_level:
             self.actual_boundary=self.get_country_boundary()
        else:
            self.actual_boundary=self.get_region_boundary()
        # self.log.info(f"Setting up the Minimum Bounding Region (MBR) for {self.region_short_code}...")
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
            
        """
        boundary_region = self.get_region_boundary()

        if boundary_region is not None:
            m = boundary_region.explore('Region', legend=True, tiles=basemap)
            
            if save:
                file_path = Path(save_path) / f"{self.region_short_code}.html"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                m.save(file_path)
                self.log.info(f">> Interactive map for '{self.region_short_code}' saved to {file_path}.")
            else:
                self.log.info(f">> Skipping the save to local directories as 'save' is set to False.")
        
        return m

    def run(self):
        """
        Executes the process of extracting boundaries and creating an interactive map.
        """
        if self.region_code_validity:
            region_gadm_gdf=self.get_region_boundary()
            self.get_bounding_box()
            self.show_regions()
            return region_gadm_gdf
        else:
            self.log.error(">> region code is not valid.")
            self.region_code_validity
            return None
