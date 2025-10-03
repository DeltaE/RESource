# Import Global Packages
from pathlib import Path
import geopandas as gpd
import pygadm
from dataclasses import dataclass
import inspect

# Import local packages
from RES.AttributesParser import AttributesParser
import RES.utility as utils
PRINT_LEVEL_BASE=4
@dataclass
class GADMBoundaries(AttributesParser):
    """
    GADM (Global Administrative Areas) boundary processor for regional analysis.
    
    This class handles the retrieval, processing, and management of administrative
    boundaries from the GADM dataset. It provides functionality to extract specific
    regional boundaries at administrative level 2 (typically states/provinces/districts)
    for renewable energy resource assessment areas.
    
    INHERITED METHODS FROM AttributesParser:
    ----------------------------------------
    - get_gadm_config() -> Dict[str, dict]: Get GADM configuration from config file
    - get_default_crs() -> str: Get default coordinate reference system ('EPSG:4326')
    - get_country() -> str: Get country name from config file
    - get_region_name() -> str: Get region name from config file using region_short_code
    - get_region_mapping() -> Dict[str, dict]: Get region mapping dictionary
    - is_region_code_valid() -> bool: Validate region short code
    - load_config() -> Dict[str, dict]: Load YAML configuration file
    - get_excluder_crs() -> int: Get recommended CRS for excluder operations
    - get_vis_dir() -> Path: Get visualization directory path
    - region_code_validity (property): Boolean property for region code validation
    - Plus other utility methods for config access
    
    OWN METHODS DEFINED IN THIS CLASS:
    ----------------------------------
    - get_country_boundary(country=None, force_update=False): Download and process complete country GADM boundaries
    - get_region_boundary(region_name=None, force_update=False): Extract and process specific regional boundary  
    - get_bounding_box(): Generate minimum bounding rectangle for region
    - show_regions(basemap='CartoDB positron', save_path='vis/regions', save=False): Create interactive map visualization
    - run(): Execute complete boundary processing workflow

    
    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing GADM settings
    region_short_code : str
        Short code identifying the target region within the country
    resource_type : str
        Resource type (passed through from parent workflow)
        
    Attributes
    ----------
    admin_level : int
        GADM administrative level (fixed at 2 for regional districts)
    gadm_root : Path
        Root directory for GADM data storage
    gadm_processed : Path
        Directory for processed regional boundary files
    crs : str
        Coordinate reference system ('EPSG:4326')
    country : str
        Country name extracted from configuration
    region_file : Path
        Path to processed regional boundary file
    boundary_datafields : dict
        Mapping of GADM fields to standardized field names
    country_file : Path
        Path to country-level GADM boundary file
    boundary_country : gpd.GeoDataFrame
        GeoDataFrame containing country-level boundaries
    boundary_region : gpd.GeoDataFrame
        GeoDataFrame containing region-specific boundaries
    actual_boundary : gpd.GeoDataFrame
        GeoDataFrame containing the actual regional boundary geometry
    bounding_box : dict
        Dictionary containing bounding box coordinates (minx, maxx, miny, maxy)
        
    Methods
    -------
    get_country_boundary(country=None, force_update=False) -> gpd.GeoDataFrame
        Download and process complete country GADM boundaries at administrative level 2
    get_region_boundary(region_name=None, force_update=False) -> gpd.GeoDataFrame
        Extract and process specific regional boundary with standardized field names
    get_bounding_box() -> tuple
        Generate minimum bounding rectangle for region and return (bounding_box_dict, boundary_gdf)
    show_regions(basemap='CartoDB positron', save_path='vis/regions', save=False) -> folium.Map
        Create interactive folium map visualization of regional boundaries
    run() -> gpd.GeoDataFrame or None
        Execute complete boundary processing workflow and return regional boundary GeoDataFrame
        
    Examples
    --------
    Extract British Columbia boundaries:
    
    >>> from RES.boundaries import GADMBoundaries
    >>> boundaries = GADMBoundaries(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind"
    ... )
    >>> bc_boundary = boundaries.get_region_boundary()
    >>> country_bounds = boundaries.get_country_boundary("Canada")
    >>> bbox, actual_boundary = boundaries.get_bounding_box()
    >>> interactive_map = boundaries.show_regions(save=True)
    >>> result = boundaries.run()  # Execute full workflow
    
    Notes
    -----
    - Uses pygadm package for GADM data access and download
    - Automatically handles data caching to avoid repeated downloads
    - Processes boundaries into GeoJSON format for efficient storage
    - Standardizes field names for consistent downstream processing
    - Administrative level 2 chosen to balance spatial resolution with data availability
    - All geometries maintained in WGS84 (EPSG:4326) for global compatibility
    - Region validation is performed using inherited region_code_validity property
    - Interactive maps are created using folium with optional save functionality
    
    Dependencies
    ------------
    - pygadm: GADM data access and processing
    - geopandas: Spatial data manipulation
    - folium: Interactive map visualization (via geopandas.explore())
    - pathlib: Path handling
    - RES.AttributesParser: Parent class for configuration management
    - RES.utility: Utility functions for logging and updates
    
    Raises
    ------
    ValueError
        If the country is not found in the GADM dataset or if region code is invalid
    Exception
        If there is an error fetching or loading the GADM data
    """
    
    def __post_init__(self):
        
     # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        self.required_args = {   #order doesn't matter
            "config_file_path" : self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
        }
        

        # Setup paths and ensure directories exist
        self.gadm_config = super().get_gadm_config()  # INHERITED METHOD from AttributesParser
        
        self.gadm_root = Path(self.gadm_config['root'])
        self.gadm_root.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        
        self.gadm_processed = Path(self.gadm_config['processed'])
        self.gadm_processed.mkdir(parents=True, exist_ok=True) # Creates parent directories if not exists.
        self.admin_level:int= self.gadm_config.get('admin_level',1) 
        self.boundary_datafields = self.gadm_config.get('datafield_mapping')

        self.country=self.get_country()  # INHERITED METHOD from AttributesParser

        self.region_file:Path = Path(self.gadm_processed) / f'gadm41_{self.country}_L{self.admin_level}_{self.region_short_code}.geojson'
        
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
        utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | Country Selected: {self.country}.")
        
        # store the user input (via method args)
        if country is not None:
            self.country = country.capitalize()

        utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | Country Selected: {self.country}.")
        self.country_file:Path=Path (self.gadm_root) /  f'gadm41_{self.country}_L{self.admin_level}.geojson'  
        
        try:
            # Load or fetch data
            if self.country_file.exists() and not force_update: # load the country gdf from local file
                utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | Loading GADM data for {self.country} from local datafile {self.country_file}.")
                self.boundary_country=gpd.read_file(self.country_file)

            else:
                # Fetch and save data if file does not exist or force_update is True
                utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | Fetching GADM data for {self.country} at Administrative Level {self.admin_level}....from source: https://gadm.org/data.html")
                
                _country_gdf_:gpd.GeoDataFrame = pygadm.AdmItems(name=self.country, content_level=self.admin_level)
                _country_gdf_.set_crs(self.crs_d)
                self.boundary_country=_country_gdf_
                # save to local file
                self.boundary_country.to_file(self.country_file, driver='GeoJSON')
                utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | GADM data saved to {self.country_file}.")
                
            return self.boundary_country

        except Exception as e:
            utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | Error fetching or loading GADM data: {e}")
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
            self.region_name =self.get_region_name()  # INHERITED METHOD from AttributesParser

        utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Region Set to >> Short Code : {self.region_short_code}, Name: {self.region_name}).")
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                           message=f"{__name__}| Collecting regional boundary...")

        if self.region_code_validity:
    
            # It should be saved in processed because the column names have been modified from source.
            
            if self.region_file.exists() and not force_update: # There is a local file and no update required
                utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__}| Loading GADM boundaries (Sub-provincial | level =2) for {self.region_name} from local file {self.region_file}.")
                
                self.boundary_region:gpd.GeoDataFrame=gpd.read_file(self.region_file)
            
            else: # When the local file for region doesn't exist, Filter region data from country file and save locally
                
                if self.multi_country_flag:
                    utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__} | Processing the Boundaries for Multi-Country Region : {self.country}.")
                    
                    _boundary_country = self.get_country_boundary(self.region_name,force_update)
                    _boundary_region_ = _boundary_country.loc[_boundary_country['NAME_0'] == self.region_name]
                    
                else:
                    _boundary_country = self.get_country_boundary(self.country,force_update)
                    _boundary_region_ = _boundary_country.loc[self.boundary_country['NAME_1'] == self.region_name]

                if _boundary_region_.empty : 
                    utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__}|  No data found for region '{self.region_name}'.")
                    utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__}| | @ LINE | Consider revising '{self.region_name}' to match source (e.g. https://gadm.org/maps.html); Select 'show sub-divisions' to get the list of Supported Regional Names")
                    raise ValueError(f"{__name__} | @ LINE {inspect.currentframe().f_lineno} | No data found for region '{self.region_name}'.")
                
                elif  self.admin_level==1:
                    _boundary_region_ = _boundary_region_[['NAME_0', 'NAME_1', 'geometry']].rename(columns={
                        'NAME_0': self.boundary_datafields['NAME_0'], 'NAME_1': self.boundary_datafields['NAME_1']
                    })
                    self.boundary_region:gpd.GeoDataFrame=_boundary_region_
                
                else:
                    _boundary_region_ = _boundary_region_[['NAME_0', 'NAME_1', 'NAME_2', 'geometry']].rename(columns={
                        'NAME_0': self.boundary_datafields['NAME_0'], 'NAME_1': self.boundary_datafields['NAME_1'], 'NAME_2': self.boundary_datafields['NAME_2']
                    })
                    self.boundary_region:gpd.GeoDataFrame=_boundary_region_
      
                
                self.boundary_region.to_file(self.region_file, driver='GeoJSON')
                utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__}| GADM data for {self.region_name} saved to {self.region_file}.")
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
        utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Processing regional bounding box...")
        
        self.actual_boundary=self.get_region_boundary()
        
        utils.print_update(level=PRINT_LEVEL_BASE+1,message=f"{__name__}| Setting up the Minimum Bounding Region (MBR) for {self.region_short_code}...")
        min_x, min_y, max_x, max_y=self.actual_boundary.geometry.total_bounds

        
        # MBR=box(min_x, min_y, max_x, max_y)
        self.bounding_box:dict={
            'minx': min_x,
            'maxx': max_x,
            'miny': min_y,
            'maxy': max_y
            }
        
        
        return self.bounding_box,self.actual_boundary
        
    def show_regions(self, 
                    basemap: str = 'CartoDB positron', 
                    save_path: str = 'vis/regions',
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
                utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Interactive map for '{self.region_short_code}' saved to {file_path}.")
            else:
                utils.print_update(level=PRINT_LEVEL_BASE+1,
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
            utils.print_update(level=PRINT_LEVEL_BASE+1,
                           message=f"{__name__}| Region code is not valid.")
            self.region_code_validity
            return None
