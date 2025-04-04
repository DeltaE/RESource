
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import warnings
from RES.hdf5_handler import DataHandler
from RES import utility as utils
from RES.era5_cutout import ERA5Cutout

class GridCells(ERA5Cutout):
    
    def __post_init__(self):
        """
        Initializes the bounding box and resolution after the parent class initialization.
        Accepts a resolution dictionary to define the x and y resolutions.
        """
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        ## Initiate the Store and Datahandler (interfacing with the Store)
        self.datahandler=DataHandler(self.store)
        self.crs=self.get_default_crs()
        

    def _check_resolution(self): # not is use for now, future scope when user up/down scales the resolution
        """Check if the resolution meets the conditions and issue warnings."""

        self.resolution=self.get_cell_resolution()
        self._check_resolution()
        # Default resolution if none provided
        if self.resolution is None:
            self.resolution = {'dx': 0.25, 'dy': 0.25}

        dx = self.resolution.get('dx', 0.25)
        dy = self.resolution.get('dy', 0.25)
        
        # Check if dx and dy are not the same
        if dx != dy:
            warnings.warn(f">> Resolution mismatch: dx ({dx}) and dy ({dy}) are not equal.\n Check 'dx' and 'dy' values of 'grid_cell_resolution' key in user config ", UserWarning)

        # Check if dx or dy are lower than 0.25
        if dx < 0.25 or dy < 0.25:
            warnings.warn(f">> Resolution too fine: dx ({dx}) or dy ({dy}) is lower than weather data resolution (0.25x0.25). Consider increasing it.", UserWarning)
       

    def generate_coords(self):
        # Get bounding box and actual boundary from parent class method
        self.bounding_box, self.actual_boundary = self.get_bounding_box()
        
        """Generate the coordinates for the grid points (centroids)."""
        minx, maxx = self.bounding_box['minx'], self.bounding_box['maxx']
        miny, maxy = self.bounding_box['miny'], self.bounding_box['maxy']
        
        # Use resolution from the dictionary for x and y
        resolution_x = self.resolution['dx']
        resolution_y = self.resolution['dy']
        
        x_values = np.arange(minx-resolution_x, maxx+resolution_x, resolution_x)
        y_values = np.arange(miny-resolution_x, maxy+resolution_x, resolution_y)
        
        self.coords = {"x": x_values, "y": y_values}
        self.shape = (len(y_values), len(x_values))  # shape as (rows, columns)

    def __get_grid__(self):
        """
        Grid with coordinates and geometries.
        The coordinates represent the centers of the grid cells.
        * Adopted from atlite.Cutout.grid method.
        
        Returns
        -------
        geopandas.GeoDataFrame
            DataFrame with coordinate columns 'x' and 'y', and geometries of the
            corresponding grid cells.
        """
        if not hasattr(self, 'coords'):
            self.generate_coords()
        
        # Create mesh grid of coordinates
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        coords = np.asarray((np.ravel(xs), np.ravel(ys))).T

        # Calculate span to determine grid cell size
        span = (coords[self.shape[1] + 1] - coords[0]) / 2

        # Generate grid cells (boxes)
        cells = [box(*c) for c in np.hstack((coords - span, coords + span))]
        self.bounding_box_grid=gpd.GeoDataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "geometry": cells},
            crs=self.crs,
        )
        # Return GeoDataFrame with centroids and grid cells
        return self.bounding_box_grid
    
    def get_custom_grid(self):
        self.bounding_box_grid=self.__get_grid__()
        _grid_cells_=self.boundary_region.overlay(self.bounding_box_grid,how='intersection',keep_geom_type=True)
        self.grid_cells=utils.assign_cell_id(_grid_cells_)
        
        self.datahandler.to_store(self.grid_cells,'cells',force_update_key=True)
        
        self.log.info(f">> {len(self.grid_cells)} Grid Cells prepared for {self.region_short_code}.")
                                  
        return self.grid_cells
    
    def get_default_grid(self):
        self.cutout,self.region_boundary=self.get_era5_cutout()
        _era5_grid_cells_gdf_=self.cutout.grid
        _resource_grid_cells_gdf_=_era5_grid_cells_gdf_.overlay(self.region_boundary)
        self.resource_grid_cells=utils.assign_cell_id(_resource_grid_cells_gdf_)
        self.datahandler.to_store(self.resource_grid_cells,'cells')
        self.datahandler.to_store(self.region_boundary,'boundary')
        return self.resource_grid_cells
    
   