import warnings

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from RESource import utility as utils
from RESource.AttributesParser import AttributesParser
from RESource.boundaries import GADMBoundaries
from RESource.era5_cutout import ERA5Cutout
from RESource.hdf5_handler import DataHandler


class GridCells(AttributesParser):
    """
    Spatial grid cell generator for renewable energy resource assessment.

    This class creates a regular spatial grid covering a specified region for
    discretized renewable energy potential analysis. It inherits from AttributesParser
    for configuration management and integrates with ERA5Cutout and GADMBoundaries
    to maintain consistency with climate data spatial resolution and regional boundaries.

    Grid cells serve as the fundamental spatial units for capacity calculations,
    land availability analysis, and resource aggregation. Each cell represents
    a homogeneous area with uniform resource characteristics and constraints.


    Parameters
    ----------
    config_file_path : str or Path
        Path to configuration file containing grid settings
    region_short_code : str
        Region identifier for boundary definition
    resource_type : str
        Resource type ('solar' or 'wind')

    Attributes
    ----------
    ERA5Cutout : ERA5Cutout
        ERA5 climate data cutout handler instance
    gadmBoundary : GADMBoundaries
        GADM boundary processor instance
    datahandler : DataHandler
        HDF5 data storage interface for grid persistence
    crs : str
        Coordinate reference system ('EPSG:4326')
    resolution : dict
        Grid resolution with 'dx' and 'dy' keys (decimal degrees)
    bounding_box : dict
        Spatial extent with 'minx', 'maxx', 'miny', 'maxy'
    actual_boundary : gpd.GeoDataFrame
        Precise regional boundary geometry
    coords : dict
        Grid coordinate arrays {'x': array, 'y': array}
    shape : tuple
        Grid dimensions (rows, columns)
    bounding_box_grid : gpd.GeoDataFrame
        Complete grid covering bounding box region
    grid_cells : gpd.GeoDataFrame
        Final grid cells intersecting with regional boundary (custom grid)
    cutout : atlite.Cutout
        ERA5 cutout object with climate data
    region_boundary : gpd.GeoDataFrame
        Regional boundary from ERA5 processing
    resource_grid_cells : gpd.GeoDataFrame
        Grid cells from default ERA5-based processing

    Methods
    -------
    generate_coords() -> None
        Create coordinate arrays based on resolution and boundary
    __get_grid__() -> gpd.GeoDataFrame
        Generate complete grid with cell geometries (private method)
    get_custom_grid() -> gpd.GeoDataFrame
        Create custom grid cells intersecting with regional boundary
    get_default_grid() -> gpd.GeoDataFrame
        Create grid using ERA5 cutout methodology with climate data alignment
    _check_resolution() -> None
        Validate resolution settings and issue warnings (private method, not currently used)

    Examples
    --------
    Generate grid for British Columbia wind assessment:

    >>> from RESource.cell import GridCells
    >>> grid = GridCells(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind"
    ... )
    >>> # Using custom grid approach
    >>> custom_cells = grid.get_custom_grid()
    >>> print(f"Generated {len(custom_cells)} custom grid cells")
    >>>
    >>> # Using default ERA5-aligned grid
    >>> default_cells = grid.get_default_grid()
    >>> print(f"Generated {len(default_cells)} ERA5-aligned grid cells")

    Custom resolution configuration:

    >>> # In configuration file (config_BC.yaml):
    >>> # grid_cell_resolution:
    >>> #   dx: 0.25  # 0.25 degrees longitude
    >>> #   dy: 0.25  # 0.25 degrees latitude
    >>> grid._check_resolution()  # Validate resolution settings

    Notes
    -----
    - Default resolution matches ERA5 climate data (0.25° x 0.25°)
    - Grid cells are represented as square polygons with centroid coordinates
    - Inherits configuration management from AttributesParser
    - Integrates with ERA5Cutout for climate data alignment
    - Uses GADMBoundaries for precise regional boundary definition
    - Uses HDF5 storage for efficient caching of large grid datasets
    - Grid generation respects regional boundaries to avoid unnecessary cells
    - Resolution warnings issued if finer than climate data resolution
    - Coordinate system maintained as WGS84 for global compatibility
    - Supports both custom grid generation and ERA5-aligned grid generation

    Grid Generation Approaches
    --------------------------
    1. Custom Grid (get_custom_grid()):
       - Creates grid based on regional bounding box
       - Intersects with precise regional boundaries
       - Stores results in HDF5 with 'cells' key

    2. Default Grid (get_default_grid()):
       - Uses ERA5 cutout grid as base
       - Aligns with climate data resolution
       - Overlays with regional boundaries
       - Stores both 'cells' and 'boundary' in HDF5

    Resolution Considerations
    -------------------------
    - Minimum recommended: 0.25° (matching ERA5 resolution)
    - Harmonized resolutions required for interpolation of climate data
    - Coarser resolutions may miss local variations in resource quality
    - Square cells assumed (dx = dy) for geometric consistency
    - Resolution validation available via _check_resolution() method

    Dependencies
    ------------
    - geopandas: Spatial data manipulation
    - numpy: Numerical operations for coordinate generation
    - shapely.geometry.box: Grid cell geometry creation
    - RES.AttributesParser: Parent class for configuration management
    - RES.boundaries.GADMBoundaries: Regional boundary processing
    - RES.era5_cutout.ERA5Cutout: Climate data cutout handling
    - RES.hdf5_handler.DataHandler: HDF5 data storage interface
    - RES.utility: Utility functions for cell ID assignment and logging
    """

    def __post_init__(self):
        """
        Initializes the bounding box and resolution after the parent class initialization.
        Accepts a resolution dictionary to define the x and y resolutions.
        """
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()

        # This dictionary will be used to pass arguments to external classes
        self.required_args = {  # order doesn't matter
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type,
        }

        self.ERA5Cutout = ERA5Cutout(**self.required_args)
        self.gadmBoundary = GADMBoundaries(**self.required_args)

        ## Initiate the Store and Datahandler (interfacing with the Store)
        self.datahandler = DataHandler(self.store)

    def _check_resolution(
        self,
    ):  # not is use for now, future scope when user up/down scales the resolution
        """Check if the resolution meets the conditions and issue warnings."""

        self.resolution = self.get_cell_resolution()
        # Default resolution if none provided
        if self.resolution is None:
            self.resolution = {"dx": 0.25, "dy": 0.25}

        dx = self.resolution.get("dx", 0.25)
        dy = self.resolution.get("dy", 0.25)

        # Check if dx and dy are not the same
        if dx != dy:
            warnings.warn(
                f">> Resolution mismatch: dx ({dx}) and dy ({dy}) are not equal.\n Check 'dx' and 'dy' values of 'grid_cell_resolution' key in user config ",
                UserWarning,
                stacklevel=2,
            )

        # Check if dx or dy are lower than 0.25
        if dx < 0.25 or dy < 0.25:
            warnings.warn(
                f">> Resolution too fine: dx ({dx}) or dy ({dy}) is lower than weather data resolution (0.25x0.25). Consider increasing it.",
                UserWarning,
                stacklevel=2,
            )

    def generate_coords(self):
        # Get bounding box and actual boundary from parent class method
        self.bounding_box, self.actual_boundary = self.gadmBoundary.get_bounding_box()

        """Generate the coordinates for the grid points (centroids)."""
        minx, maxx = self.bounding_box["minx"], self.bounding_box["maxx"]
        miny, maxy = self.bounding_box["miny"], self.bounding_box["maxy"]

        # Use resolution from the dictionary for x and y
        resolution_x = self.resolution["dx"]
        resolution_y = self.resolution["dy"]

        x_values = np.arange(minx - resolution_x, maxx + resolution_x, resolution_x)
        y_values = np.arange(miny - resolution_x, maxy + resolution_x, resolution_y)

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
        if not hasattr(self, "coords"):
            self.generate_coords()

        # Create mesh grid of coordinates
        xs, ys = np.meshgrid(self.coords["x"], self.coords["y"])
        coords = np.asarray((np.ravel(xs), np.ravel(ys))).T

        # Calculate span to determine grid cell size
        span = (coords[self.shape[1] + 1] - coords[0]) / 2

        # Generate grid cells (boxes)
        cells = [box(*c) for c in np.hstack((coords - span, coords + span))]
        self.bounding_box_grid = gpd.GeoDataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "geometry": cells},
            crs=self.crs_d,
        )
        # Return GeoDataFrame with centroids and grid cells
        return self.bounding_box_grid

    # def get_custom_grid(self):
    #     self.bounding_box_grid=self.__get_grid__()
    #     _grid_cells_=self.boundary_region.overlay(self.bounding_box_grid,how='intersection',keep_geom_type=True)
    #     self.grid_cells=utils.assign_cell_id(_grid_cells_)

    #     self.datahandler.to_store(self.grid_cells,'cells',force_update_key=True)

    #     utils.print_update(level=2,message=f"{len(self.grid_cells)} Grid Cells prepared for {self.region_short_code}.")

    #     return self.grid_cells

    def get_default_grid(self):
        self.cutout, self.region_boundary = self.ERA5Cutout.get_era5_cutout(
            weather_year=self.weather_year
        )

        resource_grid_cells = (
            self.cutout.grid
            # .to_crs(self.crs_m)
            # .assign(ERA5_cell_area_km2=lambda df: df.geometry.area / 1e6)
        )

        # region_boundary_m = self.region_boundary.to_crs(self.crs_m)

        resource_grid_cells = resource_grid_cells.overlay(self.region_boundary, how="intersection")

        self.resource_grid_cells = utils.assign_cell_id(
            resource_grid_cells, source_column=self.gadmBoundary.sub_national_unit_tag
        )

        # self.resource_grid_cells_gdf_m["geom_area_km2"] = self.resource_grid_cells_gdf_m.geometry.area / 1e6
        # self.resource_grid_cells_gdf_m["geom_area_share"] = (
        #     self.resource_grid_cells_gdf_m["geom_area_km2"] /
        #     self.resource_grid_cells_gdf_m["ERA5_cell_area_km2"]
        # )

        # self.resource_grid_cells=self.resource_grid_cells_gdf.to_crs(self.crs_d)
        self.datahandler.to_store(self.resource_grid_cells, "cells")
        self.datahandler.to_store(self.region_boundary, "boundary")

        return self.resource_grid_cells
