import logging as log
import os,sys
from collections import namedtuple
import atlite
from atlite.gis import ExclusionContainer,shape_availability
import pathlib as Path
import geopandas as gpd
import xarray as xr
import pandas as pd
from shapely.geometry import box,Point,Polygon

# Local Packages
import linkingtool.linking_utility as utils
# import linkingtool.visuals as vis
# import linkingtool.linking_solar as solar
from linkingtool.AttributesParser import AttributesParser
from linkingtool.boundaries import GADMBoundaries
from linkingtool.osm import OSMData
from linkingtool.lands import ConservationLands,LandContainer
from linkingtool.coders import CODERSData
from linkingtool.era5_cutout import ERA5Cutout
from linkingtool.find import GridNodeLocator
from linkingtool.hdf5_handler import DataHandler

# Logging Configuration
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
class CellCapacityProcessor(AttributesParser):
    
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
    
        # This dictionary will be used to pass arguments to external classes
        self.required_args = { #order doesn't matter
                "config_file_path": self.config_file_path,
                "province_short_code": self.province_short_code,
                "resource_type": self.resource_type
            }
        
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        self.resource_landuse_intensity = self.resource_disaggregation_config['landuse_intensity']
        
        ## Initiate the Store and Datahandler (interfacing with the Store)
        self.datahandler=DataHandler(store=self.store)


        ## Load geospatial data (geodataframes)
        # self.gadm=GADMBoundaries(**self.required_args)
        
        ### Exclusion Layer Container
        self.land_container=LandContainer(**self.required_args)
    
        ### ERA5 Cutout
        self.era5_cutout=ERA5Cutout(**self.required_args)
        self.cell_resolution=self.era5_cutout.cutout_config['dx']
    
    def set_composite_excluder(self):
        return self.land_container.set_excluder()
    
    # def __get_raster_path__(self, config, root, rasters_dir):
    #     return os.path.join(root, rasters_dir , config['zip_extract_direct'], config['raster'])
    
    def __get_unified_province_shape__(self):
        self.province_shape=self.province_boundary.dissolve(by='Province').drop(columns =['Region'])
        # self.province_shape=self.store_grid_cells.dissolve(by='Province').drop(columns =['Region'])
        return self.province_shape
    
    # Define a function to create bounding boxes (of cell) directly from coordinates (x, y) and resolution
    def __create_cell_geom__(self,x, y):
        half_res = self.cell_resolution / 2
        return Polygon([
            (x - half_res, y - half_res),  # Bottom-left
            (x + half_res, y - half_res),  # Bottom-right
            (x + half_res, y + half_res),  # Top-right
            (x - half_res, y + half_res)   # Top-left
        ])
        
    def get_capacity(self):
    #a. load cutout and province boundary for which the cutout has been created.
        self.cutout,self.province_boundary=self.era5_cutout.get_era5_cutout()
        # self.cutout,_=self.era5_cutout.get_era5_cutout()
        # loads complete boundary of regions.
        
    #b. load excluder
        composite_excluder=self.land_container.set_excluder()
        
    #c. create a province boundary (union of all regions) to pass it onto excluder.
        # Because passing regional geoms will increase the calculation time.
        _province_shape_gdf=self.__get_unified_province_shape__()
        _province_shape_geom=_province_shape_gdf.geometry.to_crs(composite_excluder.crs)
                
    #d. Load costs (float)
        (
            self.resource_capex, 
            self.resource_fom, 
            self.resource_vom,
            self.grid_connection_cost_per_km,
            self.tx_line_rebuild_cost
        ) = self.load_cost()
        
    ## 1. Calculate shape availability after adding the composite exclusion layer (excluder)
        masked, transform = composite_excluder.compute_shape_availability(_province_shape_geom)
        
        # The masked object is a numpy array. Eligible raster cells have a 1 and excluded cells a 0. 
        # Note that this data still lives in the projection of excluder. For calculating the eligible share we can use the following routine.
        eligible_share = masked.sum() * composite_excluder.res**2 / _province_shape_geom.area.sum()
        
        print(f"The land eligibility share is: {eligible_share:.2%}")
        
        # Visuals
        # fig, ax = plt.subplots()
        # excluder.plot_shape_availability(test.geometry)
        
    ## 2.1 Compute availability Matrix
        Availability_matrix:xr = self.cutout.availabilitymatrix(self.province_shape, composite_excluder)
        
        area = self.cutout.grid.set_index(["y", "x"]).to_crs(3035).area / 1e6 # This crs is fit for area calculation
        area = xr.DataArray(area, dims=("spatial"))

        capacity_matrix:xr.DataArray = Availability_matrix.stack(spatial=["y", "x"]) * area * self.resource_landuse_intensity
        capacity_matrix.rename('potential_capacity')

    ## 2.1 convert the Availability Matrix to dataframe.
        _provincial_cell_capacity_df:pd.DataFrame=capacity_matrix.to_dataframe('potential_capacity')
        
        # filter the cells that has no lands (i.e. no potential capacity)
        # _provincial_cell_capacity_df = _provincial_cell_capacity[_provincial_cell_capacity["potential_capacity"] != 0]

        # The xarray doesn't create cell geometries by default. We hav to create it.
        # Apply the bounding box (cell) creation to the DataFrame's x,y coordinates (centroid of the cells)
        _provincial_cell_capacity_gdf:gpd.GeoDataFrame = gpd.GeoDataFrame(
            _provincial_cell_capacity_df,
            geometry=[self.__create_cell_geom__(x, y) for x, y in zip(_provincial_cell_capacity_df['x'], _provincial_cell_capacity_df['y'])],
            crs=self.get_default_crs()
        )
        
    ## 3 Assign Static exogenous Costs after potential capacity calculation
        _provincial_cell_capacity_gdf = _provincial_cell_capacity_gdf.assign(
            capex=self.resource_capex,
            fom=self.resource_fom,
            vom = round(self.resource_vom, 4),
            grid_connection_cost_per_km=self.grid_connection_cost_per_km,
            tx_line_rebuild_cost=self.tx_line_rebuild_cost
        )
        
    ## 4 Trim the cells to sub-provincial boundaries instead of overlapping cell (boxes) in the regional boundaries.
        _provincial_cell_capacity_gdf=_provincial_cell_capacity_gdf.overlay(self.province_boundary)
        _updated_provincial_cells_=utils.assign_cell_id(_provincial_cell_capacity_gdf)
        
        self.store_grid_cells=self.datahandler.from_store('cells')
        # Add new columns to the existing DataFrame
        for column in _updated_provincial_cells_.columns:
            self.store_grid_cells[column] = _updated_provincial_cells_[column]
        
        self.datahandler.to_store(self.store_grid_cells,'cells')
        
        # era5_cell_capacity=utils.assign_cell_id(_provincial_cell_capacity_gdf,'Region',self.site_index)
        # era5_cell_capacity=_provincial_cell_capacity_gdf
        
        # Define a namedtuple
        capacity_data = namedtuple('capacity_data', ['data','matrix','cutout'])
        
        self.solar_resources_nt=capacity_data(self.store_grid_cells,capacity_matrix,self.cutout)
        
        print(f"Total ERA5 cells loaded : {len(self.store_grid_cells)} [each with .025 deg. (~30km) resolution ]")
        self.log.info(f"Saving to the local store (as HDF5 file)")
        # self.datahandler.save_to_hdf(era5_cell_capacity,'cells')
        
        return self.solar_resources_nt

## Visuals
    def show_capacity_map(self):
        
        gdf=self.solar_resourced_nt.data
        
        m = gdf.explore(
                    column='potential_capacity',  # Use potential_capacity for marker size
                    markersize='potential_capacity',  # Adjust marker size based on capacity
                    legend=True,
                    tiles='CartoDB dark_matter',
                    marker_type='circle',
                    icon='power'
                )
        return m