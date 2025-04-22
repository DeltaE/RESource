from collections import namedtuple
import geopandas as gpd
import xarray as xr
import pandas as pd
from shapely.geometry import Polygon
from typing import Dict
import matplotlib.pyplot as plt
import xarray as xr

# Local Packages
import RES.utility as utils
# from RES.AttributesParser import AttributesParser
from RES.lands import LandContainer
from RES.era5_cutout import ERA5Cutout
from RES.hdf5_handler import DataHandler
from RES.atb import NREL_ATBProcessor


class CellCapacityProcessor(LandContainer,
                            ERA5Cutout,
                            ):
    
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
        self.atb=NREL_ATBProcessor(**self.required_args)
        
        (self.utility_pv_cost, 
        self.land_based_wind_cost, 
        self.bess_cost)= self.atb.pull_data()
        
        ## Initiate the Store and Datahandler (interfacing with the Store)
        self.datahandler=DataHandler(self.store)

        ## Load geospatial data (geodataframes)
        # self.gadm=GADMBoundaries(**self.required_args)
        
        ### Exclusion Layer Container
        # self.land_container=LandContainer(**self.required_args)
    
        ### ERA5 Cutout
        # self.era5_cutout=ERA5Cutout(**self.required_args)
        self.cell_resolution=self.cutout_config['dx']
    
    def set_composite_excluder(self):
        return LandContainer.set_excluder()
    
    # def __get_raster_path__(self, config, root, rasters_dir):
    #     return os.path.join(root, rasters_dir , config['zip_extract_direct'], config['raster'])
    
    def __get_unified_province_shape__(self):
        self.province_shape=self.province_boundary.dissolve(by='Province').drop(columns =['Region'])
        # self.province_shape=self.store_grid_cells.dissolve(by='Province').drop(columns =['Region'])
        return self.province_shape
    
    def load_cost(self,
                  resource_atb:pd.DataFrame):
        
        grid_connection_cost_per_km = self.disaggregation_config.get('transmission', {}).get('grid_connection_cost_per_Km', 0)
        tx_line_rebuild_cost = self.disaggregation_config.get('transmission', {}).get('tx_line_rebuild_cost', 0)
        
        self.ATB:Dict[str,dict]=super().get_atb_config()
        source_column:str= self.ATB.get('column',{})
        cost_params_mapping:Dict[str,str]=self.ATB.get('cost_params',{})
        
        
        # capex,fom,vom in NREL is given in US$/kw and we need to convert it to million $/MW
        resource_capex:float=resource_atb[resource_atb[source_column]==cost_params_mapping.get('capex',{})].value.iloc[0]/ 1E3  # Convert to million $/MW
        resource_fom:float=resource_atb[resource_atb[source_column]==cost_params_mapping.get('fom',{})].value.iloc[0] /1E3  # Convert to million $/MW
        
        # Initialize resource_vom based on the availability of 'vom' in cost_params_mapping
        resource_vom: float = 0  # Default value if 'vom' is not found
        

        if cost_params_mapping.get('vom') is not None:
            # Check if the DataFrame 'utility_scale_cost' is not empty and get the value for 'vom'
            if not resource_atb.empty:
                vom_row = resource_atb[resource_atb[source_column] == cost_params_mapping['vom']]
                if not vom_row.empty:
                    resource_vom = vom_row['value'].iloc[0] / 1E3  # Convert to million $/MW

        return (resource_capex, # in million $/MW
                resource_vom,   # in million $/MW
                resource_fom,  # in million $/MW
                grid_connection_cost_per_km,  # in million $
                tx_line_rebuild_cost) # in million $
    
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
        self.cutout,self.province_boundary=self.get_era5_cutout()
        # self.cutout,_=self.era5_cutout.get_era5_cutout()
        # loads complete boundary of regions.
        
    #b. load excluder
        composite_excluder=self.set_excluder()
        
    #d. Load costs (float)
        (
        self.resource_capex, 
        self.resource_fom, 
        self.resource_vom,
        self.grid_connection_cost_per_km,
        self.tx_line_rebuild_cost
        ) = self.load_cost(
                resource_atb=(
                    self.utility_pv_cost if self.resource_type == 'solar' else
                    self.land_based_wind_cost if self.resource_type == 'wind' else
                    self.bess_cost if self.resource_type == 'bess' else
                    None
                    )
                )
        
    # OPTIONAL step (skipping to save memory+processing time)
        ## 1. Calculate shape availability after adding the composite exclusion layer (excluder)
        # c. create a province boundary (union of all regions) to pass it onto excluder.
        # Because passing regional geoms will increase the calculation time.
        # _province_shape_gdf=self.__get_unified_province_shape__()
        # _province_shape_geom=_province_shape_gdf.geometry.to_crs(composite_excluder.crs)
        # masked, transform = composite_excluder.compute_shape_availability(_province_shape_geom)
        # cell_processor.excluder.plot_shape_availability(cell_processor.province_boundary)
        # The masked object is a numpy array. Eligible raster cells have a 1 and excluded cells a 0. 
        # Note that this data still lives in the projection of excluder. For calculating the eligible share we can use the following routine.
        # eligible_share = masked.sum() * composite_excluder.res**2 / _province_shape_geom.area.sum()
        
        # print(f"The land eligibility share is: {eligible_share:.2%}")
        
        # Visuals
        # fig, ax = plt.subplots()
        # excluder.plot_shape_availability(test.geometry)
        
    ## 2.1 Compute availability Matrix
        self.province_shape= self.__get_unified_province_shape__()
        self.Availability_matrix:xr = self.cutout.availabilitymatrix(self.province_shape, composite_excluder)
        self.plot_ERAF5_grid_land_availability()
        self.plot_excluder_land_availability()
        
        area = self.cutout.grid.set_index(["y", "x"]).to_crs(3035).area / 1e6 # This crs is fit for area calculation
        area = xr.DataArray(area, dims=("spatial"))

        capacity_matrix:xr.DataArray = self.Availability_matrix.stack(spatial=["y", "x"]) * area * self.resource_landuse_intensity
        self.capacity_matrix=capacity_matrix.rename(f'potential_capacity_{self.resource_type}')

    ## 2.1 convert the Availability Matrix to dataframe.
        _provincial_cell_capacity_df:pd.DataFrame=self.capacity_matrix.to_dataframe()
        
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
        parameters_to_add = {
            'capex': self.resource_capex, # m$/MW
            'fom': self.resource_fom, # m$/MW
            'vom': round(self.resource_vom, 4), # m$/MW
            'grid_connection_cost_per_km': self.grid_connection_cost_per_km, # m$/km
            'tx_line_rebuild_cost': self.tx_line_rebuild_cost,  # m$/km
            'Operational_life': int(25) if self.resource_type == 'solar' else int(20) if self.resource_type == 'wind' else 0   # years
        }

        # Create a new dictionary with stylized keys
        stylized_columns = {f'{key}_{self.resource_type}': value for key, value in parameters_to_add.items()}

        # Assign the new stylized columns to the DataFrame
        _provincial_cell_capacity_gdf = _provincial_cell_capacity_gdf.assign(**stylized_columns)

    ## 4 Trim the cells to sub-provincial boundaries instead of overlapping cell (boxes) in the regional boundaries.
        _provincial_cell_capacity_gdf=_provincial_cell_capacity_gdf.overlay(self.province_boundary)
        self.provincial_cells=utils.assign_cell_id(_provincial_cell_capacity_gdf)

        ''' 
        >>> here, self.provincial_cells = our default  grid cells
        ## Future Scope, while we will have user defined grid cells 
        
        # self.store_grid_cells=self.datahandler.from_store('cells')
        # Add new columns to the existing DataFrame
        # for column in self._updated_provincial_cells_.columns:
        #     self.store_grid_cells[column] = self._updated_provincial_cells_[column]
        '''
        
        # era5_cell_capacity=utils.assign_cell_id(_provincial_cell_capacity_gdf,'Region',self.site_index)
        # era5_cell_capacity=_provincial_cell_capacity_gdf
        
        # Define a namedtuple
        capacity_data = namedtuple('capacity_data', ['data','matrix','cutout'])
        
        self.resources_nt=capacity_data(self.provincial_cells,capacity_matrix,self.cutout)
        
        print(f">> Total ERA5 cells loaded : {len(self.provincial_cells)} [each with .025 deg. (~30km) resolution ]")
        self.log.info(">> Saving to the local store (as HDF5 file)")
        
        self.datahandler.to_store(self.provincial_cells,'cells')
     
        return self.resources_nt

## Visuals
    def show_capacity_map(self):
        
        gdf=self.resources_nt.data
        
        m = gdf.explore(
                    column=f'potential_capacity_{self.resource_type}',  # Use potential_capacity for marker size
                    markersize=f'potential_capacity_{self.resource_type}',  # Adjust marker size based on capacity
                    legend=True,
                    tiles='CartoDB dark_matter',
                    marker_type='circle',
                    icon='power'
                )
        return m
    
    def plot_ERAF5_grid_land_availability(self):
        
        province_boundary = self.province_boundary
        
        # Load availability data
        _AM_=self.Availability_matrix
        A_df=_AM_.to_dataframe(name="availability").reset_index()
        
        
        # Define bins and labels
        bins = [x / 100 for x in [0, 10, 30, 60, 90, 100]]  # Define bin edges
        labels = ["0-10%", "10-30%", "30-60%", "60-90%", ">90%"]
        
        
        # Categorize availability into bins
        A_df["availability_category"] = pd.cut(A_df["availability"], bins=bins, labels=labels, include_lowest=True)
        
        # Convert to GeoDataFrame
        A_gdf:gpd.GeoDataFrame = gpd.GeoDataFrame(
                    A_df,
                    geometry=[self.__create_cell_geom__(x, y) for x, y in zip(A_df['x'], A_df['y'])],
                    crs='EPSG:4326'
                )
                
        A_gdf=A_gdf.overlay(province_boundary)

        # Categorize availability into bins
        A_gdf["availability_category"] = pd.cut(A_gdf["availability"], bins=bins, labels=labels, include_lowest=True)
        # Create figure and axes for side-by-side plotting
        fig, ax = plt.subplots(figsize=(12, 8),constrained_layout=True)

        # Set axis off for both subplots
        ax.set_axis_off()

        # Shadow effect offset
        shadow_offset = 0.008

        # Plot solar map on ax1
        # Add shadow effect for solar map
        province_boundary.geometry = province_boundary.geometry.translate(xoff=shadow_offset, yoff=-shadow_offset)
        province_boundary.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=2, alpha=0.3)  # Shadow layer

        # Plot solar cells
        if self.province_short_code in ['AB', 'SK']:
            bbox_anchor_offset=(1.25, 1) # AB and SK has skewed map, custom tweak
        else:
            bbox_anchor_offset=(1.1, 1)
        A_gdf.plot(column='availability_category', ax=ax, cmap='Greens', legend=True, 
                legend_kwds={'title': "Land Availability", 'loc': 'upper right', 'bbox_to_anchor': bbox_anchor_offset,'borderpad': 1,'frameon': False})

        # Plot actual boundary for solar map
        province_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.7, alpha=0.9)
        plt.subplots_adjust(right=0.85)  # Increase space on the right
        # Adjust layout for cleaner appearance
        plt.tight_layout()
        plt.savefig(f'vis/misc/land_availability_ERA5grid_{self.province_name}.png')
        print(f"Land availability (grid cells) map saved at vis/misc/land_availability_ERA5grid_{self.province_name}.png")

    def plot_excluder_land_availability(self):

        fig, ax = plt.subplots()
        self.excluder.plot_shape_availability(self.province_shape,
                                              plot_kwargs={'facecolor':'none','edgecolor':'black'},
                                              ax=ax)
        ax.axis("off")
        plt.savefig(f'vis/misc/land_availability_excluderResolution_{self.province_name}.png')
        print(f"Land availability map (excluder resolution) saved at vis/misc/land_availability_excluderResolution_{self.province_name}.png")