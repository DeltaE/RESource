from altair import Data
from linkingtool import era5_cutout
from linkingtool.AttributesParser import AttributesParser
from linkingtool.CellCapacityProcessor import CellCapacityProcessor
import atlite
from linkingtool.era5_cutout import ERA5Cutout
import linkingtool.linking_utility as utils
import xarray as xr
from dataclasses import dataclass, field
import pandas as pd
import geopandas as gpd
from collections import namedtuple
from linkingtool.hdf5_handler import DataHandler

@dataclass
class Timeseries(ERA5Cutout,
                 AttributesParser,
                 ):
    
    # Fields initialized later
    cell_capacity_gdf: object = field(init=False)
    cell_capacity_matrix: object = field(init=False)
    pv_panel_config: dict = field(init=False)
    layout_MW: object = field(init=False)
    
    def __post_init__(self):
        
        # Call the parent class __init__ (not __post_init__!)
        super().__post_init__()

        # Fetch the disaggregation configuration based on the resource type
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        self.cell_static_CF_tolerance=self.resource_disaggregation_config.get('cell_static_CF_tolerance',0)
        
        self.datahandler=DataHandler(store=self.store)
        
    def __process_PV_timeseries__(self):
        """ 
        A wrapper that leverage Atlite's _cutout.pv_ method to convert downward-shortwave, upward-shortwave radiation flux and ambient temperature into a pv generation time-series.
        """
        self.cutout,self.province_boundary=self.get_era5_cutout()
        self.province_grid_cells = self.cutout.grid.overlay(self.province_boundary, how='intersection',keep_geom_type=True)
        self.province_grid_cells = utils.assign_cell_id(self.province_grid_cells,'Region',self.site_index)
        self.log.info(f"Extracted {len(self.province_grid_cells)} ERA5 Grid Cells for {self.province_name} from Cutout")
        self.resource_type='solar'
        
        # Set arguments for the atlite cutout's pv method
        pv_args = {
            'panel': self.resource_disaggregation_config['atlite_panel'],
            'orientation': "latitude_optimal",
            'clearsky_model': None ,# ambient air temperature and relative humidity data not available
            'tracking': self.resource_disaggregation_config['tracking'],
            'layout': None,
            
            'matrix': None, 
            # (N x S - xr.DataArray or sp.sparse.csr_matrix or None) – If given, it is used to aggregate the grid cells to buses. 
            # N is the number of buses, S the number of spatial coordinates, in the order of cutout.grid
            
            'layout':None,
            # (X x Y - xr.DataArray) – The capacity to be build in each of the grid_cells.
            
            'shapes': self.province_grid_cells.geometry,
            #  (list or pd.Series of shapely.geometry.Polygon) – If given, matrix is constructed as indicatormatrix of the polygons, 
            # its index determines the bus index on the time-series.
            
            # 'capacity_factor_timeseries':True, # If True, the capacity factor time series of the chosen resource for each grid cell is computed.
            # 'return_capacity': False, # Additionally returns the installed capacity at each bus corresponding to layout (defaults to False).
            # 'capacity_factor':True, # If True, the static capacity factor of the chosen resource for each grid cell is computed.
            'per_unit':True, # Returns the time-series in per-unit units, instead of in MW (defaults to False).
            'show_progress': False, # Progress bar
        }

        # Generate PV timeseries profile using the atlite cutout
        self.pv_sites_profile: xr.DataArray = self.cutout.pv(**pv_args)
        
        return self.pv_sites_profile
    
    
    def get_cells_timeseries(self):
        """
        Generate the site-specific timeseries for PV sites and filter based on capacity factor (CF) thresholds.

        ### Args:
            Args are set to match the internal named tuples to pass the args as **kwargs.
        
        ### Returns:
            A named tuple 'site_data' containing:
            - cells: GeoDataFrame with grid cells and their calculated CF mean.
            - timeseries: DataFrame with timeseries data for the selected cells.
            
        ### Stores:
            Locally stores the timeseries and grid cells with CF_mean and regional mapping.
            @ 'data/store/solar_resources.h5'
        """
        
        # Step 1: Process the PV timeseries data for all sites
        self.pv_sites_profile = self.__process_PV_timeseries__()
        
        # Step 2: Extract start and end dates from the cutout configuration
        self.start_date = self.cutout_config['snapshots']['start'][0]
        self.end_date = self.cutout_config['snapshots']['end'][0]
        
        # Step 3: Convert the Xarray profile to a pandas DataFrame for easier manipulation
        sites = self.pv_sites_profile[self.site_index].to_pandas()
        
        # Create a datetime index from start to end date with hourly frequency
        datetime_index = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        _CF_ts_df = pd.DataFrame(index=datetime_index)  # Initialize an empty DataFrame for the timeseries

        # Step 4: Build a list of DataFrames for each site
        site_dfs = []
        for site in sites:
            # Create a DataFrame for each site based on its profile data and the defined datetime index
            site_df = pd.DataFrame({
                site: self.pv_sites_profile.sel(time=slice(self.start_date, self.end_date), 
                                                **{self.site_index: site}).values
            }, index=datetime_index)
            site_dfs.append(site_df)

        # Step 5: Concatenate all site-specific DataFrames into a single DataFrame
        _CF_ts_df = pd.concat(site_dfs, axis=1)

        # Step 6: Calculate the mean capacity factor (CF) for each cell and store it in 'CF_mean'
        self.log.info(f">> Calculating CF mean from the {len(_CF_ts_df)} data points for each Cell ...")
        self.province_grid_cells['CF_mean'] = _CF_ts_df.mean()

        # Step 7: Filter the grid cells based on the CF threshold (cell_static_CF_tolerance)
        # _CF_mask = self.province_grid_cells.CF_mean >= self.cell_static_CF_tolerance
        # self.province_grid_cells = self.province_grid_cells[_CF_mask]

        # Step 8: Filter the timeseries DataFrame to only include the selected cells after CF mask. This reduces less data offload to the store
        _CF_ts_df = _CF_ts_df[self.province_grid_cells.index]

        # Step 9: Convert the timeseries data to the appropriate province timezone
        self.province_timezone=self.get_province_timezone()
        self.CF_ts_df = self.__fix_timezone__(_CF_ts_df)
        self.CF_ts_df.tz_localize(None)
        
        # Step 10: Save the grid cells and timeseries to the local HDF5 store
        self.datahandler.to_store(self.province_grid_cells, 'cells')
        self.datahandler.to_store(self.CF_ts_df, 'timeseries')

        # Step 11: Define a namedtuple to store both the grid cells and the filtered timeseries
        site_data = namedtuple('site_data', ['cells', 'timeseries'])
        self.data = site_data(self.province_grid_cells, self.CF_ts_df)

        return self.data

    def __fix_timezone__(self,
        data:pd.DataFrame)->pd.DataFrame:
        '''
        This function converts the timeseries index with timezone information imputed conversion.<br>
        <b> Recommended timeseries index conversion method</b> in contrast to naive timestamp index reset method. 
        '''
        # Localize to UTC (assuming your times are currently in UTC)
        df_index_utc = data.tz_localize('UTC')

        # Convert to defined timezone (in Pandas time zones)
        df_index_converted = df_index_utc.tz_convert(self.province_timezone)
        
        df_index_converted.tz_localize(None) # without timezone conversion metadata
        
        return df_index_converted
