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
from linkingtool.tech import OEDBTurbines
import yaml
from requests import get

@dataclass
class Timeseries(ERA5Cutout,
                 AttributesParser,
                 ):
    
    def __post_init__(self):
        
        super().__post_init__()

        # Fetch the disaggregation configuration based on the resource type
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        
        # Initialize the local store
        self.datahandler=DataHandler(store=self.store)
    
    def get_timeseries(self)-> tuple:
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
        
        ### Future Scope:
            Plug-in multiple sources to fit timeseries data e.g. [NSRDB, NREL](https://nsrdb.nrel.gov/data-sets/how-to-access-data)
        """
        if self.resource_type=='solar':
            # Step 1: Set-up Technology parameters and extract the synthetic timeseries data for all sites
            self.sites_profile:xr.DataArray = self.__process_PV_timeseries__()
            
        elif self.resource_type=='wind':
            # Step 1: Set-up Technology parameters and extract the synthetic timeseries data for all sites
            self.sites_profile:xr.DataArray = self.__process_WIND_timeseries__('OEDB',2)
        
        '''
        Here, 'sites_profile_profile' is a 2D array i.e. cell(Region_xcoord_ycoord) and timestamps. 
        For xarray.DataArray.to_pandas we use DataArray.to_pandas() thus, 2D -> pandas.DataFrame
        '''
         
        # Step 2: Convert the xarray:DataArray to pandas dataframe for easier navigation to site profile via site index (id).
        '''
        >>>>> Using "to_dataframe()" and then ".unstack()" methods instead for incorporating future scopes     
        - self.pv_sites_profile.to_pandas() # We convert the Xarray to pandas df for easier access to data by using cell_indices. 
        - The array index order is (time, cell) hence in pandas 'time' will be default index and 'cell' default header.
        '''
        self._CF_ts_df_ :pd.DataFrame=self.sites_profile.to_dataframe().unstack('cell') 
        ''' 
        We already rename the Xarray to 'self.resource_type' i.e solar/wind at the end of "__process_PV_timeseries__()" method. Hence now the Xarray could be transformed to wide-format DataFrame.
        - using the to_dataframe() method in xarray.DataArray, the behavior is different from to_pandas().
        - The array index order is (time, cell) hence in pandas 'time' will be default index and 'cell' default header. But now it will have an additional "Y" index "PV" adopted from xarray name.
        - WIND profile will be stored under same index to generate a synthetic hybrid availability (correlational) profile.
        '''
        # here, "_CF_ts_df_" will provide same data formate alike .to_pandas() method, just have to use "_CF_ts_df_.PV"  ("PV" is the xarray name)
        
        # Step 3: Convert the timeseries data to the appropriate province timezone
        self.province_timezone=self.get_province_timezone()
        self.CF_ts_df = self.__fix_timezone__(self._CF_ts_df_).tz_localize(None)
        '''
        - We localize the datetime-stamp (i.e. removing the timezone information) to sync the requirements for downstream models.
        - The naive timestamps (without timezone info) found better harmonized with the other data sources.
        - This step needs to be tailored by the user to harmonize the timeseries with other operational data.
        '''
        
        # Step 4: Calculate the mean capacity factor (CF) for each cell and store it in 'CF_mean'
        self.log.info(f">> Calculating CF mean from the {len(self.CF_ts_df)} data points for each Cell ...")
        self.log.info(f">> Total Grid Cells: {len(self.province_grid_cells_store)}, "
                      f">> Timeseries Generated for: {len(self.CF_ts_df.columns)}, "
                      f">> Matched Sites: {self.CF_ts_df[self.resource_type][self.province_grid_cells_store.index].shape}")
        
        self.log.info(f">> Calculating '{self.resource_type}_CF_mean' for {len(self.province_grid_cells_store)} Cells...")
        self.province_grid_cells_store[f'{self.resource_type}_CF_mean'] = self.CF_ts_df[self.resource_type].mean(axis=0) # Mean of all rows (Hours)
        # Updates the 'CF_mean' field to stored dataframe with key 'cells. The grid cells must have matched "X(grid cell's)-Y(timeseries header)" index to do this step.
        '''
        Future Scope: Replacing CF_mean with high resolution data (likely from Global Solar Atlas/ Local data)
        '''
        
        # Step 6: Define a namedtuple to store both the grid cells and the filtered timeseries
        site_data = namedtuple('site_data', ['cells', 'timeseries'])
        self.data : tuple= site_data(self.province_grid_cells_store, self.CF_ts_df)
        '''
        @ to access return data
        Both PV and WIND are gonna go under same 'name' 
        # To access the PV timeseries, user has to use the "Y" index to access PV timeseries e.g. pv_timeseries_dataframe = data.timeseries.solar ('timeseries' is name of the tuple, 'solar' is the first level column name of the dataframe.)
        '''
            # Step 5: Save the grid cells and timeseries to the local HDF5 store
        self.datahandler.to_store(self.province_grid_cells_store, 'cells') # We don't want 'force-update' here, just need to append 'CF_mean' datafields to cells.
        self.datahandler.to_store(self.CF_ts_df, 'timeseries') 
        '''
        @ store data
        Both PV and WIND are gonna go under same 'key' 
        # To access the PV timeseries, user has to use the "Y" index to access PV timeseries e.g. pv_timeseries_dataframe = timeseries.solar ('timeseries' = key to stored data)
        '''
        return self.data 

    
    def __process_PV_timeseries__(self):
        """ 
        A wrapper that leverage Atlite's _cutout.pv_ method to convert downward-shortwave, upward-shortwave radiation flux and ambient temperature into a pv generation time-series.
        """
        
        # Step 1.1: Get the Atlite's Cutout Object loaded
        self.log.info(f">> Loading ERA5 Cutout")
        self.cutout,self.province_boundary=self.get_era5_cutout()
        self.province_grid_cells = self.cutout.grid.overlay(self.province_boundary, how='intersection',keep_geom_type=True)
        self.province_grid_cells = utils.assign_cell_id(self.province_grid_cells,'Region',self.site_index)
        
        # Step 1.2: Get the Province Grid Cells from Store. Ideally these cells should have same resolution as the Cutout (the indices are prepared from x,y coords and Region names)
        
        self.province_grid_cells_store=self.datahandler.from_store('cells')
        self.log.info(f">> {len(self.province_grid_cells_store)} Grid Cells from Store Cutout")
        
        # Step 1.3: Set arguments for the atlite cutout's pv method
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
            
            'shapes': self.province_grid_cells_store.geometry,
            #  (list or pd.Series of shapely.geometry.Polygon) – If given, matrix is constructed as indicator-matrix of the polygons, 
            # its index determines the bus index on the time-series.
            
            # 'capacity_factor_timeseries':True, # If True, the capacity factor time series of the chosen resource for each grid cell is computed.
            # 'return_capacity': False, # Additionally returns the installed capacity at each bus corresponding to layout (defaults to False).
            # 'capacity_factor':True, # If True, the static capacity factor of the chosen resource for each grid cell is computed.
            'index':self.province_grid_cells_store.index,
            'per_unit':True, # Returns the time-series in per-unit units, instead of in MW (defaults to False).
            'show_progress': False, # Progress bar
        }

        # Step 1.4: Generate PV timeseries profile using the atlite's cutout
        self.pv_profile: xr.DataArray = self.cutout.pv(**pv_args).rename(self.resource_type)
        
        return self.pv_profile
    
    def __process_WIND_timeseries__(self,
                                    turbine_model_source:str='OEDB',
                                    model:int=2):
        """ 
        - A wrapper that leverage Atlite's _cutout.wind_ method to convert wind speed to wind generation CF timeseries.
        - Extrapolates 10m wind speed with monthly surface roughness to hub height and evaluates the power curve.
        """
        
        # Step 1.1: Get the Atlite's Cutout Object loaded
        self.log.info(f">> Loading ERA5 Cutout")
        self.cutout,self.province_boundary=self.get_era5_cutout()
        self.province_grid_cells = self.cutout.grid.overlay(self.province_boundary, how='intersection',keep_geom_type=True)
        self.province_grid_cells = utils.assign_cell_id(self.province_grid_cells,'Region',self.site_index)
        
        # Step 1.2: Get the Province Grid Cells from Store. Ideally these cells should have same resolution as the Cutout (the indices are prepared from x,y coords and Region names)
        
        self.province_grid_cells_store=self.datahandler.from_store('cells')
        self.log.info(f">> {len(self.province_grid_cells_store)} Grid Cells from Store Cutout")
        self.wind_turbine_config=self.get_turbines_config()
        
        if turbine_model_source=='atlite':
            atlite_turbine_model:str=self.wind_turbine_config[turbine_model_source][model]['name'] # The default is set in Attributes parser's .get_turbine_config() method.
            hub_height_turbine=atlite.resource.get_windturbineconfig(atlite_turbine_model)['hub_height']
            
            self.turbine_config:dict = atlite.resource.get_windturbineconfig(atlite_turbine_model, {"hub_height": 100})
            self.log.info(f">> selected Wind Turbine  Model : {atlite_turbine_model} @ {hub_height_turbine}m Hub Height")
            
        elif turbine_model_source=='OEDB':
            self.OEDB_config:dict=self.wind_turbine_config[turbine_model_source]
            self.OEDB_turbines=OEDBTurbines(self.OEDB_config)
            self.OEDB_turbine_config=self.OEDB_turbines.fetch_turbine_config(model)
            self.turbine_config=self.OEDB_turbine_config
            
        # Step 1.4: Set arguments for the atlite cutout's wind method
        wind_args = {
        # .wind() method params
            'turbine': self.turbine_config,
            # 'smooth': False,
                # If True smooth power curve with a gaussian kernel as determined for the Danish wind fleet to Delta_v = 1.27 and sigma = 2.29. 
                # A dict allows to tune these values.
            'add_cutout_windspeed':True,
                # If True and in case the power curve does not end with a zero, will add zero power output at the highest wind speed in the power curve. 
                # If False, a warning will be raised if the power curve does not have a cut-out wind speed. The default is False.
                
        # '.convert_and_aggregate()' method parameters
            'layout': None, 
            # The capacity to be build in each of the grid_cells.
            
            'matrix': None, 
            # (N x S - xr.DataArray or sp.sparse.csr_matrix or None) – If given, it is used to aggregate the grid cells to buses. 
            # N is the number of buses, S the number of spatial coordinates, in the order of cutout.grid
            
            'layout':None,
            # (X x Y - xr.DataArray) – The capacity to be build in each of the grid_cells.
            
            'shapes': self.province_grid_cells_store.geometry,
            #  (list or pd.Series of shapely.geometry.Polygon) – If given, matrix is constructed as indicator-matrix of the polygons, 
            # If index' param is not set, shapes' index determines the bus index on the time-series.
            
            # 'capacity_factor_timeseries':True, # If True, the capacity factor time series of the chosen resource for each grid cell is computed.
            # 'return_capacity': False, # Additionally returns the installed capacity at each bus corresponding to layout (defaults to False).
            # 'capacity_factor':True, # If True, the static capacity factor of the chosen resource for each grid cell is computed.
            
            'index':self.province_grid_cells_store.index,
            # Index of Buses. We use grid cell indices here.
            
            'per_unit':True, # Returns the time-series in per-unit units, instead of in MW (defaults to False).
            'show_progress': False, # Progress bar
        }

        # Step 1.4: Generate PV timeseries profile using the atlite's cutout
        self.wind_profile: xr.DataArray = self.cutout.wind(**wind_args).rename(self.resource_type)
        
        return self.wind_profile
    
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
    
    def get_cluster_timeseries(self,
                               all_clusters:pd.DataFrame,
                               cells_timeseries:pd.DataFrame,
                               dissolved_indices:pd.DataFrame):

        # Initialize an empty list to store the results
        results = []

        # Iterate through each cluster
        for cluster, row in all_clusters.iterrows():
            # Extract the cluster's region and cluster number
            region = row['Region']
            cluster_no = row['Cluster_No']  # Dynamically fetch the cluster number from the row
            
            # Get the cell indices for the cluster based on the region and cluster number
            cluster_cell_indices = dissolved_indices.loc[region][cluster_no]
            
            # Calculate the mean for the timeseries data corresponding to the cluster
            cluster_ts = cells_timeseries[cluster_cell_indices].mean(axis=1)
            
            # Store the mean as a DataFrame with the cluster name as the column name
            results.append(pd.DataFrame(cluster_ts, columns=[cluster]))

        # Concatenate all results into a single DataFrame
        self.cluster_df = pd.concat(results, axis=1)
        # self.cluster_df.columns = pd.MultiIndex.from_arrays([[self.resource_type] * len(self.cluster_df.columns), self.cluster_df.columns])
        self.datahandler.to_store(self.cluster_df, f'timeseries_clusters_{self.resource_type}',force_update=True)

        # Display the final DataFrame
        return self.cluster_df

