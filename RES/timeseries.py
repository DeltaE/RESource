import atlite
import xarray as xr
from dataclasses import dataclass
import pandas as pd
from collections import namedtuple
from pathlib import Path
import geopandas as gpd
import plotly.graph_objects as go

# Linking Tool - Local Packages
from RES.hdf5_handler import DataHandler
from RES.tech import OEDBTurbines
from RES.era5_cutout import ERA5Cutout
import RES.utility as utils

@dataclass
class Timeseries(ERA5Cutout):
    
    def __post_init__(self):
        super().__post_init__()

        # Fetch the disaggregation configuration based on the resource type
        self.resource_disaggregation_config=self.get_resource_disaggregation_config()
        
        # Initialize the local store
        self.datahandler=DataHandler(self.store)
    
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
        dupes = self.sites_profile.to_dataframe().index[self.sites_profile.to_dataframe().index.duplicated()]
        print(dupes)  # Show duplicated index entries

        self._CF_ts_df_org = (
            self.sites_profile.to_dataframe()
            .groupby(["time", "cell"])
            .mean()  # You can also use .sum(), depending on your needs
            .unstack("cell")
        )
        # df_reset = self.sites_profile.to_dataframe().reset_index()
        # df_pivoted = df_reset.pivot(index="time", columns="cell", values="solar")


        self._CF_ts_df_=self._CF_ts_df_org.loc[:,self.resource_type] # Multi-index dataframe with Y_L1 index (array name i.e. "solar" or "wind")
        
        
        ''' 
        We already rename the Xarray to 'self.resource_type' i.e solar/wind at the end of "__process_PV_timeseries__()" method. Hence now the Xarray could be transformed to wide-format DataFrame.
        - using the to_dataframe() method in xarray.DataArray, the behavior is different from to_pandas().
        - The array index order is (time, cell) hence in pandas 'time' will be default index and 'cell' default header. But now it will have an additional "Y" index "PV" adopted from xarray name.
        - WIND profile will be stored under same index to generate a synthetic hybrid availability (correlational) profile.
        '''
        # here, "_CF_ts_df_" will provide same data formate alike .to_pandas() method, just have to use "_CF_ts_df_.PV"  ("solar" or "wind" is the xarray name)
        
        # Step 3: Convert the timeseries data to the appropriate region timezone
        self.region_timezone=self.get_region_timezone()
        self.CF_ts_df = self.__fix_timezone__(self._CF_ts_df_).tz_localize(None)

        '''
        - We localize the datetime-stamp (i.e. removing the timezone information) to sync the requirements for downstream models.
        - The naive timestamps (without timezone info) found better harmonized with the other data sources.
        - This step needs to be tailored by the user to harmonize the timeseries with other operational data.
        '''
        
        # Step 4: Calculate the mean capacity factor (CF) for each cell and store it in 'CF_mean'
        self.log.info(f">> Calculating CF mean from the {len(self.CF_ts_df)} data points for each Cell ...")
        self.log.info(f">> Total Grid Cells: {len(self.region_grid_cells_store)}, "
                      f">> Timeseries Generated for: {len(self.CF_ts_df.columns)}, "
                    #   f">> Matched Sites: {self.CF_ts_df[self.resource_type][self.region_grid_cells_store.index].shape}")
                      f">> Matched Sites: {self.CF_ts_df[self.region_grid_cells_store.index].shape}")
        
        self.log.info(f">> Calculating '{self.resource_type}_CF_mean' for {len(self.region_grid_cells_store)} Cells...")
        # self.region_grid_cells_store[f'{self.resource_type}_CF_mean'] = self.CF_ts_df[self.resource_type].mean(axis=0) # Mean of all rows (Hours)
        self.region_grid_cells_store[f'{self.resource_type}_CF_mean'] = self.CF_ts_df.mean(axis=0) # Mean of all rows (Hours)
        # Updates the 'CF_mean' field to stored dataframe with key 'cells. The grid cells must have matched "X(grid cell's)-Y(timeseries header)" index to do this step.
        '''
        Future Scope: Replacing CF_mean with high resolution data (likely from Global Solar Atlas/ Local data)
        '''
        
        # Step 6: Define a namedtuple to store both the grid cells and the filtered timeseries
        site_data = namedtuple('site_data', ['cells', 'timeseries'])
        self.data : tuple= site_data(self.region_grid_cells_store, self.CF_ts_df)
        '''
        @ to access return data
        Both PV and WIND are gonna go under same 'key' 
        # To access the PV timeseries, user has to use the "Y" index to access PV timeseries e.g. pv_timeseries_dataframe = data.timeseries.solar ('timeseries' is name of the tuple, 'solar' is the first level column name of the dataframe.)
        '''
            # Step 5: Save the grid cells and timeseries to the local HDF5 store
        self.datahandler.to_store(self.region_grid_cells_store, 'cells') # We don't want 'force-update' here, just need to append 'CF_mean' datafields to cells.
        self.datahandler.to_store(self.CF_ts_df, f'timeseries/{self.resource_type}') # Hierarchical data of resources under kley 'timeseries' 
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
        self.log.info(">> Loading ERA5 Cutout")
        self.cutout,self.region_boundary=self.get_era5_cutout()
        self.region_grid_cells = self.cutout.grid.overlay(self.region_boundary, how='intersection',keep_geom_type=True)
        # self.region_grid_cells = utils.assign_cell_id(self.region_grid_cells,'Region',self.site_index)
        self.region_grid_cells = utils.assign_cell_id(self.region_grid_cells,'Country',self.site_index)
        
        # Step 1.2: Get the region Grid Cells from Store. Ideally these cells should have same resolution as the Cutout (the indices are prepared from x,y coords and Region names)
        
        self.region_grid_cells_store=self.datahandler.from_store('cells')
        self.log.info(f">> {len(self.region_grid_cells_store)} Grid Cells from Store Cutout")
        
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
            
            'shapes': self.region_grid_cells_store.geometry,
            #  (list or pd.Series of shapely.geometry.Polygon) – If given, matrix is constructed as indicator-matrix of the polygons, 
            # its index determines the bus index on the time-series.
            
            # 'capacity_factor_timeseries':True, # If True, the capacity factor time series of the chosen resource for each grid cell is computed.
            # 'return_capacity': False, # Additionally returns the installed capacity at each bus corresponding to layout (defaults to False).
            # 'capacity_factor':True, # If True, the static capacity factor of the chosen resource for each grid cell is computed.
            'index':self.region_grid_cells_store.index,
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
        self.cutout,self.region_boundary=self.get_era5_cutout()
        self.region_grid_cells = self.cutout.grid.overlay(self.region_boundary, how='intersection',keep_geom_type=True)
        # self.region_grid_cells = utils.assign_cell_id(self.region_grid_cells,'Region',self.site_index)
        self.region_grid_cells = utils.assign_cell_id(self.region_grid_cells,'Country',self.site_index)
        
        # Step 1.2: Get the region Grid Cells from Store. Ideally these cells should have same resolution as the Cutout (the indices are prepared from x,y coords and Region names)
        
        self.region_grid_cells_store=self.datahandler.from_store('cells')
        self.log.info(f">> {len(self.region_grid_cells_store)} Grid Cells from Store Cutout")
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
            
            'shapes': self.region_grid_cells_store.geometry,
            #  (list or pd.Series of shapely.geometry.Polygon) – If given, matrix is constructed as indicator-matrix of the polygons, 
            # If index' param is not set, shapes' index determines the bus index on the time-series.
            
            # 'capacity_factor_timeseries':True, # If True, the capacity factor time series of the chosen resource for each grid cell is computed.
            # 'return_capacity': False, # Additionally returns the installed capacity at each bus corresponding to layout (defaults to False).
            # 'capacity_factor':True, # If True, the static capacity factor of the chosen resource for each grid cell is computed.
            
            'index':self.region_grid_cells_store.index,
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
        df_index_converted = df_index_utc.tz_convert(self.region_timezone)
        
        df_index_converted.tz_localize(None) # without timezone conversion metadata
        
        return df_index_converted
    
    def get_cluster_timeseries(self,
                               all_clusters:pd.DataFrame,
                               cells_timeseries:pd.DataFrame,
                               dissolved_indices:pd.DataFrame,
                               region_col_name:str):

        # Initialize an empty list to store the results
        results = []

        # Iterate through each cluster
        for cluster, row in all_clusters.iterrows():
            # Extract the cluster's region and cluster number
            region = row[region_col_name]
            cluster_no = row['Cluster_No']  # Dynamically fetch the cluster number from the row
            
            # Get the cell indices for the cluster based on the region and cluster number
            cluster_cell_indices = dissolved_indices.loc[region][cluster_no]
            
            # Calculate the mean for the timeseries data corresponding to the cluster
            cluster_ts = cells_timeseries[cluster_cell_indices].mean(axis=1)
            
            # Store the mean as a DataFrame with the cluster name as the column name
            results.append(pd.DataFrame(cluster_ts, columns=[cluster]))

        # Concatenate all results into a single DataFrame
        self.cluster_df = pd.concat(results, axis=1)
        self.datahandler.to_store(self.cluster_df, f'timeseries/clusters/{self.resource_type}',force_update=True)# Hierarchical data of resources under kley 'timeseries' 

        # Display the final DataFrame
        return self.cluster_df


@staticmethod
def get_timeseries_for_project_points(
    resources_store: Path,
    projects: gpd.GeoDataFrame,
    save_to: str | Path,
    show: bool = True
):
    """
    Extracts time series data for solar and wind resources for given projects.

    Parameters:
    ----------
    resources_store : Path
        Path to the directory containing the resources store (HDF5 format) that includes 
        the data for 'cells', 'timeseries/solar', and 'timeseries/wind'.

    projects : gpd.GeoDataFrame
        A GeoDataFrame containing the project points with the following structure:
        
        - CRS: Must be in 'EPSG:4326' (WGS84), i.e., latitude and longitude coordinates.
        - Columns:
          - 'geometry': Point geometries representing project locations.
          - 'resource_type': A column specifying the type of resource for each project, 
            expected values are 'solar' or 'wind'.
          - Other project-specific columns may be present but are not directly used in this function.

    save_to : str | Path
        Directory path where the output CSVs and HTML plots will be saved.

    show : bool, optional, default=True
        Whether to display the plots interactively using Plotly.

    Returns:
    -------
    None
        The function outputs two CSV files and two interactive HTML plots:
        - 'projects_solar_ts.csv': Time series for solar projects.
        - 'projects_wind_ts.csv': Time series for wind projects.
        - 'projects_solar_ts.html': Interactive plot for solar projects.
        - 'projects_wind_ts.html': Interactive plot for wind projects.
    """

    # Initialize DataHandler for loading data
    datahandler = DataHandler(resources_store)
    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    # Ensure the CRS of the projects GeoDataFrame is 'EPSG:4326'
    projects.crs = 'EPSG:4326'

    # Load data from resources store

    tss = datahandler.from_store('timeseries/solar')
    tsw = datahandler.from_store('timeseries/wind')
    

    # Perform a spatial join to assign the polygon index to the points
    if 'ERA5_cell' not in projects.columns:
        cells = datahandler.from_store('cells')
        _joined_gdf_ = gpd.sjoin(projects, cells, how="left", op="intersects")
        projects['ERA5_cell'] = _joined_gdf_.index_right 

    projects.index=projects['ERA5_cell']
    projects.drop(columns=['ERA5_cell'],inplace=True)
    
    # Filter projects based on resource type
    solar_projects = projects[projects['resource_type'] == 'solar']
    wind_projects = projects[projects['resource_type'] == 'wind']
    
    # Extract time series data for solar and wind projects
    solar_ts = tss[solar_projects.index]
    solar_ts_save_to=save_to / 'projects_solar_ts.csv'
    
    wind_ts = tsw[wind_projects.index]
    wind_ts_save_to=save_to / 'projects_wind_ts.csv'
    
    # Save the time series data as CSV files
    solar_ts.to_csv(solar_ts_save_to)
    print(f"Saved solar time series data to {solar_ts_save_to}")
    
    wind_ts.to_csv(wind_ts_save_to)
    print(f"Saved wind time series data to {wind_ts_save_to}")

    # Plotting the time series for Solar Projects
    fig_s = go.Figure()
    for col in solar_ts.columns:
        fig_s.add_trace(go.Scatter(x=solar_ts.index, y=solar_ts[col], mode='lines', name=col))
    
    fig_s.update_layout(
        title="Time Series for Solar Projects",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Sites"
    )
    fig_s.write_html(save_to / 'projects_solar_ts.html')
    print(f"Saved solar time series data to {save_to / 'projects_solar_ts.html'}")


    
    # Plotting the time series for Wind Projects
    fig_w = go.Figure()
    for col in wind_ts.columns:
        fig_w.add_trace(go.Scatter(x=wind_ts.index, y=wind_ts[col], mode='lines', name=col))
    
    fig_w.update_layout(
        title="Time Series for Wind Projects",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Sites"
    )
    fig_w.write_html(save_to / 'projects_wind_ts.html')
    print(f"Saved wind time series data to {save_to / 'projects_wind_ts.html'}")
    
    
    if show:
        fig_s.show()
        fig_w.show()

