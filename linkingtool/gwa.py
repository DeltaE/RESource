
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
import rioxarray as rxr
import xarray as xr
from linkingtool.hdf5_handler import DataHandler
import requests
import dask_geopandas as dgpd

# ---------
from linkingtool.boundaries import GADMBoundaries

@dataclass
class GWACells(GADMBoundaries):
    merged_data: xr.DataArray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.gwa_config = self.get_gwa_config()
        self.datahandler = DataHandler(self.store)

    def prepare_GWA_data(self,
                         windpseed_min=10,
                         windpseed_max=20,
                         memory_resource_limitation:bool=False) -> xr.DataArray:
        """
        Prepares the Global Wind Atlas (GWA) data by loading and merging raster files.
        Downloads files from sources if they do not exist.
        Returns a cleaned DataArray with relevant data.
        """
        data_list = []

        # Load configuration parameters
        self.gwa_datafields = self.gwa_config.get('datafields', {})
        self.gwa_rasters = self.gwa_config.get('rasters', {})
        self.gwa_sources = self.gwa_config.get('sources', {})
        self.gwa_root = Path(self.gwa_config.get('root', 'data/downloaded_data/GWA'))
        self.bounding_box, _ = self.get_bounding_box()
        # Create the root directory if it doesn't exist
        self.gwa_root.mkdir(parents=True, exist_ok=True)

        # Check for existence and download if necessary
        for key, raster_name in self.gwa_rasters.items():
            raster_path = self.gwa_root / raster_name
            if not raster_path.exists():
                source_url = self.gwa_sources[key]
                self.log.info(f">> Downloading {self.gwa_sources[key]} from {source_url}")
                self.download_file(source_url, raster_path)

            try:
                # Process each raster using a streamlined approach
                data = (
                    rxr.open_rasterio(raster_path)
                    .rio.clip_box(**self.bounding_box)
                    .rename(key)
                    .drop_vars(['band', 'spatial_ref'])
                    .isel(band=1 if '*Class*' in key else 0)  # 'IEC_Class_ExLoads' data is in band 1
                )

                data_list.append(data)
            except Exception as e:
                print(f"Error processing {key}: {e}")

        # Merge and clean the data in a more efficient way
        self.merged_data = xr.merge(data_list) if data_list else xr.DataArray() #.rename('gwa_data')

        self.merged_df = self.merged_data.to_dataframe().dropna(how='all')
        self.merged_df.reset_index(inplace=True)
        
        if memory_resource_limitation:
            self.log.info(f"Memory resource limitations enabled. Filtering GWA cells witn windspeed mask to limit the data offload processing...")
        else:
            windpseed_min:float=0 #m/s
            windpseed_max:float=50 #m/s
 
        mask=(self.merged_df['windspeed_gwa'] >= windpseed_min) & (self.merged_df['windspeed_gwa'] <= windpseed_max)
        self.merged_df_f=self.merged_df[mask]
        self.log.info(f">> {abs(len(self.merged_df_f) - self.merged_df.shape[0])} cells have been filtered due to Windspeed filter [{windpseed_min}-{windpseed_max} m/s].\n>>> Cleaned data loaded for {len(self.merged_df_f)} GWA cells")
        
        # class_mapping = {0: 'III', 1: 'II', 2: 'I', 3: 'T', 4: 'S'}
        # # Correctly modifying only one column
        # self.merged_df_f['IEC_Class_ExLoads'] = self.merged_df_f['IEC_Class_ExLoads'].map(class_mapping).fillna(0)

        return self.merged_df_f
    
    def download_file(self,
                      url: str, 
                      destination: Path) -> None:
        """Downloads a file from a given URL to a specified destination."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            with destination.open('wb') as f:
                f.write(response.content)
        except requests.RequestException as e:
            self.log.error(f">> Failed to download {destination} from {url}. Error: {e}")

    
    def load_gwa_cells(self,
                       memory_resource_limitation:bool=False):
        self.province_gwa_cells_df = self.prepare_GWA_data(memory_resource_limitation)

        # Vectorized creation of geometries
        self.gwa_cells_gdf = gpd.GeoDataFrame(
            self.province_gwa_cells_df,
            geometry=gpd.points_from_xy(self.province_gwa_cells_df['x'], self.province_gwa_cells_df['y']),
            crs=self.get_default_crs()
        ).clip(self.get_province_boundary(), keep_geom_type=False)

        # self.gwa_cells_gdf = self.calculate_common_parameters_GWA_cells()
        # self.gwa_cells_gdf = self.map_GWAcells_to_ERA5cells()
        self.log.info(f">> Global Wind Atlas (GWA) Cells loaded. Size: {len(self.province_gwa_cells_df)}")
        
        return self.gwa_cells_gdf
    

    def map_GWA_cells_to_ERA5(self,
                              memory_resource_limitation):
        
        # Load the grid cells and GWA cells as GeoDataFrames
        self.store_grid_cells = self.datahandler.from_store('cells')
        
        _era5_cells_=self.store_grid_cells.reset_index()
 
        self.gwa_cells_gdf = self.load_gwa_cells(memory_resource_limitation)

        self.log.info(f">> Mapping {len(self.gwa_cells_gdf)} GWA Cells to {len(_era5_cells_)} ERA5 Cells...")

        results = []  # List to store results for each region
        
        for region in _era5_cells_['Region'].unique():
            _era5_cells_region = _era5_cells_[_era5_cells_['Region'] == region]
            
            # Perform overlay operation
            _data_ = self.gwa_cells_gdf.overlay(_era5_cells_region, how='intersection', keep_geom_type=False)
            
            # Rename columns and select relevant data
            _data_ = _data_.rename(columns={'x_1': 'x', 'y_1': 'y'})
            selected_columns = list(_data_.columns) + [f'{self.resource_type}_CF_mean']

            regional_df=_data_.loc[:, selected_columns]
            
            self.log.info(f">> Calculating aggregated values for ERA5 Cell's...")
            # regional_mapped_gwa_cells_aggr = regional_df.groupby('cell').agg({
            #         'windspeed_gwa': 'mean',
            #         'CF_IEC2': 'mean',
            #         'CF_IEC3': 'mean',
            #         'wind_CF_mean': 'mean'
            #     }, numeric_only=True)
            numeric_cols = regional_df.select_dtypes(include='number')
            regional_mapped_gwa_cells_aggr = numeric_cols.groupby(regional_df['cell']).mean()
            
            # Store mapped GWA cells in results list
            results.append(regional_mapped_gwa_cells_aggr)
        
        # Concatenate all results into a single GeoDataFrame
        self.mapped_gwa_cells_aggr_df = pd.concat(results, axis=0)
            
        # Store the aggregated data
        self.datahandler.to_store(self.mapped_gwa_cells_aggr_df, 'cells')  
