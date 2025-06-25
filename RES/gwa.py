
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
import rioxarray as rxr
import xarray as xr
from RES.hdf5_handler import DataHandler
import requests
from typing import List, Dict, Optional
import RES.utility as utils

# ---------
from RES.boundaries import GADMBoundaries

@dataclass
class GWACells(GADMBoundaries):
    merged_data: xr.DataArray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.gwa_config = self.get_gwa_config()
        self.region_mapping=self.get_region_mapping()
        self.datahandler = DataHandler(self.store)

    def prepare_GWA_data(self,
                         windpseed_min=10,
                         windpseed_max=40,
                         region_short_code:Optional[str]=None,
                         memory_resource_limitation:bool=False) -> xr.DataArray:
        """
        Prepares the Global Wind Atlas (GWA) data by loading and merging raster files.
        Downloads files from sources if they do not exist.
        Returns a cleaned DataArray with relevant data.
        """
        data_list = []
        region_short_code = self.region_short_code.upper() if region_short_code is None else region_short_code

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
            self.gwa_country_code=self.region_mapping[region_short_code].get('GWA_country_code')
            self.raster_name=raster_name.replace("GWA_country_code",  self.gwa_country_code)
            self.raster_path = self.gwa_root / raster_name
            if not self.raster_path.exists():
                generic_source_url = self.gwa_sources[key]
                self.region_source_url = generic_source_url.replace("GWA_country_code",  self.gwa_country_code)
                self.log.info(f">> Downloading {key} from {self.region_source_url}")
                self.download_file(self.region_source_url, self.raster_path)

            try:
                # Process each raster using a streamlined approach
                data = (
                    rxr.open_rasterio(self.raster_path)
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
        
        destination=utils.ensure_path(destination)
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            with destination.open('wb') as f:
                f.write(response.content)
        except requests.RequestException as e:
            self.log.error(f">> Failed to download {destination} from {url}. Error: {e}")

    
    def load_gwa_cells(self,
                       memory_resource_limitation:Optional[bool]=False):
        self.region_gwa_cells_df = self.prepare_GWA_data(memory_resource_limitation)

        # Vectorized creation of geometries
        self.gwa_cells_gdf = gpd.GeoDataFrame(
            self.region_gwa_cells_df,
            geometry=gpd.points_from_xy(self.region_gwa_cells_df['x'], self.region_gwa_cells_df['y']),
            crs=self.get_default_crs())
            # ).clip(self.get_country_boundary(country_level=True), keep_geom_type=False)

        # self.gwa_cells_gdf = self.calculate_common_parameters_GWA_cells()
        # self.gwa_cells_gdf = self.map_GWAcells_to_ERA5cells()
        self.log.info(f">> Global Wind Atlas (GWA) Cells loaded. Size: {len(self.region_gwa_cells_df)}")
        
        return self.gwa_cells_gdf
    

    def map_GWA_cells_to_ERA5(self,
                              region_column:Optional[str]=None,
                              memory_resource_limitation:Optional[bool]=False):
        """
        Maps the GWA high resolution cells to comparatively low resolution ERA5 Cells.
        """
        
        # Load the grid cells and GWA cells as GeoDataFrames
        self.store_grid_cells = self.datahandler.from_store('cells')
        
        _era5_cells_=self.store_grid_cells.reset_index()
        _era5_cells_ = _era5_cells_.loc[:, ~_era5_cells_.columns.str.contains('_2')]
 
        self.gwa_cells_gdf = self.load_gwa_cells(memory_resource_limitation)

        self.log.info(f">> Mapping {len(self.gwa_cells_gdf)} GWA Cells to {len(_era5_cells_)} ERA5 Cells...")

        results = []  # List to store results for each region
        self.log.info(">> Updating aggregated values for ERA5 Cell's...")
        
        for region in _era5_cells_[region_column].unique():
            _era5_cells_region = _era5_cells_[_era5_cells_[region_column] == region]
            
            # Perform overlay operation
            _data_ = self.gwa_cells_gdf.overlay(_era5_cells_region, how='intersection', keep_geom_type=False)
            
            # Rename columns and select relevant data
            # _data_ = _data_.rename(columns={'x_1': 'x', 'y_1': 'y'}) # x1,y1 are the GWA coords
            _data_ = _data_.rename(columns={'x_2': 'x', 'y_2': 'y'}) #x2,y2 are the ERA5 coords
            selected_columns = list(_data_.columns)

            regional_df=_data_.loc[:, selected_columns]
            
            numeric_cols = regional_df.select_dtypes(include='number') 
            regional_mapped_gwa_cells_aggr = numeric_cols.groupby(regional_df['cell']).mean() # Aggregates he numeric columns data via mean
            
            # Store mapped GWA cells in results list
            results.append(regional_mapped_gwa_cells_aggr)
        
        # Concatenate all results into a single GeoDataFrame
        self.mapped_gwa_cells_aggr_df = pd.concat(results, axis=0)
            
        # Store the aggregated data
        self.datahandler.to_store(self.mapped_gwa_cells_aggr_df, 'cells') 
        return self.mapped_gwa_cells_aggr_df
