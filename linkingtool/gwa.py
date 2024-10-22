import pandas as pd
import geopandas as gpd
from linkingtool.boundaries import GADMBoundaries
from dataclasses import dataclass, field
from pathlib import Path
import linkingtool.linking_utility as utils
import rioxarray as rxr
import xarray as xr
from linkingtool.hdf5_handler import DataHandler
from shapely.geometry import box
from shapely.geometry import Point


@dataclass
class GWACells(GADMBoundaries):
    merged_data: xr.DataArray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.gwa_config = self.get_gwa_config()
        self.bounding_box, _ = self.get_bounding_box()
        self.datahandler = DataHandler(store=self.store)

    def prepare_GWA_data(self) -> pd.DataFrame:
        """
        Prepares the Global Wind Atlas (GWA) data by loading and merging raster files.
        Returns a cleaned DataFrame with relevant data.
        """
        data_list = []

        # Load configuration parameters
        self.gwa_datafields = self.gwa_config.get('datafields', {})
        self.gwa_rasters = self.gwa_config.get('rasters', {})
        self.gwa_root = Path(self.gwa_config.get('root', 'data/downloaded_data/GWA'))

        # Using a single try-except block to handle errors more effectively
        for key, raster_name in self.gwa_rasters.items():
            raster_path = self.gwa_root / raster_name
            if raster_path.exists():
                try:
                    # Process each raster using a streamlined approach
                    data = (
                        rxr.open_rasterio(raster_path)
                        .rio.clip_box(**self.bounding_box)
                        .rename(key)
                        .drop_vars(['band', 'spatial_ref'])
                        .isel(band = 1 if '*Class*' in key else 0) # 'IEC_Class_ExLoads' data is in band 1
                        )
              
                    data_list.append(data)
                except Exception as e:
                    print(f"Error processing {key}: {e}")

        # Merge and clean the data in a more efficient way
        self.merged_data = xr.merge(data_list) if data_list else xr.DataArray()
        self.merged_df = self.merged_data.to_dataframe().dropna(how='all')
        self.merged_df.reset_index(inplace=True)
        
        
        mask=self.merged_df['windspeed_gwa']>0
        self.merged_df_f=self.merged_df[mask]
        self.log.info(f">> {abs(len(self.merged_df_f) - self.merged_df.shape[0])} cells have no Windspeed values.\n>>> Cleaned data loaded for {len(self.merged_df_f)} GWA cells")
        
        # class_mapping = {0: 'III', 1: 'II', 2: 'I', 3: 'T', 4: 'S'}
        # # Correctly modifying only one column
        # self.merged_df_f['IEC_Class_ExLoads'] = self.merged_df_f['IEC_Class_ExLoads'].map(class_mapping).fillna(0)
        
        return self.merged_df_f

    def load_gwa_cells(self):
        self.province_gwa_cells_df = self.prepare_GWA_data()

        self.log.info(f">> Global Wind Atlas (GWA) Cells loaded. Size: {len(self.province_gwa_cells_df)}")

        # Vectorized creation of geometries
        self.gwa_cells_gdf = gpd.GeoDataFrame(
            self.province_gwa_cells_df,
            geometry=gpd.points_from_xy(self.province_gwa_cells_df['x'], self.province_gwa_cells_df['y']),
            crs=self.get_default_crs()
        ).clip(self.get_province_boundary(), keep_geom_type=False)

        # self.gwa_cells_gdf = self.calculate_common_parameters_GWA_cells()
        # self.gwa_cells_gdf = self.map_GWAcells_to_ERA5cells()
        return self.gwa_cells_gdf

    def map_GWA_cells_to_ERA5(self):
        self.store_grid_cells=self.datahandler.from_store('cells')
        _store_grid_cells_=self.store_grid_cells.reset_index()
        
        self.gwa_cells_gdf=self.load_gwa_cells()
        
        self.log.info(f">> Mapping {len(self.gwa_cells_gdf)} GWA Cells to {len(_store_grid_cells_)} ERA5 Cells...")
        
        _data_=gpd.overlay(self.gwa_cells_gdf,_store_grid_cells_,how='intersection',keep_geom_type=False)
        _data_ = _data_.rename(columns={'x_1': 'x', 'y_1': 'y'})
        selected_columns = list(_data_.columns) + [f'{self.resource_type}_CF_mean']
        
        self.mapped_gwa_cells=_data_.loc[:,selected_columns]
        self.log.info(f">> Calculating aggregated values for ERA5 Cell's...")
        
        self.mapped_gwa_cells=self.mapped_gwa_cells.loc[:, ~self.mapped_gwa_cells.columns.duplicated()]
        self.mapped_gwa_cells_aggr=self.mapped_gwa_cells.groupby('cell').agg({
                                                                    'windspeed_gwa': 'mean', 
                                                                    'CF_IEC2': 'mean', 
                                                                    'CF_IEC3': 'mean', 
                                                                    'wind_CF_mean': 'mean'
                                                                }, numeric_only=True)
        
        self.datahandler.to_store(self.mapped_gwa_cells_aggr,'cells')
    

    # def map_GWAcells_to_ERA5cells(self):
    #     """
    #     Maps GWA cells to ERA5 cells and calculates potential capacity.
    #     Returns the mapped GeoDataFrame.
    #     """
    #     era5_cells_gdf = self.datahandler.from_store('cells').reset_index()
    #     gwa_cells_mapped_gdf = gpd.sjoin(
    #         self.gwa_cells_gdf,
    #         era5_cells_gdf[['cell', 'Region', 'geometry']],
    #         how='inner',
    #         predicate='intersects'
    #     )

    #     # Efficient index handling
    #     era5_cells_gdf.set_index('cell', inplace=True)
    #     gwa_cells_mapped_gdf = gwa_cells_mapped_gdf.rename(columns={'cell': 'ERA5_cell_index'}).reset_index(drop=True)

    #     # Filter and calculate capacities
    #     GWA_unique_era5_cells = gwa_cells_mapped_gdf['ERA5_cell_index'].unique()
    #     era5_cells_gdf_filtered = era5_cells_gdf.loc[GWA_unique_era5_cells]

    #     max_cap_GWA_cell = gwa_cells_mapped_gdf['potential_capacity'].max()
        
    #     # Vectorized calculation of distributed potential capacity
    #     cell_counts = gwa_cells_mapped_gdf['ERA5_cell_index'].value_counts()
    #     for province_index in cell_counts.index:
    #         if province_index in era5_cells_gdf_filtered.index:
    #             calculated_cap = era5_cells_gdf_filtered.loc[province_index, 'potential_capacity'] / cell_counts[province_index]
    #             gwa_cells_mapped_gdf.loc[gwa_cells_mapped_gdf['ERA5_cell_index'] == province_index, 'potential_capacity'] = calculated_cap

    #     # Summary output
    #     print(f'Filtered Sites: Total Wind Potential (ERA5 Cells): {round(era5_cells_gdf_filtered.potential_capacity.sum() / 1000, 2)} GW')
    #     print(f'Filtered Sites: Total Wind Potential (GWA Cells): {round(gwa_cells_mapped_gdf.potential_capacity.sum() / 1000, 2)} GW')

    #     return gwa_cells_mapped_gdf