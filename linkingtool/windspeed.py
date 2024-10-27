import geopandas as gpd
import atlite
from linkingtool import utility

def impute_ERA5_windspeed_to_Cells(
        cutout:atlite.Cutout, 
        province_grid_cells:gpd.GeoDataFrame)->gpd.GeoDataFrame:
        """
        For each grid cells, this function finds the yearly mean windspeed from the windspeed timeseries and imputes to the cell dataframe.
        """
        
        print(f"Calculating yearly mean windspeed and imputing to provincial Grid Cells named as 'windspeed_ERA5'")

        # Calculate yearly mean windspeed
        wnd_ymean_df = cutout.data.wnd100m.groupby('time.year').mean('time').to_dataframe(name='windspeed_ERA5').reset_index()

        # Create a GeoDataFrame for spatial join
        wnd_ymean_gdf = gpd.GeoDataFrame(wnd_ymean_df, geometry=gpd.points_from_xy(wnd_ymean_df['x'], wnd_ymean_df['y']))
        wnd_ymean_gdf.crs = province_grid_cells.crs

        # Perform spatial join and drop unnecessary columns
        province_grid_cells = (
            gpd.sjoin(province_grid_cells.rename(columns={'x': 'x_bc', 'y': 'y_bc'}), wnd_ymean_gdf, predicate='intersects')
            .drop(columns=['x_bc', 'y_bc', 'lon', 'lat','index_right','year'])
        )
        
        # Handle potential duplicate indices
        # Resetting index to ensure unique index after join
        province_grid_cells = province_grid_cells.reset_index(drop=True)
        province_grid_cells = province_grid_cells.drop_duplicates(subset=['geometry'])
        province_grid_cells=utility.assign_cell_id(province_grid_cells)

        return province_grid_cells