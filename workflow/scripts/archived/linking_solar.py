import pandas as pd
import geopandas as gpd
import linkingtool.utility as utils
import atlite

def create_CF_timeseries_df(cutout,start_date,end_date,geodataframe_sites,layout_matrix,panel_config,tracking_config,Site_index='cell'):

    # _layout_MW=utils.create_layout_for_generation(cutout,geodataframe_sites,'potential_capacity')
    # cutout=atlite.Cutout(cutout_file)
    pv = cutout.pv(
        panel=panel_config,
        show_progress=True,
        orientation="latitude_optimal",
        tracking=tracking_config,
        # layout=_layout_MW,
        # matrix=layout_matrix,
        shapes=geodataframe_sites.geometry,
        capacity_factor=True,
        per_unit=True,
        return_capacity=True,
    )  # Xarray

    # Convert Xarray to Dataframe
    sites = pv[Site_index].to_pandas()

    # datetime_index = pd.date_range(start=start_date + ' 00:00:00', end=end_date + ' 23:00:00', freq='H')
    datetime_index = pd.date_range(start=start_date , end=end_date, freq='h')
    CF_ts_df = pd.DataFrame(index=datetime_index)

    # List to store DataFrames for each site
    site_dfs = []

    for site in sites:
        site_df = pd.DataFrame({site: pv.sel(time=slice(start_date, end_date), **{Site_index: site}).values}, index=datetime_index)
        site_dfs.append(site_df)

    # Concatenate all site-specific DataFrames into one DataFrame
    CF_ts_df = pd.concat(site_dfs, axis=1)
    # Assign the mean values to a new column 'CF_mean' in bc_grid_cells
    print(f">> Calculating CF mean from the {len(CF_ts_df)} data points for each Cell ...")
    geodataframe_sites['CF_mean'] = CF_ts_df.mean()

    return geodataframe_sites,CF_ts_df

def create_timeseries_for_Cluster(
    cell_cluster_gdf:gpd.GeoDataFrame, 
    dissolved_indices_per_cluster:dict, 
    ERA5_grid_cells_CF_ts:pd.DataFrame):
    print(f">>> Generating Representative Timeseries for the Clusters...")

    region_cluster_dict = {}  # Initialize a dictionary to store median DataFrames for each (region, cluster) pair

    for region in cell_cluster_gdf['Region'].unique():
        region_mask = cell_cluster_gdf['Region'] == region
        region_x_ = cell_cluster_gdf.loc[region_mask]

        region_median_values_df = pd.DataFrame()  # Initialize an empty DataFrame for the current region

        for cluster in region_x_['Cluster_No'].unique():
            # Access the list of indices for the specified region and Cluster_No
            indices_for_region_cluster = dissolved_indices_per_cluster.get(region, {}).get(cluster, [])

            # Ensure indices_for_region_cluster is a list of indices
            if not isinstance(indices_for_region_cluster, list):
                indices_for_region_cluster = [indices_for_region_cluster]

            # Create a subset DataFrame using loc
            cluster_x_ = ERA5_grid_cells_CF_ts[indices_for_region_cluster]

            # Calculate median values for the subset
            median_df = cluster_x_.mean(axis=1)

            # Rename the Series to match the desired column name
            column_name = f'{region}_{cluster}'
            median_df = median_df.rename(column_name)

            # Append the median values as a new column to region_median_values_df
            region_median_values_df = pd.concat([region_median_values_df, median_df], axis=1)
            region_median_values_df.index = pd.to_datetime(region_median_values_df.index)

        # Store the DataFrame for the current region in the dictionary
        region_cluster_dict[region] = region_median_values_df

    # Concatenate all DataFrames for each region into a single DataFrame
    cf_ts_cluster_df = pd.concat(region_cluster_dict.values(), axis=1)
    print(f">>> Representative Timeseries for the Solar Clusters Created Successfully.")
    return cf_ts_cluster_df