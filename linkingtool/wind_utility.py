# Script for Wind Module utility Functions
# Prepared by : Md Eliasinul Islam
# Affiliation : Delta E+ lab, Simon Fraser University
# Version : 1.0
# Release : 2024

import os,yaml
from scipy.spatial import cKDTree
import pandas as pd
from scipy.spatial import cKDTree
import geopandas as gpd
import rioxarray as rxr
import atlite
from shapely.geometry import box,Point
import logging as log
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import linkingtool.utility as utility

# Function for GWA data download
def download_GWA_data(parent_direct,url_source,raster_file_name):
    file_path=os.path.join(parent_direct,raster_file_name)
    response = requests.get(url_source)

    if response.status_code == 200:
        # Download the file and save it with the defined name
        with open(file_path, 'wb') as file:
            file.write(response.content)
        log.info(f"> Global Wind Atlas (GWA) raster file downloaded successfully and saved as {raster_file_name}.\n")
    else:
        log.info("> Failed to download the file from the URL. Checkand update URL in Unser Config file at '{user_config_file}'")

    return True

def calculate_ERA5_cell_area(gdf,availability_column,actual_area_ROI):
    #actual_area_ROI = Actual area of Region of Interest in Sq. km
    gdf = gdf.to_crs(epsg=2163) 

    gdf_with_area_column=gdf.copy()

    if 'land_area_sq_km' not in gdf_with_area_column.columns:
        gdf_with_area_column ['land_area_sq_km']=gdf_with_area_column['geometry'].area / 1e6 #sqkm
        gdf_with_area_column['eligible_land_area']= gdf_with_area_column['land_area_sq_km']* gdf_with_area_column[availability_column] 
    else:
        gdf_with_area_column['eligible_land_area']= gdf_with_area_column['eligible_land_area']* gdf_with_area_column[availability_column] 
    
    _total_eligible_land_area = int(gdf_with_area_column['eligible_land_area'].sum()) #sq. km

    print(
        f'\nRegion of Interest : BC\n',
        f"> Eligible Land Area of the  Grid Cells = {{:,}} Km²\n".format(_total_eligible_land_area),
        f"> Actual Land Area : {{:,}} km²\n".format(actual_area_ROI)
    )

    #Restore the projection for other usage
    gdf_with_area_column=gdf_with_area_column.to_crs(epsg=4326)
    
    return gdf_with_area_column


def IEC_class_from_pixel_values(gwa_IEC_class_layers_df,pixel_values_column_name):
    # Mapping pixel values to IEC class
    class_mapping = {0: 'III', 1: 'II', 2: 'I', 3: 'T', 4: 'S'}
    df=gwa_IEC_class_layers_df
    # Apply mapping, set 'none' for other values
    df[pixel_values_column_name] = df[pixel_values_column_name].map(class_mapping).fillna('none')
    df.reset_index(inplace=True)
    df.drop(columns=['spatial_ref','band'], inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # geometry = [Point(xy) for xy in zip(gwa_IEC_class_layers_df['x'], df['y'])]
    # gwa_IEC_class_layers_gdf = gpd.GeoDataFrame(df, geometry=geometry)

    log.info(f"IEC Classes mapped for cells. Class mapping from Pixel value as described in URL : https://globalwindatlas.info/en/about/dataset")
    return df
    
def visualize_GWA_data(gwa_dataframe, data, color, xlabel):

    ####### Parameters:
    # gwa_dataframe: The DataFrame containing the data to be visualized.
    # data: str, The column name from the DataFrame to be plotted.
    # color: The color of the histogram bars.
    # xlabel: The label for the x-axis of the plo


    # Create a histogram plot using Seaborn
    sns.histplot(gwa_dataframe[data], color=color, kde=True)

    # Check if the data column is related to windspeed and update xlabel accordingly
    if 'windspeed' in data:
        xlabel = 'windspeed (m/s)'
    else:
        xlabel = xlabel

    # Set x-axis label, y-axis label, and plot title
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(f'{data} Distribution of GWA Cells')

    # Adjust layout, add grid, and save the plot as an SVG file
    plt.tight_layout()
    plt.grid(True)
    save_to=f'vis/Wind/{data}.svg'
    plt.savefig(save_to)

    # Close the current figure to free up resources
    plt.close()

    # Log an information message indicating the successful creation and saving of the plot
    return log.info(f"{data} Distribution of All GWA Cells - Created and saved in Visualization directory")


def calculate_common_parameters_GWA_cells (GWA_dataframe,wind_cap_per_km2,cell_resolution_arc_deg=0.002499):
    res = cell_resolution_arc_deg
    log.info("Creating Polygon Geometry for GWA Cells for a Single Cell")

    # Extract centroid coordinates for the first row
    x, y = GWA_dataframe[['x', 'y']].iloc[0]

    # Generate the geometry for a single cell
    geom_single_cell = box(x - res/2, y - res/2, x + res/2, y + res/2)

    # Create the GeoDataFrame for a single cell
    GWA_single_cell_gdf = gpd.GeoDataFrame(geometry=[geom_single_cell], crs="EPSG:4326")

    # Change the CRS for area calculation
    GWA_single_cell_gdf = GWA_single_cell_gdf.to_crs(epsg=2163)  # availability values [0-1]

    # Calculate the land area for the cell and add it as a new column
    GWA_single_cell_gdf['land_area_sq_km'] = GWA_single_cell_gdf['geometry'].area / 1e6  # Converts to square kilometers

    # Restore the projection for other usage
    GWA_single_cell_gdf = GWA_single_cell_gdf.to_crs(epsg=4326)
    log.info(f"Imputing Land Area, Potential Capacity for all GWA Cells")
    GWA_dataframe['land_area_sq_km'] = GWA_single_cell_gdf['land_area_sq_km'].iloc[0]
    GWA_dataframe['potential_capacity'] = wind_cap_per_km2 * GWA_single_cell_gdf['land_area_sq_km'].iloc[0]

    return GWA_dataframe,GWA_single_cell_gdf

def update_ERA5_params_from_mapped_GWA_cells(era5_cells_gdf,mapped_GWA_cells_gdf):

    log.info(f"Updating ERA5 mean windspeed with mapped GWA Cells' for each cell, labeled as 'windspeed_GWA'")

    mean_windspeed_by_bcgrid = mapped_GWA_cells_gdf.groupby('ERA5_cell_index')['windspeed'].mean()
    mean_CF_by_bcgrid = mapped_GWA_cells_gdf.groupby('ERA5_cell_index')['CF_IEC3'].mean()
    
    era5_cells_gdf = era5_cells_gdf.copy()
    # Map the mean windspeed values to the corresponding bcgrid_id in the DataFrame
    era5_cells_gdf['windspeed_GWA'] = era5_cells_gdf.index.map(mean_windspeed_by_bcgrid)
    era5_cells_gdf['CF_mean_GWA'] = era5_cells_gdf.index.map(mean_CF_by_bcgrid)

    # # era5_cells_gdf_filtered = era5_cells_gdf_filtered.loc[era5_cells_gdf_filtered['windspeed_GWA'].notna()]
    # era5_cells_gdf_filtered.to_pickle('data/Processed_data/Wind/era5_cells_gdf_filtered.pkl')
    
    era5_cells_gdf_updated=era5_cells_gdf

    log.info(f"ERA5 mean windspeed and CF updated and labeled as 'windspeed_GWA' , 'CF_mean_GWA' ")
    return era5_cells_gdf_updated

def create_GWA_all_cells_polygon(dataframe,cell_resolution=0.002499):
    res = cell_resolution
    log.info("Creating Polygon Geometry for GWA Cells for a Single Cell")

    dataframe['geometry'] = dataframe.apply(lambda row: box(row['x'] - res/2, row['y'] - res/2, row['x'] + res/2, row['y'] + res/2), axis=1)

    return dataframe


def map_GWAcells_to_ERA5cells(gwa_cells_gdf,era5_cells_gdf,batch_process_size=1000):
    batch_size = batch_process_size  # Adjust the batch size based on your system's capacity
    
    gwa_batches = [gwa_cells_gdf.iloc[i:i + batch_size] for i in range(0, len(gwa_cells_gdf), batch_size)]

    log.info(f"Mapping {len(gwa_cells_gdf)} GWA Cells to {len(era5_cells_gdf)} ERA5 Cells")
    log.warning(f"!! Memory intensivie process initiated !!")
    
    gwa_cells_mapped_dfs = []
    
    for gwa_batch in gwa_batches:
        gwa_cells_mapped_df_batch = gpd.sjoin(gwa_batch, era5_cells_gdf[['Region', 'geometry','Region_ID','capex']], how='inner', predicate='intersects')
        gwa_cells_mapped_dfs.append(gwa_cells_mapped_df_batch)
        gwa_batch = None # Clear the memory

    gwa_cells_mapped_gdf = gpd.GeoDataFrame(pd.concat(gwa_cells_mapped_dfs), crs=gwa_cells_gdf.crs)

    gwa_cells_mapped_gdf = gwa_cells_mapped_gdf.rename(columns={'index_right': 'ERA5_cell_index'})

    gwa_cells_mapped_gdf['gwa_cell_id'] = gwa_cells_mapped_gdf['gwa_cell_id'].astype(int)

    log.info(f"Filtering the ERA5 Cells which containts atleast one GWA Cells")
    # Assuming 'ERA5_cell_index' is the column name in both GeoDataFrames
    GWA_unique_era5_cells = gwa_cells_mapped_gdf['ERA5_cell_index'].unique().tolist()

    # Filter bc_grid_cells based on unique values in 'ERA5_cell_index'
    era5_cells_gdf_filtered = era5_cells_gdf[era5_cells_gdf.index.isin(GWA_unique_era5_cells)]

    log.info(f'{len(era5_cells_gdf_filtered)} out of {len(era5_cells_gdf)} ERA5 Cells Kept for further processing.\n\
             {len(era5_cells_gdf)-len(era5_cells_gdf_filtered)}  Cells filtered out out due to non-avaailability of atleast 1 GWA Cells within\n')

    return gwa_cells_mapped_gdf,era5_cells_gdf_filtered

def create_CF_timeseries_df(cutout,start_date,end_date,geodataframe_sites,turbine_model,turbine_config_file,Site_index='cell_id',config='OEDB'):

    _layout_MW=utility.create_layout_for_generation(cutout,geodataframe_sites,'potential_capacity')
    
    log.info(f"Calculating Generation timeseries for BC Grid Cells as per the Layout Capacity (MW)...")
    # turbine_config=atlite.resource.get_windturbineconfig(turbine_model)
    if config=='atlite':
        hub_height_turbine=atlite.resource.get_windturbineconfig(turbine_model)['hub_height']
        turbine_config = atlite.resource.get_windturbineconfig(turbine_model, {"hub_height": 100})
        log.info(f"Selected Wind Turbine  Model : {turbine_model} @ {hub_height_turbine}m Hub Height")
    else:
        # Specify the path to your YAML file
        config_file = turbine_config_file #'configs/3.2M114_NES.yaml'

        # Read the YAML file into a dictionary
        with open(config_file, 'r') as file:
            turbine_config = yaml.safe_load(file)
            hub_height_turbine=turbine_config['hub_height']
            log.info(f"Selected Wind Turbine  Model : {turbine_config['name']} @ {hub_height_turbine}m Hub Height")

    wind = cutout.wind(
        turbine=turbine_config,
        show_progress=True,
        capacity_factor=True,
        layout=_layout_MW,
        add_cutout_windspeed=True,
        shapes=geodataframe_sites.geometry,
        per_unit=True,
    )  # Xarray

    # Convert Xarray to Dataframe
    sites = wind[Site_index].to_pandas()

    # start_date = configs['cutout']['start_date']
    # end_date = configs['cutout']['end_date']

    datetime_index = pd.date_range(start=start_date + ' 00:00:00', end=end_date + ' 23:00:00', freq='H')
    CF_ts_df = pd.DataFrame(index=datetime_index)

    # List to store DataFrames for each site
    site_dfs = []

    for site in sites:
        site_df = pd.DataFrame({site: wind.sel(time=slice(start_date, end_date), **{Site_index: site}).values}, index=datetime_index)
        site_dfs.append(site_df)

    # Concatenate all site-specific DataFrames into one DataFrame
    CF_ts_df = pd.concat(site_dfs, axis=1)
    # Assign the mean values to a new column 'CF_mean' in bc_grid_cells
    log.info(f"Calculating CF mean from the {len(CF_ts_df)} data points for each Cell ...")
    geodataframe_sites['CF_mean_atlite'] = CF_ts_df.mean()

    return geodataframe_sites,CF_ts_df

def clip_bc_data_from_GWA(GWA_raster_file,column_name,bounding_box):

    # Open the GeoTIFF file using rioxarray
    data = rxr.open_rasterio(GWA_raster_file).rename(column_name)
    
    min_x,max_x,min_y,max_y=bounding_box.values()

    # Clip the data to the specified bounding box
    bc_data = data.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)

    gwa_df = bc_data.to_dataframe()
    log.info(f"{column_name} data @ 100m hub height ; clipped for BC Region. Size: {len(gwa_df)}")

    return gwa_df #dataframe

def filter_GWA_cells(GWA_raster_path,bounding_box,data_low_bound,data_high_bound,column_name):

    gwa_df = clip_bc_data_from_GWA(GWA_raster_path,column_name,bounding_box)
    mask = (gwa_df[column_name] >= data_low_bound) & (gwa_df[column_name] <= data_high_bound)
    gwa_df_filtered = gwa_df[mask].sort_values(by=column_name, ascending=False)

    # Assigning a Cell id for the high resolution data to map with the lower resoltiuon BC Grid Cells.
  
    gwa_df_filtered.reset_index(inplace=True)
    # Clean the Data Fields (Columns) of the data frame
    gwa_df_filtered=gwa_df_filtered.loc[:, ( 'x', 'y',column_name)].astype('float32')

    # gwa_df_filtered.to_pickle(save_to)
    log.info(f'{column_name} data filtered with the filter range : {data_low_bound}-{data_high_bound}. Size: {len(gwa_df_filtered)}')
    log.warning(f"!! You can Configure the {column_name} filter in User config file and restart the data prepration with updated filter !!")
    return gwa_df_filtered