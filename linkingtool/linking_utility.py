# Script for Utility Functions for all modules

# Prepared by : Md Eliasinul Islam
# Affiliation : Delta E+ lab, Simon Fraser University
# Version : 1.0
# Release : 2024

import os,yaml,requests

import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import atlite
from atlite.gis import ExclusionContainer
from shapely.geometry import box
import logging as log
import json
from shapely.geometry import box,Point
import matplotlib.pyplot as plt
from typing import Dict, List,Union

import pickle
import datetime

now = datetime.datetime.now()
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

def create_log(log_path):
    if not os.path.exists('workflow/log'):
        # If it doesn't exist, create it
        os.mkdir('workflow/log')
        print("Directory 'log' created successfully.")
    else:
        # exit
        None
        
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
    with open(log_path, 'w') as file:
        pass
    file_handler = log.FileHandler(log_path)
    log.getLogger().addHandler(file_handler)
    
    return log.getLogger()

def create_directories(base_path, structure):
    for key, value in structure.items():
        # Create the main directory
        dir_path = os.path.join(base_path, key)
        if os.path.exists(dir_path):
            print(f" >> !! '{key}' already exists")
        else:
            os.makedirs(dir_path, exist_ok=True)
            print(f"- '{key}' created")
        
        # Recursively create subdirectories
        if isinstance(value, dict):
            create_directories(dir_path, value)
        elif value is None:
            print(f"'{key}' has no subdirectories")


def dict_to_pickle(
        my_dictionary:dict,
        save_to_path:str):
    """
    Takes dictionary file and saves to given local path as pickle file. Returns a NONE as
    """
    with open(save_to_path,'wb') as file:
        pickle.dump(my_dictionary,file)
    # return None

def pickle_to_dict(pickle_file_path):
    with open(pickle_file_path,'rb') as file:
        my_dictionary=pickle.load(file)
    return my_dictionary

def create_blank_yaml(file_path):
        with open(file_path, 'w'):
            pass

def save_dict_datafile(dictionary,save_to):
    with open(save_to,'w') as json_file:
        json.dump(dictionary,json_file)
        return log.info(f" saved as '{save_to}")
    
def load_dict_datafile(json_file_path:str)->dict:
    with open(json_file_path,'r') as json_file:
        dictionary_:dict=json.load(json_file)
        return dictionary_

def check_LocalCopy_and_run_function(
        directory_path:str, 
        function_to_run:str, 
        force_update:bool=False)->bool:
    """
    Check if a directory exists. If it does, execute the provided function.

    Parameters:
    - directory_path (str): The path to the directory.
    - function_to_run (callable): The function to execute if the directory exists.

    Returns:
    - bool: True if the directory exists and the function is executed, False otherwise.
    """
    if force_update:
        output=function_to_run()
        log.info(f"Forcefully ran '{function_to_run.__name__}' on '{directory_path}'.")
        return output
    else:
        if not os.path.exists(directory_path):
            output=function_to_run()
            log.info(f"Directory '{directory_path}' created.")
            return output
        else:
            # log.info(f"Directory '{directory_path}' found locally.")
            return log.info(f"Directory '{directory_path}' found locally.")

# >>>> replaced by advanced func
# def create_directories_alt(path):
#     """
#     Takes in the path string and creates the directories if non-existent.
#     """

#     if not os.path.exists(path):
#         os.mkdir(path)
#         log.info(f"Directory created: {path}")
#         return True
#     else:
#         #do nothing
#         log.info(f"Directory existis locally: {path}")
#         return False
    

# Function to Load User Configuration File
def load_config(file_path):

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def download_data(
    source_URL:str,
    file_path:str):

    # Headers (modify these if required)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Send HTTP GET request to the URL
    response = requests.get(source_URL, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        log.info(f"File downloaded successfully and saved as {file_path}")
    else:
        # log.info(f"Failed to download the Resources zip file. Status code: {response.status_code}")
        log.info(f"Please Download the data from {source_URL} and extract the files to {file_path}")
            # return file_path


def update_population_data(
    config_population: dict, 
    gadm_regions:  gpd.GeoDataFrame,
    population_csv_data_path: str, 
) -> gpd.GeoDataFrame:
    """
    Updates gadm_regions DataFrame with population data based on the mapping provided in the config.

    Parameters:
    - config (dict): Dictionary containing YAML configuration with name mappings.
    - gadm_regions (pd.DataFrame): DataFrame to be updated with population data.
    - population (pd.DataFrame): DataFrame containing population data.
    - skiprows (int): Number of rows to skip at the start of the CSV file (default is 6).

    Returns:
    - pd.DataFrame: Updated gadm_regions DataFrame with population data.
    """
    
    
    # Preprocess the population data
    df = pd.read_csv(population_csv_data_path,skiprows=config_population['skiprows']) # Work on a copy to avoid modifying the original DataFrame
    # df.columns = df.columns.str.strip()  # Remove any extra spaces from column names

    # Select relevant columns and clean the 'Pop' column
    df = df[['Regional District', 'Year', 'Pop']]
    df['Pop'] = df['Pop'].str.replace(' ', '').str.replace(',', '').astype('int32')
    df['Year'] = df['Year'].astype('int32')

    # Rename 'Regional District' to 'Region'
    df.rename(columns={'Regional District': 'Region'}, inplace=True)

    # Filter for the year 2021 and apply the name mapping
    population_2021 = df[df['Year'] == 2021].copy()  # Create a copy to avoid SettingWithCopyWarning
    mapping = config_population['different_name_mapping']
    population_2021.loc[:, 'Region'] = population_2021['Region'].replace(mapping)

    # Create a mapping from 'Region' to 'Pop' and update gadm_regions
    pop_map = population_2021.set_index('Region')['Pop']
    gadm_regions['population'] = gadm_regions['Region'].map(pop_map)
    gadm_regions_with_pop_data=gadm_regions.copy()
    return gadm_regions_with_pop_data

def calculate_potential_capacity(
        ERA5_cells_gdf:gpd.GeoDataFrame,
        tech_cap_per_km2:float,
        index_name:str)->gpd.GeoDataFrame:
    """
    Calculates the potential capacity(MW) for each ERA5 grid cells from the available land area(km2) and the landuse intensity (MW/km2) of the technology.
    """
    
    log.info(f"Calculating Potential Capacity for BC Grid Cells based on eligible land area")

    ERA5_cells_gdf['potential_capacity'] = ERA5_cells_gdf['eligible_land_area'] * tech_cap_per_km2 # MW

    # Calculate the total PV potential based on available land (in GW)
    total_resource_potential_GW = ERA5_cells_gdf['potential_capacity'].sum() / 1000  #GW

    # Print information about the PV potential
    print(f'\nAssuming a literature-based installable tech capacity per unit area: {tech_cap_per_km2} MW/km²')
    print(f'Total Resource Potential (based on available land): {round(total_resource_potential_GW,2)} GW \n')
    

    ERA5_cells_gdf.reset_index(inplace=True,drop=True)
    
    ERA5_cells_gdf_new=assign_regional_cell_ids(ERA5_cells_gdf,'Region',index_name)

    
    log.info(f"Potential Capacity for Provincial Grid Cells based on available Land calculated")
    
    return ERA5_cells_gdf_new

def distance_cost(
        distance:float,
        grid_connection_cost_per_Km:float,
        tx_line_rebuild_cost:float)->float:
    """
    xxx 
    """
    
    distance_cost=distance * grid_connection_cost_per_Km/1.60934 * tx_line_rebuild_cost/1.60934    # 1.60934 mile to km conversion
    return distance_cost  # M$

def calculate_cost(
        distance:float,
        grid_connection_cost_per_Km:float,
        tx_line_rebuild_cost:float,
        capex_tech:float)->float:
    """
    xxx 
    """
    capex_1_MW = capex_tech # M$/MW
    total_cost =  capex_1_MW + distance_cost(distance,grid_connection_cost_per_Km,tx_line_rebuild_cost)  # M$

    return total_cost # M$

def calculate_cell_score(
        dataframe:pd.DataFrame,
        grid_connection_cost_per_Km:float,
        tx_line_rebuild_cost:float,
        CF_column:str,
        capex_tech:float)->pd.DataFrame:
    
    """
    xxx 
    """
    dataframe=dataframe.copy()
    print(f">> Calculating Score for each Cell ...")

    dataframe['p_lcoe'] = dataframe.apply(
            lambda x: 8760 * x[CF_column] / calculate_cost(x['nearest_station_distance_km'],grid_connection_cost_per_Km,tx_line_rebuild_cost,capex_tech),
            axis=1
        )  # MWh/CA$    # higher p_lcoe means better cells

    scored_dataframe=dataframe.sort_values(by='p_lcoe', ascending=False).copy()
    scored_dataframe.loc[:, 'capex'] = capex_tech  # Use .loc to avoid SettingWithCopyWarning
    
    return scored_dataframe

def find_grid_nodes_ERA5_cells(
        current_region:dict,
        buses_gdf:gpd.GeoDataFrame,
        cells_gdf:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    
    """
    takes in the provincial grid nodes, a proximity filter ( <= km) and the ERA5 cells dataframe. Calcuates the nearest nodde and distance to that node, imputes the 
    data to the ERA5 cells dataframe and returns as geodataframe. 
    """
    
    buses_gdf.sindex
    # Create a KDTree for bus station geometries
    bus_tree = cKDTree(buses_gdf['geometry'].apply(lambda x: (x.x, x.y)).tolist())
    log.info(f"> Calculating Nearest Grid Nodes for Grid Cells of {current_region['code']}")
    cells_gdf[['nearest_station', 'nearest_station_distance_km']] = cells_gdf['geometry'].apply(find_nearest_station, buses_gdf=buses_gdf,bus_tree=bus_tree).apply(pd.Series)
    cells_gdf_with_station_data=cells_gdf.copy()
    return cells_gdf_with_station_data

# Function to Extract BC Grid Cells from ERA5 Cutout and GADM Regional Boundary
# >>>>>>>>>>>>>>>>>>>>> NOT IN USE , provides wrong data
# def extract_BC_grid_cells_within_Regions(
#     cutout:atlite.Cutout,
#     province_gadm_regions_gdf:gpd.GeoDataFrame)->gpd.GeoDataFrame:

#     log.info(f"Extracting BC Grid Cells from ERA5 Cutout and GADM Region Boundaries...")

#     province_grid_cells = gpd.sjoin(cutout.grid, province_gadm_regions_gdf, predicate='intersects')
#     province_grid_cells.drop(columns=['index_right'],inplace=True)

#     return province_grid_cells

def merge_LandData_xarray_with_geodataframe(
            land_availability_xr:xr,
            cells_gdf:gpd.GeoDataFrame,
            column_name:str):
    
        land_availability_df=land_availability_xr.to_dataframe(name=column_name).reset_index().loc[lambda x: x[column_name] > 0]
        geometry = [Point(xy) for xy in zip(land_availability_df['x'], land_availability_df['y'])]

        # Create a new GeoDataFrame with the geometry column
        land_availability_gdf = gpd.GeoDataFrame(land_availability_df, geometry=geometry)
        # Create spatial indices for both GeoDataFrames
        cells_gdf_sindex = cells_gdf.sindex

        # Perform spatial join using spatial indices
        matched_rows = []
        for idx, row in land_availability_gdf.iterrows():
            x_geometry = row.geometry
            possible_matches_index = list(cells_gdf_sindex.intersection(x_geometry.bounds))
            possible_matches = cells_gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(x_geometry)]
            matched_rows.extend([(idx, i) for i in precise_matches.index])

        # Merge the two GeoDataFrames based on the matched rows
        merged_gdf = gpd.GeoDataFrame(matched_rows, columns=['x_index', 'cells_gdf_index'])
        merged_gdf = merged_gdf.merge(land_availability_gdf, left_on='x_index', right_index=True)
        cells_df_with_land_availability = merged_gdf.merge(cells_gdf, left_on='cells_gdf_index', right_index=True)
        cells_df_with_land_availability.drop(columns=['x_index', 'cells_gdf_index', '_x_', 'y_x', 'x_x',
            'geometry_x'],inplace=True)

        # Now, merged_gdf contains the merged GeoDataFrame based on spatial join using spatial indices

        cells_df_with_land_availability = cells_df_with_land_availability.rename(columns={'x_y': 'y', 'y_y': 'x','geometry_y':'geometry'})
        cells_gdf_with_land_availability=gpd.GeoDataFrame(cells_df_with_land_availability,geometry='geometry')
        return cells_gdf_with_land_availability

def calculate_land_availability_vector_data(
        cutout:atlite.Cutout,
        cells_gdf:gpd.GeoDataFrame,
        exclusion_gdf:gpd.GeoDataFrame,
        msg:str,
        column_name:str,
        current_region:dict)-> gpd.GeoDataFrame:
    
    """
    Calculates land availability (%) for given Geodataframe Cells and a vector filter.nEReturns a GeoDataFrame with land availability (%)
    """

    log.info(f"Calculating land availability by - {msg}...")

    excluder = ExclusionContainer()
    excluder.add_geometry(exclusion_gdf.geometry)
    
    cells_gdf_union = cells_gdf.dissolve(by='Province').loc[["British Columbia"]].geometry.to_crs(excluder.crs)

    #  Ensure cells_gdf_union contains only one geometry
    if len(cells_gdf_union) == 1:
        # Calculate eligibility share
        eligible_share = (excluder.compute_shape_availability(cells_gdf_union)[0].sum() * excluder.res**2 / cells_gdf_union.geometry.iloc[0].area).item()
        print(f"The eligibility share is: {eligible_share:.2%}")
    else:
        print("Error: cells_gdf_union contains more than one geometry. Please ensure it contains only one geometry.")

    if eligible_share==1:
        cells_gdf_with_land_availability=cells_gdf.copy()
        cells_gdf_with_land_availability[column_name]=1
    else:
        # Calculate land availability matrix
        land_availability_xr = cutout.availabilitymatrix(cells_gdf_union, excluder).rename({'Province': '_x_'})

        # Merge land availability with the original GeoDataFrame
        cells_gdf_with_availability=merge_LandData_xarray_with_geodataframe(land_availability_xr,cells_gdf,column_name)
        # log.info(f"Calculated Land Availability data loaded for  {len(province_ERA5_cells_with_land_availability)} Cells in Col.:{'eligible_land_area'}.\n")

        cells_gdf_with_availability=calculate_cell_area(cells_gdf_with_availability,column_name,current_region)
    
    return cells_gdf_with_availability

def calculate_land_availability_raster(
        cutout:atlite.Cutout,
        province_ERA5_cells:gpd.GeoDataFrame,
        msg:str,
        gaez_raster:str,
        column_name:str,
        land_class_selection:list,
        current_region:dict,
        buffer:float,
        exclusion=True)->gpd.GeoDataFrame:

    """
    Calculates land availability (%) for given Geodataframe Cells and a raster filter.

    Parameters:
        ERA5_cells_BC : (GeoDataFrame): GeoDataFrame containing grid cells.
        gaez_raster : (Directory Str): Raster File Directory
        column_name : (Str) : Data (%) to be loaded in this new column.
        land_class_selection : A list of land classes to be included/excluded based on 'exclusion' criteria
        exclusion : Boolean Field; 
            True : land_class_selection list will excluded
            False : land_class_selection list will included

    Returns:
        GeoDataFrame: GeoDataFrame with land availability (%)
    """
        
    log.info(f"Calculating land availability by - {msg}...")
    excluder = ExclusionContainer()

    if exclusion:
        excluder.add_raster(gaez_raster,land_class_selection,buffer=buffer)
    else:
        #for INCLUSION layers
        excluder.add_raster(gaez_raster,land_class_selection,buffer=buffer, invert=True)

    # Dissolve by 'Province' and select British Columbia
    province_ERA5_cells_union = province_ERA5_cells.dissolve(by='Province').loc[["British Columbia"]].geometry.to_crs(excluder.crs)

    # Calculate eligibility share
    eligible_share = (excluder.compute_shape_availability(province_ERA5_cells_union)[0].sum() * excluder.res**2 / province_ERA5_cells_union.geometry.item().area).item()
    print(f"The eligibility share is: {eligible_share:.2%}")
    if eligible_share==1:
        province_ERA5_cells_with_land_availability=province_ERA5_cells.copy()
        province_ERA5_cells_with_land_availability[column_name]=1
    else:
        # Calculate land availability matrix
        land_availability_xr = cutout.availabilitymatrix(province_ERA5_cells_union, excluder).rename({'Province': '_x_'})

        province_ERA5_cells_with_land_availability=merge_LandData_xarray_with_geodataframe(land_availability_xr,province_ERA5_cells,column_name)

    log.info(f"Calculated Land Availability data loaded for  {len(province_ERA5_cells_with_land_availability)} Cells in Col.:{column_name}.\n")

    province_ERA5_cells_with_eligible_land_area = calculate_cell_area(province_ERA5_cells_with_land_availability,column_name,current_region)

    return province_ERA5_cells_with_eligible_land_area


def calculate_cell_area(
        gdf:gpd.GeoDataFrame,
        availability_column:str,
        current_region:dict)->gpd.GeoDataFrame:

    """
    Takes the ERA5 Grid geodataframe,land availability column name and information for the selected region and calculates area for individual grid (ERA5) cells. 
    Returns a geodataframe with the calculated land area.
    """
    
    gdf = gdf.to_crs(epsg=2163) 

    gdf_with_area_column=gdf.copy()

    if 'land_area_sq_km' not in gdf_with_area_column.columns:
        gdf_with_area_column ['land_area_sq_km']=gdf_with_area_column['geometry'].area / 1e6 #sq.km
        gdf_with_area_column['eligible_land_area']= gdf_with_area_column['land_area_sq_km']* gdf_with_area_column[availability_column] 
    else:
        gdf_with_area_column['eligible_land_area']= gdf_with_area_column['eligible_land_area']* gdf_with_area_column[availability_column] 
    
    _total_eligible_land_area = int(gdf_with_area_column['eligible_land_area'].sum()) #sq.km

    print(
        f"\nRegion of Interest : {current_region['code']}\n",
        f"> Eligible Land Area of the  Grid Cells = {{:,}} Km²\n".format(_total_eligible_land_area),
        f"> Actual Land Area : {{:,}} km²\n".format(current_region['area'])
    )

    #Restore the projection for other usage
    gdf_with_area_column=gdf_with_area_column.to_crs(epsg=4326)
    
    return gdf_with_area_column


# Function to Generate Cell Index from Region name
def assign_regional_cell_ids(cells_dataframe, Source_Column, index_name):
    unique_values = cells_dataframe[Source_Column].unique()

    # If there's only one unique value, return the original DataFrame
    if len(unique_values) == 1:
        return cells_dataframe

    region_dfs = []

    for x in unique_values:
        _mask = cells_dataframe[Source_Column] == x
        _x_df_ = cells_dataframe[_mask].reset_index(drop=True)

        # Create a new column with given index name, for each region
        _x_df_[index_name] = [f'{x}_{index + 1}' for index in range(len(_x_df_))]

        region_dfs.append(_x_df_)

    # Concatenate all site-specific DataFrames into one DataFrame
    dataframe_with_cell_ids = pd.concat(region_dfs, ignore_index=True)

    # Set the index to the newly created column if it exists
    if index_name in dataframe_with_cell_ids.columns:
        dataframe_with_cell_ids.set_index(index_name, inplace=True)

    return dataframe_with_cell_ids

def create_layout_for_generation(cutout,cells_gdf,capacity_column):
    log.info(f"Creating Layout for PV generation from BC Grid Cells...")
    resource_layout_MW = cutout.layout_from_capacity_list(
        cells_gdf, col=capacity_column)
    resource_layout_MW = resource_layout_MW.where(resource_layout_MW != 0, drop=True)

    return resource_layout_MW  #xarray


def find_nearest_station(cell_geometry, buses_gdf, bus_tree):
    _, index = bus_tree.query((cell_geometry.centroid.x, cell_geometry.centroid.y))
    nearest_bus_row = buses_gdf.iloc[index]
    distance_km = cell_geometry.distance(nearest_bus_row['geometry']) * 111.32  # Degrees to km conversion
    nearest_station_name = nearest_bus_row['name']
    return nearest_station_name, distance_km




def find_optimal_K(
        data_for_clustering:pd.DataFrame, 
        region_id:int, 
        wcss_tolerance:float, 
        max_k  :int)->pd.DataFrame:
    
    """
    takes the scored grid cells and gives the optimal number of k-means cluster from the elbow chart approach. 
    The Within Cluster Sum of Squares (WCSS) is a tolerance measure to find that opitmal number of k. Higher the WCSS, higher the aggregation of cells i.e. less number of cluster.
    WCSS measures the squared average distance (difference between cluster's mean CF and individual CF) of all the points within a cluster to the cluster centroid (here - mean CF)  
    """
    
    print(f">> Estimating optimal number of Clusters for each region based on the Score for each Cell ...")

    # Remove duplicate points
    data_for_clustering = data_for_clustering.drop_duplicates()

    scaler = StandardScaler()
    data = scaler.fit_transform(data_for_clustering)

    # Initialize empty list to store the within-cluster sum of squares (WCSS)
    wcss_data = []

    # Try different values of k (number of clusters)
    for k in range(1, min(max_k, len(data_for_clustering))):
        kmeans_data = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
        # Inertia is the within-cluster sum of squares
        wcss_data.append(kmeans_data.inertia_)

    # Calculate the total WCSS
    total_wcss_data = sum(wcss_data)

    # Calculate the tolerance as a percentage of the total WCSS
    tolerance_data = wcss_tolerance * total_wcss_data

    # Initialize the optimal k
    optimal_k_data = next((k for k, wcss_value in enumerate(wcss_data, start=1) if wcss_value <= tolerance_data), None)

# Plot and save the elbow charts
    plt.plot(range(1, min(max_k, len(data_for_clustering))), wcss_data, marker='o', linestyle='-', label='p_lcoe')
    if optimal_k_data is not None:
        plt.axvline(x=optimal_k_data, color='r', linestyle='--',
                    label=f"Optimal k = {optimal_k_data}; K-means with {round(wcss_tolerance*100)}% of WCSS")

    plt.title(f"Elbow plot of K-means Clustering with 'p_lcoe' for Region-{region_id}")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.legend()

    # Ensure x-axis ticks are integers
    plt.xticks(range(1, min(max_k, len(data_for_clustering))))

    plt.tight_layout()

    # Print the optimal k
    print(f"Zone {region_id} - Optimal k for p_lcoe based clustering: {optimal_k_data}\n")

    return optimal_k_data

def pre_process_cluster_mapping(
        cells_scored:pd.DataFrame,
        vis_directory:str,
        wcss_tolerance:float)->tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    xxx 
    """

    unique_regions = cells_scored['Region'].unique()
    elbow_plot_directory=os.path.join(vis_directory,'Regional_cluster_Elbow_Plots')

    region_optimal_k_list = []

    # Loop over unique regions
    for region in unique_regions:
        # Select data for the current region_id
        data_for_clustering = cells_scored[cells_scored['Region'] == region][['p_lcoe']]
        
        # Call the function for K-means clustering and elbow plot
        optimal_k = find_optimal_K(data_for_clustering, region, wcss_tolerance, max_k=15)
        
        # Append values to the list
        region_optimal_k_list.append({'Region': region, 'Optimal_k': optimal_k})

        # Save the elbow plot
        plot_name = f'elbow_plot_region_{region}.png'
        plt.savefig(os.path.join(elbow_plot_directory, plot_name))
        plt.close()  # Close the plot to avoid overlapping
    ##################################################################
    print(f">>> K-means clustering Elbow plots generated for each region based on the Score for each Cell ...")

    # Create a DataFrame from the list
    region_optimal_k_df = pd.DataFrame(region_optimal_k_list)
    region_optimal_k_df['Optimal_k'].fillna(0, inplace=True)
    region_optimal_k_df['Optimal_k'] = region_optimal_k_df['Optimal_k'].astype(int)
    
    NonZeroClustersmask=region_optimal_k_df['Optimal_k']!=0
    region_optimal_k_df=region_optimal_k_df[NonZeroClustersmask]

    _x = cells_scored.merge(region_optimal_k_df, on='Region', how='left')
    cells_scored = assign_regional_cell_ids(_x,'Region', 'cell')#.set_index('cell')
    

    print(f"Optimal-k based on 'p_lcoe' clustering calculated for {len(unique_regions)} zones and saved to cell dataframe.\n")
    cells_scored_cluster_mapped=cells_scored.copy()

    return cells_scored_cluster_mapped,region_optimal_k_df

def cells_to_cluster_mapping(
        cells_scored:pd.DataFrame,
        vis_directory:str,
        wcss_tolerance:float, 
        sort_columns:list=['p_lcoe', 'potential_capacity'])-> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Clustering requires a base prior to perform the aggregation/clustering similar data. Here, we are doing spatial clustering which aggregates the spatial regions
    based on some common features. As a common feature, the modellers can design tailored metric find/calculate a feature which performs as a good proxy of the suitablity of the cells as a
    potential renewable energy site. For this tool (this version), we are calculating a composite scoring approach to find the suitablity of the sites. The composite score has been labelled as 
    as "p_lcoe" i.e. proxy Levelized Cost of Electricity (LCOE) which denotes the energy yield (MWh) per dollar of investment in a cell. We have calculated this composite score for each cell and
    here with this function, this score will be acting as the common feature for the spatial clustering.

    Does the following sequential tasks:
  Processess the GWA cells beofe we approach for clustering . An output of this function is a region vs optimal cluster for the region and another output denotes cell vs mapped region vs cluster no. (in which the cells will be merged into).
    
    """
    dataframe,optimal_k_df=pre_process_cluster_mapping(cells_scored,vis_directory,wcss_tolerance)

    print(f">>> Mapping the Optimal Number of Clusters for Each region ...")

    clusters = []
    dataframe_filtered=dataframe[dataframe['Region'].isin(list(optimal_k_df['Region']))]
    
    for region, group in dataframe_filtered.groupby('Region'):
        group = group.sort_values(by=sort_columns, ascending=False)
        region_rows = len(group)
        
        optimal_k = optimal_k_df[optimal_k_df['Region'] == region]['Optimal_k'].iloc[0]
        region_step_size = region_rows // optimal_k
        
        clusters.extend([group.iloc[i:i+region_step_size].copy() for i in range(0, region_rows, region_step_size)])
        
        if len(clusters[-1]) < region_step_size:
            clusters[-2] = pd.concat([clusters[-2], clusters.pop()], ignore_index=False)
        
        cluster_no_counter = 1  # Reset cluster_no_counter for each region
        for cluster_df in clusters[-optimal_k:]:
            cluster_df['Cluster_No'] = cluster_no_counter
            cluster_no_counter += 1
    cells_cluster_map_df=pd.concat(clusters, ignore_index=False)

    return cells_cluster_map_df,optimal_k_df

def create_cells_Union_in_clusters(
        cluster_map_gdf:gpd.GeoDataFrame, 
        region_optimal_k_df:pd.DataFrame
            )->tuple[pd.DataFrame,dict]:

    """
    Dissolve a GeoDataFrame based on 'Bucket_No' and aggregate columns as specified.

    Parameters:
    - cluster_map_gdf: GeoDataFrame, the input GeoDataFrame to be dissolved and aggregated.
    - region_optimal_k_df: DataFrame, a DataFrame containing region information.

    Returns:
    - dissolved_gdf: GeoDataFrame, the dissolved and aggregated GeoDataFrame.
    - dissolved_indices: dict, a dictionary containing the indices of the dissolved rows for each region and each Cluster_No.
    """
    log.info(f" Preparing Clusters...")
    
    # Initialize an aggregation dictionary
    # agg_dict = {'p_lcoe': lambda x: x.iloc[len(x) // 2], 'capex':'first','Cluster_No':'first','potential_capacity': 'sum','Region_ID': 'first', 'Region': 'first', 'CF_mean': 'mean',}
    agg_dict = {'p_lcoe': lambda x: x.iloc[len(x) // 2], 
                'capex':'first',
                'Cluster_No':'first',
                'potential_capacity': 'sum',
                'Region_ID': 'first',
                'Region': 'first',
                # 'CF_mean': 'mean',
                'nearest_station':'first',
                'nearest_station_distance_km':'first'}

    # Initialize an empty list to store the dissolved results
    dissolved_gdf_list = []
    
    # Initialize an empty dictionary to store dissolved indices for each region and each Cluster_No
    dissolved_indices = {}
    i=0
    # Loop through each region
    for region in region_optimal_k_df['Region']:
        i+=1
        log.info(f" Creating cluster for {region} {i}/{len(region_optimal_k_df['Region'])}")
        region_mask = cluster_map_gdf['Region'] == region
        region_cells = cluster_map_gdf[region_mask]

        # Initialize dictionary for the current region
        dissolved_indices[region] = {}

        # Loop through each Cluster_No in the current region
        for cluster_no, group in region_cells.groupby('Cluster_No'):
            # Store the indices of the rows before dissolving
            dissolved_indices[region][cluster_no] = group.index.tolist()

            # Dissolve by 'Bucket_No' and aggregate using the agg_dict
            region_dissolved = group.dissolve(by='Cluster_No', aggfunc=agg_dict)

            # Append the dissolved GeoDataFrame to the list
            dissolved_gdf_list.append(region_dissolved)

        # Concatenate all GeoDataFrames in the list
        dissolved_gdf = pd.concat(dissolved_gdf_list, ignore_index=True)
        
        # Keep only the specified columns
        # columns_to_keep = ['Region','Region_ID','Cluster_No', 'capex', 'potential_capacity','p_lcoe','nearest_station','nearest_station_distance_km','geometry' ] #'CF_mean',
        # dissolved_gdf = dissolved_gdf[columns_to_keep]

        dissolved_gdf=assign_regional_cell_ids(dissolved_gdf,'Region','cluster_id')

        dissolved_gdf['Cluster_No'] = dissolved_gdf['Cluster_No'].astype(int)
        dissolved_gdf.sort_values(by='p_lcoe', ascending=False, inplace=True)
        dissolved_gdf['Site_ID'] = range(1, len(dissolved_gdf)+1)
    log.info(f" Culsters Created and a list generated to map the Cells inside each Cluster...")
    return dissolved_gdf, dissolved_indices

def clip_cluster_boundaries_upto_regions(
        cell_cluster_gdf:gpd.GeoDataFrame,
        gadm_regions_gdf:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    """
    xxx 
    """
    cell_cluster_gdf_clipped=cell_cluster_gdf.clip(gadm_regions_gdf,keep_geom_type=False)
    cell_cluster_gdf_clipped.sort_values(by=['p_lcoe'], ascending=False, inplace=True) #, 'CF_mean'

    return cell_cluster_gdf_clipped

def select_top_sites(
    all_scored_sites_gdf:gpd.GeoDataFrame, 
    resource_max_capacity:float)-> gpd.GeoDataFrame:
    print(f">>> Selecting TOP Sites to for {resource_max_capacity} GW Capacity Investment in BC...")
    """
    Select the top sites based on potential capacity and a maximum resource capacity limit.

    Parameters:
    - sites_gdf: GeoDataFrame containing  cell and bucket information.
    - resource_max_capacity : Maximum allowable  capacity in GW.

    Returns:
    - selected_sites: GeoDataFrame with the selected top sites.
    """
    print(f"{'_'*50}")
    print(f"Selecting the Top Ranked Sites to invest in {resource_max_capacity} GW PV in BC")
    print(f"{'_'*50}\n")

    # Initialize variables
    selected_rows:list = []
    total_capacity:float = 0.0

    top_sites:gpd.GeoDataFrame = all_scored_sites_gdf.copy()

    if top_sites['potential_capacity'].iloc[0] < resource_max_capacity * 1000:
        # Iterate through the sorted GeoDataFrame
        for index, row in top_sites.iterrows():
            # Check if adding the current row's capacity exceeds resource capacity
            if total_capacity + row['potential_capacity'] <= resource_max_capacity * 1000:
                selected_rows.append(index)  # Add the row to the selection
                # Update the total capacity
                total_capacity += row['potential_capacity']
            # If adding the current row's capacity would exceed max resource capacity, stop the loop
            else:
                break

        # Create a new GeoDataFrame with the selected rows
        top_sites:gpd.GeoDataFrame = top_sites.loc[selected_rows]

        # Apply the additional logic
        mask = all_scored_sites_gdf['Site_ID'] > top_sites['Site_ID'].max()
        selected_additional_sites:gpd.GeoDataFrame = all_scored_sites_gdf[mask].head(1)
        
        remaining_capacity:float = resource_max_capacity * 1000 - top_sites['potential_capacity'].sum()

        if remaining_capacity > 0:
            
            # selected_additional_sites['capex'] = capex* remaining_capacity
            print(f"\n!! Note: The Last cluster originally had {round(selected_additional_sites['potential_capacity'].iloc[0] / 1000,2)} GW potential capacity."
                 f"To fit the maximum capacity investment of {resource_max_capacity} GW, it has been adjusted to {round(remaining_capacity / 1000,2)} GW\n")
            
            selected_additional_sites['potential_capacity'] = remaining_capacity
        # Concatenate the DataFrames
        top_sites = pd.concat([top_sites, selected_additional_sites])
    else:
        original_capacity = all_scored_sites_gdf['potential_capacity'].iloc[0]

        print(f"\n!! Note: The first cluster originally had {round(original_capacity / 1000,2)} GW potential capacity."
              f"To fit the maximum capacity investment of {resource_max_capacity} GW, it has been adjusted. \n")

        top_sites = top_sites.iloc[:1]  # Keep only the first row
        # Adjust the potential_capacity of the first row
        top_sites.at[top_sites.index[0], 'potential_capacity'] = resource_max_capacity * 1000

    return top_sites  # gdf


def select_top_sites(all_scored_sites_gdf, resource_max_capacity):
    print(f">>> Selecting TOP Sites to for {resource_max_capacity} GW Capacity Investment in Province...")
    """
    Select the top sites based on potential capacity and a maximum capacity limit.

    Parameters:
    - sites_gdf: GeoDataFrame containing cell and bucket information.
    - Maximum allowable resource capacity in GW.

    Returns:
    - selected_sites: GeoDataFrame with the selected top sites.
    """
    print(f"{'_'*50}")
    print(f"Selecting the Top Ranked Sites to invest in {resource_max_capacity} GW resource in Province")
    print(f"{'_'*50}\n")

    selected_rows = []
    total_capacity = 0.0

    top_sites = all_scored_sites_gdf.copy()

    if top_sites['potential_capacity'].iloc[0] < resource_max_capacity * 1000:
        # Iterate through the sorted GeoDataFrame
        for index, row in top_sites.iterrows():
            # Check if adding the current row's capacity exceeds max_resource_capacity
            if total_capacity + row['potential_capacity'] <= resource_max_capacity * 1000:
                selected_rows.append(index)  # Add the row to the selection
                # Update the total capacity
                total_capacity += row['potential_capacity']
            # If adding the current row's capacity would exceed max_resource_capacity, stop the loop
            else:
                break

        # Create a new GeoDataFrame with the selected rows
        top_sites = top_sites.loc[selected_rows]

        # Apply the additional logic
        mask = all_scored_sites_gdf['Site_ID'] > top_sites['Site_ID'].max()
        selected_additional_sites = all_scored_sites_gdf[mask].head(1)
        
        remaining_capacity = resource_max_capacity * 1000 - top_sites['potential_capacity'].sum()

        if remaining_capacity > 0:
            
            # selected_additional_sites['capex'] = capex* remaining_capacity
            print(f"\n!! Note: The Last cluster originally had {round(selected_additional_sites['potential_capacity'].iloc[0] / 1000,2)} GW potential capacity."
                 f"To fit the maximum capacity investment of {resource_max_capacity} GW, it has been adjusted to {round(remaining_capacity / 1000,2)} GW\n")
            
            selected_additional_sites['potential_capacity'] = remaining_capacity
        # Concatenate the DataFrames
        top_sites = pd.concat([top_sites, selected_additional_sites])
    else:
        original_capacity = all_scored_sites_gdf['potential_capacity'].iloc[0]

        print(f"\n!! Note: The first cluster originally had {round(original_capacity / 1000,2)} GW potential capacity."
              f"To fit the maximum capacity investment of {resource_max_capacity} GW, it has been adjusted. \n")

        top_sites = top_sites.iloc[:1]  # Keep only the first row
        # Adjust the potential_capacity of the first row
        top_sites.at[top_sites.index[0], 'potential_capacity'] = resource_max_capacity * 1000

    return top_sites  # gdf

def print_module_title(text,Length_Char_inLine=60):
    print(f"{Length_Char_inLine*'_'}\n"
        f"{5*' ' }{text}\n"
        f"{Length_Char_inLine*'_'}")
    
def fix_df_ts_index(
    df:pd.DataFrame, 
    snapshot_timezone_region:dict[list],
    snapshot_serial:int)->pd.DataFrame:
    '''
    This function resets the timeseries index with timezone conversion.<br>
    <b>!! Caution </b>: The converted timeseries is an index reset only, with a naive timestamp without timezone info. 
    '''
    
    new_indices = pd.date_range(start = snapshot_timezone_region['start'][snapshot_serial], end = snapshot_timezone_region['end'][snapshot_serial], freq='h')
    
    df.index = new_indices
    
    return df

def tz_convert_ts_index(
    ts_df:pd.DataFrame,
    timezone:str
    )->pd.DataFrame:
    '''
    This function converts the timeseries index with timezone information imputed conversion.<br>
    <b> Recommended timeseries index conversion method</b> in contrast to naive timestamp index reset method. 
    '''
    # Localize to UTC (assuming your times are currently in UTC)
    ts_df_index_utc = ts_df.tz_localize('UTC')

    # Convert to defined timezone (in Pandas time zones)
    ts_df_index_converted = ts_df_index_utc.tz_convert(timezone)

    # ts_df.index=ts_df_index_converted.index
    
    return ts_df_index_converted