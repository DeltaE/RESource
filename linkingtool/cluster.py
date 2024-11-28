
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import geopandas as gpd
import logging as log
import matplotlib.pyplot as plt
import linkingtool.utility as utils

def assign_cluster_id(cells: gpd.GeoDataFrame, 
                  source_column: str = 'Region', 
                  index_name: str = 'cell') -> gpd.GeoDataFrame:
    """
    Assigns unique cell IDs to each region in the specified GeoDataFrame.

    Parameters:
    cells (gpd.GeoDataFrame): Input GeoDataFrame containing spatial data with 'x' and 'y' coordinates.
    source_column (str): Column name in the GeoDataFrame that contains regional names.
    index_name (str): Name for the new index column to be created.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame with a new column of unique cell IDs for each region.
    """
    # Ensure the source column exists
    if source_column not in cells.columns:
        raise ValueError(f"'{source_column}' does not exist in the GeoDataFrame.")

    # Remove spaces in the region names for consistency
    cells[source_column] = cells[source_column].str.replace(" ", "", regex=False)

    # Check if 'x' and 'y' coordinates exist
    if 'x' not in cells.columns or 'y' not in cells.columns:
        raise ValueError("Columns 'x' and 'y' must exist in the GeoDataFrame.")

    # Generate unique cell IDs using a combination of the region name and coordinates
    cells[index_name] = (
        cells.apply(
            lambda row: f"{row[source_column]}_{row['x']}_{row['y']}",
            axis=1
        )
    )

    # Set the index to the newly created column
    cells.set_index(index_name, inplace=True)

    return cells

def find_optimal_K(
        resource_type:str,
        data_for_clustering:pd.DataFrame, 
        region:str, 
        wcss_tolerance:float, 
        max_k  :int
        )->pd.DataFrame:
    
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
    plt.plot(range(1, min(max_k, len(data_for_clustering))), wcss_data, marker='o', linestyle='-', label=f'lcoe_{resource_type}')
    if optimal_k_data is not None:
        plt.axvline(x=optimal_k_data, color='r', linestyle='--',
                    label=f"Optimal k = {optimal_k_data}; K-means with {round(wcss_tolerance*100)}% of WCSS")

    plt.title(f"Elbow plot of K-means Clustering with 'lcoe_{resource_type}' for Region-{region}")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.legend()

    # Ensure x-axis ticks are integers
    plt.xticks(range(1, min(max_k, len(data_for_clustering))))

    plt.tight_layout()

    # Print the optimal k
    print(f"Zone {region} - Optimal k for lcoe_{resource_type} based clustering: {optimal_k_data}\n")

    return optimal_k_data

def pre_process_cluster_mapping(
        cells_scored:pd.DataFrame,
        vis_directory:str,
        wcss_tolerance:float,
        resource_type:str)->tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    xxx 
    """

    unique_regions = cells_scored['Region'].unique()
    elbow_plot_directory=os.path.join(vis_directory,'Regional_cluster_Elbow_Plots')
    if not os.path.exists(elbow_plot_directory):
        os.makedirs(elbow_plot_directory)

    region_optimal_k_list = []

    # Loop over unique regions
    for region in unique_regions:
        # Select data for the current region
        data_for_clustering = cells_scored[cells_scored['Region'] == region][[f'lcoe_{resource_type}']]
        
        # Call the function for K-means clustering and elbow plot
        optimal_k = find_optimal_K(resource_type,data_for_clustering, region, wcss_tolerance, max_k=15)
        
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
    cells_scored = assign_cluster_id(_x,'Region', 'cell')#.set_index('cell')
    

    print(f"Optimal-k based on 'lcoe' clustering calculated for {len(unique_regions)} zones and saved to cell dataframe.\n")
    cells_scored_cluster_mapped=cells_scored.copy()

    return cells_scored_cluster_mapped,region_optimal_k_df

def cells_to_cluster_mapping(
        cells_scored:pd.DataFrame,
        vis_directory:str,
        wcss_tolerance:float,
        resource_type:str,
        sort_columns:list)-> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Clustering requires a base prior to perform the aggregation/clustering similar data. Here, we are doing spatial clustering which aggregates the spatial regions
    based on some common features. As a common feature, the modellers can design tailored metric find/calculate a feature which performs as a good proxy of the suitablity of the cells as a
    potential renewable energy site. For this tool (this version), we are calculating a composite scoring approach to find the suitablity of the sites. The composite score has been labelled as 
    as "lcoe" i.e. proxy Levelized Cost of Electricity (LCOE) which denotes the energy yield (MWh) per dollar of investment in a cell. We have calculated this composite score for each cell and
    here with this function, this score will be acting as the common feature for the spatial clustering.

    Does the following sequential tasks:
  Processess the GWA cells beofe we approach for clustering . An output of this function is a region vs optimal cluster for the region and another output denotes cell vs mapped region vs cluster no. (in which the cells will be merged into).
    
    """
    dataframe,optimal_k_df=pre_process_cluster_mapping(cells_scored,vis_directory,wcss_tolerance,resource_type)

    print(f">>> Mapping the Optimal Number of Clusters for Each region ...")

    clusters = []
    dataframe_filtered=dataframe[dataframe['Region'].isin(list(optimal_k_df['Region']))]
    
    for region, group in dataframe_filtered.groupby('Region'):
        group = group.sort_values(by=sort_columns, ascending=True)
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
        region_optimal_k_df:pd.DataFrame,
        resource_type:str
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
    agg_dict = {f'lcoe_{resource_type}': lambda x: x.iloc[len(x) // 2], 
                f'capex_{resource_type}':'first',
                f'fom_{resource_type}':'first',
                f'vom_{resource_type}':'first',
                f'{resource_type}_CF_mean':'mean',
                f'Cluster_No':'first',
                f'potential_capacity_{resource_type}': 'sum',
                f'Region': 'first',
                f'nearest_station':'first',
                f'nearest_station_distance_km':'first'}

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
        # columns_to_keep = ['Region','Region','Cluster_No', 'capex', 'potential_capacity','lcoe','nearest_station','nearest_station_distance_km','geometry' ]
        # dissolved_gdf = dissolved_gdf[columns_to_keep]

        dissolved_gdf=utils.assign_regional_cell_ids(dissolved_gdf,'Region','cluster_id')

        dissolved_gdf['Cluster_No'] = dissolved_gdf['Cluster_No'].astype(int)
        dissolved_gdf.sort_values(by=f'lcoe_{resource_type}', ascending=True, inplace=True)
        dissolved_gdf['Rank'] = range(1, len(dissolved_gdf)+1)
    log.info(f" Culsters Created and a list generated to map the Cells inside each Cluster...")
    return dissolved_gdf, dissolved_indices

def clip_cluster_boundaries_upto_regions(
        cell_cluster_gdf:gpd.GeoDataFrame,
        gadm_regions_gdf:gpd.GeoDataFrame,
        resource_type)->gpd.GeoDataFrame:
    """
    xxx 
    """
    cell_cluster_gdf_clipped=cell_cluster_gdf.clip(gadm_regions_gdf,keep_geom_type=False)
    cell_cluster_gdf_clipped.sort_values(by=[f'lcoe_{resource_type}'], ascending=True, inplace=True) 

    return cell_cluster_gdf_clipped
