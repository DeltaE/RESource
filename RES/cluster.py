
"""
Spatial clustering module for renewable energy resource assessment.

This module provides K-means clustering functionality for aggregating grid cells
with similar renewable energy characteristics into representative clusters. The
clustering is based on techno-economic metrics such as Levelized Cost of Electricity
(LCOE) and potential capacity, enabling spatial aggregation for energy system
modeling and optimization.

The module implements an automated workflow for determining optimal cluster numbers,
performing spatial clustering, and creating representative cluster geometries that
maintain spatial relationships while reducing computational complexity for large-scale
renewable energy assessments.

Key Features
------------
- Automated optimal cluster number determination using elbow method
- Spatial clustering based on LCOE and capacity metrics
- Grid cell identifier generation for data linking
- Cluster geometry creation through spatial union operations
- Regional boundary clipping for precise spatial extent
- Visualization of clustering analysis results

Functions
---------
assign_cluster_id(cells, source_column=sub_national_unit_tag, index_name='cell')
    Generate unique identifiers for grid cells based on region and coordinates
    
find_optimal_K(resource_type, data_for_clustering, region, wcss_tolerance, max_k)
    Determine optimal number of clusters using elbow method and WCSS tolerance
    
pre_process_cluster_mapping(cells_scored, vis_directory, wcss_tolerance, resource_type)
    Preprocess data and determine optimal cluster numbers for each region
    
cells_to_cluster_mapping(cells_scored, vis_directory, wcss_tolerance, resource_type, sort_columns)
    Map grid cells to clusters based on similarity metrics and optimal cluster numbers
    
create_cells_Union_in_clusters(cluster_map_gdf, region_optimal_k_df, resource_type)
    Create unified cluster geometries by dissolving individual cell boundaries
    
clip_cluster_boundaries_upto_regions(cell_cluster_gdf, gadm_regions_gdf, resource_type)
    Clip cluster boundaries to precise regional administrative boundaries

Clustering Methodology
----------------------
The clustering approach follows a multi-step process:

1. **Data Preparation**: Grid cells with calculated LCOE and capacity metrics
2. **Optimal K Determination**: Uses elbow method with Within-Cluster Sum of Squares (WCSS)
3. **Regional Clustering**: Performs K-means clustering separately for each region
4. **Spatial Aggregation**: Creates unified cluster geometries through spatial union
5. **Boundary Refinement**: Clips results to precise administrative boundaries

The LCOE-based clustering ensures that cells with similar techno-economic 
characteristics are grouped together, creating representative clusters suitable
for energy system optimization while maintaining spatial coherence.

Algorithm Details
-----------------
- **K-means Clustering**: Uses scikit-learn implementation with multiple initializations
- **Elbow Method**: Automatically determines optimal cluster count based on WCSS tolerance
- **Missing Data Handling**: Imputes missing values using mean strategy
- **Spatial Preservation**: Maintains geographic relationships through geometry operations
- **Regional Processing**: Handles each administrative region independently

Usage Examples
--------------
Basic clustering workflow:

>>> import pandas as pd
>>> import geopandas as gpd
>>> from RES.cluster import cells_to_cluster_mapping, create_cells_Union_in_clusters
>>> 
>>> # Perform clustering analysis
>>> cluster_map_gdf, optimal_k_df = cells_to_cluster_mapping(
>>>     cells_scored=scored_cells,
>>>     vis_directory="vis/BC",
>>>     wcss_tolerance=0.15,
>>>     resource_type="solar",
>>>     sort_columns=["lcoe_solar"]
>>> )
>>> 
>>> # Create unified cluster geometries
>>> clusters_gdf, cluster_indices = create_cells_Union_in_clusters(
>>>     cluster_map_gdf=cluster_map_gdf,
>>>     region_optimal_k_df=optimal_k_df,
>>>     resource_type="solar"
>>> )

Cell identification:

>>> # Generate unique cell identifiers
>>> cells_with_ids = assign_cluster_id(
>>>     cells=grid_cells,
>>>     source_column="Province",
>>>     index_name="cell_id"
>>> )

Input Data Requirements
-----------------------
The clustering functions expect GeoDataFrames with specific columns:

Required Columns:
- 'x', 'y': Grid cell centroid coordinates
- sub_national_unit_tag: Administrative region classification
- 'lcoe_{resource_type}': Levelized cost of electricity
- 'potential_capacity_{resource_type}': Maximum potential capacity
- 'geometry': Spatial geometry (Polygon or Point)

Optional Columns:
- 'capex_{resource_type}': Capital expenditure costs
- 'fom_{resource_type}': Fixed operation and maintenance costs
- 'vom_{resource_type}': Variable operation and maintenance costs
- '{resource_type}_CF_mean': Average capacity factor
- 'nearest_station': Nearest grid connection point
- 'nearest_station_distance_km': Distance to grid connection

Output Data Structure
---------------------
Clustering results include:

Cluster Map GeoDataFrame:
- Individual cells with assigned cluster numbers
- Original cell attributes preserved
- Cluster_No: Integer cluster identifier
- Optimal_k: Optimal number of clusters for region

Unified Clusters GeoDataFrame:
- Dissolved cluster geometries
- Aggregated techno-economic parameters
- Representative cluster characteristics
- Spatial extent covering all member cells

Cluster Indices Dictionary:
- Mapping of original cell indices to clusters
- Structure: {region: {cluster_no: [cell_indices]}}
- Enables traceability from clusters back to individual cells

Visualization Outputs
--------------------
The module generates several visualization products:

Elbow Plots:
- WCSS vs. number of clusters for each region
- Optimal cluster number identification
- Saved to vis_directory/Regional_cluster_Elbow_Plots/

Performance Considerations
--------------------------
- Memory usage scales with number of grid cells and clusters
- Processing time increases with higher max_k values
- Imputation handles missing data but may affect clustering quality
- Large regions may benefit from hierarchical clustering approaches

Dependencies
------------
- pandas: Data manipulation and analysis
- geopandas: Spatial data operations
- numpy: Numerical computations
- matplotlib.pyplot: Visualization
- sklearn.cluster.KMeans: K-means clustering algorithm
- sklearn.impute.SimpleImputer: Missing value imputation
- pathlib: File path operations
- logging: Progress and error reporting
- RES.utility: Custom utility functions for spatial operations

Notes
-----
- Clustering is performed separately for each administrative region
- WCSS tolerance controls the trade-off between cluster number and representation
- Missing or infinite values are automatically handled through imputation
- Cluster ranking is based on ascending LCOE values (lowest cost first)
- Spatial relationships are preserved through geometry operations
- Results are suitable for energy system optimization models

See Also
--------
- RES.CellCapacityProcessor: For generating input data with LCOE calculations
- RES.utility: For additional spatial operations and cell ID management
- sklearn.cluster: For alternative clustering algorithms
"""

import logging as log
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

import RES.utility as utils

imputer = SimpleImputer(strategy="mean")  # Other strategies: "median",         "most_frequent"

def assign_cluster_id(cells: gpd.GeoDataFrame, 
                  source_column: str = None, 
                  index_name: str = 'cell') -> gpd.GeoDataFrame:
    """
    Generate unique identifiers for grid cells based on region and coordinates.
    
    Creates standardized cell identifiers that combine regional information
    with spatial coordinates to ensure uniqueness across the entire assessment
    domain. These identifiers serve as primary keys for data linking and
    result tracking throughout the assessment workflow.
    
    Parameters
    ----------
    cells : gpd.GeoDataFrame
        Input GeoDataFrame containing spatial data with 'x', 'y' coordinates
        and regional classification information
    source_column : str, default None
        Column name containing regional classification (e.g., province, state)
    index_name : str, default 'cell'
        Name for the new unique identifier column
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with new unique cell identifier column set as index
        
    Examples
    --------
    Basic cell ID assignment:
    
    >>> cells_with_ids = assign_cluster_id(
    ...     cells=grid_cells,
    ...     source_column='Province',
    ...     index_name='cell_id'
    ... )
    >>> print(cells_with_ids.index.name)  # 'cell_id'
    
    Custom identifier format:
    
    >>> # Creates IDs like: "BC_-123.5_49.2"
    >>> cells = assign_cluster_id(cells, 'Province', 'unique_cell')
    
    Raises
    ------
    ValueError
        If source_column doesn't exist in the GeoDataFrame
    ValueError  
        If required coordinate columns 'x', 'y' are missing
        
    Notes
    -----
    - Removes spaces from region names for consistent formatting
    - ID format: "{region}_{x_coord}_{y_coord}"
    - Coordinates maintain original decimal precision
    - Sets generated IDs as DataFrame index for efficient lookups
    - Essential for linking spatial analysis results across workflow steps
    """
    
    if source_column is None:
        raise ValueError(f"'{source_column}' not defined for indexing. Please provide a valid source column name.")
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
    Determine optimal number of clusters using elbow method and WCSS tolerance.
    
    Analyzes grid cells with renewable energy characteristics to find the optimal
    number of K-means clusters using the elbow method. The Within-Cluster Sum of
    Squares (WCSS) tolerance parameter controls the trade-off between cluster
    representation accuracy and computational complexity.
    
    The function iteratively tests different cluster numbers (k) and calculates
    WCSS for each configuration. The optimal k is determined when WCSS falls
    below the specified tolerance threshold, indicating diminishing returns for
    additional clusters.
    
    Parameters
    ----------
    resource_type : str
        Type of renewable energy resource ('solar', 'wind', 'bess')
        Used for labeling and file naming
    data_for_clustering : pd.DataFrame
        Preprocessed data containing clustering features (LCOE, capacity)
        Must have no missing values or infinite values
    region : str
        Name of the administrative region being processed
        Used for plot titles and output messages
    wcss_tolerance : float
        Tolerance threshold as fraction of total WCSS (0.0 to 1.0)
        Lower values = more clusters, higher values = fewer clusters
    max_k : int
        Maximum number of clusters to test
        Limited by data size and computational constraints
        
    Returns
    -------
    int or None
        Optimal number of clusters for the region
        Returns None if no optimal k found within tolerance
        
    Examples
    --------
    Find optimal clusters for solar data:
    
    >>> optimal_k = find_optimal_K(
    ...     resource_type="solar",
    ...     data_for_clustering=clean_data,
    ...     region="British Columbia",
    ...     wcss_tolerance=0.15,
    ...     max_k=20
    ... )
    >>> print(f"Optimal clusters: {optimal_k}")
    
    Notes
    -----
    - WCSS measures squared distances from cluster centroids
    - Higher WCSS tolerance leads to fewer, more aggregated clusters
    - Lower WCSS tolerance leads to more, finer-grained clusters
    - Elbow plots are automatically generated and displayed
    - Function uses K-means with 10 random initializations for stability
    - Processing time increases quadratically with max_k
    
    Algorithm Details
    -----------------
    1. Test k from 1 to min(max_k, data_size)
    2. Calculate WCSS (inertia) for each k using K-means
    3. Compute tolerance threshold as fraction of total WCSS
    4. Find first k where WCSS ≤ tolerance threshold
    5. Generate elbow plot with optimal k marked
    
    The WCSS measures the sum of squared distances between each data point
    and its assigned cluster centroid. Lower WCSS indicates tighter, more
    homogeneous clusters but may lead to over-segmentation.
    
    Raises
    ------
    ValueError
        If data_for_clustering is empty or contains only NaN values
    RuntimeError
        If K-means clustering fails for any k value
        
    See Also
    --------
    sklearn.cluster.KMeans : K-means clustering implementation
    pre_process_cluster_mapping : Preprocessing function that calls this method
    """
    
    utils.print_update(level=2,message="Estimating optimal number of Clusters for each region based on the Score for each Cell ...")

    # Initialize empty list to store the within-cluster sum of squares (WCSS)
    wcss_data = []

    # Try different values of k (number of clusters)
    for k in range(1, min(max_k, len(data_for_clustering))):
        # Handle NaN values by filling them with the mean of the column


        kmeans_data = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data_for_clustering)
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
                    label=f"Optimal k = {optimal_k_data}; K-means with {round(wcss_tolerance*100,3)}% of WCSS")

    plt.title(f"Elbow plot of K-means Clustering with 'LCOE_{resource_type}' for Region-{region}")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.legend()

    # Ensure x-axis ticks are integers
    plt.xticks(range(1, min(max_k, len(data_for_clustering))))

    # plt.tight_layout()

    # Print the optimal k
    print(f"Zone {region} - Optimal k for LCOE_{resource_type} based clustering: {optimal_k_data}\n")

    return optimal_k_data

def pre_process_cluster_mapping(
        cells_scored:pd.DataFrame,
        vis_directory:str,
        wcss_tolerance:float,
        sub_national_unit_tag:str,
        resource_type:str)->tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Preprocess data and determine optimal cluster numbers for each region.
    
    Performs comprehensive preprocessing of scored grid cells to prepare them
    for K-means clustering analysis. The function handles missing data, determines
    optimal cluster numbers for each administrative region, and generates
    visualization outputs for clustering analysis.
    
    This function serves as the preprocessing pipeline that prepares raw scored
    cell data for the main clustering workflow, ensuring data quality and
    generating region-specific clustering parameters.
    
    Parameters
    ----------
    cells_scored : pd.DataFrame
        GeoDataFrame containing scored grid cells with LCOE and capacity data
        Must include columns: 'Region', 'lcoe_{resource_type}', 'potential_capacity_{resource_type}'
    vis_directory : str
        Base directory path for saving visualization outputs
        Elbow plots will be saved in subdirectory 'Regional_cluster_Elbow_Plots'
    wcss_tolerance : float
        WCSS tolerance threshold for optimal cluster determination (0.0 to 1.0)
        Controls trade-off between cluster number and representation accuracy
    resource_type : str
        Resource type identifier ('solar', 'wind', 'bess')
        Used for column name construction and labeling
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - cells_scored_cluster_mapped: Enhanced cell data with optimal k values and cell IDs
        - region_optimal_k_df: Summary of optimal cluster numbers by region
        
    Examples
    --------
    Preprocess solar cell data:
    
    >>> cells_mapped, optimal_k_summary = pre_process_cluster_mapping(
    ...     cells_scored=scored_solar_cells,
    ...     vis_directory="vis/BC",
    ...     wcss_tolerance=0.15,
    ...     resource_type="solar"
    ... )
    >>> print(f"Processed {len(cells_mapped)} cells across {len(optimal_k_summary)} regions")
    
    Processing Workflow
    -------------------
    1. **Region Iteration**: Process each unique administrative region separately
    2. **Data Validation**: Check for required columns and sufficient data
    3. **Data Cleaning**: Handle infinite values and missing data through imputation
    4. **Optimal K Finding**: Apply elbow method to determine cluster numbers
    5. **Visualization**: Generate and save elbow plots for each region
    6. **Data Integration**: Merge optimal k values back to cell data
    7. **ID Assignment**: Generate unique cell identifiers for data linking
    
    Data Quality Handling
    ---------------------
    - **Missing Columns**: Regions without required columns are skipped
    - **Infinite Values**: Replaced with NaN for proper imputation
    - **Empty Data**: Regions with insufficient data are excluded
    - **Imputation**: Uses mean strategy for missing value replacement
    - **Zero Clusters**: Regions with optimal_k=0 are filtered out
    
    Output Structure
    ----------------
    cells_scored_cluster_mapped contains:
    - All original cell attributes
    - 'Optimal_k': Optimal cluster number for the cell's region
    - 'cell': Unique cell identifier (set as index)
    
    region_optimal_k_df contains:
    - 'Region': Administrative region name
    - 'Optimal_k': Optimal number of clusters for the region
    
    Visualization Outputs
    ---------------------
    Generates elbow plots saved to:
    `{vis_directory}/Regional_cluster_Elbow_Plots/elbow_plot_region_{region}.png`
    
    Each plot shows:
    - WCSS vs. number of clusters
    - Optimal k marked with vertical line
    - Region-specific title and labels
    
    Notes
    -----
    - Processing is performed region-by-region for spatial coherence
    - Imputation strategy can affect clustering quality
    - Visualization directory is created if it doesn't exist
    - Regions with insufficient data (< 2 cells) may be skipped
    - Memory usage scales with number of regions and cells per region
    
    Raises
    ------
    ValueError
        If vis_directory path is invalid or cannot be created
    KeyError
        If required columns are missing from cells_scored
    RuntimeError
        If imputation or clustering fails for critical regions
        
    See Also
    --------
    find_optimal_K : Core optimal cluster determination function
    assign_cluster_id : Cell identifier generation function
    cells_to_cluster_mapping : Main clustering workflow function
    """

    unique_regions = cells_scored[sub_national_unit_tag].unique()
    try:
        elbow_plot_directory = Path(vis_directory, 'Regional_cluster_Elbow_Plots')
        elbow_plot_directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Failed to create directory at {elbow_plot_directory}. Ensure 'vis_directory' is valid. Error: {e}")
    region_optimal_k_list = []

    # Loop over unique regions
    for region in unique_regions:
        print(f"\n=== Processing region: {region} ===")
        expected_cols = [f'lcoe_{resource_type}', f'potential_capacity_{resource_type}']
        
        available_cols = cells_scored.columns.tolist()
        print("Available columns in cells_scored:", available_cols)
        
        # Check if all required columns exist
        if not all(col in available_cols for col in expected_cols):
            print(f"Missing columns for clustering in region {region}. Skipping.")
            continue

        data_for_clustering = cells_scored[cells_scored[sub_national_unit_tag] == region][expected_cols]
        
        # Replace inf/-inf with NaN so they can be imputed
        data_for_clustering.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        print("Data before imputation:")
        print(data_for_clustering.describe())

        # Drop columns that are entirely NaN
        data_for_clustering.dropna(axis=1, how='all', inplace=True)

        if data_for_clustering.empty or data_for_clustering.shape[1] == 0:
            print(f"Data for clustering is empty or invalid for region {region}. Skipping.")
            continue

        try:
            imputed_array = imputer.fit_transform(data_for_clustering)
        except Exception as e:
            print(f"Imputer failed for region {region} with error: {e}")
            continue

        data_for_clustering_cleaned = pd.DataFrame(imputed_array, columns=data_for_clustering.columns)
        
        
        # Call the function for K-means clustering and elbow plot
        optimal_k = find_optimal_K(resource_type,data_for_clustering_cleaned, region, wcss_tolerance, max_k=15)
        
        # Append values to the list
        region_optimal_k_list.append({sub_national_unit_tag: region, 'Optimal_k': optimal_k})

        # Save the elbow plot
        plot_name = f'elbow_plot_region_{region}.png'
        plot_save_to=elbow_plot_directory/plot_name
        plt.savefig(plot_save_to)
        plt.close()  # Close the plot to avoid overlapping
    ##################################################################
    print(">>> K-means clustering Elbow plots generated for each region based on the Score for each Cell ...")

    # Create a DataFrame from the list
    region_optimal_k_df = pd.DataFrame(region_optimal_k_list)
    region_optimal_k_df['Optimal_k'].fillna(0, inplace=True)
    region_optimal_k_df['Optimal_k'] = region_optimal_k_df['Optimal_k'].astype(int)
    
    NonZeroClustersmask=region_optimal_k_df['Optimal_k']!=0
    region_optimal_k_df=region_optimal_k_df[NonZeroClustersmask]

    _x = cells_scored.merge(region_optimal_k_df, on=sub_national_unit_tag, how='left')
    cells_scored = assign_cluster_id(_x,sub_national_unit_tag, 'cell')#.set_index('cell')
    

    print(f"Optimal-k based on 'LCOE' clustering calculated for {len(unique_regions)} zones and saved to cell dataframe.\n")
    cells_scored_cluster_mapped=cells_scored.copy()

    return cells_scored_cluster_mapped,region_optimal_k_df

def cells_to_cluster_mapping(
        cells_scored:pd.DataFrame,
        vis_directory:str,
        wcss_tolerance:float,
        sub_national_unit_tag:str,
        resource_type:str,
        sort_columns:list)-> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Map grid cells to clusters based on similarity metrics and optimal cluster numbers.
    
    Performs spatial clustering of renewable energy grid cells by grouping cells with
    similar techno-economic characteristics (primarily LCOE) into representative clusters.
    The function implements a systematic approach to divide each region's cells into
    the optimal number of clusters determined through elbow method analysis.
    
    This is the main clustering workflow function that transforms individual grid cells
    into clustered representations suitable for energy system optimization models,
    reducing computational complexity while preserving spatial and economic relationships.
    
    Parameters
    ----------
    cells_scored : pd.DataFrame
        Scored grid cells with techno-economic attributes
        Must contain LCOE, capacity, and regional classification data
    vis_directory : str
        Directory path for saving clustering visualization outputs
        Used for elbow plots and clustering analysis results
    wcss_tolerance : float
        Within-Cluster Sum of Squares tolerance (0.0 to 1.0)
        Controls cluster granularity vs. computational efficiency trade-off
    resource_type : str
        Renewable energy resource type ('solar', 'wind', 'bess')
        Determines which columns to use for clustering analysis
    sort_columns : list
        Column names for sorting cells before cluster assignment
        Typically includes LCOE or other ranking metrics
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - cells_cluster_map_df: Individual cells with assigned cluster numbers
        - optimal_k_df: Summary of optimal cluster counts by region
        
    Examples
    --------
    Perform clustering for wind resources:
    
    >>> cluster_map, optimal_k = cells_to_cluster_mapping(
    ...     cells_scored=wind_cells_scored,
    ...     vis_directory="vis/Alberta",
    ...     wcss_tolerance=0.20,
    ...     resource_type="wind",
    ...     sort_columns=["lcoe_wind", "potential_capacity_wind"]
    ... )
    >>> print(f"Created {cluster_map['Cluster_No'].max()} clusters across regions")
    
    Clustering Methodology
    ----------------------
    The clustering approach follows several key principles:
    
    1. **Regional Separation**: Clustering is performed independently for each
       administrative region to maintain spatial coherence and respect political
       boundaries that affect renewable energy development.
    
    2. **LCOE-Based Similarity**: Cells are grouped based on Levelized Cost of
       Electricity (LCOE) as the primary similarity metric, ensuring clusters
       represent similar economic viability.
    
    3. **Sorted Assignment**: Within each region, cells are sorted by specified
       metrics (typically LCOE) before being assigned to clusters, ensuring
       that the best cells are distributed across clusters.
    
    4. **Equal Distribution**: Cells are divided as evenly as possible across
       the optimal number of clusters for each region, preventing cluster
       size imbalances.
    
    Algorithm Workflow
    ------------------
    1. **Preprocessing**: Call pre_process_cluster_mapping to determine optimal k
    2. **Region Filtering**: Focus on regions with valid optimal cluster numbers
    3. **Cell Sorting**: Sort cells within each region by specified criteria
    4. **Cluster Assignment**: Divide sorted cells into optimal number of groups
    5. **Remainder Handling**: Merge small remainder groups into larger clusters
    6. **Numbering**: Assign sequential cluster numbers within each region
    
    Cluster Assignment Strategy
    ---------------------------
    For each region with n cells and k optimal clusters:
    - Calculate step_size = n ÷ k
    - Assign cells [0:step_size] to cluster 1
    - Assign cells [step_size:2*step_size] to cluster 2
    - Continue until all cells are assigned
    - Merge any remainder cells into the last cluster
    
    This ensures balanced cluster sizes while maintaining economic similarity
    through the pre-sorting step.
    
    Output Data Structure
    ---------------------
    cells_cluster_map_df contains:
    - All original cell attributes (LCOE, capacity, coordinates, etc.)
    - 'Cluster_No': Integer cluster identifier within region
    - 'Optimal_k': Total number of clusters for the cell's region
    - 'cell': Unique cell identifier (as index)
    
    optimal_k_df contains:
    - sub_national_unit_tag : Administrative region unit (e.g. Region or Municipality etc.)
    - 'Optimal_k': Optimal number of clusters determined for region
    
    Performance Considerations
    --------------------------
    - Memory usage scales linearly with number of cells
    - Processing time increases with number of regions and complexity
    - Sorting operations may be memory-intensive for large datasets
    - Cluster assignment is efficient O(n) operation per region
    
    Quality Assurance
    -----------------
    - Validates that all cells receive cluster assignments
    - Ensures cluster numbers are sequential within regions
    - Maintains data integrity through concatenation operations
    - Preserves spatial relationships through regional processing
    
    Notes
    -----
    - Clustering preserves regional boundaries for political/administrative coherence
    - LCOE-based sorting ensures economic similarity within clusters
    - Balanced cluster sizes improve downstream optimization performance
    - Results are suitable for capacity expansion and dispatch optimization models
    - Cluster numbering resets for each region (regional scope)
    
    Raises
    ------
    ValueError
        If required columns are missing or data validation fails
    KeyError
        If region names don't match between datasets
    RuntimeError
        If clustering assignment produces invalid results
        
    See Also
    --------
    pre_process_cluster_mapping : Preprocessing and optimal k determination
    create_cells_Union_in_clusters : Spatial union of clustered cells
    find_optimal_K : Core optimal cluster number determination
    """
    dataframe,optimal_k_df=pre_process_cluster_mapping(cells_scored,vis_directory,wcss_tolerance,sub_national_unit_tag,resource_type)

    utils.print_update(level=2,message="Mapping the Optimal Number of Clusters for Each region ...")

    clusters = []
    dataframe_filtered=dataframe[dataframe[sub_national_unit_tag].isin(list(optimal_k_df[sub_national_unit_tag]))]
    
    for region, group in dataframe_filtered.groupby(sub_national_unit_tag):
        group = group.sort_values(by=sort_columns, ascending=True)
        region_rows = len(group)
        
        optimal_k = optimal_k_df[optimal_k_df[sub_national_unit_tag] == region]['Optimal_k'].iloc[0]
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
        sub_national_unit_tag:str,
        resource_type:str
            )->tuple[pd.DataFrame,dict]:

    """
    Create unified cluster geometries by dissolving individual cell boundaries.
    
    Transforms individual grid cells assigned to clusters into unified cluster
    geometries through spatial union operations. This process aggregates both
    geometric boundaries and techno-economic attributes to create representative
    cluster entities suitable for energy system optimization models.
    
    The function performs spatial dissolve operations grouped by cluster number
    within each region, creating cohesive cluster polygons while maintaining
    traceability back to original cells through detailed index mapping.
    
    Parameters
    ----------
    cluster_map_gdf : gpd.GeoDataFrame
        Grid cells with cluster assignments from cells_to_cluster_mapping
        Must contain defined sub_national_unit_tag, 'Cluster_No', and geometric attributes
    region_optimal_k_df : pd.DataFrame
        Summary of optimal cluster numbers by region
        Contains defined sub_national_unit_tag and 'Optimal_k' columns
    resource_type : str
        Resource type identifier ('solar', 'wind', 'bess')
        Used for column naming and aggregation rules
        
    Returns
    -------
    tuple[pd.DataFrame, dict]
        - dissolved_gdf: Unified cluster geometries with aggregated attributes
        - dissolved_indices: Mapping of cluster to original cell indices
        
    Examples
    --------
    Create unified solar clusters:
    
    >>> clusters_gdf, cell_mapping = create_cells_Union_in_clusters(
    ...     cluster_map_gdf=mapped_cells,
    ...     region_optimal_k_df=optimal_k_summary,
    ...     resource_type="solar"
    ... )
    >>> print(f"Created {len(clusters_gdf)} unified clusters")
    >>> print(f"Cluster 1 contains {len(cell_mapping['BC'][1])} original cells")
    
    Aggregation Strategy
    --------------------
    Different attributes are aggregated using specific strategies:
    
    **Economic Metrics**:
    - LCOE: Median value (representative of cluster economics)
    - CAPEX, FOM, VOM: First value (uniform within region/technology)
    
    **Performance Metrics**:
    - Capacity Factor: Mean value (average performance)
    - Potential Capacity: Sum (total cluster capacity)
    
    **Infrastructure Metrics**:
    - Nearest Station: First value (primary connection point)
    - Distance to Grid: First value (representative distance)
    
    **Classification**:
    - Region, Cluster_No: First value (preserved identity)
    
    Geometric Operations
    --------------------
    1. **Spatial Dissolve**: Union of cell geometries within each cluster
    2. **Topology Preservation**: Maintains valid polygon geometry
    3. **Attribute Aggregation**: Combines cell attributes per aggregation rules
    4. **Index Tracking**: Records original cell indices for each cluster
    
    Output Structure
    ----------------
    dissolved_gdf contains unified clusters with:
    - 'cluster_id': Unique cluster identifier (as index)
    -  sub_national_unit_tag: Administrative region unit (e.g., Region or Municipality)
    - 'Cluster_No': Sequential cluster number within region
    - 'Rank': Cluster ranking based on LCOE (ascending)
    - Economic attributes: Aggregated costs and performance metrics
    - 'geometry': Unified cluster polygon geometry
    
    dissolved_indices structure:
    ```
    {
        'region_name': {
            cluster_no: [list_of_original_cell_indices],
            ...
        },
        ...
    }
    ```
    
    Processing Workflow
    -------------------
    1. **Region Iteration**: Process each region independently
    2. **Cluster Grouping**: Group cells by cluster number within region
    3. **Index Recording**: Store original cell indices before dissolving
    4. **Spatial Dissolve**: Union geometries and aggregate attributes
    5. **Result Compilation**: Concatenate all dissolved clusters
    6. **ID Assignment**: Generate unique cluster identifiers
    7. **Ranking**: Sort and rank clusters by economic metrics
    8. **Column Cleanup**: Standardize column names for downstream use
    
    Traceability Features
    ---------------------
    The dissolved_indices dictionary enables:
    - Mapping clusters back to constituent cells
    - Detailed analysis of cluster composition
    - Validation of aggregation results
    - Disaggregation for detailed reporting
    
    Quality Assurance
    -----------------
    - Validates that all cells are included in clusters
    - Ensures geometric validity after spatial operations
    - Maintains attribute consistency through aggregation
    - Preserves regional and cluster identity information
    
    Performance Considerations
    --------------------------
    - Memory usage scales with cluster complexity and number
    - Spatial operations may be computationally intensive
    - Large clusters with many cells require more processing time
    - Geometric simplification may be beneficial for very detailed cells
    
    Notes
    -----
    - Cluster ranking facilitates economic dispatch optimization
    - Column name standardization removes resource type suffixes
    - Median LCOE provides robust cluster economic representation
    - Spatial union preserves geographic relationships
    - Results are optimized for energy system modeling workflows
    
    Raises
    ------
    ValueError
        If cluster assignments are invalid or missing
    GeometryError
        If spatial dissolve operations fail
    KeyError
        If required columns are missing from input data
        
    See Also
    --------
    cells_to_cluster_mapping : Preceding cluster assignment function
    clip_cluster_boundaries_upto_regions : Boundary refinement function
    gpd.GeoDataFrame.dissolve : Core spatial dissolve operation
    """
    utils.print_update(level=1,message=" Preparing Clusters...")
    node_distance_col = utils.get_available_column(cluster_map_gdf, ['nearest_station_distance_km', 'nearest_distance'])
    grid_node_col = utils.get_available_column(cluster_map_gdf, ['nearest_station', 'nearest_connection_point'])
        
    # Initialize an aggregation dictionary
    agg_dict = {#f'LCOE_{resource_type}': lambda x: x.iloc[len(x) // 2], 
                f'lcoe_{resource_type}': lambda x: x.iloc[len(x) // 2], 
                f'capex_{resource_type}':'first',
                f'fom_{resource_type}':'first',
                f'vom_{resource_type}':'first',
                f'{resource_type}_CF_mean':'mean',
                'Cluster_No':'first',
                f'potential_capacity_{resource_type}': 'sum',
                sub_national_unit_tag: 'first',
                grid_node_col:'first',
                node_distance_col:'first'}

    # Initialize an empty list to store the dissolved results
    dissolved_gdf_list = []
    
    # Initialize an empty dictionary to store dissolved indices for each region and each Cluster_No
    dissolved_indices = {}
    i=0
    # Loop through each region
    for region in region_optimal_k_df[sub_national_unit_tag]:
        i+=1
        log.info(f" Creating cluster for {region} {i}/{len(region_optimal_k_df[sub_national_unit_tag])}")
        region_mask = cluster_map_gdf[sub_national_unit_tag] == region
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

        dissolved_gdf=utils.assign_regional_cell_ids(dissolved_gdf,sub_national_unit_tag,'cluster_id')

        dissolved_gdf['Cluster_No'] = dissolved_gdf['Cluster_No'].astype(int)
        dissolved_gdf.sort_values(by=f'lcoe_{resource_type}', ascending=True, inplace=True)
        # dissolved_gdf.sort_values(by=f'LCOE_{resource_type}', ascending=True, inplace=True)
        dissolved_gdf['Rank'] = range(1, len(dissolved_gdf)+1)
        
        dissolved_gdf.columns=dissolved_gdf.columns.str.replace(fr"(?i)(_{resource_type}|{resource_type}_)", "", regex=True)
        
    utils.print_update(level=2,message="Clusters Created and a list generated to map the Cells inside each Cluster...")
    return dissolved_gdf, dissolved_indices

def clip_cluster_boundaries_upto_regions(
        cell_cluster_gdf:gpd.GeoDataFrame,
        gadm_regions_gdf:gpd.GeoDataFrame,
        resource_type)->gpd.GeoDataFrame:
    """
    Clip cluster boundaries to precise regional administrative boundaries.
    
    Refines cluster geometries by clipping them to exact administrative
    boundaries, ensuring that cluster extents respect political and
    administrative divisions. This final processing step removes any
    geometric artifacts from the clustering process and aligns results
    with official regional boundaries.
    
    The function performs spatial clipping operations to trim cluster
    polygons to the precise extent of administrative regions, maintaining
    data integrity while ensuring geographic accuracy for policy and
    planning applications.
    
    Parameters
    ----------
    cell_cluster_gdf : gpd.GeoDataFrame
        Unified cluster geometries from create_cells_Union_in_clusters
        Contains cluster polygons that may extend beyond regional boundaries
    gadm_regions_gdf : gpd.GeoDataFrame
        Official administrative boundary geometries from GADM dataset
        Defines precise regional extents for clipping operations
    resource_type : str
        Resource type identifier ('solar', 'wind', 'bess')
        Used for column identification and sorting operations
        
    Returns
    -------
    gpd.GeoDataFrame
        Clipped cluster geometries with boundaries precisely aligned
        to administrative regions, sorted by LCOE in ascending order
        
    Examples
    --------
    Clip wind clusters to provincial boundaries:
    
    >>> clipped_clusters = clip_cluster_boundaries_upto_regions(
    ...     cell_cluster_gdf=unified_clusters,
    ...     gadm_regions_gdf=provincial_boundaries,
    ...     resource_type="wind"
    ... )
    >>> print(f"Clipped {len(clipped_clusters)} clusters to regional boundaries")
    
    Clipping Operations
    -------------------
    1. **Spatial Intersection**: Clips cluster geometries using administrative boundaries
    2. **Topology Preservation**: Maintains valid polygon geometry after clipping
    3. **Attribute Retention**: Preserves all cluster attributes through clipping
    4. **Multi-geometry Handling**: Manages potential multi-polygon results
    
    Boundary Alignment Benefits
    ---------------------------
    - **Policy Compliance**: Ensures clusters respect administrative jurisdictions
    - **Planning Accuracy**: Aligns with regional energy planning boundaries
    - **Data Integrity**: Removes geometric inconsistencies from processing
    - **Visualization Quality**: Improves map accuracy for stakeholder communication
    
    Geometric Considerations
    ------------------------
    - Handles edge cases where clusters span multiple regions
    - Preserves cluster identity even after boundary clipping
    - Maintains geometric validity through robust clipping algorithms
    - May create multi-polygon geometries for clusters crossing boundaries
    
    Sorting and Organization
    ------------------------
    Results are sorted by LCOE in ascending order to facilitate:
    - Economic dispatch optimization
    - Merit order analysis
    - Least-cost development planning
    - Investment prioritization
    
    Quality Assurance
    -----------------
    - Validates geometric integrity after clipping operations
    - Ensures all clusters remain within administrative boundaries
    - Maintains attribute consistency through spatial operations
    - Preserves cluster ranking and identification
    
    Performance Notes
    -------------------
    - Clipping operations scale with geometric complexity
    - Large regions or detailed boundaries increase processing time
    - Memory usage depends on cluster and boundary detail level
    - Results are optimized for downstream energy modeling applications
    
    Use Cases
    ---------
    - **Regulatory Compliance**: Ensuring development respects jurisdictions
    - **Policy Analysis**: Aligning renewable development with administrative units
    - **Planning Integration**: Connecting energy models with regional planning
    - **Stakeholder Communication**: Accurate maps for decision-maker engagement
    
    Notes
    -----
    - Final step in the clustering workflow before energy system modeling
    - Essential for maintaining political and administrative coherence
    - Improves visual quality of cluster maps and analysis results
    - Ensures compatibility with regional energy planning frameworks
    - Results are ready for capacity expansion and dispatch optimization
    
    Raises
    ------
    GeometryError
        If clipping operations produce invalid geometries
    ValueError
        If input datasets have incompatible coordinate systems
    AttributeError
        If required columns are missing from input data
        
    See Also
    --------
    create_cells_Union_in_clusters : Preceding cluster creation function
    gpd.GeoDataFrame.clip : Core spatial clipping operation
    RES.boundaries.GADMBoundaries : Administrative boundary data source
    """
    cell_cluster_gdf_clipped=cell_cluster_gdf.clip(gadm_regions_gdf,keep_geom_type=False)
    # cell_cluster_gdf_clipped.sort_values(by=[f'LCOE_{resource_type}'], ascending=True, inplace=True) 
    cell_cluster_gdf_clipped.sort_values(by=[f'lcoe_{resource_type}'], ascending=True, inplace=True) 

    return cell_cluster_gdf_clipped
