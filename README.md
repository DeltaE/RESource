## Resource Builder

The heart of RES package is __RESources.py__ module. The module contains a class 'RESources_builder'. The core method `build()` steers the workflow to run sequential tasks to populate necessary datafields required for producing the stepwise resource assessment results.

## Steps (Methods)
### Step 1: `get_grid_cells()`
Returns a GeoDataframe with ERA5 resolution grid cells with coordinates, geometry and unique cell ids.

 1. The `get_default_grid()` creates several attributes e.g. the atlite's `cutout` object, the `region_boundary`.
 2. Uses `cutout.grid` attribute to create our test beds for the analysis i.e. the grid cells (geodataframe)

The resulting GeoDataFrame from `get_grid_cells()` includes the following columns:

| Attribute   | x                                              | y                                             | Country                                      | Province                                               | Region                                               | geometry                                                      |
| ----------- | ---------------------------------------------- | --------------------------------------------- | -------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------- |
| Description | Longitude coordinate (center) of the grid cell | Latitude coordinate (center) of the grid cell | Name of the country containing the grid cell | Name of the province or state containing the grid cell | Name of the specific region containing the grid cell | Polygon geometry defining the spatial extent of the grid cell |

### Step 2: `get_cell_capacity()`
Calculates the maximum installable capacity for each grid cell based on available area, land use constraints, and technology-specific parameters.

- 1. Determines % of usable area within each grid cell after applying exclusion criteria (e.g., protected areas, water bodies).
- 2. Applies technology landuse intensity (e.g., MW/km² for wind or solar) to estimate the potential capacity.
- 3. Adds a new column to the GeoDataFrame indicating the calculated capacity for each cell.

### Step 3a: `extract_weather_data()` , `update_gwa_scaled_params()`
- 1. Extracts relevant weather data (e.g., wind speed, solar irradiance) for each grid cell, required for resource assessment.
- 2. Updates grid cell parameters using Global Wind Atlas (GWA) data, scaling them as needed for accurate modeling.

### Step 4: `get_CF_timeseries()`
Generates capacity factor (CF) time series for each grid cell using weather data and technology characteristics.

### Step 5: `find_grid_nodes()`
Identifies and assigns grid nodes to each cell, facilitating network analysis and connection cost estimation.

### Step 6: `score_cells()`
Scores each grid cell based on multiple criteria (e.g., resource quality, proximity to grid), supporting site selection.

### Step 7.1: `get_clusters()`
Groups grid cells into clusters based on spatial or resource characteristics to enable aggregated analysis.

### Step 7.2: `get_cluster_timeseries()`
Produces time series data for each cluster, summarizing the resource and capacity factor information at the cluster level.
