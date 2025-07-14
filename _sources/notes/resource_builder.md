## RESource Builder Module

```{warning}
Typical Sequential Steps involves the following process as mentioned below. Note that data-source errors, configuration error may break the workflow which may necessitate additional methods to be used as intermediate steps.
```

```{tip}
Check the open-access publication on [RESource]()
```

```{seealso}
For detailed API documentation of the methods, see {doc}`api` and apis under {ref}`RESource Builder`.
```

<insert flow diagram of typical resource assessment>


### Step 1: Prepare Spatial Grid Cells

```{seealso}
`get_grid_cells()` at {ref}`RESource Builder` of {doc}`api`.
```

- This method collects the sub-national administrative boundaries. 
- Using that boundary, we calculate the Minimum Bounding Rectangle (MBR). 
- We use that MBR as a cutout to source weather resources data from ERA5 via CDSAPI. The ERA5's cutout is then stored as a netcdf `.nc' file.
- We load that cutout as `atlite`'s `cutout` object.
- We then use `atlite`'s `cutout.grid` attribute to create our test beds for the analysis i.e. the grid cells (geodataframe)
- We get the following key visuals out of this step:
  - Land availability map at excluder's highest resolution and another one at ERA5's resolution.
    > To harmonize the grid cells with the weather resources data, we use the ERA5's resolution data for further stages

```{important}
Data supply-chain for this method Requires [CDS-API setup](https://cds.climate.copernicus.eu/how-to-api)
```

The resulting GeoDataFrame from `get_grid_cells()` includes the following columns:

|Attribute   | Description|
|------------|---------------------------------------------------------------|
|`x `          | Longitude coordinate (center) of the grid cell|
|`y`           | Latitude coordinate (center) of the grid cell|
|`Country`    | Name of the country containing the grid cell. This is datafield 'NAME_0' in GADM's dataset |
|`Province`   | Name of the province or state containing the grid cell. This is datafield 'NAME_1' in GADM's dataset |
|`Region`    | Name of the specific region containing the grid cell. This is datafield 'NAME_2' in GADM's dataset |
|`geometry`   | Polygon geometry defining the spatial extent of the grid cell. The default CRS is EPSG:4326, to be set via config file|


### Step 2: Calculate Potential Capacity

```{seealso}
`RES.cell_processor.get_capacity()` of {doc}`api`.
```
- This method loads the cutout (atlite's cutout object), regional boundary (GeoDataFrame), loads the cost parameters and  also initiates a __composite excluder__
  - The ([`atlite`'s exclusion container](https://atlite.readthedocs.io/en/master/ref_api.html#atlite.Cutout.availabilitymatrix)) to merge all the spatial layers.
- the `cutout.availabilitymatrix` method calculates % of usable area within each grid cell after applying exclusion criteria (e.g., protected areas, water bodies) and returns an [`AvaliabilityMatrix`](https://atlite.readthedocs.io/en/master/ref_api.html#atlite.Cutout.availabilitymatrix)
- We apply technology landuse intensity (e.g., MW/km² for wind or solar) to translate this to potential capacity data.
  
```{attention}
- Current results gives a percentage of availability for each grid cell. It does not tell specifically which spatial area inside a grid cell is unavailable.
- The _potential capacity_ translation processing involves `area` calculation. The area calculation method is integrated to `RES.cell_processor.get_capacity()`. That method is sensitive to area calculation specific coordinate-system projection of the geodataframe. It is recommended to be cautious about choosing this crs.
```
- We get the maximum installable capacity for each grid cell based on available area, land use constraints, and technology-specific parameters.

- This method returns a named tuple with 'data' (a GeoDataFrame) and 'matrix' (availability matrix xarray).

```{tip}
- You can access the individual elements of the named tuple using dot notation, e.g., it you named the methods results as `result` then `result.data`, `result.matrix`
```
- Resulting GeoDataFrame includes the following new fields for each resource type (e.g., wind, solar):

| Attribute                                      | Description                                                      |
|------------------------------------------------|------------------------------------------------------------------|
| `potential_capacity_<resource_type>`           | Maximum installable capacity (MW) for the resource in the grid cell |
| `capex_<resource_type>`                        | Capital expenditure per MW for the resource                      |
| `fom_<resource_type>`                          | Fixed operation & maintenance cost per MW                        |
| `vom_<resource_type>`                          | Variable operation & maintenance cost per MWh                    |
| `grid_connection_cost_per_km_<resource_type>`  | Estimated grid connection cost per km for the resource           |
| `tx_line_rebuild_cost_<resource_type>`         | Transmission line rebuild cost per km for the resource           |
| `Operational_life_<resource_type>`             | Expected operational lifetime (years) for the resource           |

```{hint}
- Working enhancement includes VRE farm layout calculation models to make the potential capacity calculation more closer to the real world scenarios.
- Future improvements are planned to identify spatial areas (geometry) inside a grid cell.
```

### Step 3a: Get CF and Windspeed from Higher Resolution Data

```{attention}
- Currently configured for Wind Resources only. 
```

### Why wind resources' ERA5 data are rescaled ?
Wind resources (windspeed) are known to have significant variations across ERA5's ~30km resolution. To account for this, we rescaled the windspeed using higher resolution data from the Global Wind Atlas (GWA). This allows us to better estimate the windspeed at the grid cell level. However, GWA does not provide hourly profiles, so we source the profile from ERA5.


```{figure} ../_static/ERA5_resolution_windspeed_distribution_ERA5vsGWA_British_Columbia.png
:width: 600px
:name: era5-vs-gwa-windspeed-bc

Windspeed Distribution - ERA5 vs GWA (resampled to ERA5 Resolution) | Example from _British Columbia_ Study 
```

```{attention}
-  Working enhancement includes similar rescaling method for solar resources.
```

```{seealso}
`extract_weather_data()` , `update_gwa_scaled_params()` at {ref}`RESource Builder`.
```


- Extracts relevant weather data (e.g., wind speed, solar irradiance) for each grid cell. This calculation has been used for validation purposes. However, the available CF parameters (from different methods) could be used for scoring metric, energy calculations etc. 
  - We compared CF for IEC Class 2,3 turbines sourced from GWA and compared with RESource's result CFs.
- Updates grid cell parameters using Global Wind Atlas (GWA) data, scaling them as needed for accurate modeling.


### Step 4: Get Timeseries

```{seealso}
`RES.timeseries.get_timeseries()` at {ref}`RESource Builder`.
```
This method generates capacity factor (CF) time series for each grid cell using weather data and technology characteristics.

- We define technology attributes.
- We extract timeseries using weather resources data from ERA5's cutout.
  - The timeseries calculation method currently configured with [atlite.cutout.pv](https://atlite.readthedocs.io/en/master/ref_api.html#atlite.Cutout.pv) and [atlite.cutout.wind](https://atlite.readthedocs.io/en/master/ref_api.html#atlite.Cutout.wind) methods.

```{attention}
- Configure the timezone conversion information carefully to ensure proper usage of the timeseries in downstream modelling. 
- ERA5 provides naive timezone index data. We use the timezone information from config file to enable the timezone shift of the timeseries.
- However, after conversion we removed the timezone awareness from the datetime index to harmonize with pypsa supported timeseries index.
```

```{tip}
`RES.timeseries.__fix_timezone__()` method could be leveraged to reconfigure timezone awareness, if it is critical for your use-case of the timeseries.
```

- 
### Step 5: Find Grid Proximity
> This information is critical for downstream operational analysis with this resource options.

```{attention}
- Currently configured for Transmission Lines and/or Grid Substations.
- We do not know the specific project point of a resource. Hence, the resource to grid-node distance has been calculated from the centroid of each grid to the grid node. 
- If you have a specific project point, you should recalculate this distance with your specific project point.
```

- Identifies and assigns grid nodes to each cell. 
- Calculates distance (in km) from each grid cell to the nearest grid node (e.g., transmission line, substation) to assess connectivity and feasibility for energy transport.

    
```{tip}
If your use case of the resource options are to be plugged in to a downstream operational model (e.g. PyPSA), use harmonized nodes to populate this data.
> harmonized nodes i.e. same data that are intended to be used as _bus_ nodes at your operational model. 
```
```{seealso}
`find_grid_nodes()` at {ref}`RESource Builder`.
```

### Step 6: Scoring Metric to Rank the Sites

```{attention}
Currently scoring calculation is configured as simplified LCOE formula.
```

```{seealso}
`RES.score.CellScorer.get_cell_score()` at {ref}`RESource Builder`.
```
- Scores each grid cell based on multiple criteria (e.g., resource quality, proximity to grid), supporting site selection.

```{hint}
- Working enhancement includes several scoring metrics to serve scenario analysis.
- Future improvements are planned to include Multi-criteria Decision Analysis (MCDA) methods to make scoring more robust.

```

### Step 7: Clusterized Representation of the Sites
```{seealso}
`get_clusters()`, `get_cluster_timeseries()` at {ref}`RESource Builder`.
```
- Groups grid cells into clusters based on spatial or resource characteristics to enable aggregated analysis.
- Produces time series data for each cluster, summarizing the resource and capacity factor information at the cluster level.
