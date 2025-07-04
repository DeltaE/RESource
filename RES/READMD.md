## Resource Builder

The heart of RES package is __RESources.py__ module. The module contains a class 'RESources_builder'. The core method 'build()' steers the workflow to run sequential tasks to populate necessary datafields required for producing the stepwise resource assessment results.

## Steps (Methods)
### Step 1:
`get_grid_cells()` method is wrapper of `cell.py` module's `get_default_grid()` method. Returns a GeoDataframe.

> 1. The `get_default_grid()` creates several attributes e.g. the atlite's `cutout` object, the `region_boundary`.
> 2. Uses `cutout.grid` 

        self.get_cell_capacity()
        self.extract_weather_data()
        self.update_gwa_scaled_params(self.memory_resource_limitation) # testing, 2025 04 21
        self.get_CF_timeseries(cells=self.store_grid_cells)
        self.find_grid_nodes(cells=self.cells_with_ts_nt.cells,
                             use_pypsa_buses=use_pypsa_buses)
        self.score_cells(cells=self.region_grid_cells_cap_with_nodes)
        self.get_clusters(scored_cells=self.scored_cells)
        self.get_cluster_timeseries()
        self.units.create_units_dictionary()