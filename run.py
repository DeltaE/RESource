
#!/usr/bin/env python3
"""
Canadian Renewable Energy Resource Analysis and Processing Pipeline

This script provides a comprehensive workflow for analyzing renewable energy resources 
(wind and solar) across Canadian provinces. It processes spatial data, calculates 
potential capacity, generates time series data, and exports results for downstream 
energy system modeling.

Main Workflow:
1. Iterates through Canadian provinces and resource types (wind/solar)
2. Builds resource datasets using the RESources_builder class
3. Processes grid cells, capacity calculations, and clustering
4. Generates time series data for resource availability
5. Selects optimal sites based on capacity constraints
6. Exports results in standardized formats

Provinces Processed:
    - QC: Quebec
    - AB: Alberta  
    - SK: Saskatchewan
    - ON: Ontario
    - NS: Nova Scotia
    - MB: Manitoba

Resource Types:
    - Wind: Wind power potential analysis
    - Solar: Solar photovoltaic potential analysis

Outputs:
    - HDF5 data stores containing processed resource data
    - CSV files with selected optimal sites and time series
    - Clustered resource representations for energy system modeling

Dependencies:
    - RES.RESources: Core resource analysis module
    - RES.hdf5_handler: Data storage and retrieval
    - Configuration files in config/ directory
    - Spatial data sources (ERA5, Global Wind Atlas, etc.)

Usage:
    python run.py

Notes:
    - Requires proper conda environment setup (see environment.yml)
    - Configuration parameters defined in config/config_CAN.yaml
    - Results stored in data/store/ and results/ directories
    - Processing time varies by province size and data availability

Author: RESource Development Team
Date: 2025
Version: Latest
"""

import RES.RESources as RES
from RES.hdf5_handler import DataHandler

# Iterate over provinces for both solar and wind resources
resource_types = ['wind','solar'] 
provinces=['QC','AB','SK','ON','NS','MB']  # 'BC','QC','AB','SK','ON','NS','MB'
for province_code in provinces:
    for resource_type in resource_types:
        required_args = {
            "config_file_path": 'config/config_CAN.yaml',
            "region_short_code": province_code,
            "resource_type": resource_type
        }
        
        # Create an instance of Resources and execute the module
        RES_module = RES.RESources_builder(**required_args)
        RES_module.build(select_top_sites=True,
                         use_pypsa_buses=False)
        

        # Explore the outputs from Store
        res_store=DataHandler(f'data/store/resources_{province_code}.h5')

        cells=res_store.from_store('cells')
        boundary=res_store.from_store('boundary')
        solar_clusters=res_store.from_store('clusters/solar')
        wind_clusters=res_store.from_store('clusters/wind')
        solar_clusters_ts=res_store.from_store('timeseries/clusters/solar')
        wind_clusters_ts=res_store.from_store('timeseries/clusters/wind')

        ## Example: Top Site Selection
        resource_clusters_solar,cluster_timeseries_solar=RES_module.select_top_sites(solar_clusters,
                                                                        solar_clusters_ts,
                                                                            resource_max_capacity=5) # GW

        resource_clusters_wind,cluster_timeseries_wind=RES_module.select_top_sites(wind_clusters,
                                                                        wind_clusters_ts,
                                                                            resource_max_capacity=20) # GW

        RES_module.export_results('wind',
                            province_code,
                            resource_clusters_wind,
                            cluster_timeseries_wind,)


        RES_module.export_results('solar',
                            province_code,
                            resource_clusters_solar,
                            cluster_timeseries_solar,)
