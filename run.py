# # Set Required Args to Activate Modules

import RES.RESources as RES
from RES.hdf5_handler import DataHandler

#Iterate over provinces for both solar and wind resources
resource_types = ['wind',]  # 'solar'
provinces=['BC']  #QC,'AB','SK','ON','NS','MB'
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

## Playground for Top Site Selection
resource_clusters_solar,cluster_timeseries_solar=RES_module.select_top_sites(solar_clusters,
                                                                solar_clusters_ts,
                                                                    resource_max_capacity=10)

resource_clusters_wind,cluster_timeseries_wind=RES_module.select_top_sites(wind_clusters,
                                                                wind_clusters_ts,
                                                                    resource_max_capacity=50)

# %%
RES_module.export_results('wind',
                    resource_clusters_wind,
                    cluster_timeseries_wind,)

# %%
RES_module.export_results('solar',
                    resource_clusters_solar,
                    cluster_timeseries_solar,)
