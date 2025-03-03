from workflow.scripts import RESources as RES

# Iterate over provinces for both solar and wind resources
resource_types = ['wind','solar']  #
provinces=['BC'] # ,'AB','SK','ON','NS','MB'
for province_code in provinces:
    for resource_type in resource_types:
        required_args = {
            "config_file_path": 'config/config.yaml',
            "province_short_code": province_code,
            "resource_type": resource_type
        }
        
        # Create an instance of Resources and execute the module
        RES_module = RES.RESources_builder(**required_args)
        RES_module.build(select_top_sites=True,
                         use_pypsa_buses=False)
        