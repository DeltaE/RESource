from workflow.scripts.resources import Resources

## single resource

# Iterate over provinces for both solar and wind resources
resource_types = ['solar','wind'] 

provinces=[ 'BC']# 'AB','SK','ON','NS' ]  #''BC',
for province_code in provinces:
    for resource_type in resource_types:
        required_args = {
            "config_file_path": 'config/config.yaml',
            "province_short_code": province_code,
            "resource_type": resource_type
        }
        
        # Create an instance of Resources and execute the module
        resource_module = Resources(**required_args)
        resource_module.execute_module(memory_resource_limitation=False)