import RES.RESources as RES

def main(countries:list[str]=['AL', 'BA','XK','ME','MK','RS'], #'AL, 'BA','XK','ME','MK','RS']
         resource_types:list[str] = ['wind','solar'],
         ):
    
    for country in countries:
        for resource_type in resource_types:
            required_args = {
                "config_file_path": 'config/config.yaml',
                "region_short_code": country,
                "resource_type": resource_type
            }
            
            # Create an instance of Resources and execute the module
            RES_module = RES.RESources_builder(**required_args)
            # RES_module.build(select_top_sites=True,
            #                  use_pypsa_buses=False)
            RES_module.get_cell_capacity()

if __name__ == "__main__":
    main()