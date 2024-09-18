import argparse
import subprocess

def create_resources(config_path, resource_type):
    normalized_resource_type = resource_type.lower()
    
    if normalized_resource_type == 'solar':
        script_path = 'workflow/scripts/create_resource_options_solar.py'
    elif normalized_resource_type == 'bess':
        script_path = 'workflow/scripts/create_resource_options_bess.py'
    elif normalized_resource_type == 'wind':
        script_path = 'workflow/scripts/create_resource_options_wind.py'
    else:
        print("Unknown resource type. Linking tool currently supports only one of the following resources: Solar, Wind, BESS")
        return
    
    try:
        subprocess.run(['python', script_path, config_path, normalized_resource_type], check=True)
        print(f">>>> Successfully executed {script_path} for resource type {normalized_resource_type} with config: {config_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing {script_path}: {e}")

def prepare_data(config_path):
    script_path = 'workflow/scripts/prepare_data.py'
    try:
        subprocess.run(['python', script_path, config_path], check=True)
        print(f">>> Successfully executed !!!! {script_path} with config: {config_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing {script_path}: {e}")
        
def select_top_sites(config_path, normalized_resource_type, resource_max_total_capacity=None):
    script_path = 'workflow/scripts/resource_options_select_top_sites.py'
    
    # Construct the command, including the optional capacity if provided
    cmd = ['python', script_path, config_path, normalized_resource_type]
    if resource_max_total_capacity is not None:
        cmd.append(str(resource_max_total_capacity))
    try:
        subprocess.run(cmd, check=True)
        print(f">>> Successfully executed {script_path} for resource type {normalized_resource_type} with config: {config_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing {script_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Linking Tool CLI: A tool for managing resources and configurations.',
        epilog='Examples:\n'
               '  linkingtool create_resources /path/to/config.yml solar\n'
               '  linkingtool prepare_data /path/to/config.yml\n'
               '  linkingtool top_sites /path/to/config.yml solar 1000\n'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Subparser for 'create_resources' command
    create_resources_parser = subparsers.add_parser(
        'create_resources',
        help='Create resources based on the provided configuration file and resource type'
    )
    create_resources_parser.add_argument(
        'config_path',
        type=str,
        help="Path to the configuration file '*.yml'"
    )
    create_resources_parser.add_argument(
        'resource_type',
        type=str,
        choices=['solar', 'bess', 'wind'],
        help='Type of resource to create (solar, bess, wind)'
    )
    
    # Subparser for 'prepare_data' command
    prepare_data_parser = subparsers.add_parser(
        'prepare_data',
        help='Prepare data based on the provided configuration file'
    )
    prepare_data_parser.add_argument(
        'config_path',
        type=str,
        help="Path to the configuration file '*.yml'"
    )
    
    # Subparser for 'top_sites' command
    top_sites_parser = subparsers.add_parser(
        'top_sites',
        help='Select TOP SITES based on the provided configuration file'
    )
    top_sites_parser.add_argument(
        'config_path',
        type=str,
        help="Path to the configuration file '*.yml'"
    )
    top_sites_parser.add_argument(
        'resource_type',
        type=str,
        choices=['solar', 'bess', 'wind'],
        help='Type of resource to create (solar, bess, wind)'
    )
    # Adding the optional 'resource_max_total_capacity' argument
    top_sites_parser.add_argument(
        '--resource_max_total_capacity',
        type=float,
        default=None,
        help="<Optional> Maximum total capacity for the resource (e.g., 10 for 10 GW). If none given, value will be absorbed from the user config"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.command == 'create_resources':
        create_resources(args.config_path, args.resource_type)
    elif args.command == 'prepare_data':
        prepare_data(args.config_path)
    elif args.command == 'top_sites':
        select_top_sites(args.config_path, args.resource_type, args.resource_max_total_capacity)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
