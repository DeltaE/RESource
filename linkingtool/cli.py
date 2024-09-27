import argparse
import subprocess
from pathlib import Path

class ResourceCreator:
    def __init__(self, config_path, resource_type):
        self.config_path = Path(config_path).resolve()
        self.resource_type = resource_type.lower()
    
    def create(self):
        # Base directories where scripts may exist
        base_dirs = [
            Path('workflow/scripts'),
            Path('Linking_tool/workflow/scripts')
        ]
        
        # Map resource types to script file names (without hardcoding full paths)
        script_map = {
            'solar': 'solar_module_v2.py',
            'bess': 'bess_module_v1.py',
            'wind': 'wind_module_v2.py'
        }

        if self.resource_type == 'all':
            # Run all scripts for all resource types
            for resource, script_name in script_map.items():
                script_path = self.find_script(base_dirs, script_name)
                if script_path:
                    self.run_script(script_path, resource)
                else:
                    print(f"Could not find a valid script for {resource}")
        else:
            # Run a single script based on resource_type
            script_name = script_map.get(self.resource_type)
            if not script_name:
                print("Unknown resource type. Supported resources: Solar, Wind, BESS, All")
                return
            
            script_path = self.find_script(base_dirs, script_name)
            if script_path:
                self.run_script(script_path, self.resource_type)
            else:
                print(f"Could not find a valid script for resource type {self.resource_type}")

    def find_script(self, base_dirs, script_name):
        """ Search for the script in the base directories and return the first valid path. """
        for base_dir in base_dirs:
            script_path = base_dir / script_name
            if script_path.exists():
                return script_path.resolve()
        return None

    def run_script(self, script_path, resource_type):
        """ Run the script with the provided resource type. """
        try:
            subprocess.run(['python', str(script_path), str(self.config_path), resource_type], check=True)
            print(f">>>> Successfully executed {script_path} for resource type {resource_type} with config: {self.config_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {script_path} for resource type {resource_type}: {e}")


class DataPreparer:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
    
    def prepare(self):
        # Base directories where scripts may exist
        base_dirs = [
            Path('workflow/scripts'),
            Path('Linking_tool/workflow/alternative')
        ]
        
        script_name = 'prepare_data_v2.py'
        script_path = self.find_script(base_dirs, script_name)
        if script_path:
            self.run_script(script_path)
        else:
            print(f"Could not find a valid script for data preparation")

    def find_script(self, base_dirs, script_name):
        """ Search for the script in the base directories and return the first valid path. """
        for base_dir in base_dirs:
            script_path = base_dir / script_name
            if script_path.exists():
                return script_path.resolve()
        return None

    def run_script(self, script_path):
        """ Run the script. """
        try:
            subprocess.run(['python', str(script_path), str(self.config_path)], check=True)
            print(f">>> Successfully executed {script_path} with config: {self.config_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {script_path}: {e}")


class TopSitesSelector:
    def __init__(self, config_path, resource_type, max_capacity=None):
        self.config_path = Path(config_path).resolve()
        self.resource_type = resource_type.lower()
        self.max_capacity = max_capacity

    def select(self):
        # Base directories where scripts may exist
        base_dirs = [
            Path('workflow/scripts'),
            Path('Linking_tool/workflow/alternative')
        ]
        
        script_name = 'top_sites_v2.py'
        script_path = self.find_script(base_dirs, script_name)
        if script_path:
            self.run_script(script_path)
        else:
            print(f"Could not find a valid script for top sites selection")

    def find_script(self, base_dirs, script_name):
        """ Search for the script in the base directories and return the first valid path. """
        for base_dir in base_dirs:
            script_path = base_dir / script_name
            if script_path.exists():
                return script_path.resolve()
        return None

    def run_script(self, script_path):
        """ Run the script with the provided resource type and max capacity. """
        cmd = ['python', str(script_path), str(self.config_path), self.resource_type]
        if self.max_capacity is not None:
            cmd.append(str(self.max_capacity))
        
        try:
            subprocess.run(cmd, check=True)
            print(f">>> Successfully executed {script_path} for resource type {self.resource_type} with config: {self.config_path}")
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
        choices=['solar', 'bess', 'wind', 'all'],
        help='Type of resource to create (solar, bess, wind, all)'
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
        choices=['solar', 'bess', 'wind', 'all'],
        help='Type of resource to create (solar, bess, wind, all)'
    )
    top_sites_parser.add_argument(
        '--resource_max_total_capacity',
        type=float,
        default=None,
        help="<Optional> Maximum total capacity for the resource (e.g., 10 for 10 GW). If none given, value will be absorbed from the user config"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.command == 'create_resources':
        creator = ResourceCreator(args.config_path, args.resource_type)
        creator.create()
    elif args.command == 'prepare_data':
        preparer = DataPreparer(args.config_path)
        preparer.prepare()
    elif args.command == 'top_sites':
        selector = TopSitesSelector(args.config_path, args.resource_type, args.resource_max_total_capacity)
        selector.select()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
