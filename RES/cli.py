#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from . import RESources as RES

@staticmethod
def find_project_root(marker="RES"):
    """
    Traverse upwards to find the root directory containing the specified marker.
    """
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:  # Stop at the filesystem root
        if (current_dir / marker).exists():  # Check for the marker
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")


def run_RESources(config_file_path, region_code, resource_type):
    """
    This function runs the RESources builder for the specified configuration, region, and resource type.
    """
    required_args = {
        "config_file_path": config_file_path,
        "region_short_code": region_code,
        "resource_type": resource_type
    }
    
    # Create an instance of Resources and execute the module
    RES_module = RES.RESources_builder(**required_args)
    RES_module.build(use_pypsa_buses=False)

def display_help():
    """
    Display the custom help message for RES CLI.
    """
    print("""
    RES - Resource Builder CLI Tool. 
        - Currently supports : 'wind' (land-based-wind), 'solar'

    Usage:
    RES [options]

    Options:
    --config           Path to the configuration YAML file (default: config/config.yaml)
    --regions       List of regions to process (default: BC, Supports all regions)
    --resources       List of resource types to generate (default: wind solar)

    Examples:
    RES --config=config.yaml --regions BC AB --resources wind solar
    RES --help-resources

    Use '--help-resources' for more detailed information about the RESources module.
    """)

def display_resources_help():
    """
    Display help information for the RESources module.
    """
    print("""
    RESources Module Help:

    This module contains the RESources_builder class used to generate resources for regions and resource types.
    
    Usage:
    RESources_builder(config_file_path, region_short_code, resource_type)

    Arguments:
    - config_file_path: Path to the configuration file (e.g., 'config/config.yaml')
    - region_short_code: region short code (e.g., 'BC')
    - resource_type: Type of resource to generate (e.g., 'wind', 'solar')

    Example:
    RESources_builder(config_file_path='config.yaml', region_short_code='BC', resource_type='wind')
    """)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(prog='RES', description='Run resource generation for regions and resource types')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--regions', type=str, nargs='+', default=['BC'], help='List of regions')
    parser.add_argument('--resources', type=str, nargs='+', default=['wind', 'solar'], help='List of resource types')
    parser.add_argument('--help-resources', action='store_true', help='Display help for RESources module')

    # Parse arguments
    args = parser.parse_args()

    # If --help-resources flag is passed, display RESources help
    if args.help_resources:
        display_resources_help()
        sys.exit(0)

    # If no arguments are passed or just '--help', show custom help message
    if len(sys.argv) == 1 or '--help' in sys.argv:
        display_help()
        sys.exit(0)

    # Detect the root directory
    try:
        project_root = find_project_root("RES")
        print(f"Project root detected at: {project_root}")
        # Set the working directory to the root
        os.chdir(project_root)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Iterate over regions and resource types
    for region_code in args.regions:
        for resource_type in args.resources:
            print(f"Running for {region_code} and {resource_type}")
            run_RESources(args.config, region_code, resource_type)

if __name__ == "__main__":
    main()
