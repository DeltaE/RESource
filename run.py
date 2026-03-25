
#!/usr/bin/env python3
# coding: utf-8
"""
Renewable Energy Resource Analysis and Processing Pipeline

This script provides a comprehensive workflow for analyzing renewable energy resources 
(wind and solar) across multiple regions. It processes spatial data, calculates 
potential capacity, generates time series data, and exports results for downstream 
energy system modeling.

Main Workflow:
1. Loads configuration file and extracts available regions
2. Validates region arguments against config file
3. Iterates through specified/all regions and resource types (wind/solar)
4. Builds resource datasets using the RESources_builder class
5. Processes grid cells, capacity calculations, and clustering
6. Generates time series data for resource availability
7. Selects optimal sites based on capacity constraints
8. Exports results in standardized formats

Supported Region Sets:
    - Canadian provinces: BC, QC, AB, SK, ON, NS, MB, etc.
    - Western Balkans: AL, BA, XK, ME, MK, RS
    - Any regions defined in configuration files

Resource Types:
    - Wind: Wind power potential analysis
    - Solar: Solar photovoltaic potential analysis

Outputs:
    - HDF5 data stores containing processed resource data
    - CSV files with selected optimal sites and time series
    - Clustered resource representations for energy system modeling

Dependencies:
    - RES.RESources: Core resource analysis module
    - RES.utility: Configuration loading utilities
    - RES.hdf5_handler: Data storage and retrieval
    - Configuration files in config/ directory
    - Spatial data sources (ERA5, Global Wind Atlas, etc.)

Usage:
    python run.py CONFIG_FILE [--regions REGION1 REGION2 ...]
    
Arguments:
    CONFIG_FILE      Path to configuration file (required)
    --regions, -r    Specific regions to process (default: all regions from config)

Examples:
    python run.py config/config_CAN_baseline.yaml          # Use Canadian config with all regions
    python run.py config/config_WB6.yaml                   # Use Western Balkan config with all regions
    python run.py config/config_WB6.yaml -r AL BA XK       # Use WB6 config with specific regions
    python run.py config/config_CAN_baseline.yaml -r BC QC AB  # Use Canadian config with specific regions

Notes:
    - Requires proper conda environment setup (see env/environment.yml)
    - Configuration parameters defined in specified config file
    - Regions are validated against the region_mapping section in config
    - Invalid regions will display available options and exit
    - Results stored in data/store/ and results/ directories
    - Processing time varies by region size and data availability

Author: RESource Development Team
Date: 2025
Version: 2.0 - Enhanced with flexible region selection
"""

import argparse
import sys
import RES.RESources as RES
from RES.utility import load_config
from datetime import datetime
import time
from pathlib import Path

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not installed
    COLORAMA_AVAILABLE = False
    print("Note: colorama not installed. Install with 'pip install colorama' for colored output.")
    class MockColor:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ''
        BRIGHT = DIM = ''
    Fore = Back = Style = MockColor()

def write_runtime_log(config_path, 
                      regions, 
                      status, 
                      start_dt, 
                      end_dt, 
                      runtime_seconds,
                      log_file=None):
    """Append runtime information for the script to a text file."""
    log_path = Path(log_file) if log_file else Path("results/logs/runtime_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    region_str = ", ".join(regions) if regions else "None"
    line = (
        f"--------------------------------------------------------------------------------\n"
        f"[{end_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"status={status} | "
        f"start={start_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"end={end_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"runtime_s={runtime_seconds:.2f} | "
        f"runtime_hms={int(runtime_seconds//3600):02d}:"
        f"{int((runtime_seconds%3600)//60):02d}:"
        f"{int(runtime_seconds%60):02d} | "
        f"config={config_path} | "
        f"regions=[{region_str}]\n"
        f"--------------------------------------------------------------------------------\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
        
def print_error(message):
    """Print error message in red."""
    print("{}{}{}".format(Fore.RED + Style.BRIGHT, message, Style.RESET_ALL))


def print_success(message):
    """Print success message in green."""
    print("{}{}{}".format(Fore.GREEN + Style.BRIGHT, message, Style.RESET_ALL))


def print_warning(message):
    """Print warning message in yellow."""
    print("{}{}{}".format(Fore.YELLOW + Style.BRIGHT, message, Style.RESET_ALL))


def print_info(message):
    """Print info message in cyan."""
    print("{}{}{}".format(Fore.CYAN + Style.BRIGHT, message, Style.RESET_ALL))


def print_suggestion(message):
    """Print suggestion message in magenta."""
    print("{}{}{}".format(Fore.MAGENTA + Style.BRIGHT, message, Style.RESET_ALL))


def main():
    """Main function to execute the renewable energy resource analysis pipeline.
    """
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
            description='Renewable Energy Resource Analysis and Processing Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=
    """
    Examples:
    python run.py config/config_CAN_baseline.yaml          # Use Canadian config with all regions
    python run.py config/config_WB6.yaml                   # Use Western Balkan config with all regions
    python run.py config/config_WB6.yaml -r AL BA XK       # Use WB6 config with specific regions
    python run.py config/config_CAN_baseline.yaml -r BC QC AB  # Use Canadian config with specific regions
    """
    )
    
    parser.add_argument(
        'config',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--regions', '-r',
        nargs='*',
        help='Specific regions to process. If not provided, all regions from config will be used.'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Load configuration file
    try:
        config = load_config(args.config)
        print_success("? Configuration file loaded: {}".format(args.config))
    except FileNotFoundError:
        print_error("? Configuration file '{}' not found.".format(args.config))
        print_suggestion("? Available config files:")
        print_info("   - config/config_CAN_baseline.yaml (Canadian provinces - baseline)")
        print_info("   - config/config_CAN_policy1.yaml (Canadian provinces - policy scenario)")
        print_info("   - config/config_WB6.yaml (Western Balkans)")
        print_suggestion("? Example: python3 run.py config/config_WB6.yaml")
        sys.exit(1)
    except Exception as e:
        print_error("? Error loading configuration file: {}".format(e))
        sys.exit(1)
    
    # Extract available regions from config
    if 'region_mapping' not in config:
        print_error("? 'region_mapping' not found in configuration file.")
        print_suggestion("? Please check your config file format - it should contain a 'region_mapping' section.")
        sys.exit(1)
    
    available_regions = list(config['region_mapping'].keys())
    print_info("- Available regions in config: {}".format(available_regions))
    
    # Determine which regions to process
    if args.regions is None:
        # Use all regions from config
        regions = available_regions
        print_success("* Processing all regions from config: {}".format(regions))
    else:
        # Validate provided regions
        invalid_regions = [r for r in args.regions if r not in available_regions]
        if invalid_regions:
            print_error("? Invalid region(s): {}".format(invalid_regions))
            print_warning("!  Available regions in config: {}".format(available_regions))
            print_suggestion("? Examples of valid commands:")
            
            # Generate helpful suggestions based on available regions
            if len(available_regions) >= 3:
                sample_regions = available_regions[:3]
                print_info("   - python3 run.py {} --regions {}".format(
                    args.config, ' '.join(sample_regions)))
            if len(available_regions) >= 1:
                print_info("   - python3 run.py {} --regions {}".format(
                    args.config, available_regions[0]))
            print_info("   - python3 run.py {} (process all regions)".format(args.config))
            sys.exit(1)
        regions = args.regions
        print_success("> Processing specified regions: {}".format(regions))
    
    # Display processing banner
    print("\n" + "="*70)
    print_info("> Starting RESource Analysis Pipeline")
    print_info("? Config: {}".format(args.config))
    print_info("* Regions: {}".format(regions))
    print_info("? Resources: wind, solar")
    print("="*70 + "\n")
    
    # Iterate over regions for both solar and wind resources
    resource_types = ['wind','solar']

    for region in regions:
        for resource_type in resource_types:
            print_info("? Processing {} {} resources...".format(region, resource_type))
            required_args = {
                "config_file_path": args.config,
                "region_short_code": region,
                "resource_type": resource_type
            }
            
            try:
                # Create an instance of Resources and execute the module
                Builder = RES.RESources_builder(**required_args)
                # Builder.clean_data_store()
                Builder.build(select_top_sites=True,
                                 use_pypsa_buses=False,
                                 use_grid_lines=True,
                                 get_clusters=False,
                                 clean_store=False)
                print_success("? Completed {} {} processing".format(region, resource_type))
            except Exception as e:
                print_error("? Failed processing {} {}: {}".format(region, resource_type, str(e)))
                print_warning("!  Continuing with next resource/region...")
                continue
    
    print("\n" + "="*70)
    print_success("? RESource Analysis Pipeline Completed!")
    print_info("? Results stored in data/store/ and results/ directories")
    print("="*70)

    # =====================
    script_end_dt = datetime.now()
    runtime_seconds = time.perf_counter() - script_start_perf

    write_runtime_log(
        config_path=args.config,
        regions=regions,
        status="SUCCESS",
        start_dt=script_start_dt,
        end_dt=script_end_dt,
        runtime_seconds=runtime_seconds,
        log_file="runtime_log.txt"
    )


if __name__ == '__main__':
    
    script_start_dt = datetime.now()
    script_start_perf = time.perf_counter()
    
    try:
        main()
    except KeyboardInterrupt:
        end_dt = datetime.now()
        runtime_seconds = time.perf_counter() - script_start_perf
        print_warning("\n!  Process interrupted by user (Ctrl+C)")
        print_info("? Partial results may be available in data/store/ directory")
        write_runtime_log(
            config_path="unknown",
            regions=[],
            status="INTERRUPTED",
            start_dt=script_start_dt,
            end_dt=end_dt,
            runtime_seconds=runtime_seconds,
            log_file="runtime_log.txt"
        )
        sys.exit(130)
    except Exception as e:
        end_dt = datetime.now()
        runtime_seconds = time.perf_counter() - script_start_perf
        print_error("? Unexpected error: {}".format(str(e)))
        write_runtime_log(
            config_path="unknown",
            regions=[],
            status=f"FAILED: {str(e)}",
            start_dt=script_start_dt,
            end_dt=end_dt,
            runtime_seconds=runtime_seconds,
            log_file="runtime_log.txt"
        )
        sys.exit(1)