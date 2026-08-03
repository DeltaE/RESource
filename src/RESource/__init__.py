"""
RESource - A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.

RESource is a comprehensive Python package that provides tools for assessing renewable energy potential
at sub-national scales. It integrates geospatial, temporal, economic, and regulatory data to evaluate
site suitability for solar and wind energy development.

Key Features:
    - Modular workflow for reproducible energy assessments
    - Integration with climate data (ERA5, GWA)
    - Geospatial analysis using GADM boundaries
    - Economic evaluation including LCOE calculations
    - Clustering and scoring mechanisms for site prioritization
    - Interactive visualizations and reporting

Modules:
    - RESources: Main builder class orchestrating the assessment workflow
    - boundaries: GADM boundary processing and regional data handling
    - cell: Grid cell generation and spatial discretization
    - timeseries: Climate data processing and capacity factor calculations
    - score: Economic scoring and LCOE calculations
    - cluster: Site clustering and aggregation methods
    - visuals: Plotting and visualization utilities

Example:
    >>> from RESource.RESources import RESources_builder
    >>> builder = RESources_builder(
    ...     config_file_path="config/config.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind"
    ... )
    >>> results = builder.run_full_workflow()

Author: Md Eliasinul Islam
Version: 2025.07
License: MIT
"""

__version__ = "2025.7.0"
__author__ = "Md Eliasinul Islam"


# import os
# import importlib

# # Dynamically import all modules in the current directory except __init__.py
# current_dir = os.path.dirname(__file__)
# for filename in os.listdir(current_dir):
#     if filename.endswith(".py") and filename != "__init__.py":
#         modulename = filename[:-3]
#         try:
#             globals()[modulename] = importlib.import_module(f".{modulename}", __package__)
#         except ImportError:
#             globals()[modulename] = None
