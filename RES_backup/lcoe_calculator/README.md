# ATB Calculator

This repository contains tools to extract data from the __NREL's ATB Excel workbook__, calculate Levelized Cost of Energy (LCOE) and Capital Expenditures (CAPEX) for various technologies, and export the results in flat or pivoted formats.

> **Note:** Python may require permission to interact with Excel. A prompt will appear the first time this script is executed.

## Source

This methods are originally sourced/inspired from: [NREL ATB-calc](https://github.com/NREL/ATB-calc/tree/main/lcoe_calculator).

## Files Overview

The following files are listed in approximate order of importance and ease of use:

- **`process_all.py`**  
    A class that processes all ATB technologies with a built-in command-line interface (CLI). Refer to the root README for CLI usage examples.

- **`tech_processors.py`**  
    Contains classes for processing individual technologies. Add new ATB technologies to this file.

- **`base_processor.py`**  
    A base processor class that serves as a parent for individual technology processors.

- **`config.py`**  
    Defines constants such as the base year and scenario names.

- **`extractor.py`**  
    Extracts values from the ATB workbook.

- **`abstract_extractor.py`**  
    An abstract version of the extractor, enabling the use of mock values in tests.

- **`macrs.py`**  
    Implements MACRS depreciation schedules.