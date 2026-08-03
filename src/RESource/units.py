from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from RESource import utility as utils
from RESource.AttributesParser import AttributesParser
from RESource.hdf5_handler import DataHandler


@dataclass
class Units(AttributesParser):
    """
    Unit conversion and metadata management system for renewable energy analysis.

    This class provides standardized unit definitions and conversion capabilities
    for various parameters used throughout the renewable energy resource assessment
    workflow. It maintains a comprehensive dictionary of units for economic,
    technical, and energy parameters, enabling consistent data interpretation
    and reporting across different analysis modules.

    Key Functionality:
    - Defines standard units for economic parameters (CAPEX, OPEX, LCOE)
    - Establishes energy and power unit conventions
    - Provides data persistence through HDF5 storage
    - Exports unit dictionaries for external reference
    - Ensures consistent unit usage across the analysis pipeline



    Attributes:
        SAVE_TO_DIR (Path): Directory for saving unit reference files
        EXCEL_FILENAME (str): Filename for CSV export of unit dictionary
        datahandler (DataHandler): HDF5 storage interface for data persistence

    Standard Units Defined:
        - capex: Million USD per MW (Mil. USD/MW)
        - fom: Million USD per MW (Mil. USD/MW) - Fixed O&M
        - vom: Million USD per MW (Mil. USD/MW) - Variable O&M
        - potential_capacity: Megawatts (MW)
        - p_lcoe: Megawatt-hours per USD (MWH/USD)
        - energy: Megawatt-hours (MWh)
        - energy_demand: Petajoules (Pj)

    Example:
        >>> units_manager = Units(
        ...     config_file_path="config/config.yaml",
        ...     SAVE_TO_DIR=Path("data/units"),
        ...     EXCEL_FILENAME="units_reference.csv"
        ... )
        >>> units_manager.create_units_dictionary()
        >>> # Creates standardized unit reference for project

    Notes:
        - Units follow international energy industry standards
        - Economic parameters use million USD to match typical project scales
        - Energy units align with grid-scale renewable energy reporting
        - HDF5 storage enables efficient data access and version control
    """

    SAVE_TO_DIR: Path = Path("data")
    EXCEL_FILENAME: str = "units.csv"

    def __post_init__(self):
        """
        Initialize inherited attributes and unit management configuration.

        This method:
        1. Initializes HDF5 data handler for persistent unit storage
        2. Sets up data management infrastructure for unit dictionaries

        The inherited attributes from AttributesParser include:
        - Configuration file parsing and validation
        - Data store configuration for HDF5 persistence
        - Regional and project-specific settings

        Raises:
            ConfigurationError: If store configuration is invalid
            FileSystemError: If HDF5 store cannot be initialized
        """
        super().__post_init__()
        self.datahandler: DataHandler = DataHandler(self.store)

    def create_units_dictionary(self):
        """
        Create and persist a comprehensive dictionary of standardized units for renewable energy analysis.

        This method establishes the authoritative unit reference for all parameters
        used in renewable energy resource assessment and economic analysis. It creates
        a standardized dictionary mapping parameter names to their corresponding units,
        ensuring consistency across all analysis modules and facilitating data
        interpretation and reporting.

        Unit Categories:
        1. Economic Parameters:
           - Capital expenditures (CAPEX) in Million USD/MW
           - Fixed and variable O&M costs in Million USD/MW
           - LCOE productivity metrics in MWH/USD

        2. Technical Parameters:
           - Power capacity in Megawatts (MW)
           - Energy production in Megawatt-hours (MWh)
           - Energy demand in Petajoules (Pj)

        Process:
        1. Defines comprehensive units dictionary with industry-standard units
        2. Converts dictionary to pandas DataFrame for structured storage
        3. Persists data to HDF5 store for efficient access
        4. Exports human-readable CSV file for external reference

        Data Storage:
        - HDF5 format: Efficient binary storage for programmatic access
        - CSV format: Human-readable reference for documentation and verification

        Raises:
            FileSystemError: If output directory cannot be created
            PermissionError: If CSV file cannot be written
            StorageError: If HDF5 store operation fails

        Example:
            >>> units_manager = Units(**config)
            >>> units_manager.create_units_dictionary()
            INFO: Units information created and saved to 'data/units.csv'

        Notes:
            - Units follow international energy industry conventions
            - Economic units scaled to typical renewable project magnitudes
            - CSV export includes parameter names and corresponding units
            - HDF5 storage enables version control and audit trails
        """
        units_dict = {
            "capex": "Mil. USD/MW",
            "fom": "Mil. USD/MW",
            "vom": "Mil. USD/MW",
            "potential_capacity": "MW",
            "p_lcoe": "MWH/USD",
            "energy": "MWh",
            "energy demand": "Pj",
        }

        # Convert the dictionary to a DataFrame
        units_df = pd.DataFrame.from_dict(units_dict, orient="index", columns=["Unit"])

        # Save DataFrame to HDF5 using DataHandler
        self.datahandler.to_store(units_df, "units")

        # Ensure the save directory exists
        self.SAVE_TO_DIR.mkdir(parents=True, exist_ok=True)
        excel_path = self.SAVE_TO_DIR / self.EXCEL_FILENAME

        # Save DataFrame to Excel
        try:
            units_df.reset_index().rename(columns={"index": "Parameter"}).to_csv(
                excel_path, index=False
            )
            utils.print_update(
                level=2, message=f">> Units information created and saved to '{excel_path}'"
            )
            return units_df
        except PermissionError as e:
            utils.print_update(
                level=3, message=f">> Permission denied when trying to save Excel file: {e}"
            )
        except Exception as e:
            utils.print_update(level=3, message=f">> Failed to save Excel file: {e}")
