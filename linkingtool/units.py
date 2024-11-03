import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from linkingtool.AttributesParser import AttributesParser
from linkingtool.hdf5_handler import DataHandler

@dataclass
class Units(AttributesParser):
    save_to_dir: Path = Path('data')
    excel_filename: str = 'units_dictionary.xlsx'
    
    def __post_init__(self):
        """
        Initializes bounding box, resolution, and other parameters 
        after the parent class initialization.
        """
        super().__post_init__()
        self.datahandler = DataHandler(self.store)

    def create_units_dictionary(self):
        """
        Creates a dictionary of units for various parameters, saves it 
        as a DataFrame in HDF5 format, and exports to Excel.
        """
        units_dict = {
            'capex': 'Mil. USD/MW',
            'fom': 'Mil. USD/MW',
            'vom': 'Mil. USD/MW',
            'potential_capacity': 'MW',
            'p_lcoe': 'MWH/USD',
            'energy': 'MWh',
            'energy demand': 'Pj'
        }
        
        # Convert the dictionary to a DataFrame
        units_df = pd.DataFrame.from_dict(units_dict, orient='index', columns=['Unit'])
        
        # Save DataFrame to HDF5 using DataHandler
        self.datahandler.to_store(units_df, 'units')
        
        # Ensure the save directory exists
        self.save_to_dir.mkdir(parents=True, exist_ok=True)
        excel_path = self.save_to_dir / self.excel_filename
        
        # Save DataFrame to Excel
        try:
            units_df.to_excel(excel_path, index=False)
            self.log.info(f">> Units dictionary created and saved to '{excel_path}'")
        except Exception as e:
            self.log.error(f">> Failed to save Excel file: {e}")
