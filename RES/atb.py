import pandas as pd
from pathlib import Path
from RES import utility as utils
from RES.AttributesParser import AttributesParser
from RES.hdf5_handler import DataHandler
print_level_base=2

class NREL_ATBProcessor(AttributesParser):
    """ 
    NREL_ATBProcessor is a class from RESource module, designed to process the Annual Technology Baseline (ATB) data 
    sourced from the National Renewable Energy Laboratory (NREL). This class provides methods 
    to pull, process, and store cost data for various renewable energy technologies, including 
    utility-scale photovoltaic (PV) systems, land-based wind turbines, and battery energy 
    storage systems (BESS).
    
    Attributes:
        atb_config (dict): Configuration dictionary containing paths and settings for ATB data.
        atb_data_save_to (Path): Path to the directory where ATB data will be saved.
        atb_parquet_source (str): Source URL or path for the ATB Parquet file.
        atb_datafile (str): Name of the ATB data file.
        atb_file_path (Path): Full path to the ATB data file.
        datahandler (DataHandler): Instance of DataHandler for storing processed data.
        
    Methods:
        __post_init__():
            Initializes the processor by loading configurations, setting up paths, 
            and creating necessary directories.
        pull_data():
            Pulls and processes the ATB data, extracting cost data for utility-scale PV, 
            land-based wind, and BESS. Returns the processed data as a tuple.
        _check_and_download_data():
            Checks for the existence of the ATB data file locally and downloads it if 
            necessary.
        _process_solar_cost(atb_cost):
            Filters and processes solar cost data from the ATB dataset based on specific 
            criteria. Saves the processed data to a CSV file and stores it in the data handler.
        _process_wind_cost(atb_cost):
            Filters and processes land-based wind cost data from the ATB dataset based on 
            specific criteria. Saves the processed data to a CSV file and stores it in the 
            data handler.
        _process_bess_cost(atb_cost):
            Filters and processes battery energy storage system (BESS) cost data from the 
            ATB dataset based on specific criteria. Saves the processed data to a CSV file 
            and stores it in the data handler.
    """
    def __post_init__(self):
        """
        Post-initialization method for setting up ATB-related configurations and paths.
        
        This method performs the following tasks:
        - Calls the parent class's `__post_init__` method to ensure proper initialization.
        - Retrieves the ATB configuration using `get_atb_config`.
        - Sets up paths for saving ATB data and accessing ATB parquet files.
        - Ensures that the directory for the ATB data file exists, creating it if necessary.
        - Initializes a `DataHandler` instance with the provided store.
        
        Attributes initialized:
        - atb_config: Configuration dictionary for ATB settings.
        - atb_data_save_to: Path object representing the root directory for saving ATB data.
        - atb_parquet_source: Path to the source parquet file as specified in the configuration.
        - atb_datafile: Name of the ATB data parquet file as specified in the configuration.
        - atb_file_path: Full path to the ATB data file.
        - datahandler: Instance of `DataHandler` initialized with the provided store.
        """
        utils.print_update(level=print_level_base,message='NREL_ATBProcessor initiated...')
        super().__post_init__()
        
        self.atb_config=self.get_atb_config()
        self.atb_data_save_to = Path(self.atb_config['root'])
        self.atb_parquet_source = self.atb_config['source']['parquet']
        self.atb_datafile = self.atb_config['datafile']['parquet']
        self.atb_file_path = Path (self.atb_data_save_to) / self.atb_datafile
        
        # Create the parent directories if they do not exist
        self.atb_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.res_data=DataHandler(self.store)
        
    def pull_data(self):
        """
        Pulls and processes the Annual Technology Baseline (ATB) data sourced from NREL.
        This method performs the following steps:
        1. Logs the start of the ATB data processing.
        2. Checks and downloads the required data file if not already available.
        3. Reads the ATB cost data from a Parquet file.
        4. Processes the ATB cost data to extract:
            - Utility-scale photovoltaic (PV) cost.
            - Land-based wind cost.
            - Battery energy storage system (BESS) cost.
        5. Logs the successful loading of the ATB cost data file.
        Returns:
            tuple: A tuple containing the processed cost data for:
                - Utility-scale PV (self.utility_pv_cost)
                - Land-based wind (self.land_based_wind_cost)
                - BESS (self.bess_cost)
        """
        
        utils.print_update(level=print_level_base+1,message=f"{__name__}| Processing Annual Technology Baseline (ATB) data sourced from NREL...")
        self.check_and_download_data()
        
        atb_cost = pd.read_parquet(self.atb_file_path)
        
        utils.print_update(level=print_level_base+1,message=f"{__name__}| ATB cost datafile: {self.atb_file_path.name} loaded")

        
        utils.print_update(level=print_level_base+1,message="Extracting technology baseline costs...")
        f"ATB cost datafile: {self.atb_file_path.name} loaded"
        self.utility_pv_cost=self.process_solar_cost(atb_cost)
        self.land_based_wind_cost=self.process_wind_cost(atb_cost)
        self.bess_cost=self.process_bess_cost(atb_cost)
        
        return (self.utility_pv_cost, 
                self.land_based_wind_cost, 
                self.bess_cost)

    def check_and_download_data(self):
        """
        Checks the existence of a local copy of the ATB data file and downloads it if necessary.

        This method ensures that the required ATB data file is available locally. If the file
        does not exist, it triggers a download from the specified source. The download is
        skipped if the file already exists and `force_update` is set to False.

        Uses:
            - `utils.check_LocalCopy_and_run_function` to verify the local copy and execute
              the download function if needed.
            - `utils.download_data` to handle the actual data download process.

        Parameters:
            None

        Returns:
            None
        """
        utils.check_LocalCopy_and_run_function(
            self.atb_file_path,
            lambda: utils.download_data(self.atb_parquet_source,self.atb_file_path),
            force_update=False
        )

    def process_solar_cost(self, atb_cost):
        """
        Processes solar cost data from the ATB (Annual Technology Baseline) dataset.

        This method filters the ATB cost data for utility-scale photovoltaic (PV) systems
        based on specific criteria such as technology alias, core metric parameters, scenario,
        core metric case, technology details, and other attributes. The filtered data is then
        saved to a specified file and stored using the data handler.

        Args:
            atb_cost (pd.DataFrame): The ATB cost dataset containing cost information
                                     for various technologies and scenarios.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered and processed solar cost data.

        Raises:
            KeyError: If required keys are missing in the configuration dictionary.
            FileNotFoundError: If the specified save path is invalid or inaccessible.

        Notes:
            - The method assumes the configuration dictionary (`self.config`) contains
              the necessary keys and paths for filtering and saving the data.
            - The `self.datahandler.to_store` method is used to store the processed data
              for further use.
        """
        utils.print_update(level=print_level_base+2,message="Extracting Solar PV technology cost...")
        pv_cost_mask = (
            (atb_cost['technology_alias'] == 'Utility PV') &
            (atb_cost['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M', 'Variable O&M','OCC'])) &
            (atb_cost['scenario'] == 'Moderate') &
            (atb_cost['core_metric_case'] == 'Market') &
            (atb_cost['techdetail'] == self.config['capacity_disaggregation']['solar']['NREL_ATB_type']) &
            (atb_cost['crpyears'] == '20') & # capital recovery period
            (atb_cost['core_metric_variable'] == 2022)
        )

        utility_pv_cost = atb_cost[pv_cost_mask].sort_values('core_metric_variable')

        save_to=Path(self.config['capacity_disaggregation']['solar']['cost_data'])
        save_to.parent.mkdir(parents=True, exist_ok=True)
        utility_pv_cost.to_csv(save_to, index=False)
        self.res_data.to_store(utility_pv_cost,'cost/atb/solar',force_update = True)
        return utility_pv_cost

    def process_wind_cost(self, atb_cost):
        """
        Processes wind cost data from the ATB (Annual Technology Baseline) dataset 
        and saves the filtered results to a CSV file and a data store.
        Args:
            atb_cost (pd.DataFrame): A DataFrame containing ATB cost data with 
                various attributes such as technology alias, core metric parameters, 
                scenarios, and more.
        Returns:
            pd.DataFrame: A filtered and sorted DataFrame containing land-based wind 
            cost data based on the specified criteria.
        Filtering Criteria:
            - Technology alias is 'Land-Based Wind'.
            - Core metric parameter is one of ['CAPEX', 'Fixed O&M', 'Variable O&M', 'OCC'].
            - Scenario is 'Moderate'.
            - Core metric case is 'Market'.
            - Technology detail matches the NREL_ATB_type specified in the configuration.
            - CRP years is '20'.
            - Core metric variable is 2024.
        Side Effects:
            - Saves the filtered DataFrame to a CSV file at the path specified in 
              the configuration under 'capacity_disaggregation' -> 'wind' -> 'cost_data'.
            - Stores the filtered DataFrame in the data handler under the key 
              'cost/atb/wind' with force update enabled.
        """
        utils.print_update(level=print_level_base+2,message="Extracting Wind Turbine technology cost...")
        land_based_wind_cost_mask = (
            (atb_cost['technology_alias'] == 'Land-Based Wind') &
            (atb_cost['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M', 'Variable O&M','OCC'])) &
            (atb_cost['scenario'] == 'Moderate') &
            (atb_cost['core_metric_case'] == 'Market') &
            (atb_cost['techdetail2'] == self.config['capacity_disaggregation']['wind']['turbines']['NREL_ATB_type']) &
            (atb_cost['crpyears'] == '20') &
            (atb_cost['core_metric_variable'] == 2024)
        )

        land_based_wind_cost = atb_cost[land_based_wind_cost_mask].sort_values('core_metric_variable')
        
        save_to=Path(self.config['capacity_disaggregation']['wind']['cost_data'])
        save_to.parent.mkdir(parents=True, exist_ok=True)
        land_based_wind_cost.to_csv(save_to, index=False)
        self.res_data.to_store(land_based_wind_cost,'cost/atb/wind',force_update = True)
        return land_based_wind_cost

    def process_bess_cost(self, atb_cost):
        """
        Processes the cost data for Utility-Scale Battery Storage (BESS) from the ATB dataset.

        This method filters the ATB cost data based on specific criteria such as technology alias,
        core metric parameters, scenario, core metric case, technology details, and other attributes.
        The filtered data is then sorted, saved to a CSV file, and stored in the data handler.

        Args:
            atb_cost (pd.DataFrame): The ATB cost dataset containing cost information for various technologies.

        Returns:
            pd.DataFrame: The filtered and processed cost data for Utility-Scale Battery Storage (BESS).

        Raises:
            KeyError: If required configuration keys are missing in `self.config`.
            FileNotFoundError: If the specified save path is invalid or inaccessible.

        Notes:
            - The method assumes that the configuration dictionary (`self.config`) contains the necessary
              keys and values for filtering and saving the data.
            - The filtered data is stored in the data handler under the key 'cost/atb/bess'.
        """
        utils.print_update(level=print_level_base+2,message="Extracting BESS technology cost...")
        bess_cost_mask = (
            (atb_cost['technology_alias'] == 'Utility-Scale Battery Storage') &
            (atb_cost['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M', 'Variable O&M','OCC'])) &
            (atb_cost['scenario'] == 'Moderate') &
            (atb_cost['core_metric_case'] == 'Market') &
            (atb_cost['techdetail'] == self.config['capacity_disaggregation']['bess']['NREL_ATB_type']) &
            (atb_cost['crpyears'] == '20') &
            (atb_cost['core_metric_variable'] == 2024)
        )

        bess_cost = atb_cost[bess_cost_mask].sort_values('core_metric_variable')
        save_to=Path(self.config['capacity_disaggregation']['bess']['cost_data'])
        save_to.parent.mkdir(parents=True, exist_ok=True)
        bess_cost.to_csv(save_to, index=False)
        self.res_data.to_store(bess_cost,'cost/atb/bess',force_update = True)
        
        return bess_cost

