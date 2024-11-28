import pandas as pd
import logging
from pathlib import Path
from linkingtool import utility as utils
from linkingtool.AttributesParser import AttributesParser
from linkingtool.hdf5_handler import DataHandler

class NREL_ATBProcessor(AttributesParser):
    
    def __post_init__(self):

        super().__post_init__()
        
        self.atb_config=self.get_atb_config()
        self.atb_data_save_to = Path(self.atb_config['root'])
        self.atb_parquet_source = self.atb_config['source']['parquet']
        self.atb_datafile = self.atb_config['datafile']['parquet']
        self.atb_file_path = Path (self.atb_data_save_to) / self.atb_datafile
        # Create the parent directories if they do not exist
        self.atb_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.datahandler=DataHandler(self.store)
        
    def pull_data(self):
        self.log.info("Processing Annual Technology Baseline (ATB) data sourced from NREL...")
        self._check_and_download_data()
        atb_cost = pd.read_parquet(self.atb_file_path)
        self.log.info(f"ATB cost datafile: {self.atb_file_path.name} loaded")
        
        self._process_solar_cost(atb_cost)
        self._process_wind_cost(atb_cost)
        self._process_bess_cost(atb_cost)
        
        return atb_cost

    def _check_and_download_data(self):
        utils.check_LocalCopy_and_run_function(
            self.atb_file_path,
            lambda: utils.download_data(self.atb_parquet_source,self.atb_file_path),
            force_update=False
        )

    def _process_solar_cost(self, atb_cost):
        pv_cost_mask = (
            (atb_cost['technology_alias'] == 'Utility PV') &
            (atb_cost['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M', 'Variable O&M'])) &
            (atb_cost['scenario'] == 'Moderate') &
            (atb_cost['core_metric_case'] == 'Market') &
            (atb_cost['techdetail'] == self.config['capacity_disaggregation']['solar']['NREL_ATB_type']) &
            (atb_cost['crpyears'] == '20') &
            (atb_cost['core_metric_variable'] == 2022)
        )

        utility_pv_cost = atb_cost[pv_cost_mask].sort_values('core_metric_variable')

        save_to=Path(self.config['capacity_disaggregation']['solar']['cost_data'])
        save_to.parent.mkdir(parents=True, exist_ok=True)
        utility_pv_cost.to_csv(save_to, index=False)
        self.datahandler.to_store(utility_pv_cost,'cost/atb/wind')

    def _process_wind_cost(self, atb_cost):
        land_based_wind_cost_mask = (
            (atb_cost['technology_alias'] == 'Land-Based Wind') &
            (atb_cost['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M', 'Variable O&M'])) &
            (atb_cost['scenario'] == 'Moderate') &
            (atb_cost['core_metric_case'] == 'Market') &
            (atb_cost['techdetail2'] == self.config['capacity_disaggregation']['wind']['turbines']['NREL_ATB_type']) &
            (atb_cost['crpyears'] == '20') &
            (atb_cost['core_metric_variable'] == 2022)
        )

        land_based_wind_cost = atb_cost[land_based_wind_cost_mask].sort_values('core_metric_variable')
        
        save_to=Path(self.config['capacity_disaggregation']['wind']['cost_data'])
        save_to.parent.mkdir(parents=True, exist_ok=True)
        land_based_wind_cost.to_csv(save_to, index=False)
        self.datahandler.to_store(land_based_wind_cost,'cost/atb/wind')

    def _process_bess_cost(self, atb_cost):
        bess_cost_mask = (
            (atb_cost['technology_alias'] == 'Utility-Scale Battery Storage') &
            (atb_cost['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M', 'Variable O&M'])) &
            (atb_cost['scenario'] == 'Moderate') &
            (atb_cost['core_metric_case'] == 'Market') &
            (atb_cost['techdetail'] == self.config['capacity_disaggregation']['bess']['NREL_ATB_type']) &
            (atb_cost['crpyears'] == '20') &
            (atb_cost['core_metric_variable'] == 2022)
        )

        bess_cost = atb_cost[bess_cost_mask].sort_values('core_metric_variable')
        save_to=Path(self.config['capacity_disaggregation']['bess']['cost_data'])
        save_to.parent.mkdir(parents=True, exist_ok=True)
        bess_cost.to_csv(save_to, index=False)
        self.datahandler.to_store(bess_cost,'cost/atb/bess')

