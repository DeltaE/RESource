import time
import argparse
import logging as log
import os
import pandas as pd
import geopandas as gpd


# Local Packages
try:
    # Try importing from the submodule context
    import linkingtool.linking_utility as utils
    import linkingtool.linking_vis as vis
    import linkingtool.linking_solar as solar
    from linkingtool.attributes_parser import AttributesParser
    from linkingtool.cell_capacity_processor import cell_capacity_processor
except ImportError:
    # Fallback for when running as a standalone script or outside the submodule
    import Linking_tool.linkingtool.linking_utility as utils
    import Linking_tool.linkingtool.linking_vis as vis
    import Linking_tool.linkingtool.linking_solar as solar
    from Linking_tool.linkingtool.attributes_parser import AttributesParser
    from Linking_tool.linkingtool.cell_capacity_processor import cell_capacity_processor

class TopSiteSelection:
    def __init__(self, config_file_path: str, resource_type: str, resource_max_total_capacity: float = None):
        self.config_file_path = config_file_path
        self.resource_type = resource_type
        self.resource_max_total_capacity = resource_max_total_capacity
        self.config = utils.load_config(config_file_path)
        self.current_region = self.config['regional_info']['region_1']
        self._CRC_ = self.current_region['code']
        self.disaggregation_config = self.config['capacity_disaggregation'][f'{resource_type}']
        self.linking_data = self.config['processed_data']['linking']
        self.result_files = self.config['results']['linking']
        self.vis_dir = os.path.join(self.config['visualization']['linking'], resource_type)
        self.resource_max_capacity = self.resource_max_total_capacity or self.disaggregation_config['max_capacity']  # GW

    def setup_logging(self):
        log_path = f'workflow/log/{self.resource_type}_top_sites_module_log.txt'
        utils.create_log(log_path)
        utils.print_module_title(f'{self.resource_type} Top Sites Selection Module Initiated')

    def load_data(self):
        log.info(f"Loading linking data files...")
        self.cell_clusters = gpd.GeoDataFrame(
            pd.read_pickle(os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['cell_clusters']))
        )
        if self.resource_type != 'bess':
            self.dissolved_indices = pd.read_pickle(
                os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['dissolved_indices'])
            )
            self.province_grid_CF_ts_df = gpd.GeoDataFrame(
                pd.read_pickle(os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['ERA5_CF_ts']))
            )
            self.scored_cells = gpd.GeoDataFrame(
                pd.read_pickle(os.path.join(self.linking_data['root'], self.resource_type, self.linking_data[f'{self.resource_type}']['scored_cells']))
            )

    def select_top_sites(self):
        selected_sites = utils.select_top_sites(self.cell_clusters, self.resource_max_capacity)
        if not selected_sites.empty:
            print(f"> {len(selected_sites)} Top Sites Selected.\n>> Total Capacity: {round(selected_sites['potential_capacity'].sum() / 1000, 2)} GW")
        else:
            print("No sites selected.")
        return selected_sites

    def create_timeseries(self, selected_sites):
        if self.resource_type == "bess":
            log.info(f"Time-slice creation and visuals not required for BESS.")
        else:
            if self.resource_type == "wind":
                CF_ts_clusters_df, within_cluster_cells_ts_df = wind.create_timeseries_for_Cluster(
                    selected_sites, self.dissolved_indices, self.scored_cells, self.province_grid_CF_ts_df
                )
            elif self.resource_type == "solar":
                CF_ts_clusters_df = solar.create_timeseries_for_Cluster(
                    selected_sites, self.dissolved_indices, self.province_grid_CF_ts_df
                )
            return CF_ts_clusters_df

    def prepare_timeseries(self, CF_ts_clusters_df, selected_sites):
        CF_ts_df_Top_sites = CF_ts_clusters_df[selected_sites.index]
        CF_ts_df_Top_sites = utils.fix_df_ts_index(CF_ts_df_Top_sites, self.current_region['snapshots_tz_BC'], snapshot_serial=0)
        CF_ts_df_Top_sites.to_pickle(
            os.path.join(self.result_files['root'], self.result_files['clusters_CFts_topSites'][f'{self.resource_type}'])
        )
        return CF_ts_df_Top_sites

    def visualize_timeseries(self, CF_ts_df_Top_sites):
        plots_save_to = os.path.join(self.vis_dir, 'Site_timeseries')
        vis.create_timeseries_interactive_plots(CF_ts_df_Top_sites, plots_save_to)
        vis.create_sites_ts_plots_all_sites(self.resource_type, CF_ts_df_Top_sites, plots_save_to)

    def save_results(self, selected_sites, CF_ts_df_Top_sites):
        selected_sites['CF_mean'] = selected_sites.index.map(CF_ts_df_Top_sites.mean())
        selected_sites.to_pickle(
            os.path.join(self.result_files['root'], self.result_files['clusters_topSites'][f'{self.resource_type}'])
        )

    def run(self):
        self.setup_logging()
        self.load_data()
        selected_sites = self.select_top_sites()
        if not selected_sites.empty:
            CF_ts_clusters_df = self.create_timeseries(selected_sites)
            CF_ts_df_Top_sites = self.prepare_timeseries(CF_ts_clusters_df, selected_sites)
            self.visualize_timeseries(CF_ts_df_Top_sites)
            self.save_results(selected_sites, CF_ts_df_Top_sites)
        log.info(f"Top Sites Selection for {self.resource_type} - Execution Completed!")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run data preparation script')
    parser.add_argument('config', type=str, help="Path to the configuration file 'config_master.yml'.")
    parser.add_argument('resource_type', choices=['wind', 'solar', 'bess'], help="Type of resource: 'wind', 'solar', or 'bess'")
    parser.add_argument('resource_max_total_capacity', nargs='?', type=float, default=None, help="Maximum total capacity (GW) for the resource (optional)")

    args = parser.parse_args()

    # Run the main function with arguments
    top_site_selection = TopSiteSelection(args.config, args.resource_type, args.resource_max_total_capacity)
    top_site_selection.run()
