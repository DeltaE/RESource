import os
import matplotlib.pyplot as plt
import seaborn as sns
import linkingtool.utility as utility
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
file_handler = log.FileHandler('log/visualizations.txt')
# Add the FileHandler to the root logger
log.getLogger().addHandler(file_handler)

# Load data from YAML file
# configs=utility.load_config('config/config_linking_tool.yml')
# solar_vis_directory=configs['solar']['solar_vis_directory']

def plot_data_in_GADM_regions(
        dataframe,
        data_column_df,
        gadm_regions_gdf,
        color_map,
        dpi,
        plt_title,
        plt_file_name,
        vis_directory):
    
    ax = dataframe.plot(column=data_column_df, edgecolor='white',linewidth=0.2,legend=True,cmap=color_map)
    gadm_regions_gdf.plot(ax=ax, alpha=0.6, color='none', edgecolor='k', linewidth=0.7)
    ax.set_title(plt_title)
    plt_save_to=os.path.join(vis_directory,plt_file_name)
    plt.tight_layout()
    plt.savefig(plt_save_to,dpi=dpi)
    plt.close
    return log.info(f"Plot Created for {plt_title} for Potential PV Plants and Save to {plt_save_to}")