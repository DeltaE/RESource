import os
import matplotlib.pyplot as plt
# import seaborn as sns
import logging as log
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib as mpl

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
log_name=f'workflow/log/linking_vis.txt'
# with open(log_name, 'w') as file:
#     pass
# file_handler = log.FileHandler(log_name)
# log.getLogger().addHandler(file_handler)


def plot_with_matched_cells(ax, cells: gpd.GeoDataFrame, filtered_cells: gpd.GeoDataFrame, column: str, cmap: str,
                            background_cell_linewidth: float, selected_cells_linewidth: float,font_size:int=9):
    """Helper function to plot cells with matched cells overlay."""
    # Plot the main cells layer
    vmin = cells[column].min()  # Minimum value for color mapping
    vmax = cells[column].max()  # Maximum value for color mapping

    # Create the main plot
    cells.plot(
        column=column,
        cmap=cmap,
        edgecolor='white',
        linewidth=background_cell_linewidth,
        ax=ax,
        alpha=1,
        vmin=vmin,  # Set vmin for color normalization
        vmax=vmax   # Set vmax for color normalization
    )

    # Overlay matched_cells with edge highlight
    filtered_cells.plot(
        ax=ax,
        edgecolor='black',
        color='None',
        linewidth=selected_cells_linewidth,
        alpha=1
    )

    # Create a colorbar for the plot
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Only needed for older Matplotlib versions
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label(column, fontsize=font_size)  # Label for the colorbar
    cbar.ax.tick_params(labelsize=font_size) 

def get_selected_vs_missed_visuals(cells: gpd.GeoDataFrame,
                                  province_short_code,
                                  resource_type,
                                   lcoe_threshold: float,
                                   CF_threshold: float,
                                   capacity_threshold: float,
                                   text_box_x=.4,
                                   text_box_y=.95,
                                   title_y=1,
                                   title_x=0.6,
                                   font_size=10,
                                   dpi=1000,
                                   figsize=(12, 7),
                                   save=False):
    
    mask=(cells[f'{resource_type}_CF_mean']>=CF_threshold)&(cells[f'potential_capacity_{resource_type}']>=capacity_threshold)&(cells[f'lcoe_{resource_type}']<=lcoe_threshold)
    filtered_cells=cells[mask]
    
    # Create a high-resolution side-by-side plot in a 2x2 grid
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)

    # Define the message
    msg = (f"Cell thresholds @ lcoe >= {lcoe_threshold} $/kWH, "
           f"CF >={CF_threshold}, MW >={capacity_threshold}")



    # First plot: CF_mean Visualization (top left)
    plot_with_matched_cells(axs[0, 0], cells, filtered_cells, f'{resource_type}_CF_mean', 'YlOrRd', 
                            background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
    axs[0, 0].set_title('CF_mean Overview', fontsize=font_size)
    axs[0, 0].set_xlabel('Longitude', fontsize=font_size-3)
    axs[0, 0].set_ylabel('Latitude', fontsize=font_size-3)
    axs[0, 0].set_axis_off()

    # Second plot: Potential Capacity Visualization (top right)
    plot_with_matched_cells(axs[0, 1], cells, filtered_cells, f'potential_capacity_{resource_type}', 'Blues',
                            background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
    axs[0, 1].set_title('Potential Capacity Overview', fontsize=font_size)
    axs[0, 1].set_xlabel('Longitude', fontsize=font_size-3)
    axs[0, 1].set_ylabel('Latitude', fontsize=font_size-3)
    axs[0, 1].set_axis_off()

    # Third plot: Nearest Station Distance Visualization (bottom left)
    plot_with_matched_cells(axs[1, 0], cells, filtered_cells, f'nearest_station_distance_km', 'coolwarm',
                            background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
    axs[1, 0].set_title('Nearest Station Distance Overview', fontsize=font_size)
    axs[1, 0].set_xlabel('Longitude', fontsize=font_size-3)
    axs[1, 0].set_ylabel('Latitude', fontsize=font_size-3)
    axs[1, 0].set_axis_off()

    # Fourth plot: LCOE Visualization (bottom right)
    plot_with_matched_cells(axs[1, 1], cells, filtered_cells, f'lcoe_{resource_type}', 'summer',
                            background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
    axs[1, 1].set_title('LCOE Overview', fontsize=font_size)
    axs[1, 1].set_xlabel('Longitude', fontsize=font_size-3)
    axs[1, 1].set_ylabel('Latitude', fontsize=font_size-3)
    axs[1, 1].set_axis_off()

    # Add a super title for the figure
    fig.suptitle(f'{resource_type}- Selected Cells Overview - {province_short_code}', fontsize=font_size+2,fontweight='bold', x=title_x,y=title_y)
    # Add a text box with grey background for the message
    fig.text(text_box_x, text_box_y, msg, ha='center', va='top', fontsize=font_size-3,
             bbox=dict(facecolor='lightgrey', edgecolor='grey', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    # Save the plot
    if save:
        plt.savefig(f"vis/linking/solar/Selected_cells_solar_{province_short_code}.png", bbox_inches='tight')
    plt.tight_layout()
    plt.show()  # Optional: Show the plot if desired


def create_raster_image_with_legend(
        raster:str, 
        cmap:str, 
        title:str, 
        plot_save_to:str):
    with rasterio.open(raster) as src:
        # Read the raster data
        raster_data = src.read(1)

        # Get the spatial information
        transform = src.transform
        min_x = transform[2]
        max_y = transform[5]
        max_x = min_x + transform[0] * src.width
        min_y = max_y + transform[4] * src.height

        # Get unique values (classes) in the raster
        unique_classes = np.unique(raster_data)

        # Create a colormap with a unique color for each class
        cmap = plt.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=unique_classes.min(), vmax=unique_classes.max())
        colormap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # Display the raster using imshow
        fig, ax = plt.subplots()
        im = ax.imshow(colormap.to_rgba(raster_data), extent=[min_x, max_x, min_y, max_y], interpolation='none')

        # Create legend patches
        legend_patches = [mpatches.Patch(color=colormap.to_rgba(cls), label=f'Class {cls}') for cls in unique_classes]

        # Add legend
        ax.legend(handles=legend_patches, title='Land Classes', loc='upper left', bbox_to_anchor=(1.05, 1))

        # Set labels for x and y axes
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Show the plot
        plt.title(title)
        plt.tight_layout()

        # Save the plot
        plt.savefig(plot_save_to, dpi=300)
        plt.close()  # Close the plot to avoid superimposing

        return log.info(f"Raster images created for {title}. Please check GAEZ model documentation for raster class descriptions")

# Load data from YAML file
# configs=utility.load_config('config/config_linking_tool.yml')
# solar_vis_directory=configs['solar']['solar_vis_directory']
#  %%
# def visualize_GADM_regions(
#         gadm_regions_gdf:gpd.GeoDataFrame,
#         color_palette:list,
#         plot_save_to:str):
#     """
#     Takes in the GADM  regions (with given admin level) and plots with given color coded regions, saving the plot to given path.
#     """
#     palette = sns.color_palette(color_palette, n_colors=len(gadm_regions_gdf))

#     # Create a dictionary to map Map_IDs to colors
#     color_map = {map_id: color for map_id, color in zip(gadm_regions_gdf['Region_ID'], palette)}

#     # Sort gadm_regions_gdf by Map_ID
#     gadm_regions_gdf = gadm_regions_gdf.sort_values(by='Region_ID')

#     # Plot the GeoDataFrame with Map_ID numbers
#     fig, ax = plt.subplots(figsize=(12, 12))
#     gadm_regions_gdf.plot(ax=ax, color=[color_map[region] for region in gadm_regions_gdf['Region_ID']])

#     # Annotate each region with its Map_ID
#     for idx, row in gadm_regions_gdf.iterrows():
#         plt.annotate(text=str(row['Region_ID']), xy=row['geometry'].centroid.coords[0], ha='center', color='black')

#     # Create legend with region names and Map_IDs
#     handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in palette]
#     labels = [f"{row['Region']} ({row['Region_ID']})" for idx, row in gadm_regions_gdf.iterrows()]
#     plt.legend(handles, labels, title='Regions', loc='upper left', bbox_to_anchor=(1, 1))

#     # Set plot title
#     plt.title('Map of Regions with Map_ID Numbers')
#     plt.tight_layout()

#     # Show the plot or save it to a file
#     plt.savefig(plot_save_to)
    
#     fig = plt.gcf()
#     plt.close(fig)
#     log.info(f"GADM Regions visualization reated and saved locally.\n")

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
    return log.info(f"Plot Created for {plt_title} for Potential Plants and Save to {plt_save_to}")

def visualize_ss_nodes(substations_gdf,
                       provincem_gadm_regions_gdf:gpd.GeoDataFrame, 
                           plot_name):
        """
        Visualizes transmission nodes (buses) on a map with different colors based on substation types.

        Parameters:
        - gadm_regions_gdf (GeoDataFrame): GeoDataFrame containing base regions to plot.
        - buses_gdf (GeoDataFrame): GeoDataFrame containing buses with 'substation_type' column.
        - plot_name (str): File path to save the plot image.

        Returns:
        - None
        """
        
        fig, ax = plt.subplots(figsize=(10, 8))
        provincem_gadm_regions_gdf.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.8,alpha=0.2)
        substations_gdf.plot('substation_type',ax=ax,legend=True,cmap='viridis',marker='x',markersize=10,linewidth=1,alpha=0.6)

        # Finalize plot details
        plt.title('Buses with Colormap of Substation Types')
        plt.tight_layout()
        
        # Save and close the plot
        plt.savefig(plot_name)
        plt.close()
        
        # Logging success message
        log.info(f"Plot for Grid Nodes Generated and saved as {plot_name}")
        
def create_timeseries_plots(cells_df, CF_timeseries_df, max_resource_capacity, dissolved_indices, resampling, representative_color_palette, std_deviation_gradient, vis_directory):
    print(f">>> Generating CF timeseries PLOTs for TOP Sites for {max_resource_capacity} GW Capacity investment in province...")

    for index, row in cells_df.iterrows():
        region = row['Region']
        cluster_no = row['Cluster_No']

        # Ensure dissolved_indices is a dictionary
        if isinstance(dissolved_indices, dict):
            # Get representative_ts_list with error handling
            representative_ts_list = dissolved_indices.get(region, {}).get(cluster_no, [])
            if not isinstance(representative_ts_list, list):
                representative_ts_list = []
        else:
            representative_ts_list = []
        filtered_ts_list = [col for col in representative_ts_list if col in CF_timeseries_df.columns]

        df = CF_timeseries_df[filtered_ts_list]

        # Resample the data to given frequency (mean)
        _data = df.resample(resampling).mean()

        # Calculate mean and standard deviation across all columns
        mean_values = _data.mean(axis=1)
        std_values = _data.std(axis=1)

        # Create a plot with shaded areas representing standard deviations
        plt.figure(figsize=(16, 3))
        sns.lineplot(data=_data, x=_data.index, y=mean_values, label=f'Cluster ({region}_{cluster_no})', alpha=1, color=representative_color_palette)

        # Plot the shaded areas for standard deviations
        plt.fill_between(_data.index, mean_values - std_values, mean_values + std_values, alpha=0.4, color=std_deviation_gradient, edgecolor='None', label=f"Cells' inside the Cluster ({region}_{cluster_no})")
        plt.legend()
        plt.title(f'Site Capacity Factor  (Resample Span: {resampling}) - {region}_{cluster_no}  [site {cluster_no}/{len(cells_df)}]')
        plt.xlabel('Time')
        plt.ylabel('CF')
        plt.grid(True)
        plt.tight_layout()
        
        plt_name = f'Site Capacity Factor (Resample Span: {resampling}) - {region}_{cluster_no}.png'
        plt.savefig(os.path.join(vis_directory,plt_name))
        plt.close()

    log.info(f">>> Plots generated for CF timeseries of TOP Sites for {max_resource_capacity} GW Capacity Investment in Province...")


def create_timeseries_plots_solar(cells_df,CF_timeseries_df, dissolved_indices,max_solar_capacity,resampling,solar_vis_directory):

    print(f">>> Generating CF timeseries for TOP Sites for {max_solar_capacity} GW Capacity Investment in BC...")
    
    for _index,row in cells_df.iterrows():
        region = row['Region']
        cluster_no = row['Cluster_No']

        resample_span = resampling
        df = CF_timeseries_df[dissolved_indices[region][cluster_no]]

        # Resample the data to monthly frequency (mean)
        _data = df.resample(resample_span).mean()

        # Calculate mean and standard deviation across all columns
        mean_values = _data.mean(axis=1)
        std_values = _data.std(axis=1)

        # Create a plot with shaded areas representing standard deviations
         # Adjust the figure size if needed
        plt.figure(figsize=(16, 3)) 
        # Plot the mean lines for both datasets with different colors for each plot
        sns.lineplot(data=_data, x=_data.index, y=mean_values, label=f'Cluster ({region}_{cluster_no})', alpha=0.6, color=sns.color_palette("dark", 1)[0])

        # Plot the shaded areas for standard deviations
        plt.fill_between(
            _data.index,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
            # color='red',
            label=f"Cells' inside the Cluster ({region}_{cluster_no})"
        )
        plt.legend()
        cluster_no = row['Cluster_No']
        plt.title(f'Solar CF timeseries (Resample Span :{resample_span}) - {region}_{int(cluster_no)}[site {int(cluster_no)}/{len(cells_df)}]')
        plt.xlabel('Date')
        plt.ylabel('Column Values')
    
        plt.grid(True)
        plt.tight_layout()
        plt_name=f'Solar CF timeseries (Resample Span :{resample_span}) - {region}_{cluster_no}.png'
        plt.savefig(os.path.join(solar_vis_directory,'Site_timeseries',plt_name))

    print(f">>> Plots generated for CF timeseries of TOP Sites for {max_solar_capacity} GW Capacity Investment in BC...")
    
def create_timeseries_interactive_plots(
    ts_df:pd.DataFrame,
    save_to_dir:str):
    
    sites=ts_df.columns.to_list()
    
    for site in sites:
        site_df = ts_df[site]  # Select only the column for the current site
        
        hourly_df = site_df
        daily_df = site_df.resample('D').mean()
        weekly_df = site_df.resample('W').mean()
        monthly_df = site_df.resample('ME').mean()
        quarterly_df = site_df.resample('QE').mean()

        # Create a figure
        fig = make_subplots(rows=1, cols=1)

        # Add traces for each aggregation type
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df, mode='lines', name='Hourly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df, mode='lines', name='Daily', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=weekly_df.index, y=weekly_df, mode='lines', name='Weekly', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df, mode='lines', name='Monthly', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=quarterly_df.index, y=quarterly_df, mode='lines', name='Quarterly', visible='legendonly'), row=1, col=1)

        # Define labels and ticks
        hourly_ticks = hourly_df.index[::12]    # Every 12 hours
        daily_ticks = daily_df.index[::10]      # Every 10 days
        weekly_ticks = weekly_df.index[::3]     # Every 3 weeks
        monthly_ticks = monthly_df.index[::1]   # Every month
        quarterly_ticks = quarterly_df.index    # Every quarter
        title=f"Availability of site {site}"
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': 'Hourly', 'method': 'update', 'args': [
                        {'visible': [True, False, False, False, False]},
                        {'xaxis': {'title': 'Time', 'tickvals': hourly_ticks, 'ticktext': hourly_ticks.strftime('%Y-%m-%d %H:%M:%S')}},
                        {'yaxis': {'title': title}}
                    ]},
                    {'label': 'Daily', 'method': 'update', 'args': [
                        {'visible': [False, True, False, False, False]},
                        {'xaxis': {'title': 'Date', 'tickvals': daily_ticks, 'ticktext': daily_ticks.strftime('%Y-%m-%d')}},
                        {'yaxis': {'title': title}}
                    ]},
                    {'label': 'Weekly', 'method': 'update', 'args': [
                        {'visible': [False, False, True, False, False]},
                        {'xaxis': {'title': 'Week', 'tickvals': weekly_ticks, 'ticktext': weekly_ticks.strftime('%Y-W%U')}},
                        {'yaxis': {'title': title}}
                    ]},
                    {'label': 'Monthly', 'method': 'update', 'args': [
                        {'visible': [False, False, False, True, False]},
                        {'xaxis': {'title': 'Month', 'tickvals': monthly_ticks, 'ticktext': monthly_ticks.strftime('%Y-%m')}},
                        {'yaxis': {'title': title}}
                    ]},
                    {'label': 'Quarterly', 'method': 'update', 'args': [
                        {'visible': [False, False, False, False, True]},
                        {'xaxis': {'title': 'Quarter', 'tickvals': quarterly_ticks, 'ticktext': quarterly_ticks.strftime('%Y-Q%q')}},
                        {'yaxis': {'title': title}}
                    ]}
                ],
                'direction': 'down',
                'showactive': True
            }],
            title=f'CF over Time for {site}',
            xaxis_title='Time',
            yaxis_title='CF'
        )

        # Save the plot to an HTML file
        fig.write_html(f'{save_to_dir}/Timeseries_{site}.html')

    # # Display the plot
    # pio.show(fig)

def create_key_data_map_interactive(
    province_gadm_regions_gdf:gpd.GeoDataFrame,
    provincial_conservation_protected_lands: gpd.GeoDataFrame,
    aeroway_with_buffer_solar:gpd.GeoDataFrame,
    aeroway_with_buffer_wind:gpd.GeoDataFrame,
    aeroway:gpd.GeoDataFrame,
    provincial_bus_gdf:gpd.GeoDataFrame,
    current_region:dict,
    about_OSM_data:dict[dict],
    map_html_save_to:str
    ):
    buffer_distance_m:dict[dict]=about_OSM_data['aeroway_buffer']
    
    m = province_gadm_regions_gdf.explore('Region', color='grey',style_kwds={'fillOpacity': 0.1}, name=f"{current_region['code']} Regions")
    provincial_conservation_protected_lands.explore(m=m,color='red', style_kwds={'fillOpacity': 0.05}, name='Conservation and Protected lands')
    aeroway_with_buffer_solar.explore(m=m, color='orange', style_kwds={'fillOpacity': 0.5}, name=f"aeroway with {buffer_distance_m['solar']}m buffer")
    aeroway_with_buffer_wind.explore(m=m, color='skyblue', style_kwds={'fillOpacity': 0.5}, name=f"aeroway with {buffer_distance_m['wind']}m buffer")
    aeroway.explore(m=m,color='blue', marker_kwds={'radius': 2}, name='aeroway')
    provincial_bus_gdf.explore(m=m, color='black', style_kwds={'fillOpacity': 0.5}, name=f'{current_region['code']} Grid Nodes')


    # Add layer control
    folium.LayerControl().add_to(m)

    # Display the map
    m.save(map_html_save_to)
    log.info(f"Key data map create for BC and saved to {map_html_save_to}")
    # return m
    

def create_sites_ts_plots_all_sites(
    resource_type:str,
    CF_ts_df:pd.DataFrame,
    save_to_dir:str):
    
    # Create a plot using plotly.express
    fig = px.line(CF_ts_df, x=CF_ts_df.index, y=CF_ts_df.columns[0:], title=f'Hourly timeseries for {resource_type} sites',
                labels={'value': 'CF', 'datetime': 'DateTime'}, template='plotly_dark')
    # Update the layout to move the legend to the top
    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Aligns the legend at the bottom of the top position
            y=1.02,           # Moves the legend up (outside the plot area)
            xanchor="center",  # Centers the legend horizontally
            x=0.5             # Sets the x position of the legend to be centered
        )
    )
    # Display the plot
    fig.write_html(f'{save_to_dir}/Timeseries_top_sites_{resource_type}.html')
    # fig.write_html(f'results/linking/Timeseries_top_sites_{resource_type}.html')


def create_sites_ts_plots_all_sites_2(
    resource_type: str,
    CF_ts_df: pd.DataFrame,
    save_to_dir: str):
    
    # Resample data for different time intervals
    hourly_df = CF_ts_df
    daily_df = CF_ts_df.resample('D').mean()
    weekly_df = CF_ts_df.resample('W').mean()
    monthly_df = CF_ts_df.resample('ME').mean()
    quarterly_df = CF_ts_df.resample('QE').mean()

    # Create the plot using plotly express for the hourly data
    fig = px.line(hourly_df, x=hourly_df.index, y=hourly_df.columns[0:], title=f'Hourly timeseries for {resource_type} sites',
                  labels={'value': 'CF', 'datetime': 'DateTime'}, template='ggplot2')

    # Add traces for other time intervals (daily, weekly, etc.) with dotted lines
    fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df[daily_df.columns[0]], mode='lines', name='Daily', visible='legendonly',
                             line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=weekly_df.index, y=weekly_df[weekly_df.columns[0]], mode='lines', name='Weekly', visible='legendonly',
                             line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df[monthly_df.columns[0]], mode='lines', name='Monthly', visible='legendonly',
                             line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=quarterly_df.index, y=quarterly_df[quarterly_df.columns[0]], mode='lines', name='Quarterly', visible='legendonly',
                             line=dict(dash='dot')))

    # Update the layout to move the legend to the right, make it scrollable, and shrink the font size
    fig.update_layout(
        legend=dict(
            orientation="v",   # Vertical legend
            yanchor="top",      # Aligns the legend at the top
            y=1,                # Moves the legend up (inside the plot area)
            xanchor="left",     # Aligns the legend on the right
            x=1.02,             # Slightly outside the plot area
            font=dict(size=10),  # Make the font size smaller
            itemwidth=30        # Reduce the width of legend items
        ),
        xaxis_title='DateTime',
        yaxis_title='CF',
        hovermode='x unified',  # Unified hover info across traces
        autosize=False,  # Allow custom sizing
        width=800,       # Adjust plot width
        height=500,      # Adjust plot height
    )

    # Add scrollable legend using CSS styling
    fig.update_layout(
        legend_title=dict(text=f'{resource_type} sites'),
        legend=dict(
            title=dict(font=dict(size=12)),  # Title size
            traceorder='normal',
            itemclick='toggleothers',
            itemdoubleclick='toggle',
            bordercolor="grey",
            borderwidth=1,
        ),
    )

    fig.update_traces(hoverinfo='name+x+y')  # Improve hover info

    # Add range selector and range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),  # Add a range slider
            type="date"
        )
    )

    # Save the plot to an HTML file
    fig.write_html(f'{save_to_dir}/Timeseries_top_sites_{resource_type}.html')

    # Optionally display the plot
    # fig.show()
# 
