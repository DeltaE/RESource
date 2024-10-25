import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging as log
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import geopandas as gpd
import folium


# local scripts

import linkingtool.linking_utility as utility

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S')
log_name=f'workflow/log/linking_vis.txt'
with open(log_name, 'w') as file:
    pass
file_handler = log.FileHandler(log_name)
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
    return log.info(f"Plot Created for {plt_title} for Potential Plants and Save to {plt_save_to}")


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
