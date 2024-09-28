# %%
import requests
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import yaml
import logging as log
import os

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# %%

# Local Packages

try:
    # Try importing from the submodule context
    import linkingtool.linking_utility as utils
    import linkingtool.linking_vis as vis
    import linkingtool.linking_solar as solar
    from linkingtool.attributes_parser import AttributesParser
except ImportError:
    # Fallback for when running as a standalone script or outside the submodule
    import Linking_tool.linkingtool.linking_utility as utils
    import Linking_tool.linkingtool.linking_vis as vis
    import Linking_tool.linkingtool.linking_solar as solar
    from Linking_tool.linkingtool.attributes_parser import AttributesParser

# %%
import geopandas as gpd
from shapely.geometry import Point

# %%


# %%
def convert_coders_df_to_gdf(df):
    df=df.copy()
    # Create a geometry column
    df.loc[:,'geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry').set_crs(epsg=4326, inplace=True)

    # Set a coordinate reference system (CRS) if known, e.g., WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

# %% [markdown]
# - Load CONFIG

# %%
config_path:str='config/config_linking_tool.yml'
config:dict=utils.load_config(config_path)
current_region=config['regional_info']['region_1']
_CRC_=current_region['code'] # Current Region Code (CRC), later to be configured with the user config file.

# %%
CODERS_data:dict= config['CODERS']
url=CODERS_data['url_1']
api_elias=CODERS_data['api_key']['Elias']
query="?key="+api_elias

data_pull:dict=config['CODERS']['data_pull']

# %% [markdown]
# -  Create Directories

# %% [markdown]
# ## Tables

# %% [markdown]
# -  Create Directories

# %%
# 'coders or 'cef'

# %%
tables_list = [t for t in requests.get(url+"/tables/cef"+query).json()]
tables_list
print(f"CEF data available:\n {tables_list}")

# %%
tables_list = [t for t in requests.get(url+"/tables/coders"+query).json()]
tables_list
print(f"CODERS data available:\n {tables_list}")

# %% [markdown]
# ### transmission_generic

# %%
table_name='transmission_generic'
transmission_generic = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())

data=transmission_generic
file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# %%
# table_name='grid_cell_info'
# grid_cell_info = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())
# grid_cell_info_gdf=convert_coders_df_to_gdf(grid_cell_info)
# # grid_cell_info_gdf.explore('balancing_area')

# %% [markdown]
# ### macro_indicators

# %%
table_name='macro_indicators'
macro_indicators = pd.DataFrame.from_dict(requests.get(url+f"/cef/{table_name}"+query).json())
for scenario in macro_indicators.scenario:
    data=macro_indicators
    scenario_data=data[data['scenario']==scenario]
    file_path:str=os.path.join(data_pull['root'],'macro_indicators',f'{table_name}_{scenario}.csv')
    scenario_data.to_csv(file_path)

# %% [markdown]
# ### Emission

# %%
table_name='greenhouse_gas_emissions'
greenhouse_gas_emissions = pd.DataFrame.from_dict(requests.get(url+f"/cef/{table_name}"+query).json())

for scenario in greenhouse_gas_emissions.scenario:
    data=greenhouse_gas_emissions
    scenario_data=data[data['scenario']==scenario]
    file_path:str=os.path.join(data_pull['root'],'emission',f'{table_name}_{scenario}.csv')
    scenario_data.to_csv(file_path)
    print(f"{table_name} data saved to:\n {file_path}")

# %% [markdown]
# ### end_use_demand

# %%
table_name='end_use_demand'
end_use_demand = pd.DataFrame.from_dict(requests.get(url+f"/cef/{table_name}"+query).json())

region_mask=end_use_demand['region']==current_region['name']
regional_end_use_demand=end_use_demand[region_mask]
regional_end_use_demand_elec=regional_end_use_demand[regional_end_use_demand['variable']=='Electricity']

for scenario in regional_end_use_demand['scenario'].unique():
    data=regional_end_use_demand
    scenario_data=data[data['scenario']==scenario]
    file_path:str=os.path.join(data_pull['root'],'demand',f'{_CRC_}_{table_name}_{scenario}.csv')
    scenario_data.to_csv(file_path)

# Iterate over unique scenarios
for scenario in regional_end_use_demand_elec['scenario'].unique():
    # Filter data for the current scenario and year >= 2021
    df_filtered = (regional_end_use_demand_elec[regional_end_use_demand_elec['scenario'] == scenario]
                   .query('year >= 2021')
                   .assign(value=pd.to_numeric(regional_end_use_demand_elec['value'], errors='coerce'))
                   .groupby(['sector', 'year'], as_index=False)['value'].sum())

    # Clean sector names and pivot the DataFrame
    df_filtered['sector'] = df_filtered['sector'].str.replace('\r', '', regex=False)
    df_pivot = df_filtered.pivot_table(index='year', columns='sector', values='value', aggfunc='sum').fillna(0).reset_index()

    # Reorder columns if present
    sectors = ['Commercial', 'Industrial', 'Residential', 'Transportation'] #, 'Total End-Use'
    df_pivot = df_pivot[['year'] + [sector for sector in sectors if sector in df_pivot.columns]]

    # Save the DataFrame to CSV
    file_path = os.path.join(data_pull['root'], 'demand', f'{_CRC_}_sectoral_elec_{table_name}_{scenario}.csv')
    df_pivot.to_csv(file_path, index=False)
    
    # Create and save a stacked area chart
    fig = px.area(df_pivot, x='year', y=df_pivot.columns[1:], 
                  labels={'value': 'Pj', 'year': 'Year'}, 
                  title=f'Stacked Area Chart by Sector ({scenario})')
    chart_save_to = os.path.join('vis', f'{_CRC_}_sectoral_elec_{table_name}_{scenario}.html')
    fig.write_html(chart_save_to)



# %% [markdown]
# ### benchmark_prices

# %%
table_name='benchmark_prices'
benchmark_prices = pd.DataFrame.from_dict(requests.get(url+f"/cef/{table_name}"+query).json())

# %%
for scenario in benchmark_prices.scenario:
    data=benchmark_prices
    scenario_data=data[data['scenario']==scenario]
    file_path:str=os.path.join(data_pull['root'],'fuel_price',f'{table_name}_{scenario}.csv')
    scenario_data.to_csv(file_path)
    print(f"{table_name} data saved to:\n {file_path}")

# %% [markdown]
# ### Generators

# %%
table_name='generators'
generators = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())
data=generators

file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# Provincial data trimming
province_mask=generators['province']==_CRC_
province_generators=generators[province_mask]
data=province_generators
file_path:str=os.path.join(data_pull['root'],'supply',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")

province_generators_gdf=convert_coders_df_to_gdf(province_generators)

# %% [markdown]
# ### hydro_cascade

# %%
table_name='hydro_cascade'
hydro_cascade = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())
data=hydro_cascade

file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# Provincial data trimming
province_mask=hydro_cascade['province']==current_region['name']
province_hydro_cascade=hydro_cascade[province_mask]

data=province_hydro_cascade
file_path:str=os.path.join(data_pull['root'],'supply',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")

province_hydro_cascade_gdf=convert_coders_df_to_gdf(province_hydro_cascade)

# %% [markdown]
# ### Hydro (existing)

# %%
table_name='hydro_existing'
hydro_existing = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())

data=hydro_existing
file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# Provincial data trimming
province_mask=hydro_existing['Province']==_CRC_
province_hydro_existing=hydro_existing[province_mask]

data=province_hydro_existing
file_path:str=os.path.join(data_pull['root'],'supply',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")

# %% [markdown]
# ### Wind Generators

# %%
table_name='wind_generators'
wind_generators = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())

data=wind_generators
file_path:str=os.path.join(data_pull['root'],'supply',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# Provincial data trimming
province_mask=wind_generators['province']==_CRC_
province_wind_generators=wind_generators[province_mask]

data=province_wind_generators
file_path:str=os.path.join(data_pull['root'],'supply',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")
# %% [markdown]
# ### Forecasted Annual Demand

# %%
table_name='forecasted_annual_demand'
forecasted_annual_demand:pd.DataFrame=pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())
data=forecasted_annual_demand
file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# Provincial data trimming
province_mask=forecasted_annual_demand['province']==_CRC_
province_forecasted_annual_demand=forecasted_annual_demand[province_mask]

data=province_forecasted_annual_demand
file_path:str=os.path.join(data_pull['root'],'demand',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")

# %% [markdown]
# ### Forecasted Peak Demand

# %%
table_name='forecasted_peak_demand'
forecasted_peak_demand:pd.DataFrame=pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())

data=forecasted_peak_demand
file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")

# Provincial data trimming
province_mask=forecasted_peak_demand['province']==_CRC_
province_forecasted_peak_demand=forecasted_peak_demand[province_mask]

data=province_forecasted_annual_demand
file_path:str=os.path.join(data_pull['root'],'demand',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")

# %% [markdown]
# ### Demand Profile

# %% [markdown]
# -  Create Directories

# %%
table_name='provincial_demand'
demand_dataset = pd.DataFrame.from_dict(requests.get(url + f"/{table_name}" + query).json())

# %%
province_mask=demand_dataset['province']==_CRC_
province_demand_dataset=demand_dataset[province_mask] 
print(F"Demand Profile dataset for province - {current_region['name']} available for the following years -\n{province_demand_dataset.year.values}")


year=2021
print(f" Year {year} selected")
province_demand_profile_yr = pd.DataFrame.from_dict(requests.get(url + f"/{table_name}" + query + f"&year=2020&province={_CRC_}").json())

province_demand_profile_yr.set_index('local_time',inplace=True)
province_demand_profile_yr.index = pd.to_datetime(province_demand_profile_yr.index)
province_demand_profile_yr

data=province_demand_profile_yr

file_path:str=os.path.join(data_pull['root'],'demand',f'{_CRC_}_{table_name}_profile.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")
# %% [markdown]
# #### Visualize demand profile

# %%
# Resampling the data
df=province_demand_profile_yr
hourly_df = df['demand_MWh']
daily_df = df['demand_MWh'].resample('D').mean()
weekly_df = df['demand_MWh'].resample('W').mean()
monthly_df = df['demand_MWh'].resample('ME').mean()
quarterly_df = df['demand_MWh'].resample('QE').mean()

# Create a figure
fig = make_subplots(rows=1, cols=1)

# Add traces for each aggregation type
fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df, mode='lines', name='Hourly'), row=1, col=1)
fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df, mode='lines', name='Daily', visible='legendonly'), row=1, col=1)
fig.add_trace(go.Scatter(x=weekly_df.index, y=weekly_df, mode='lines', name='Weekly', visible='legendonly'), row=1, col=1)
fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df, mode='lines', name='Monthly', visible='legendonly'), row=1, col=1)
fig.add_trace(go.Scatter(x=quarterly_df.index, y=quarterly_df, mode='lines', name='Quarterly', visible='legendonly'), row=1, col=1)


# Define labels and ticks
daily_ticks = hourly_df.index[::12]    # Every 36 hours
daily_ticks = daily_df.index[::10]    # Every 10 days
weekly_ticks = weekly_df.index[::3]  # Every 3 weeks
monthly_ticks = monthly_df.index[::1]  # Every month

# Add dropdown menu
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {'label': 'Hourly', 'method': 'update', 'args': [
                {'visible': [True, False, False, False, False]},
                {'xaxis': {'title': 'Time', 'tickvals': daily_ticks, 'ticktext': daily_ticks.strftime('%Y-%m-%d %H:%M:%S')}},
                {'yaxis': {'title': 'Demand (MWh)'}}
            ]},
            {'label': 'Daily', 'method': 'update', 'args': [
                {'visible': [False, True, False, False, False]},
                {'xaxis': {'title': 'Date', 'tickvals': daily_ticks, 'ticktext': daily_ticks.strftime('%Y-%m-%d')}},
                {'yaxis': {'title': 'Demand (MWh)'}}
            ]},
            {'label': 'Weekly', 'method': 'update', 'args': [
                {'visible': [False, False, True, False, False]},
                {'xaxis': {'title': 'Week', 'tickvals': weekly_ticks, 'ticktext': weekly_ticks.strftime('%Y-W%U')}},
                {'yaxis': {'title': 'Demand (MWh)'}}
            ]},
            {'label': 'Monthly', 'method': 'update', 'args': [
                {'visible': [False, False, False, True, False]},
                {'xaxis': {'title': 'Month', 'tickvals': monthly_ticks, 'ticktext': monthly_ticks.strftime('%Y-%m')}},
                {'yaxis': {'title': 'Demand (MWh)'}}
            ]},
            {'label': 'Quarterly', 'method': 'update', 'args': [
                {'visible': [False, False, False, False, True]},
                {'xaxis': {'title': 'Quarter', 'tickvals': quarterly_df.index, 'ticktext': quarterly_df.index.strftime('%Y-Q%q')}},
                {'yaxis': {'title': 'Demand (MWh)'}}
            ]}
        ],
        'direction': 'down',
        'showactive': True
    }],
    title='Demand in MWh over Time',
    xaxis_title='Time',
    yaxis_title='Demand (MWh)'
)

# Save the plot to an HTML file
fig.write_html(f'vis/Demand_profile_{_CRC_}_{year}.html')
print(f"Provincial demand profile visuals saved as:\n vis/Demand_profile_{_CRC_}_{year}.html")

# Display the plot
pio.show(fig)

# %% [markdown]
# ### Storage

# %%
table_name='storage'
storage_df = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())


province_mask=storage_df['province']==_CRC_
province_storage_df=storage_df[province_mask]
if(len(province_storage_df)==0):
    print(f"No storage found for province - {current_region['name']}")
else:
    data=province_storage_df

    file_path:str=os.path.join(data_pull['root'],'supply',f'{_CRC_}_{table_name}_profile.pkl')
    data.to_pickle(file_path)
    print(f"Provincial {table_name} data saved to:\n {file_path}")
    

# %% [markdown]
# # Generation Planning Reserve

# %%
table_name='generation_planning_reserve'
generation_planning_reserve = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())

province_mask=generation_planning_reserve['province']==current_region['name']
province_generation_planning_reserve=generation_planning_reserve[province_mask]

data=province_generation_planning_reserve
file_path:str=os.path.join(data_pull['root'],'reserve',f'{_CRC_}_{table_name}.csv')
data.to_csv(file_path)
print(f"Provincial {table_name} data saved to:\n {file_path}")

# %%
# table_name='annual_demand_and_efficiencies'
# annual_demand_and_efficiencies = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query+"/attributes").json())

# %% [markdown]
# ### generation_generic

# %%
table_name='generation_generic'
generation_generic = pd.DataFrame.from_dict(requests.get(url+f"/{table_name}"+query).json())

data=generation_generic
file_path:str=os.path.join(data_pull['root'],data_pull[f'{table_name}'])
data.to_csv(file_path)
print(f"{table_name} data saved to:\n {file_path}")


# %%
dropdown_columns=['typical_plant_size_MW',
       'capital_cost_CAD_per_kW', 'capital_overhead_CAD_per_kW',
       'overnight_capital_cost_CAD_per_kW',
       'interest_during_construction_CAD_per_kW',
       'implementation_costs_CAD_per_kW',
       'project_definition_costs_CAD_per_kW',
       'total_project_cost_2020_CAD_per_kW', 'economic_life', 'service_life',
       'annualized_capital_cost_CAD_per_MWyear',
       'fixed_om_cost_CAD_per_MWyear', 'variable_om_cost_CAD_per_MWh',
       'construction_time', 'development_time',
       'average_fuel_price_CAD_per_MMBtu', 'average_fuel_price_CAD_per_GJ',
       'carbon_emissions', 'heat_rate', 'efficiency', 'min_plant_load',
       'min_capacity_factor', 'max_capacity_factor', 'time_to_full_capacity',
       'min_up_time_hours', 'min_down_time_hours', 'ramp_rate_percent_per_min',
       'spinning_reserve_capability', 'forced_outage_rate',
       'planned_outage_rate', 'startup_cost', 'shutdown_cost',
       ]

# %%
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
color_palette = {
    'typical_plant_size_MW': 'blue',
    'capital_cost_CAD_per_kW': 'green',
    'capital_overhead_CAD_per_kW': 'red',
    'overnight_capital_cost_CAD_per_kW': 'purple',
    'interest_during_construction_CAD_per_kW': 'orange',
    'implementation_costs_CAD_per_kW': 'cyan',
    'project_definition_costs_CAD_per_kW': 'magenta',
    'total_project_cost_2020_CAD_per_kW': 'yellow',
    'economic_life': 'pink',
    'service_life': 'brown',
    'annualized_capital_cost_CAD_per_MWyear': 'grey',
    'fixed_om_cost_CAD_per_MWyear': 'teal',
    'variable_om_cost_CAD_per_MWh': 'olive',
    'construction_time': 'maroon',
    'development_time': 'navy',
    'average_fuel_price_CAD_per_MMBtu': 'lime',
    'average_fuel_price_CAD_per_GJ': 'salmon',
    'carbon_emissions': 'coral',
    'heat_rate': 'gold',
    'efficiency': 'plum',
    'min_plant_load': 'khaki',
    'min_capacity_factor': 'tan',
    'max_capacity_factor': 'silver',
    'time_to_full_capacity': 'indigo',
    'min_up_time_hours': 'violet',
    'min_down_time_hours': 'wheat',
    'ramp_rate_percent_per_min': 'crimson',
    'spinning_reserve_capability': 'azure',
    'forced_outage_rate': 'beige',
    'planned_outage_rate': 'lightgrey',
    'startup_cost': 'peachpuff',
    'shutdown_cost': 'lightblue'
}

# %%
# Create an empty figure
fig = make_subplots(rows=1, cols=1)

# Create traces for each selected column
traces = []
for col in dropdown_columns:
    # Sort the DataFrame by 'gen_type_copper' and the current column
    sorted_df = generation_generic.sort_values(by=[col])

    trace = go.Box(
        x=sorted_df['gen_type_copper'],
        y=sorted_df[col],
        name=col,
        marker_color=color_palette[col],
        boxmean='sd',  # Show mean and standard deviation
        whiskerwidth=0.5,
        line=dict(width=2),
        fillcolor=color_palette[col],
        opacity=0.6,
        hoverinfo='x+y+name'  # Show x, y, and trace name on hover
    )
    traces.append(trace)
    fig.add_trace(trace)

# Set the first trace to be visible initially
fig.data[0].visible = True

# Create dropdown menu
dropdown_buttons = [
    {'label': col, 'method': 'update', 'args': [{'visible': [col == trace.name for trace in traces]}, {'title': f'gen_type_copper vs {col}'}]}
    for col in dropdown_columns
]

# Add dropdown menu to the layout
fig.update_layout(
    title='gen_type_copper vs Data Fields [Source: CODERS/generation_generic]',
    updatemenus=[{
        'buttons': dropdown_buttons,
        'direction': 'down',
        'showactive': True,
        'x': 1.1, # Position dropdown
        'xanchor': 'left',
        'y': 1.15, # Position dropdown
        'yanchor': 'top'
    }],
    xaxis_title='gen_type_copper',
    yaxis_title='Value',
    title_font=dict(size=24, family='Arial', color='rgba(0,0,0,0.8)'),
    xaxis=dict(tickangle=-45),  # Rotate x-axis labels
    plot_bgcolor='rgba(245,245,245,0.8)',  # Background color
    paper_bgcolor='white',  # Overall background color
    font=dict(family='Arial', size=12, color='rgba(0,0,0,0.8)'),
    margin=dict(l=40, r=40, t=80, b=40)  # Adjust margins
)

# Save the figure to an HTML file and show it
fig.write_html('vis/Generation_Generic_Datafields_from_CODERS_Enhanced.html')
print(f"Generic Generation Data visuals saved to:\n vis/Generation_Generic_Datafields_from_CODERS_Enhanced.html")
# Display the plot
pio.show(fig)


