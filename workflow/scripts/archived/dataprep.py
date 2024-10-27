# %% [markdown]
# # 1- Import Packages

# %% [markdown]
# ## i. global packages

# %%

from pathlib import Path
import os
import requests
from requests import get
import logging as log

import yaml
from zipfile import ZipFile

import pandas as pd
import geopandas as gpd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

import pygadm

import rasterio
from shapely.geometry import box, Point
from shapely.ops import unary_union
from rasterio.mask import mask

import cartopy.feature as cfeature
import cartopy.crs as ccrs

from pyrosm import OSM, get_data
from pyrosm.data import sources
from pyrosm.config import Conf


# %% [markdown]
# ## ii. import local packages

# %%
import linkingtool.utility as utils
import linkingtool.windspeed as wind
import linkingtool.visuals as vis

# %% [markdown]
# # 2- Supporting Functions
# %% [markdown]
# ## ii. Units Directory

# %%
#--------------------[ Create Units Dictionary in Excel ]---------------------
def create_units_dictionary(
      units_file_path:str):
    
   Units_dct={
      'capex' : 'Mil. USD/MW',
      'fom':  'Mil. USD/MW',
      'vom':  'Mil. USD/MW',
      'potential_capacity' : 'MW',
      'p_lcoe' : 'MWH/USD',
      'energy': 'MWh',
      'energy demand':'Pj'
   }

   # Convert the dictionary to a DataFrame
   Units_df = pd.DataFrame.from_dict(Units_dct,orient='index', columns=['Unit'])

   # Save the DataFrame to an Excel file
   Units_df.to_excel(units_file_path, index=False)
   log.info(f"Units Dictionary Created and saved to '{units_file_path}'")


""" >>> recheck the usability of this code
def get_bus_name_x_y(line, node_code, df_substations):
    '''
    This function finds the correct name for a bus from the node dataset
    line: Line row from CODERS lines dataframe.
    node_code: Name of start/end node from CODERS lines dataframe.
    df_substations: Substations dataframe from CODERS.
    return bus_name: Unique name to use for the bus in PyPSA (i.e. 230_AAL_DSS, 230=nominal_voltage, AAL=unique substation name in CODERS, DSS=Substation type)
    return bus_x: Bus longitude. 
    return bus_y: Bus latitude.
    '''
    # (1) identical match
    for idx,substation in df_substations.iterrows():
        if substation["node_code"] == node_code:
            bus_name = str(line["voltage"]) + "_" + "_".join(node_code.split('_')[1:]) # (i.e. 230_AAL_DSS)
            bus_x = substation["longitude"]
            bus_y = substation["latitude"]
            return bus_name, bus_x, bus_y

    # print(f"Did not find exact match for line node: {node_code}") # To-be logged

    # # (2) International and Interprovincial nodes
    # if node_code.split('_')[-1] in ["IPT","INT"]:
    #     bus_name = str(line["voltage"]) + "_" + "_".join(node_code.split('_')[1:])
    #     if node_code == "PP_BCAB3_IPT":
    #         # ~ 50 km east
    #         bus_y, bus_x = 50.247937, -114.2 
    #     elif node_code == "PP_BCAB1_IPT":
    #         # ~ 18 km east
    #         bus_y, bus_x = 49.735535, -114.6
    #     elif node_code == "PP_BCAB4_IPT":
    #         # ~21 km east
    #         bus_y, bus_x = 58.64525, -119.7
    #     elif node_code == "XX_BCUS2_INT":
    #         # ~ 1 km south
    #         bus_y, bus_x = 48.9974, -117.341514
    #     elif node_code == "XX_BCUS1_INT":
    #         # ~ 21 km south
    #         bus_y, bus_x = 48.97 , -122.873948
    #     elif node_code == "PP_BCAB2_IPT":
    #         # ~ 108 km east
    #         bus_y, bus_x = 49.500543, -114.08
    #     return bus_name, bus_x, bus_y

    # (3) Find first matching 3-middle characters
    for idx,substation in df_substations.iterrows():
        if node_code.split('_')[1] == substation['node_code'].split('_')[1]:
            bus_name = str(line["voltage"]) + "_" + "_".join(node_code.split('_')[1:])
            bus_x = substation["longitude"]
            bus_y = substation["latitude"]
            return bus_name, bus_x, bus_y
    # print(f"Did not find partial match for: {node_code}") # To-be logged

    # (4) No matching 3 middle characters (i.e. BC_WAX_GSS).. these are special cases..
    print(f"There is no information to create bus for: {node_code}") # To-be logged

    return None,None,None

def create_bus_df(df_lines, 
                  df_substations,
                  save_to_file_path):
    '''
    This function will create an initial DataFrame of buses from a DataFrame of lines.
    When creating the buses it
    The lines DF contains the node names for the buses, nominal voltage, and the carrier is implicitly added.
    '''
    # name = []
    # x = []
    # y = []
    # type = []
    # v_nom = []
    data_dict = {}

    # (1) Add buses based on line nodes
    for idx,line in df_lines.iterrows():
        # Search for match between line and substation
        for node_code in [line["starting_node_code"], line["ending_node_code"]]:
            bus_name, bus_x, bus_y = get_bus_name_x_y(line, node_code, df_substations)
            if bus_name not in data_dict: # Avoid duplication (Change to dictionary)
                data_dict[bus_name] = {'x':bus_x, 'y':bus_y, 'type':line['current_type'], 'v_nom':line['voltage']}
                # name.append(bus_name) # i.e. 230_AAL
                # x.append(bus_x)
                # y.append(bus_y)
                # type.append(line['type'])
                # v_nom.append(line['v_nom'])
    
    df_buses = pd.DataFrame.from_dict(data_dict,orient='index').reset_index().rename(columns={'index':'name'})
    df_buses['substation_type'] = df_buses['name'].str.split('_').str[2].str[:3]
    # df_buses = pd.DataFrame()
    # df_buses = pd.DataFrame()
    # df_buses['name'] = name
    # df_buses['x'] = x
    # df_buses['y'] = y
    # df_buses['type'] = type
    # df_buses['v_nom'] = v_nom
    print(f"{50*'_'}")
    df_buses.to_csv(save_to_file_path)

    return df_buses

"""

# %% [markdown]
# ## v. Open Energy Database (OEDB) [industry turbine configurations]

# %% [markdown]
# * Saving the pulled data as config file

# %%
def format_and_save_turbine_config(
        turbine_data:dict, 
        save_to:str):
    """ 
    takes in a Turbine's specification data (dict) and saves it to config file of that turbine.
    """
    # Extracted information
    name = turbine_data['name']
    manufacturer = turbine_data['manufacturer']
    source = turbine_data['source']
    hub_heights = list(map(float, turbine_data['hub_height'].split(';')))
    power_curve_wind_speeds = eval(turbine_data['power_curve_wind_speeds'])
    power_curve_values = eval(turbine_data['power_curve_values']) #kW
    power_curve_values = [value / 1000 for value in power_curve_values]

    nominal_power = turbine_data['nominal_power']/1000  #MW

    name_for_directory = name.replace(' ', '_')

    # Create a dictionary for YAML output
    formatted_data = {
        'name': name,
        'manufacturer': manufacturer,
        'hub_height': hub_heights[0],
        'V': power_curve_wind_speeds,
        'POW': power_curve_values,
        'source': source,
        'P': nominal_power,
    }
    
    output_file = os.path.join(save_to, f"{name_for_directory}.yaml")
    utils.create_blank_yaml(output_file)
    with open(output_file, 'a') as file:
        yaml.dump(formatted_data, file, default_flow_style=False)

    log.info(f"Formatted data saved to '{output_file}'")

# %% [markdown]
# * Pull the data

# %%

def get_OEDB_dict(
        OEDB_turbines_dict:dict, 
        key:str, 
        value:str):
    """ 
    Takes in OEDB turbine configuration dictionary and the search criterion datafields name:'id' as key and turbine_id (int) as 'value' and returns the search results as the trimmed dictionary of specific turbine's specifications.
    """
    for entry in OEDB_turbines_dict:
        if entry.get(key) == value:
            return entry
    return None  # Return None if no match is found

# %% [markdown]
# ## vii. Global Agro-Ecological Zones (GAEZ) [for land and terrain's raster resources]

# %% [markdown]
# * Pull raster files' ZIP
# %%
def download_resources_zip_file_from_GAEZ(parent_direct,zip_file):
    url = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR.zip"
    response = requests.get(url)
        
    if response.status_code == 200:
        with open(os.path.join(parent_direct, zip_file), 'wb') as zip_file:
            zip_file.write(response.content)
        log.info(f"Zip file downloaded and saved to: {parent_direct}")
        return zip_file
    else:
        return log.info(f"Failed to download the Resources zip file from GAEZ. Status code: {response.status_code}")

# %% [markdown]
# * Extract the required land,exclusion and terrain resources from the ZIP

# %%
def extract_rasters(
        parent_direct:str,
        Rasters_in_use_direct:str,
        raster_files:list,
        zip_file:str,
        zip_extract_directories:list):
    
    with ZipFile(os.path.join(parent_direct, zip_file), 'r') as zip_ref:
            for raster_file, zip_direct in zip(raster_files, zip_extract_directories):
                file_in_zip = os.path.join(zip_direct, raster_file)
                if not os.path.exists(os.path.join(parent_direct, Rasters_in_use_direct,zip_direct, raster_file)):
                    if file_in_zip in zip_ref.namelist():
                        zip_ref.extract(file_in_zip, path=os.path.join(parent_direct, Rasters_in_use_direct))
                        log.info(f"Raster file '{raster_file}' extracted from the Resources zip file and saved to : {os.path.join(parent_direct, Rasters_in_use_direct,zip_direct)}")
                    else:
                        log.error(f"Raster file '{raster_file}' not found in the zip file.")
                else:
                     log.info(f"Raster file '{raster_file}' exists locally @ : {os.path.join(parent_direct, Rasters_in_use_direct,zip_direct)}")

# %% [markdown]
# * Sequential Processing of the Raster

# %%
def process_raster_files(
        root_dir:str, 
        zip_file:str, 
        raster_files:dict, 
        zip_extract_directories:dict, 
        Rasters_in_use_direct:str):
    """
    Download and extract raster files from a zip file.
    
    Args:
    - parent_direct: Path to the parent directory where files will be downloaded and extracted.
    - zip_file: Name of the zip file to download.
    - raster_files: List of names of raster files to extract from the zip.
    - zip_directories: List of directories within the zip where raster files are located.
    - Rasters_in_use_direct: Path to the directory where raster files will be extracted.
    """

    # Check if the zip file exists
    if not os.path.exists(os.path.join(root_dir, zip_file)):
        # If the zip file doesn't exist, download it
        download_resources_zip_file_from_GAEZ(root_dir,zip_file)
        extract_rasters(root_dir,Rasters_in_use_direct,raster_files,zip_file,zip_extract_directories)
    
    else:
        # The zip file exists
        extract_rasters(root_dir,Rasters_in_use_direct,raster_files,zip_file,zip_extract_directories)

# %% [markdown]
# * Visualize Raster

# %%

# %% [markdown]
# * Trim the Raster to Required GADM Boundaries

# %%
def CLIP_n_PLOT_GAEZraster_for_province(
        province_short_code:str,
        parent_direct:str,
        Rasters_in_use_direct:str,
        type:str,
        zip_direct:str,
        raster:str,
        cmap:str,
        boundary_geom):

    ip_raster_file=os.path.join(parent_direct,Rasters_in_use_direct,zip_direct,raster)
    raster_plot_name=f'{province_short_code}_{raster}'
    raster_dump_direct=os.path.join(parent_direct,Rasters_in_use_direct,zip_direct,province_short_code)
    os.makedirs(raster_dump_direct,exist_ok=True)

    # Replace 'your_output_clipped_raster.tif' with the desired output path for the clipped raster
    output_clipped_raster_path = os.path.join(raster_dump_direct,raster_plot_name)
    # Open the input raster file
    with rasterio.open(ip_raster_file) as src:
        # Clip the raster with the bounding box
        clipped_raster, clipped_transform = mask(src, boundary_geom, crop=True, indexes=src.indexes)
        
        # Update the metadata for the clipped raster
        clipped_meta = src.meta.copy()

        clipped_meta.update({
            'height': clipped_raster.shape[1],
            'width': clipped_raster.shape[2],
            'transform': clipped_transform
        })

        # Write the clipped raster to a new file
        with rasterio.open(output_clipped_raster_path, 'w', **clipped_meta) as dst:
            dst.write(clipped_raster)

        plot_save_to=os.path.join('vis/misc',str(raster_plot_name).replace('.tif',"_raster.png"))
        # create_raster_image(output_clipped_raster_path,cmap,type,plot_save_to)
        vis.create_raster_image_with_legend(output_clipped_raster_path,cmap,type,plot_save_to)

    return log.info(f"Raster plot for BC cropped from '{raster}' and saved to : '{plot_save_to}'")

# %% [markdown]
# * Prepare the data query for rasters and sequentially process the raster data

# %%
def prepare_GAEZ_raster_files(
        config:dict,
        GADM_gdf:gpd.GeoDataFrame,
        province_short_code:str):

    root_dir = config['GAEZ']['root']
    zip_file =config['GAEZ']['zip_file']
    raster_types= config['GAEZ']['raster_types']
    raster_filenames = [config['GAEZ'][key]['raster'] for key in raster_types]
    zip_extract_directories = [config['GAEZ'][key]['zip_extract_direct'] for key in raster_types]
    Rasters_in_use_direct = config['GAEZ']['Rasters_in_use_direct']
    process_raster_files(root_dir, zip_file, raster_filenames, zip_extract_directories, Rasters_in_use_direct)

    color_map=[config['GAEZ'][key]['color_map'] for key in raster_types]
    gadm_boundary=GADM_gdf
    buffer_distance_km= 0 #km #future explore feature
    gadm_boundary_buffer=gadm_boundary.copy()
    gadm_boundary_buffer['geometry'] = gadm_boundary.buffer(buffer_distance_km / 111.32)  # 1 degree is approximately 111.32 km

    for type,raster,zip_direct,cmap in zip(raster_types,raster_filenames,zip_extract_directories,color_map):
        CLIP_n_PLOT_GAEZraster_for_province(province_short_code,root_dir,Rasters_in_use_direct,type,zip_direct,raster,cmap,gadm_boundary_buffer.geometry)

    return log.info ("Required rasters for GAEZ processed and plotted successfully !")


# %%
def extract_provincial_data_from_gov_conservation_lands(
    province_gadm_regions_gdf:gpd.GeoDataFrame,
    gov_conservation_lands_config:dict,
    province_name:str,
    save_to:str
    ):
    cfg=gov_conservation_lands_config
    
    gdb_file_name = cfg['geodatabase_file']
    data_root=cfg['data_root']#.replace('data',"data/downloaded_data")
    gdb_file_path=os.path.join(data_root,gdb_file_name)
    zip_file_path=str(gdb_file_path+'.zip')

    # Directory to extract the contents of the zip file
    extraction_direct = cfg['zip_expand'].replace('data',"data/downloaded_data")

    # Extract the zip file
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_direct)

    protectedConservedArea:gpd.GeoDataFrame=gpd.read_file(os.path.join(extraction_direct,gdb_file_name))
    protectedConservedArea = protectedConservedArea.to_crs(province_gadm_regions_gdf.crs)

    conservation_lands_provincial_id=cfg['LOCATION_mapping'][province_name]

    provincial_mask=protectedConservedArea[protectedConservedArea['LOC']==conservation_lands_provincial_id]
    IUCN_CAT:dict=cfg['IUCN_CAT_mapping']

    # Check for invalid geometries
    invalid_geoms = provincial_mask[~provincial_mask.is_valid]

    # Fix invalid geometries
    if not invalid_geoms.empty:
        print("Fixing invalid geometries...")
        provincial_mask['geometry'] = provincial_mask['geometry'].buffer(0)

    # Retry the dissolve operation
    try:
        provincial_mask = provincial_mask.dissolve(by='IUCN_CAT')
        # Proceed with further operations...
    except Exception as e:
        print("Error:", e)

    provincial_mask.reset_index(inplace=True)
    provincial_conservation_protected_lands=provincial_mask
    provincial_conservation_protected_lands['IUCN_CAT_desc'] = provincial_conservation_protected_lands['IUCN_CAT'].map(IUCN_CAT) #Geographical Constrains
    # provincial_conservation_protected_lands.to_pickle(save_to)
    provincial_conservation_protected_lands.to_parquet(save_to)

    provincial_conservation_protected_lands_union= unary_union(provincial_conservation_protected_lands['geometry'])
    provincial_conservation_protected_lands_union = gpd.GeoDataFrame(geometry=[provincial_conservation_protected_lands_union], crs=province_gadm_regions_gdf.crs)
    provincial_conservation_protected_lands_union.to_parquet(save_to.replace(".parquet","_union.parquet"))

# %% [markdown]
# * Visualize data

# %%

def create_plots_protected_lands(
        provincial_conservation_protected_lands:pd.DataFrame,
        gov_conservation_lands:dict,
        current_region:dict,
        province_gadm_regions_gdf:gpd.GeoDataFrame,
        plot_name:str,
        ):
    data_df=provincial_conservation_protected_lands
    IUCN_CAT=gov_conservation_lands['IUCN_CAT']
    # Create the figure and axes
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Specify distinct colors for each category
    category_colors = ['red', 'green', 'navy', 'brown', 'purple', 'orange', 'olive', 'magenta', 'lime']  # Add more colors as needed

    # Plot the conservation lands with the specified colors
    data_df.plot(ax=ax, column='IUCN_CAT', color=[category_colors[i - 1] for i in data_df['IUCN_CAT']])

    # Overlay the gadm_regions_gdf
    province_gadm_regions_gdf.plot(ax=ax, color='none', linewidth=0.4, edgecolor='grey')

    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    for i, (code, category) in enumerate(IUCN_CAT.items()):
        legend_handles.append(mpatches.Patch(color=category_colors[i], label=category))
        legend_labels.append(category)

    # Add legend with handles and labels
    plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # Set the title
    plt.title(f"GOV. Conservation Lands - {current_region['code']}")

    # Change the baseline map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color="azure")
    ax.add_feature(cfeature.LAND, color="cornsilk",alpha=1)
    ax.add_feature(cfeature.RIVERS, alpha=0.6)
    ax.add_feature(cfeature.LAKES, alpha=0.7)
    # Include the Texts in Plot
    plt.text(1.02, 0.60,f"Canadian Protected and\nConserved Areas Database\n(CPCAD-2023)",
                        #   f"Plot Generated on : {date_time_str}",
            transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='gray', alpha=0.2, edgecolor='none'))

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{plot_name}", dpi=600, bbox_inches='tight')
    # plt.savefig('data/Gov/BC GOV. Conservation and Protected Lands.png', dpi=600, bbox_inches='tight')

# %% [markdown]
# ## ix. Global Wind Atlas (GWA)

# %% [markdown]
# * Pull Windspeed Data

# %%
def create_regional_GWA_data(
        GWA_raster_paths:str,
        filter_params:dict, 
        datafields:list[str],
        bounding_box,
        save_to)-> pd.DataFrame:
    """
    Takes in Global Wind Atlas (GWA) Raster source paths, filtering parameters defined in user config file and the list of datafields 
    that needs to be extracted from raster. Saves the tailored dataframe in given path in local directory.
    """

    # Dictionary to map datafields' names to corresponding colors
    # color_mapping = {'windspeed': 'skyblue', 'CF_IEC2': 'orange', 'CF_IEC3': 'red'}

    # Separate Dataframes will be created based on the given datafields with applicable filters on each of the datafields. Later we will merge the dataframes into one.
    
    gwa_df_combined:pd.DataFrame = None

    for datafield, GWA_raster_path in zip(datafields, GWA_raster_paths):
        if datafield == 'windspeed':
            gwa_df_windspeed = wind.filter_GWA_cells(GWA_raster_path, bounding_box, filter_params['windspeed_low'], filter_params['windspeed_high'], datafield)

        elif datafield == 'CF_IEC2':
            gwa_df_CF_IEC2 = wind.filter_GWA_cells(GWA_raster_path, bounding_box, filter_params['CF_low'], filter_params['CF_high'], datafield)
        
        elif datafield == 'CF_IEC3':
            gwa_df_CF_IEC3 = wind.filter_GWA_cells(GWA_raster_path, bounding_box, filter_params['CF_low'], filter_params['CF_high'], datafield)

    # Merge the DataFrames based on 'x' and 'y' coordinates
    gwa_df_combined = gwa_df_windspeed.copy()  # Start with windspeed DataFrame
    if 'gwa_df_CF_IEC2' in locals():
        gwa_df_combined = pd.merge(gwa_df_combined, gwa_df_CF_IEC2, on=['x', 'y'], how='inner', validate='one_to_one')

    if 'gwa_df_CF_IEC3' in locals():
        gwa_df_combined = pd.merge(gwa_df_combined, gwa_df_CF_IEC3, on=['x', 'y'], how='inner', validate='one_to_one')

    # Save the merged DataFrame to a pickle file
    if gwa_df_combined is not None:
        gwa_df_combined['gwa_cell_id'] = range(1, len(gwa_df_combined) + 1)
        gwa_df_combined.to_pickle(save_to)

    log.info("GWA data and filtered dataset' distribution plots generated")

    return gwa_df_combined


# %% [markdown]
# * Pull IEC Turbine Load Class Mapping data

# %%
def create_IEC_loadClass_mapping_dataframe(
        GWA_IEC_Classes_raster:str,
        bounding_box:gpd.GeoDataFrame,
        GWA_root_direct:str)->pd.DataFrame:
    """
    Description
    """
    IEC_class_raster_path=os.path.join(GWA_root_direct,GWA_IEC_Classes_raster)
    _df_IEC_class:pd.DataFrame = wind.clip_bc_data_from_GWA(IEC_class_raster_path, 'IEC_ExtermeLoad_class', bounding_box)
    gwa_df_IEC_class:pd.DataFrame = wind.IEC_class_from_pixel_values(_df_IEC_class,'IEC_ExtermeLoad_class')

    return gwa_df_IEC_class


# %% [markdown]
# * Visualize IEC Load Class Mapping

# %%
def Create_IEC_load_class_mapping_plot(
        province_gadm_regions_gdf:pd.DataFrame,
        GWA_IEC_Classes_raster:str,
        bounding_box:gpd.GeoDataFrame,
        GWA_root_direct:str,
        plt_save_to:str):
    """
    Description
    """
    if not os.path.exists:
        gwa_df_IEC_class=create_IEC_loadClass_mapping_dataframe(GWA_IEC_Classes_raster,bounding_box,GWA_root_direct)

        gwa_df_IEC_class = gwa_df_IEC_class.loc[:, ['x', 'y', 'IEC_ExtermeLoad_class']]
        none_mask=gwa_df_IEC_class.IEC_ExtermeLoad_class!='none'

        gwa_df_IEC_class=gwa_df_IEC_class[none_mask]
        gwa_df_IEC_class.IEC_ExtermeLoad_class.value_counts()

        geometry = [Point(xy) for xy in zip(gwa_df_IEC_class['x'], gwa_df_IEC_class['y'])]
        gwa_IEC_class_layers_gdf = gpd.GeoDataFrame(gwa_df_IEC_class, geometry=geometry)

        gwa_IEC_class_layers_gdf.crs=province_gadm_regions_gdf.crs
        gwa_IEC_class_layers_gdf=gwa_IEC_class_layers_gdf.clip(province_gadm_regions_gdf,keep_geom_type=False)
        gwa_IEC_class_layers_gdf.plot(column='IEC_ExtermeLoad_class', legend=True,edgecolor='white',linewidth=0.02)

        plt.savefig(plt_save_to,dpi=600)

        log.info(f"{plt_save_to} created")
    else:
        log.info(f"Plot found locally @ {plt_save_to}")


# %% [markdown]
# ## x. Open Street Map (OSM) Data

# %% [markdown]
# * Pull Provincial Data

# %%
def prepare_province_OSM_datafile(
    province_osm_data_root:str,
    province_osm_data_file_userdefined:str):
    
    if os.path.exists(province_osm_data_file_userdefined):
        log.info(f"{province_osm_data_file_userdefined} found locally." )
    else:
        fp = get_data("british_columbia", update=False, directory=province_osm_data_root)
        province_osm_data_file_default = os.path.join(province_osm_data_root,os.path.basename(fp))
        province_osm_data_file_userdefined_path=os.path.join(province_osm_data_root,province_osm_data_file_userdefined)
        # Revise the default file name to a user defined one for easy file navigation

        os.rename(province_osm_data_file_default, province_osm_data_file_userdefined_path)


# %% [markdown]
# * Filter Required Datafields and save locally

# %%
# Function to create buffered GeoDataFrame and save as a parquet file
def create_and_save_buffered_gdf(
        aeroway:gpd.GeoDataFrame, 
        buffer_distance:dict[str,float], 
        VRE_type:str, 
        current_region_code:str, 
        save_path:os.path):
    # Convert the aeroway GeoDataFrame to the appropriate CRS (e.g., UTM zone 33N, EPSG:32633)

    aeroway = aeroway.to_crs(epsg=32633)

    aeroway_with_buffer = aeroway.copy()
    aeroway_with_buffer['buffer'] = aeroway['geometry'].buffer(buffer_distance)
    aeroway_with_buffer = aeroway_with_buffer.to_crs(epsg=4326)
    file_name = f"aeroway_OSM_{current_region_code}_with_buffer_{VRE_type}.parquet"
    file_path = os.path.join(save_path, file_name)
    utils.check_LocalCopy_and_run_function(file_path, aeroway_with_buffer.to_parquet(file_path), force_update=False)
