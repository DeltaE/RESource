"""
Visualization and plotting utilities for renewable energy resource assessment.

This module provides comprehensive visualization tools for displaying renewable energy
assessment results including spatial maps, time series plots, capacity distributions,
economic analysis charts, and interactive dashboards. It supports both static 
publication-quality figures and interactive web-based visualizations.

The visualization tools are designed to facilitate analysis interpretation, result
communication, and workflow debugging through clear, informative graphics that
highlight spatial patterns, temporal variations, and economic trade-offs in
renewable energy development potential.

Key Functions:
    - Spatial mapping: Choropleth maps of resource potential and constraints
    - Time series visualization: Capacity factor profiles and seasonal patterns  
    - Economic analysis: LCOE distributions and cost component breakdowns
    - Cluster visualization: Site groupings and representative characteristics
    - Interactive dashboards: Web-based exploration interfaces
    - Export utilities: High-resolution figure generation for publications

Dependencies:
    - matplotlib/seaborn: Static plotting and publication graphics
    - plotly: Interactive visualizations and dashboards  
    - folium: Web-based interactive maps
    - geopandas: Spatial data visualization
    - xarray: Multi-dimensional data plotting
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import seaborn as sns
import xarray
import xarray as xr
from atlite import ExclusionContainer
from IPython.display import display
from matplotlib import lines as mlines
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    to_rgba,
)
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, RegularPolygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from plotly.subplots import make_subplots
from rasterio.warp import Resampling, calculate_default_transform, reproject

import RES.lands as lands
import RES.utility as utils
import RES.visual_styles as styles

style_path = Path(styles.__file__).parent / "elsevier.mplstyle"
plt.style.use(style_path)# Custom style for publication quality figures


def size_for_legend(mw):
    """
    Calculate bubble size for capacity-based map legends.
    
    Converts megawatt capacity values to appropriate bubble sizes for
    proportional symbol maps, ensuring visual clarity and proper scaling
    across different capacity ranges.
    
    Parameters
    ----------
    mw : float
        Capacity value in megawatts
        
    Returns
    -------
    float
        Scaled bubble size for mapping visualization
        
    Examples
    --------
    >>> size_for_legend(100)  # 100 MW site
    50.0
    >>> size_for_legend(500)  # 500 MW site  
    150.0
    """
    """Calculate the size of the bubble for the legend based on megawatts (MW).
    Args:
        mw (float): The megawatt value to convert to bubble size.
    Returns:
        float: The size of the bubble in points.
    """
    return np.sqrt(mw / 100)  # since s = mw / 100 in scatter

def add_compass_arrow(ax, 
                      x:float=0.9, 
                      y:float=0.9,  
                      fontsize:float=9, 
                      color:str='grey',
                      length:float=0.05, 
                      text_offset:float=0.01,
                      arrow_head_width:float=6,
                      arrow_width=1.5
                      ):
    """
    Adds a simple north arrow to the plot.
    Parameters:
        ax (matplotlib.axes.Axes): The plot axes to annotate.
        x (float): X position in axes fraction coordinates.
        y (float): Y position in axes fraction coordinates.
        length (float): Length of the arrow in axes fraction units.
        text_offset (float): Offset for the 'N' label below the arrow.
    """
    ax.annotate(
        '', 
        xy=(x, y), 
        xytext=(x, y - length),
        xycoords='axes fraction',
        arrowprops=dict(
            facecolor=color,         # Fill color of the arrow head
            edgecolor=color,         # Edge color of the arrow
            width=arrow_width,       # Width of the arrow shaft
            headwidth=arrow_head_width, # Width of the arrow head
            headlength=arrow_head_width * 1.5, # Length of the arrow head
            shrink=0,                # Do not shrink the arrow
            lw=0.5,                  # Line width of the arrow edge
            alpha=0.8,               # Transparency
            linestyle='-',           # Line style
            arrowstyle='|>',        # Arrow style
            mutation_scale=12        # Scale of the arrow head
        )
    )
    ax.text(x, y - length - text_offset, 'N', transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, fontweight='bold', color=color)
def add_compass_arrow_custom(ax, 
                            x: float = 0.9, 
                            y: float = 0.9,  
                            fontsize: float = 9, 
                            color: str = 'grey',
                            length: float = 0.01, 
                            text_offset: float = 0.01,
                            arrow_head_width: float = 6,
                            arrow_border_width: float = 0.5,
                            text: str = 'N'
                            ):
    """
    Alternative version with more arrow head customization.
    Uses the older arrow method for more control over head dimensions.
    """
    # Option 2: Without arrowstyle (allows headwidth/headlength parameters)
    ax.annotate(
        '', 
        xy=(x, y), 
        xytext=(x, y - length),
        xycoords='axes fraction',
        arrowprops=dict(
            facecolor=color,
            edgecolor='k',
            headwidth=arrow_head_width,
            headlength=arrow_head_width * 1.5,
            shrink=0,
            lw=arrow_border_width,
            alpha=0.8,
        )
    )
    # Add the text
    ax.text(x, y - length - text_offset, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, fontweight='bold', color=color)
    
def add_compass_to_plot(ax, x_offset=0.76, y_offset=0.92, size=14, triangle_size=0.02):
    """
    Adds a simple upward-pointing triangle with an 'N' label below it as a North indicator.

    Parameters:
        ax (matplotlib.axes.Axes): The plot axes to annotate.
        x_offset (float): X position in axes fraction coordinates.
        y_offset (float): Y position in axes fraction coordinates.
        size (int): Font size for the 'N' label.
        triangle_size (float): Radius of the triangle (in axes fraction units).
    """
    # Add upward triangle (north arrow)
    triangle = RegularPolygon(
        (x_offset, y_offset),           # center of triangle
        numVertices=3,
        radius=triangle_size,
        orientation=0,                  # pointing up
        transform=ax.transAxes,
        facecolor='grey',
        edgecolor='k',
        lw=0.1
    )
    ax.add_patch(triangle)

    # Add "N" label slightly below the triangle
    ax.text(x_offset, y_offset - triangle_size * 1.5, 'N',
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=size, fontweight='bold',
            color='grey')


def extract_region(col_name:str):
    """
    Extracts the region code from a column name formatted as 'Region_SiteID'.
    Parameters:
        col_name (str): The column name string.
    Returns:
        str: The extracted region code.
    """
    return col_name.split('_')[0]


def plot_region_complementarity(timeseries_solar: pd.DataFrame,
                                timeseries_wind: pd.DataFrame,
                                region_code: str,
                                RUN_ID: str = 'default',
                                region: str | None = None,
                                clusters: bool = True,
                                aggregate: bool = False,   # kept for signature compatibility
                                show: bool = True,
                                font_family: str | None = None,
                                metric: str = "pearson"    # "pearson" or "diff"
                                ):
    """
    Plots complementarity heatmaps for solar and wind.

    - metric="pearson": hour×month Pearson r between (mean across selected sites) solar and wind.
                        Negative r → stronger complementarity (anti-correlation).
    - metric="diff":     normalized (solar - wind) dominance map (your previous visualization).
                         **Keeps the older annotation block** (rectangles & labels).
    If `region` is provided, site columns are filtered by `extract_region(col) == region` before aggregation.
    """

    # --- Global style ---
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set2")
    if font_family is not None:
        plt.rcParams['font.family'] = font_family
    try:
        plt.style.use(style_path)  # optional external style
    except Exception:
        pass

    # --- Diverging cmap for "diff" metric (as in your previous code) ---
    colors_neg = plt.cm.PuBu(np.linspace(0.3, 1, 128))
    colors_pos = plt.cm.OrRd(np.linspace(0.3, 1, 128))
    colors = np.vstack((colors_neg[::-1], colors_pos))
    custom_cmap = LinearSegmentedColormap.from_list('WindSolarDiv', colors)

    # --- Safety: align indices ---
    idx = timeseries_solar.index.intersection(timeseries_wind.index)
    if len(idx) == 0:
        raise ValueError("Solar and wind time indices do not overlap.")
    ts_solar_full = timeseries_solar.loc[idx]
    ts_wind_full  = timeseries_wind.loc[idx]

    # --- Region filter helper ---
    def _select_cols(cols, region_label):
        if region_label is None:
            return list(cols)
        return [c for c in cols if extract_region(c) == region_label]

    solar_cols = _select_cols(ts_solar_full.columns, region)
    wind_cols  = _select_cols(ts_wind_full.columns,  region)
    common_cols = sorted(set(solar_cols).intersection(set(wind_cols)))
    if len(common_cols) == 0:
        raise ValueError(f"No common site columns for region={region!r}. Check column names / extract_region().")

    ts_solar = ts_solar_full[common_cols]
    ts_wind  = ts_wind_full[common_cols]

    # --- Aggregate across selected sites ---
    solar_mean = ts_solar.mean(axis=1)
    wind_mean  = ts_wind.mean(axis=1)

    # --- Pearson r in hour×month bins ---
    def hour_month_corr(a: pd.Series, b: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'solar': a, 'wind': b}).dropna()
        df['hour'] = df.index.hour
        df['month'] = df.index.month

        def _corr(g):
            if (g['solar'].std(ddof=0) == 0) or (g['wind'].std(ddof=0) == 0):
                return np.nan
            return g['solar'].corr(g['wind'])

        r = df.groupby(['hour', 'month'], observed=True).apply(_corr).unstack('month')
        return r.reindex(index=range(24), columns=range(1, 13))

    # --- Build matrix and plotting params ---
    if metric.lower() == "pearson":
        mat = hour_month_corr(solar_mean, wind_mean)              # [-1, 1]
        vmin, vmax, center = -1.0, 1.0, 0.0
        cmap = "coolwarm"
        cbar_label = "Pearson r (negative = complementary)"
        comp_score = np.nanmean(np.clip(-mat.values, 0, 1))       # average negative r (0–1)
        title_metric = "Pearson r"
    elif metric.lower() == "diff":
        # Your dominance metric
        solar_df = pd.DataFrame({'value': solar_mean})
        solar_df['hour'] = solar_df.index.hour
        solar_df['month'] = solar_df.index.month
        wind_df = pd.DataFrame({'value': wind_mean})
        wind_df['hour'] = wind_df.index.hour
        wind_df['month'] = wind_df.index.month

        solar_matrix = solar_df.groupby(['hour','month'], observed=True)['value'].mean().unstack('month')
        wind_matrix  = wind_df.groupby(['hour','month'],  observed=True)['value'].mean().unstack('month')

        solar_norm = solar_matrix / np.nanmax(solar_matrix.values)
        wind_norm  = wind_matrix  / np.nanmax(wind_matrix.values)

        mat = solar_norm - wind_norm
        vmin, vmax, center = -0.8, 0.8, 0.0
        cmap = custom_cmap
        cbar_label = 'Resource Dominance\n(Purple: Wind | Orange: Solar)'
        comp_score = np.nanmean(np.abs(mat.values))
        title_metric = "Solar–Wind Dominance"
    else:
        raise ValueError("metric must be 'pearson' or 'diff'.")

    # --- Plot ---
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(9, 5), dpi=500)
    sns.heatmap(mat, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                linewidths=0, cbar_kws={'label': cbar_label, 'shrink': 0.8}, ax=ax)

    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(month_names, rotation=90, fontsize=9)
    ax.set_yticks(np.arange(0, 24, 4) + 0.5)
    ax.set_yticklabels(np.arange(0, 24, 4), fontsize=9)
    ax.set_xlabel('Month', fontsize=9, fontweight='bold')
    ax.set_ylabel('Hour of Day', fontsize=9, fontweight='bold')

    region_label = region if region is not None else "All Regions"
    ax.set_title(f'{region_label} | {title_metric} | Complementarity Score: {comp_score:.2f}',
                 fontsize=14, fontweight='bold')

    # Annotation badge (shared)
    annotation_custom = 'Representative Clustered Sites Nos:' if clusters else 'ERA5 Cells Nos:'
    ax.text(0.02, 0.97, f'{annotation_custom} {len(common_cols)}', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgrey', alpha=0.7))

    # --- KEEP OLD ANNOTATIONS when metric == "diff" ---
    if metric.lower() == "diff":
        # Peak solar and wind periods (original rectangles & labels)
        ax.add_patch(Rectangle((0,10),12,6, linewidth=1.5, edgecolor='orange',
                               facecolor='none', linestyle='--', alpha=0.8))
        ax.text(6, 8.5, 'Solar Peak\nDay', ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.7))

        ax.add_patch(Rectangle((0,0),12,6, linewidth=1.5, edgecolor='lightblue',
                               facecolor='none', linestyle='--', alpha=0.8))
        ax.text(6, 3, 'Wind Dominant\nNight', ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

        ax.add_patch(Rectangle((0,18),12,6, linewidth=1.5, edgecolor='lightblue',
                               facecolor='none', linestyle='--', alpha=0.8))
        ax.text(6, 21, 'Wind Dominant\nEve', ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

        # Seasonal annotations (kept as in your prior code)
        ax.text(1.5, 28.2, 'Winter', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
        ax.text(5.5, 28.2, 'Summer', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.text(9.5, 28.2, 'Fall', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    if show:
        plt.show()

    return mat, comp_score


# def plot_region_complementarity(timeseries_solar:pd.DataFrame, 
#                                 timeseries_wind:pd.DataFrame,
#                                 region_code:str,
#                                 RUN_ID:str='default',
#                                 region:str=None, 
#                                 clusters: bool=True, 
#                                 aggregate: bool=False,
#                                 show:bool=True,
#                                 font_family:str=None):
#     """
#         Plots complementarity heatmaps for solar and wind resources.

#     Parameters:
#         - timeseries_solar: pd.DataFrame with solar site time series (columns = sites)
#         - timeseries_wind: pd.DataFrame with wind site time series (columns = sites)
#         - region: str or None. If str, plots only that region. If None, plots all regions individually.
#         - clusters: bool, whether using clustered representative sites
#         - aggregate: bool, if True, aggregate all regions into a single heatmap
#         - show: bool, if True, display the plot
#     """
#     regions = sorted(list(set(extract_region(c) for c in timeseries_solar.columns)))
    
#     # --- Global style ---
#     sns.set_theme(style="whitegrid")
#     sns.set_palette("Set2")
    
#     # --- Custom diverging colormap ---
#     colors_neg = plt.cm.PuBu(np.linspace(0.3, 1, 128))
#     colors_pos = plt.cm.OrRd(np.linspace(0.3, 1, 128))
#     colors = np.vstack((colors_neg[::-1], colors_pos))
#     custom_cmap = LinearSegmentedColormap.from_list('WindSolarDiv', colors)

#     plt.style.use(style_path)  
#     if font_family is not None:  
#      plt.rcParams['font.family'] = font_family
     
#     if aggregate:
#         print("Aggregating all regions into a single complementarity plot...")
#         solar_cols = timeseries_solar.columns
#         wind_cols = timeseries_wind.columns

#         # Compute mean time series across all sites
#         solar_mean = timeseries_solar[solar_cols].mean(axis=1)
#         wind_mean = timeseries_wind[wind_cols].mean(axis=1)

#         # Hour & month
#         solar_df = pd.DataFrame({'value': solar_mean})
#         solar_df['hour'] = solar_df.index.hour
#         solar_df['month'] = solar_df.index.month
#         wind_df = pd.DataFrame({'value': wind_mean})
#         wind_df['hour'] = wind_df.index.hour
#         wind_df['month'] = wind_df.index.month

#         # Aggregate & normalize
#         solar_matrix = solar_df.groupby(['hour','month'])['value'].mean().unstack()
#         wind_matrix = wind_df.groupby(['hour','month'])['value'].mean().unstack()
#         solar_norm = solar_matrix / solar_matrix.max().max()
#         wind_norm = wind_matrix / wind_matrix.max().max()
#         diff_matrix = solar_norm - wind_norm

#         # Complementarity score
#         complementarity_score = np.abs(diff_matrix).mean().mean()
#         num_sites = len(solar_cols)

#         # Plot
#         fig, ax = plt.subplots(figsize=(9,5), dpi=500)
#         sns.heatmap(diff_matrix,
#                     cmap=custom_cmap,
#                     center=0,
#                     linewidths=0,
#                     vmin=-0.8, vmax=0.8,
#                     cbar_kws={'label':'Resource Dominance\n(Purple: Wind | Orange: Solar)', 'shrink':0.8},
#                     ax=ax)

#         # Axes
#         month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#         ax.set_xticks(np.arange(12)+0.5)
#         ax.set_xticklabels(month_names, rotation=90, fontsize=9)
#         ax.set_yticks(np.arange(0,24,4)+0.5)
#         ax.set_yticklabels(np.arange(0,24,4), fontsize=9)
#         ax.set_xlabel('Month',fontsize=9,fontweight='bold' )
#         ax.set_ylabel('Hour of Day',fontsize=9,fontweight='bold')

#         # Annotations
#         ax.set_title(f'All Regions | Complementarity Score: {complementarity_score:.2f}', fontsize=14, fontweight='bold')
#         annotation_custom = 'Representative Clustered Sites Nos:' if clusters else 'ERA5 Cells Nos:'
#         ax.text(0.1, 0.2, f'{annotation_custom} {num_sites}', 
#                 ha='left', va='top', fontsize=8, fontweight='bold', color='black',
#                 bbox=dict(boxstyle='round,pad=0.1', facecolor='lightgrey', alpha=0.7))
#         # Peak solar and wind periods
#         ax.add_patch(Rectangle((0,10),12,6, linewidth=1.5, edgecolor='orange', facecolor='none', linestyle='--', alpha=0.8))
#         ax.text(6, 8.5, 'Solar Peak\nDay', ha='center', va='center', fontsize=7, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.7))

#         ax.add_patch(Rectangle((0,0),12,6, linewidth=1.5, edgecolor='lightblue', facecolor='none', linestyle='--', alpha=0.8))
#         ax.text(6, 3, 'Wind Dominant\nNight', ha='center', va='center', fontsize=7, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

#         ax.add_patch(Rectangle((0,18),12,6, linewidth=1.5, edgecolor='lightblue', facecolor='none', linestyle='--', alpha=0.8))
#         ax.text(6, 21, 'Wind Dominant\nEve', ha='center', va='center', fontsize=7, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

#         # Seasonal annotations
#         ax.text(1.5, 28.2, 'Winter', ha='center', va='center', fontsize=10,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
#         ax.text(5.5, 28.2, 'Summer', ha='center', va='center', fontsize=10,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
#         ax.text(9.5, 28.2, 'Fall', ha='center', va='center', fontsize=10,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
#         plt.tight_layout()
#         plt.show()
#         return

    # Else plot individually or selected region
    if region:
        regions = [region]
        print(f"Plotting only for region: {region}")
    else:
        print(f"Plotting for all regions: {regions}")

    for region_name in regions:
        # Region-specific columns
        region_cols_solar = [c for c in timeseries_solar.columns if extract_region(c) == region_name]
        region_cols_wind = [c for c in timeseries_wind.columns if extract_region(c) == region_name]
        
        if not region_cols_solar or not region_cols_wind:
            print(f"Skipping {region_name} (no matching columns).")
            continue

        # Compute mean time series
        solar_mean = timeseries_solar[region_cols_solar].mean(axis=1)
        wind_mean = timeseries_wind[region_cols_wind].mean(axis=1)

        # Hour & month
        solar_df = pd.DataFrame({'value': solar_mean})
        solar_df['hour'] = solar_df.index.hour
        solar_df['month'] = solar_df.index.month
        wind_df = pd.DataFrame({'value': wind_mean})
        wind_df['hour'] = wind_df.index.hour
        wind_df['month'] = wind_df.index.month

        # Aggregate & normalize
        solar_matrix = solar_df.groupby(['hour','month'])['value'].mean().unstack()
        wind_matrix = wind_df.groupby(['hour','month'])['value'].mean().unstack()
        solar_norm = solar_matrix / solar_matrix.max().max()
        wind_norm = wind_matrix / wind_matrix.max().max()
        diff_matrix = solar_norm - wind_norm

        # Complementarity score
        complementarity_score = np.abs(diff_matrix).mean().mean()
        num_sites = len(region_cols_solar)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(9,5), dpi=500)

        sns.heatmap(diff_matrix,
                    cmap=custom_cmap,
                    center=0,
                    linewidths=0,
                    vmin=-0.8, vmax=0.8,
                    cbar_kws={'label':'Resource Dominance\n(Purple: Wind | Orange: Solar)', 'shrink':0.8},
                    ax=ax)

        # Axes
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        ax.set_xticks(np.arange(12)+0.5)
        ax.set_xticklabels(month_names, rotation=90, fontsize=9)
        ax.set_yticks(np.arange(0,24,4)+0.5)
        ax.set_yticklabels(np.arange(0,24,4), fontsize=9)
        ax.set_xlabel('Month',fontsize=9,fontweight='bold' )
        ax.set_ylabel('Hour of Day',fontsize=9,fontweight='bold')

        # Peak solar and wind periods
        ax.add_patch(Rectangle((0,10),12,6, linewidth=1.5, edgecolor='orange', facecolor='none', linestyle='--', alpha=0.8))
        ax.text(6, 8.5, 'Solar Peak\nDay', ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.7))

        ax.add_patch(Rectangle((0,0),12,6, linewidth=1.5, edgecolor='lightblue', facecolor='none', linestyle='--', alpha=0.8))
        ax.text(6, 3, 'Wind Dominant\nNight', ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

        ax.add_patch(Rectangle((0,18),12,6, linewidth=1.5, edgecolor='lightblue', facecolor='none', linestyle='--', alpha=0.8))
        ax.text(6, 21, 'Wind Dominant\nEve', ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

        # Seasonal annotations
        ax.text(1.5, 28.2, 'Winter', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
        ax.text(5.5, 28.2, 'Summer', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.text(9.5, 28.2, 'Fall', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        
        # Annotation: Number of common sites
        if clusters:
                annotation_custom = 'Representative Clustered Sites Nos:'
        else:
                annotation_custom = 'ERA5 Cells Nos:'
        ax.text(0.1, 0.2, f'{annotation_custom} {num_sites}', 
                ha='left', va='top', fontsize=8, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='lightgrey', alpha=0.7))
        # Title
        ax.set_title(f'{region_name} | Complementarity Score: {complementarity_score:.2f}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'../vis/{region_code}/complementarity_{region_name}_{RUN_ID}.png', dpi=300)
        print(f"Saved figure for {region_name} with complementarity score {complementarity_score:.2f}")
        if show:
           plt.show()


def plot_resources_scatter_metric_combined(
    solar_clusters:pd.DataFrame,
    wind_clusters:pd.DataFrame,
    bubbles_GW:list= [1, 5, 10],
    bubbles_scale:float= 0.4,
    lcoe_threshold:float= 200,
    font_family=None,
    figsize=(3.5, 2.5),
    dpi= 1000, # this falls under lineart
    save_to_root:str='vis',
    set_transparent:bool=False,
    
):
    """
    Plot combined scatter metrics for solar and wind resources.

    Args:
        solar_clusters (pd.DataFrame): DataFrame containing solar cluster data.
        wind_clusters (pd.DataFrame): DataFrame containing wind cluster data.
        bubbles_GW (list, optional): List of bubble sizes in GW. Defaults to [1, 5, 10].
        bubbles_scale (float, optional): Scaling factor for bubble sizes. Defaults to 0.4.
        lcoe_threshold (float, optional): LCOE threshold for filtering. Defaults to 200.
        font_family (str, optional): Font family for the plot. Defaults to 'sans-serif'.
        save_to_root (str, optional): Directory to save the plot. Defaults to 'vis'.
        set_transparent (bool, optional): Whether to set the background transparent. Defaults to False.
    """

    plt.style.use(style_path)  
    if font_family is not None:  
     plt.rcParams['font.family'] = font_family
     
    # Filter by LCOE threshold
    solar = solar_clusters[solar_clusters['lcoe'] <= lcoe_threshold]
    wind = wind_clusters[wind_clusters['lcoe'] <= lcoe_threshold]

    fig, ax = plt.subplots(figsize=figsize,dpi=dpi)

    # Solar scatter
    ax.scatter(
        solar['CF_mean'],
        solar['lcoe'],
        s=solar['potential_capacity']*bubbles_scale,  # Scale down for better visibility
        alpha=0.7,
        c='darkorange',
        edgecolors='w',
        linewidth=0.5,
        label='Solar'
    )

    # Wind scatter
    ax.scatter(
        wind['CF_mean'],
        wind['lcoe'],
        s=wind['potential_capacity']*bubbles_scale,  # Scale down for better visibility
        alpha=0.7,
        c='purple',
        edgecolors='w',
        linewidth=0.5,
        label='Wind'
    )

    ax.set_xlabel('Average Capacity Factor', fontweight='bold')
    ax.set_ylabel('Relative Cost Score ($/MWh)', fontweight='bold')
    ax.set_title('CF vs Score for Solar and Wind resources', fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Bubble size legend
    size_labels = bubbles_GW  # GW
    size_values = [s * 1000 for s in size_labels]
    legend_handles = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                      markersize=np.sqrt(size*bubbles_scale), alpha=0.7, label=f'{label} GW')
        for size, label in zip(size_values, size_labels)
    ]
    # Resource legend
    resource_handles = [
        mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='Solar'),
        mlines.Line2D([], [], color='purple', marker='o', linestyle='None', label='Wind')
    ]

    ax.legend(handles=legend_handles + resource_handles, loc='upper right', framealpha=0,)

    ax.grid(True, ls=":", linewidth=0.3)
    # Add note below axes using figtext
    fig.text(0.5, -0.03,
            "Note: The Scoring is calculated to reflect Dollar investment required to get an unit of Energy yield (MWh). "
            "\nTo reflect market competitiveness and incentives, the Score ($/MWh) needs financial adjustment factors to be considered on top of it.",
            ha='center', va='bottom', fontsize=7, color='gray',
            wrap=True,
            bbox=dict(facecolor='None', linewidth=0.2, edgecolor='grey', boxstyle='round,pad=0.3'))


    plt.tight_layout()
        
    save_to_root = Path(save_to_root)
    save_to_root.mkdir(parents=True, exist_ok=True)
    file_path = save_to_root / "Resources_CF_vs_LCOE_combined.png"
    plt.savefig(file_path,transparent=set_transparent)
    utils.print_update(level=1, message=f"Combined CF vs LCOE plot created and saved to: {file_path}")
    # return fig


def get_CF_wind_check_plot(cells: gpd.GeoDataFrame,
                           gwa_raster_data: xarray.DataArray,
                           boundary: gpd.GeoDataFrame,
                           region_code: str,
                           region_name: str,
                           columns: list,
                           figure_height: int = 7,
                           font_family:str=None,
                           save_to: str | Path = None):
    """
    Plots GWA benchmark (left), CF_IEC3 (middle), and wind_CF_mean (right).
    """
      # assumes vis.add_compass_to_plot() exists
    plt.style.use(style_path)  
    if font_family is not None:  
        plt.rcParams['font.family'] = font_family
    assert len(columns) == 2, "Expected exactly two columns: CF_IEC3 and wind_CF_mean"
    col_mid, col_right = columns

    # Color scale
    vmin = cells[columns].min().min()
    vmax = cells[columns].max().max()

    # Layout: 1 row × 3 columns
    fig = plt.figure(figsize=(13, figure_height), constrained_layout=True,dpi=500)
    spec = GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1], figure=fig)


    axes = []

    # LEFT: GWA benchmark
    ax_gwa = fig.add_subplot(spec[0, 0])
    gwa_raster_data.plot(ax=ax_gwa, cmap='BuPu', vmin=vmin, vmax=vmax, add_colorbar=False)
    boundary.plot(ax=ax_gwa, facecolor='none', edgecolor='white', linewidth=0.5)
    ax_gwa.set_title('GWA CF-IEC3 Reference (High-res)', fontsize=11)
    ax_gwa.axis('off')
    axes.append(ax_gwa)

    # MIDDLE: CF_IEC3
    ax_mid = fig.add_subplot(spec[0, 1])
    shadow_offset = 0.02
    cells_shadow = cells.copy()
    cells_shadow['geometry'] = cells_shadow['geometry'].translate(xoff=-shadow_offset, yoff=shadow_offset)
    cells_shadow.plot(column=col_mid, cmap='Greys', ax=ax_mid, edgecolor='white', alpha=1, linewidth=0.2, zorder=1)

    cells.plot(
        column=col_mid,
        ax=ax_mid,
        cmap='BuPu',
        vmin=vmin, vmax=vmax,
        linewidth=0.2,
        legend=False
    )
    ax_mid.set_title(col_mid.replace('_', ' '), fontsize=10)
    ax_mid.axis('off')
    axes.append(ax_mid)

    # RIGHT: wind_CF_mean
    ax_right = fig.add_subplot(spec[0, 2])
    col = col_right
    cells_shadow = cells.copy()
    cells_shadow['geometry'] = cells_shadow['geometry'].translate(xoff=-shadow_offset, yoff=shadow_offset)
    cells_shadow.plot(column=col, cmap='Greys', ax=ax_right, edgecolor='white', alpha=1, linewidth=0.2, zorder=1)

    cells.plot(
        column=col,
        ax=ax_right,
        cmap='BuPu',
        vmin=vmin, vmax=vmax,
        linewidth=0.2,
        legend=False
    )
    ax_right.set_title(col.replace('_', ' '), fontsize=10)
    ax_right.axis('off')
    axes.append(ax_right)
    add_compass_to_plot(ax_right)

    # Unified colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap='BuPu', norm=norm)
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.02,shrink=0.6)
    cbar.set_label('Capacity Factor', fontsize=11)

    # Title and notes
    plt.suptitle(f"Wind Capacity Factor Comparison for {region_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.figtext(
        0.01, 0.01,
        "* CF_IEC3 is rescaled from GWA to ERA5 resolution.\n"
        "* wind_CF_mean is computed using atlite (ERA5 adjusted with GWA).",
        ha='left', fontsize=9, color='gray'
    )

    plt.rcParams['font.family']=font_family


    if save_to is not None:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        save_to_file = save_to / "Wind_CF_comparison.png"
        plt.savefig(save_to_file, dpi=300, bbox_inches='tight', transparent=False)
        utils.print_update(level=1, message=f"Wind CF comparison plot created and saved to: {save_to_file}")
    # Summary table
    display(cells[columns].describe().style.format(precision=2).set_caption("Summary Statistics for CF_IEC3 and calibrated Wind CF_mean"))


def plot_resources_scatter_metric(resource_type:str,
                                  clusters_resources:gpd.GeoDataFrame,
                                  lcoe_threshold:float=999,
                                  color=None,
                                  save_to_root:str|Path='vis'):
    """
    Generate a scatter plot visualizing the relationship between Capacity Factor (CF) and Levelized Cost of Energy (LCOE) 
    for renewable energy resources (solar or wind). The plot highlights clusters of resources based on their potential capacity.
    Args:
        resource_type (str): The type of renewable resource to plot. Must be either 'solar' or 'wind'.
        clusters_resources (gpd.GeoDataFrame): A GeoDataFrame containing resource cluster data. 
            Expected columns include:
                - 'CF_mean': Average capacity factor of the resource cluster.
                - 'lcoe': Levelized Cost of Energy for the resource cluster.
                - 'potential_capacity': Potential capacity of the resource cluster (used for bubble size).
        lcoe_threshold (float): The maximum LCOE value to include in the plot. Clusters with LCOE above this threshold are excluded.
        color (optional): Custom color for the scatter plot bubbles. Defaults to 'darkorange' for solar and 'navy' for wind.
        save_to_root (str | Path, optional): Directory path where the plot image will be saved. Defaults to 'vis'.
    Returns:
        None: The function saves the generated plot as a PNG image in the specified directory.
    Notes:
        - The size of the bubbles in the scatter plot represents the potential capacity of the resource clusters.
        - The x-axis (CF_mean) is formatted as percentages for better readability.
        - A legend is included to indicate the bubble sizes in gigawatts (GW).
        - The plot includes an annotation explaining the scoring methodology for LCOE.
        - The plot is saved as a transparent PNG image with a resolution of 600 dpi.
    Example:
        >>> plot_resources_scatter_metric(
        ...     resource_type='solar',
        ...     clusters_resources=solar_clusters_gdf,
        ...     lcoe_threshold=50,
        ...     save_to_root='output/plots'
        ... )
     
    """
    
    resource_type=resource_type.lower()
    save_to_root=Path(save_to_root)
    clusters_resources=clusters_resources[clusters_resources['lcoe']<=lcoe_threshold]
    bubble_color= 'darkorange' if resource_type=='solar' else 'navy'
    
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter( 
        clusters_resources['CF_mean'],
        clusters_resources['lcoe'],
        s=clusters_resources['potential_capacity'] / 100,  # Adjust the size for better visualization
        alpha=0.7,
        c=bubble_color,
        edgecolors='w',
        linewidth=0.5
    )

    # Set labels and title
    ax.set_xlabel(f'Average Capacity Factor for {resource_type.capitalize()} resources', fontweight='bold')
    ax.set_ylabel('Score ($/MWh)', fontweight='bold')
    ax.set_title(f'CF vs Score for {resource_type.capitalize()} resources')

    # Customize x-axis ticks to show more levels and as percentages
    ax.xaxis.set_major_locator(MultipleLocator(0.01 if resource_type=='solar' else 0.04))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))

    size_labels = [1, 5, 10]  # GW
    size_values = [s * 1000 for s in size_labels]  # Convert GW to same scale as scatter

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    legend_handles = [
        mlines.Line2D([], [], color=bubble_color, marker='o', linestyle='None',
                      markersize=np.sqrt(size / 100), alpha=0.7,label=f'{label} GW')
        for size, label in zip(size_values, size_labels)
    ]

    ax.legend(handles=legend_handles, loc='upper right', framealpha=0, prop={'size': 12, 'weight': 'bold'})

    # Remove all grids
    ax.grid(True,ls=":",linewidth=0.3)
    # Add annotation to the figure
    fig.text(0.5, -0.04, 
         "Note: The Scoring is calculated to reflect Dollar investment required to get an unit of Energy yield (MWh). "
         "\nTo reflect market competitiveness and incentives, the Score ($/MWh) needs financial adjustment factors to be considered on top of it.",
         ha='center', va='center', fontsize=9.5, color='gray', bbox=dict(facecolor='None', linewidth=0.2,edgecolor='grey', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the plot as a transparent image with 600 dpi
    save_to_root.mkdir(parents=True, exist_ok=True)
    file_path=save_to_root/f"Resources_CF_vs_LCOE_{resource_type}.png"
    
    plt.savefig(file_path, dpi=600, transparent=True)
    utils.print_update(level=1,message=f"CF vs LCOE plot for {resource_type} resources created and saved to : {file_path}")
    return fig
    
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

# def get_selected_vs_missed_visuals(cells: gpd.GeoDataFrame,
#                                   province_short_code,
#                                   resource_type,
#                                    lcoe_threshold: float,
#                                    CF_threshold: float,
#                                    capacity_threshold: float,
#                                    text_box_x=.4,
#                                    text_box_y=.95,
#                                    title_y=1,
#                                    title_x=0.6,
#                                    font_size=10,
#                                    dpi=1000,
#                                    figsize=(12, 7),
#                                    save=False):
#     """Generate visualizations for selected vs missed cells.

#     Args:
#         cells (gpd.GeoDataFrame): GeoDataFrame containing cell data.
#         province_short_code (str): Short code for the province.
#         resource_type (str): Type of renewable resource (e.g., 'solar', 'wind').
#         lcoe_threshold (float): _description_
#         CF_threshold (float): _description_
#         capacity_threshold (float): _description_
#         text_box_x (float, optional): _description_. Defaults to .4.
#         text_box_y (float, optional): _description_. Defaults to .95.
#         title_y (int, optional): _description_. Defaults to 1.
#         title_x (float, optional): _description_. Defaults to 0.6.
#         font_size (int, optional): _description_. Defaults to 10.
#         dpi (int, optional): _description_. Defaults to 1000.
#         figsize (tuple, optional): _description_. Defaults to (12, 7).
#         save (bool, optional): _description_. Defaults to False.
#     """
#     mask=(cells[f'{resource_type}_CF_mean']>=CF_threshold)&(cells[f'potential_capacity_{resource_type}']>=capacity_threshold)&(cells[f'lcoe_{resource_type}']<=lcoe_threshold)
#     filtered_cells=cells[mask]
    
#     # Create a high-resolution side-by-side plot in a 2x2 grid
#     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)

#     # Define the message
#     msg = (f"Cell thresholds @ lcoe >= {lcoe_threshold} $/kWH, "
#            f"CF >={CF_threshold}, MW >={capacity_threshold}")



#     # First plot: CF_mean Visualization (top left)
#     plot_with_matched_cells(axs[0, 0], cells, filtered_cells, f'{resource_type}_CF_mean', 'YlOrRd', 
#                             background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
#     axs[0, 0].set_title('CF_mean Overview', fontsize=font_size)
#     axs[0, 0].set_xlabel('Longitude', fontsize=font_size-3)
#     axs[0, 0].set_ylabel('Latitude', fontsize=font_size-3)
#     axs[0, 0].set_axis_off()

#     # Second plot: Potential Capacity Visualization (top right)
#     plot_with_matched_cells(axs[0, 1], cells, filtered_cells, f'potential_capacity_{resource_type}', 'Blues',
#                             background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
#     axs[0, 1].set_title('Potential Capacity Overview', fontsize=font_size)
#     axs[0, 1].set_xlabel('Longitude', fontsize=font_size-3)
#     axs[0, 1].set_ylabel('Latitude', fontsize=font_size-3)
#     axs[0, 1].set_axis_off()

#     # Third plot: Nearest Station Distance Visualization (bottom left)
#     plot_with_matched_cells(axs[1, 0], cells, filtered_cells, f'nearest_station_distance_km', 'coolwarm',
#                             background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
#     axs[1, 0].set_title('Nearest Station Distance Overview', fontsize=font_size)
#     axs[1, 0].set_xlabel('Longitude', fontsize=font_size-3)
#     axs[1, 0].set_ylabel('Latitude', fontsize=font_size-3)
#     axs[1, 0].set_axis_off()

#     # Fourth plot: LCOE Visualization (bottom right)
#     plot_with_matched_cells(axs[1, 1], cells, filtered_cells, f'lcoe_{resource_type}', 'summer',
#                             background_cell_linewidth=0.2, selected_cells_linewidth=0.5,font_size=font_size-3)
#     axs[1, 1].set_title('LCOE Overview', fontsize=font_size)
#     axs[1, 1].set_xlabel('Longitude', fontsize=font_size-3)
#     axs[1, 1].set_ylabel('Latitude', fontsize=font_size-3)
#     axs[1, 1].set_axis_off()

#     # Add a super title for the figure
#     fig.suptitle(f'{resource_type}- Selected Cells Overview - {province_short_code}', fontsize=font_size+2,fontweight='bold', x=title_x,y=title_y)
#     # Add a text box with grey background for the message
#     fig.text(text_box_x, text_box_y, msg, ha='center', va='top', fontsize=font_size-3,
#              bbox=dict(facecolor='lightgrey', edgecolor='grey', boxstyle='round,pad=0.2'))
#     plt.tight_layout()
#     # Save the plot
#     if save:
#         plt.savefig(f"vis/linking/solar/Selected_cells_solar_{province_short_code}.png", bbox_inches='tight')
#     plt.tight_layout()
#     plt.show()  # Optional: Show the plot if desired


def create_raster_image_with_legend(
        raster:str, 
        cmap:str, 
        title:str, 
        plot_save_to:str):
    """Creates a raster image with a legend for land classes."""
    
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

def plot_data_in_GADM_regions(
        dataframe,
        data_column_df,
        gadm_regions_gdf,
        color_map,
        dpi,
        plt_title,
        plt_file_name,
        vis_directory):
    """
    Plots data from a DataFrame on GADM regions using GeoPandas and Matplotlib.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data to plot.
        data_column_df (str): Name of the column in the DataFrame to plot.
        gadm_regions_gdf (gpd.GeoDataFrame): GeoDataFrame containing the GADM regions.
        color_map (str): Name of the color map to use for the plot.
        dpi (int): Dots per inch for the plot.
        plt_title (str): Title of the plot.
        plt_file_name (str): File name for saving the plot.
        vis_directory (str): Directory for saving the visualization.
    """
    
    ax = dataframe.plot(column=data_column_df, edgecolor='white',linewidth=0.2,legend=True,cmap=color_map)
    gadm_regions_gdf.plot(ax=ax, alpha=0.6, color='none', edgecolor='k', linewidth=0.7)
    ax.set_title(plt_title)
    plt_save_to=os.path.join(vis_directory,plt_file_name)
    plt.tight_layout()
    plt.savefig(plt_save_to,dpi=dpi)
    plt.close()


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


def create_timeseries_plots_solar(cells_df,CF_timeseries_df, dissolved_indices,max_solar_capacity,resampling,solar_vis_directory):
    """ Generates time series plots for solar capacity factor (CF) data.
    Args:
        cells_df (pd.DataFrame): DataFrame containing cell information.
        CF_timeseries_df (pd.DataFrame): DataFrame containing capacity factor time series data.
        dissolved_indices (dict): Dictionary mapping regions and cluster numbers to indices in CF_timeseries_df.
        max_solar_capacity (float): Maximum solar capacity for investment.
        resampling (str): Resampling frequency for the time series data.
        solar_vis_directory (str): Directory to save the generated plots.
    """

    print(f">>> Generating CF timeseries for TOP Sites for {max_solar_capacity} GW Capacity Investment ...")
    
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

    print(f">>> Plots generated for CF timeseries of TOP Sites for {max_solar_capacity} GW Capacity Investment...")
    
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
    
def get_data_in_map_plot(cells, 
                    resource_type:str=None,
                    datafield:str=None,
                    title:str=None,
                    ax=None,
                    compass_size:float=10,
                    font_family:str=None,
                    score_threshold:float=200,
                    show=True):
    
    """
    Plots a map of renewable energy resources (solar or wind) with capacity factor, potential capacity, or LCOE.
    Args:
        cells (gpd.GeoDataFrame): GeoDataFrame containing the resource data.
        resource_type (str, optional): Type of renewable resource ('solar' or 'wind'). Defaults to None.
        datafield (str, optional): Data field to plot ('CF', 'CAPACITY', or 'SCORE'). Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created. Defaults to None.
        compass_size (float, optional): Size of the compass in the plot. Defaults to 10.
        font_family (str, optional): Font family for text in the plot. Defaults to 'sans-serif'.
        discalimers (bool, optional): Whether to include disclaimers in the plot. Defaults to False.
        show (bool, optional): Whether to display the plot. Defaults to True.
    Returns:
        ax (matplotlib.axes.Axes): The axes with the plotted map.
    """
    
    plt.style.use(style_path)  
    if font_family is not None:  
     plt.rcParams['font.family'] = font_family
     
    column_keyword=datafield.upper()
    resource_type = resource_type.lower()
    
    columns={'CF':f"{resource_type}_CF_mean", 
             'CAPACITY':f"potential_capacity_{resource_type}", 
             'SCORE':f"lcoe_{resource_type}"}
    
    legend_labels = {
        'CF': f'{resource_type.capitalize()} Capacity Factor (annual mean)',
        'CAPACITY': f'{resource_type.capitalize()} Potential Capacity (MW)',
        'SCORE': f'{resource_type.capitalize()} Relative Cost Score ($/MWh)'
    }
    
    if column_keyword is not None:
        if column_keyword not in columns.keys():
            raise ValueError("datafield must be one of 'CF', 'CAPACITY', or 'LCOE'.\n Given datafield need not to be case sensitive")

    if resource_type is not None:
        if resource_type not in ['solar', 'wind']:
            raise ValueError("resource_type must be either 'solar' or 'wind'.\n Given resource_type need not to be case sensitive")
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))  # fallback if no ax passed
            else:
                fig = ax.figure

            cmap = 'YlOrRd' if resource_type == 'solar' else 'BuPu'
            column = columns[column_keyword]
            if column_keyword == 'SCORE':
                cells=cells[cells[column]<=score_threshold]
            
            vmin = cells[column].min()
            vmax = cells[column].max()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # Shadow layer
            shadow_offset = 0.016
            cells_shadow = cells.copy()
            cells_shadow['geometry'] = cells_shadow['geometry'].translate(xoff=-shadow_offset, yoff=shadow_offset)
            cells_shadow.plot(column=column, cmap='Greys', ax=ax, edgecolor='white', alpha=1, linewidth=0.2, zorder=1)

            # Main layer

            
            cells.plot(column=column, cmap=cmap, ax=ax, edgecolor='k', alpha=1, linewidth=0.15, zorder=2)

            # Colorbar
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
            cbar.set_label(legend_labels[column_keyword], fontsize=10)
            cbar.ax.tick_params(labelsize=12)
            # Set font weight for colorbar tick labels
            for label in cbar.ax.get_yticklabels():
                label.set_fontweight('bold')

            if title is not None:
                ax.set_title(title, fontsize=14, fontweight='bold', loc='center')
            else:
                ax.set_title(f'{resource_type.capitalize()} Resources', fontsize=14, fontweight='bold', loc='center')
            ax.set_axis_off()
            if resource_type=='solar':
                utils.print_update(level=2,message= "Please cross check with Solar CF map with GLobal Solar Atlas Data from : https://globalsolaratlas.info/download/country_name")
            # if column_keyword == 'SCORE':
            #     # Add disclaimer text at the bottom of the plot
            #     ax.text(
            #         0.5, 0,
            #         f"Note: The Scoring is calculated to reflect Dollar investment required to get an unit of Energy yield (MWh).\nTo reflect market competitiveness and incentives, the Score ($/MWh) needs financial adjustment factors to be considered on top of it.\nScore Higher than {score_threshold} $/MWh are assumed to be not feasible and not shown in this map.",
            #         transform=ax.transAxes, ha='center', va='top', fontsize=10, color='gray'
            #     )
            if show:
                plt.show()

            # add_compass_to_plot(ax, size=compass_size, triangle_size=0.014)
        
    return ax

def plot_grid_lines(
    region_code: str,
    region_name: str,
    lines: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    font_family: str = None,
    figsize: tuple = (10, 8),
    dpi=500,
    save_to: str | Path = None,
    show: bool = True,
):
    """
    Plots transmission lines with binned voltage levels in a specified region.
    """
    lines = lines.copy() # Avoid modifying the original GeoDataFrame
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.suptitle("Transmission Lines by Voltage Levels", fontsize=16, fontweight='bold')
    plt.style.use(style_path)
    if font_family is not None:
        plt.rcParams['font.family'] = font_family

    boundary.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=1, alpha=0.1)

    if 'voltage' in lines.columns:
        # Convert to numeric
        lines['voltage_kv'] = pd.to_numeric(lines['voltage'], errors='coerce') / 1000

        # Define voltage bins
        bins = [0, 12, 25, 132, 220, float("inf")]
        labels = ["<12 kV", "12–25 kV", "25–132 kV", "132–220 kV", "≥220 kV"]
        lines['voltage_class'] = pd.cut(lines['voltage_kv'], bins=bins, labels=labels, right=False)

        # Color map (enough distinct colors)
        cmap = plt.colormaps.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(labels))]
        color_map = {label: colors[i] for i, label in enumerate(labels)}


        # Plot by class
        for label in labels:
            mask = lines['voltage_class'] == label
            if mask.any():
                lines[mask].plot(ax=ax, color=color_map[label], linewidth=1, alpha=0.8)

        # Legend
        legend_patches = [mpatches.Patch(color=color_map[label], label=label) for label in labels if label in lines['voltage_class'].unique()]
        ax.legend(handles=legend_patches, frameon=False, fontsize=11, loc='upper right')

    else:
        lines.plot(ax=ax, color='blue', linewidth=1,alpha=0.7)

    ax.set_axis_off()
    plt.tight_layout()

    if save_to is None:
        save_to = Path("vis") / region_code / "network"
    else:
        save_to = Path(save_to)
    
    save_to.mkdir(parents=True, exist_ok=True)
    save_to_file = save_to / f"transmission_lines_{region_code}.png"
    plt.savefig(save_to_file, bbox_inches='tight', dpi=300)
    
    utils.print_update(level=2, message=f"Transmission Lines for {region_name} saved to {save_to_file}")
    if show:
        plt.show()

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
    """
    Creates an interactive map with key data for a specific province, including regions, conservation lands, aeroways, and bus nodes.

    Args:
        province_gadm_regions_gdf (gpd.GeoDataFrame): GeoDataFrame containing the province's administrative regions.
        provincial_conservation_protected_lands (gpd.GeoDataFrame): GeoDataFrame containing conservation and protected lands.
        aeroway_with_buffer_solar (gpd.GeoDataFrame): GeoDataFrame containing solar aeroways with buffer zones.
        aeroway_with_buffer_wind (gpd.GeoDataFrame): GeoDataFrame containing wind aeroways with buffer zones.
        aeroway (gpd.GeoDataFrame): GeoDataFrame containing aeroways.
        provincial_bus_gdf (gpd.GeoDataFrame): GeoDataFrame containing provincial bus routes.
        current_region (dict): Dictionary containing information about the current region.
        about_OSM_data (dict[dict]): Dictionary containing information about OSM data.
        map_html_save_to (str): _description_
    """
    buffer_distance_m:dict[dict]=about_OSM_data['aeroway_buffer']
    
    m = province_gadm_regions_gdf.explore('Region', color='grey',style_kwds={'fillOpacity': 0.1}, name=f"{current_region['code']} Regions")
    provincial_conservation_protected_lands.explore(m=m,color='red', style_kwds={'fillOpacity': 0.05}, name='Conservation and Protected lands')
    aeroway_with_buffer_solar.explore(m=m, color='orange', style_kwds={'fillOpacity': 0.5}, name=f"aeroway with {buffer_distance_m['solar']}m buffer")
    aeroway_with_buffer_wind.explore(m=m, color='skyblue', style_kwds={'fillOpacity': 0.5}, name=f"aeroway with {buffer_distance_m['wind']}m buffer")
    aeroway.explore(m=m,color='blue', marker_kwds={'radius': 2}, name='aeroway')
    provincial_bus_gdf.explore(m=m, color='black', style_kwds={'fillOpacity': 0.5}, name=f"{current_region['code']} Grid Nodes")


    # Add layer control
    folium.LayerControl().add_to(m)

    # Display the map
    m.save(map_html_save_to)
    

def create_sites_ts_plots_all_sites(
    resource_type:str,
    CF_ts_df:pd.DataFrame,
    save_to_dir:str):
    """
    Creates an interactive timeseries plot for the top sites of a given resource type.
    Args:
        resource_type (str): The type of resource (e.g., 'solar', 'wind').
        CF_ts_df (pd.DataFrame): DataFrame containing the capacity factor timeseries data.
        save_to_dir (str): Directory to save the plot.
    """
    
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


def get_conservation_lands_plot(CPCAD_actual:gpd.GeoDataFrame, CPCAD_with_buffer:gpd.GeoDataFrame,
                                save_to:Path|str,
                                font_family:str='sans-serif'):
    """
    Creates a plot comparing original and buffered conservation lands.
    """

    plt.rcParams['font.family'] =font_family
    
    # 1. Define colormap and normalization
    unique_cats = CPCAD_actual['IUCN_CAT'].unique()
    cmap = plt.cm.get_cmap('tab10', len(unique_cats))

    # 2. Setup subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)

    # 3. Original geometries
    CPCAD_actual.plot(
        ax=axes[0],
        column='IUCN_CAT_desc',
        cmap=cmap,
        linewidth=0.2,
        edgecolor='k',
        facecolor=None,
        legend=False
    )
    axes[0].set_title("Original Conservation Lands")
    axes[0].axis('off')

    # 4. Buffered geometries
    CPCAD_with_buffer.plot(
        ax=axes[1],
        column='IUCN_CAT_desc',
        cmap=cmap,
        linewidth=0.5,
        edgecolor='none',
        alpha=0.6,
        legend=False
    )
    axes[1].set_title("Buffered Conservation Lands")
    axes[1].axis('off')
    # 5. Add shared legend
    legend_labels = CPCAD_actual[['IUCN_CAT', 'IUCN_CAT_desc']].drop_duplicates().sort_values('IUCN_CAT')
    handles = [
        Line2D([0], [0], color=cmap(i-1), lw=4, label=desc)
        for i, desc in zip(legend_labels['IUCN_CAT'], legend_labels['IUCN_CAT_desc'])
    ]

    title_font = FontProperties(weight='bold', size=14)
    fig.legend(
        handles=handles,
        title="IUCN Category",
        loc='lower center',
        ncol=4,
        frameon=False,
        fontsize=12,
        title_fontproperties=title_font
    )

    add_compass_arrow(ax=axes[1],length=0.03)
    # 6. Final layout
    plt.suptitle("Comparison of Original vs Buffered Conservation Areas", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])   
    save_to=Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
    utils.print_update(level=3, message=f"Conservation Lands Plot saved to {save_to}")


def get_stepwise_availability_plots(excluder:ExclusionContainer,
                                    region_shape:gpd.GeoDataFrame,
                                    raster_configs:list[dict], 
                                    vector_configs:list[dict],
                                    save_to:str|Path):
    
    plt.rcParams['font.family']='serif'
    
    n_rasters = len(raster_configs)
    n_vectors = len(vector_configs)

    # 2. Plot setup
    total_layers = n_rasters + n_vectors
    fig, axes = plt.subplots(1, total_layers, figsize=(6 * total_layers, 8))

    # Helper function
    def plot_exclusion_layer(ax, 
                             geometry, 
                             title, 
                             invert=False, 
                             is_raster=False, 
                             filepath=None, 
                             codes=None):
        if is_raster:
            excluder.add_raster(filepath, codes, invert=invert)
        else:
            excluder.add_geometry(geometry)
        
        eligible_share, eligible_area, region_area = lands.get_eligible_share(region_shape, excluder)
        
        excluder.plot_shape_availability(
            geometry=region_shape,
            ax=ax,
            set_title=False,
            show_kwargs={"interpolation": "nearest", 'alpha': 0.7},
            plot_kwargs={"edgecolor": "black", "linewidth": 0.4, "facecolor": "none", "zorder": 3},
        )

        ax.set_title(f"{title} ({eligible_share:.2%})")
        ax.axis("off")

    # 3. Raster layers
    for i, r in enumerate(raster_configs):
        plot_exclusion_layer(
            ax=axes[i],
            geometry=None,
            title=r["title"],
            invert=r["invert"],
            is_raster=True,
            filepath=r["filepath"],
            codes=r["codes"],
        )

    # 4. Vector layers
    for i, v in enumerate(vector_configs):
        # Assert that the geometries in vector_configs are in the same CRS as excluder
        if v["gdf"].crs != excluder.crs:
                v["gdf"] = v["gdf"].to_crs(excluder.crs)
        plot_exclusion_layer(
            ax=axes[n_rasters + i],
            geometry=v["gdf"].geometry,
            title=v["title"],
            invert=v.get("invert", False),
            is_raster=False,
        )

    plt.tight_layout()
    fig.suptitle("Land Availability for Exclusion/Inclusion Layers", fontsize=16, y=1.05)

    # Save the figure
    if isinstance(save_to, str):
        save_to = Path(save_to)
    if not save_to.parent.exists():
        save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
    utils.print_update(level=3, message=f"Stepwise Availability Plots saved to {save_to}")


def plot_gaez_raster_with_boundary(raster_path, legend_csv, gdf_path,
                                   dst_crs="EPSG:4326", figsize=(12, 7),compass_length=0.1,
                                   font_family='serif',
                                   title=None,
                                   plot_save_to=None):
    """
    Plot a GAEZ categorical raster with a shadowed boundary layer using colors from CSV.
    """
    plt.rcParams['font.family'] = font_family
    # Load legend and exclude class 0
    legend_df = pd.read_csv(legend_csv)
    legend_df = legend_df[legend_df['class'] != 0]
    class_map = dict(zip(legend_df['class'], legend_df['description']))
    
    # Create a ListedColormap from the CSV colors
    cmap = ListedColormap(legend_df['color'].tolist())
    
    # Load and reproject GeoDataFrame
    gdf = gpd.read_file(gdf_path)
    gdf = gdf.to_crs(dst_crs)
    
    # Open and reproject raster
    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        data_reproj = np.empty((height, width), dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, 1),
            destination=data_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    # Mask NoData and class 0
    nodata_val = kwargs.get("nodata", None)
    data_masked = np.ma.masked_equal(data_reproj, nodata_val) if nodata_val is not None else np.ma.masked_invalid(data_reproj)
    data_masked = np.ma.masked_equal(data_masked, 0)
    
    # Normalization
    bounds = np.arange(0.5, len(class_map) + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    


    fig, ax = plt.subplots(figsize=figsize)

    
    # Shadow layer (thicker light grey)
    gdf.boundary.plot(ax=ax, edgecolor='grey', linewidth=1, zorder=0)
    
    extent = (
        kwargs["transform"][2],
        kwargs["transform"][2] + kwargs["transform"][0] * kwargs["width"],
        kwargs["transform"][5] + kwargs["transform"][4] * kwargs["height"],
        kwargs["transform"][5]
    )
    
    # Plot raster
    im = ax.imshow(data_masked, cmap=cmap, norm=norm, extent=extent)
    
    # Overlay actual boundaries (thin black line)
    gdf.boundary.plot(ax=ax, edgecolor='k', linewidth=0.2)
    

    # Colorbar
    cbar = plt.colorbar(im, ticks=range(1, len(class_map) + 1), shrink=0.5)
    cbar.ax.set_yticklabels([f"{v}: {class_map.get(v, 'Unknown')}" for v in range(1, len(class_map) + 1)],
                        fontsize=8.5)
    
    # Remove axes
    ax.set_axis_off()
    
    # Title
    if title is None:
        ax.set_title(f"GAEZ Raster: {raster_path.split('/')[-1]}", fontsize=14, fontweight="bold", pad=15)
    
    else:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    
    # vis.add_compass_arrow(ax,length=compass_length)
    plt.tight_layout()
    
    if plot_save_to is None:
        utils.print_update(level=2,message="No path provided to save the plot. Displaying the plot instead.")
    else:
        plot_save_to=Path(plot_save_to)
        plot_save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_save_to,dpi=300)
        utils.print_update(level=2,message=f"GAEZ Raster plot saved to {plot_save_to}")

def get_existing_committed_VRE_plot(
    ax: plt.Axes,
    existing_VREs_gdf: gpd.GeoDataFrame,
    committed_VREs_gdf: gpd.GeoDataFrame = None,
    existing_VRE_type_column: str = "gen_type",
    existing_marker_col: str = "facility_installed_capacity",
    committed_marker_col: str = "potential_capacity",
    target_crs: str = "EPSG:4326",
    marker_scale_existing: float = 10.0,
    marker_scale_committed: float = 2.0,
    sites_legend_handle_scale: float = 8.0,
    marker_highlight_width: float = 5.0,
):
    """
    Plots existing and committed Variable Renewable Energy (VRE) sites on a given Axes.
    Handles missing marker columns gracefully by using constant marker size.
    """

    # === CRS Handling ===
    vre_proj = None
    committed_proj = None
    if target_crs is None:
        utils.print_update(2, f"No target CRS provided. Using default ({target_crs}) of VRE GeoDataFrames.")
    else:
        utils.print_update(2, f"Reprojecting VRE GeoDataFrames to target CRS: {target_crs}")

    # === Existing VREs ===
    if existing_VREs_gdf is not None and not existing_VREs_gdf.empty:
        vre_proj = existing_VREs_gdf.to_crs(target_crs).copy()
        if existing_VRE_type_column not in vre_proj.columns:
            vre_proj[existing_VRE_type_column] = ""
        vre_proj[existing_VRE_type_column] = vre_proj[existing_VRE_type_column].fillna("").astype(str)

        # --- Safe size extraction ---
        if existing_marker_col in vre_proj.columns:
            sizes = pd.to_numeric(vre_proj[existing_marker_col], errors="coerce").fillna(1.0)
            utils.print_update(2, f"Using '{existing_marker_col}' for marker sizes.")
        else:
            sizes = pd.Series(1.0, index=vre_proj.index)
            utils.print_update(2, f"⚠️ '{existing_marker_col}' not found. Using constant marker size = 1.0.")

        # Scale sizes
        sizes = sizes * marker_scale_existing

        # Defensive: ensure array-like
        if np.isscalar(sizes) or len(sizes) != len(vre_proj):
            sizes = np.full(len(vre_proj), marker_scale_existing)

        # --- Type filters ---
        is_wind = vre_proj[existing_VRE_type_column].str.lower().str.contains("wind", regex=False)
        is_solar = vre_proj[existing_VRE_type_column].str.lower().str.contains("solar", regex=False)

    else:
        utils.print_update(2, "No existing VREs data provided or GeoDataFrame is empty.")
        vre_proj = None

    # === Committed VREs ===
    if committed_VREs_gdf is not None and not committed_VREs_gdf.empty:
        committed_proj = committed_VREs_gdf.to_crs(target_crs).copy()
        if existing_VRE_type_column not in committed_proj.columns:
            committed_proj[existing_VRE_type_column] = ""
        committed_proj[existing_VRE_type_column] = committed_proj[existing_VRE_type_column].fillna("").astype(str)

        if committed_marker_col in committed_proj.columns:
            sizes_c = pd.to_numeric(committed_proj[committed_marker_col], errors="coerce").fillna(1.0)
            utils.print_update(2, f"Using '{committed_marker_col}' for committed marker sizes.")
        else:
            sizes_c = pd.Series(1.0, index=committed_proj.index)
            utils.print_update(2, f"⚠️ '{committed_marker_col}' not found. Using constant marker size = 1.0.")

        sizes_c = sizes_c * marker_scale_committed
        if np.isscalar(sizes_c) or len(sizes_c) != len(committed_proj):
            sizes_c = np.full(len(committed_proj), marker_scale_committed)

        is_wind_c = committed_proj["resource_type"].str.lower().str.contains("wind", regex=False)
        is_solar_c = committed_proj["resource_type"].str.lower().str.contains("solar", regex=False)
    else:
        utils.print_update(2, "No committed VREs data provided or GeoDataFrame is empty.")
        committed_proj = None

    # === Nothing to plot ===
    if vre_proj is None and committed_proj is None:
        utils.print_update(2, "No VRE data available to plot.")
        return ax, []

    utils.print_update(2, "Plotting existing and committed VREs.")
    legend_handles = []

    # === Plot Existing VREs ===
    if vre_proj is not None and not vre_proj.empty:
        if is_wind.any():
            vre_proj.loc[is_wind].plot(
                ax=ax, facecolor="None", edgecolor="blue",
                markersize=sizes[is_wind], marker="s", alpha=1, zorder=4,
                path_effects=[pe.withStroke(linewidth=marker_highlight_width, foreground="yellow", alpha=0.6)],
            )
            legend_handles.append(Line2D([0], [0], marker="s", color="blue", linestyle="None",
                                         markersize=8, markerfacecolor="None", label="Existing Wind"))

        if is_solar.any():
            vre_proj.loc[is_solar].plot(
                ax=ax, facecolor="None", edgecolor="red",
                markersize=sizes[is_solar], marker="s", alpha=1, zorder=4,
                path_effects=[pe.withStroke(linewidth=marker_highlight_width, foreground="yellow", alpha=0.6)],
            )
            legend_handles.append(Line2D([0], [0], marker="s", color="red", linestyle="None",
                                         markersize=8, markerfacecolor="None", label="Existing Solar"))

    # === Plot Committed VREs ===
    if committed_proj is not None and not committed_proj.empty:
        if is_wind_c.any():
            committed_proj.loc[is_wind_c].plot(
                ax=ax, facecolor="None", edgecolor="fuchsia",
                markersize=sizes_c[is_wind_c], marker="^", alpha=1, zorder=5,
                path_effects=[pe.withStroke(linewidth=marker_highlight_width, foreground="yellow", alpha=0.6)],
            )
            legend_handles.append(Line2D([0], [0], marker="^", color="fuchsia", linestyle="None",
                                         markersize=sites_legend_handle_scale, markerfacecolor="None", label="Committed Wind"))

        if is_solar_c.any():
            committed_proj.loc[is_solar_c].plot(
                ax=ax, facecolor="None", edgecolor="coral",
                markersize=sizes_c[is_solar_c], marker="D", alpha=1, zorder=5,
                path_effects=[pe.withStroke(linewidth=marker_highlight_width, foreground="yellow", alpha=0.6)],
            )
            legend_handles.append(Line2D([0], [0], marker="D", color="coral", linestyle="None",
                                         markersize=sites_legend_handle_scale, markerfacecolor="None", label="Committed Solar"))

    return ax, legend_handles


def plot_developable_land_and_vres(
    *,
    target_crs: str = 'EPSG:4326',
    raster_data: xr.DataArray,
    raster_legends: pd.DataFrame,
    classes_to_plot: Mapping[str, Sequence[int]] | None = None,  # {"solar":[...], "wind":[...]} or None -> ALL
    boundary: gpd.GeoDataFrame,
    existing_VREs_gdf: gpd.GeoDataFrame | None = None,
    include_tags: Iterable[str] = ("solar", "wind"),
    existing_marker_col: str = "wind_turbine_capacity",
    committed_VREs_gdf: gpd.GeoDataFrame | None = None,
    committed_marker_col: str = "potential_capacity",
    marker_scale_existing: float = 7.0,
    marker_scale_committed: float = 4.0,
    marker_highlight_width: float = 1.0,
    area_labels: bool = False,
    title: str = "Developable Land with Existing & Committed VREs",
    fallback_crs: str = "EPSG:4326",
    label_column: str = "Country",
    vre_type_column: str = "gen_type",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 12),
    legend_anchor: tuple[float, float] | None = None,
    legend_fontsize: float = 10.0,
    dpi: int = 500,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, Path | None]:
    """
    Plot developable land from a categorical raster with boundaries and existing/committed VRE sites.
    Highlights excluded land classes in semi-transparent red if a subset of classes is provided.
    """

    # ---------- Legend helpers ----------
    id_to_name = dict(zip(raster_legends["class"].astype(int),
                          raster_legends["description"].astype(str)))
    id_to_hex = dict(zip(raster_legends["class"].astype(int),
                         raster_legends["color"].astype(str)))

    # ---------- CRS handling ----------
    if raster_data.rio.crs is None:
        raster_data = raster_data.rio.write_crs(target_crs, inplace=False)
    if raster_data.rio.crs != target_crs:
        try:
            raster_plot = raster_data.rio.reproject(target_crs)
        except Exception:
            print("⚠️ Reprojection failed, using original CRS instead.")
            raster_plot = raster_data
    else:
        raster_plot = raster_data

    if boundary.crs != target_crs:
        boundary_proj = boundary.to_crs(target_crs)
        
    else:
        boundary_proj = boundary
    
    # ---------- Clip raster to boundary ----------
    try:
        # Ensure boundary CRS matches raster CRS
        if boundary_proj.crs != raster_plot.rio.crs:
            boundary_for_clip = boundary_proj.to_crs(raster_plot.rio.crs)
        else:
            boundary_for_clip = boundary_proj

        raster_clipped = raster_plot.rio.clip(
                boundary_for_clip.geometry,
                boundary_for_clip.crs,
                drop=True,
                invert=False,
                all_touched=False,   # <— STRICT geometry clipping
            )
        print(f"✅ Raster clipped to boundary '{label_column}' region, shape: {raster_clipped.shape}")
        raster_plot = raster_clipped
    except Exception as e:
        print(f"⚠️ Raster clipping failed, using full raster extent instead. Reason: {e}")


    # ---------- Raster + masking ----------
    # raster = raster_plot.values.squeeze().astype("int32", copy=False)
    raster = np.nan_to_num(raster_plot.values.squeeze(), nan=0).astype("int32")

    codes_present = np.unique(raster)

    if classes_to_plot is None:
        selected_codes = set(int(c) for c in codes_present.tolist())
        selected_codes.discard(0)
        use_masking = False
    else:
        selected_codes = set()
        for tag in include_tags:
            if tag in classes_to_plot:
                selected_codes.update(int(c) for c in classes_to_plot[tag])
        selected_codes.discard(0)
        use_masking = True

    # Mask raster for plotting
    if use_masking:
        mask = np.isin(raster, list(selected_codes))
        raster_masked = raster.copy()
        raster_masked[~mask] = -1  # mark excluded
        excluded_mask = (~mask) & (raster != 0)
        values_to_color = sorted(selected_codes)
    else:
        raster_masked = raster
        excluded_mask = np.zeros_like(raster, dtype=bool)
        values_to_color = sorted(set(int(v) for v in codes_present.tolist()))

    # ---------- Colormap ----------
    color_map_dict: dict[int, tuple] = {}
    for c in values_to_color:
        color_map_dict[c] = to_rgba(id_to_hex.get(c, "#FFFFFF"))
    if use_masking:
        color_map_dict[-1] = (0, 0, 0, 0)  # transparent base for excluded

    all_vals = sorted(color_map_dict.keys())
    cmap = ListedColormap([color_map_dict[v] for v in all_vals])
    last_edge = (all_vals[-1] + 1) if all_vals else 1
    norm = BoundaryNorm(np.array(all_vals + [last_edge]), cmap.N)

    # ---------- Extent ----------
    x = raster_plot.coords["x"].values
    y = raster_plot.coords["y"].values
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    origin = "upper" if y[0] > y[-1] else "lower"

    xmin, ymin, xmax, ymax = boundary_proj.total_bounds
    aspect_ratio = (ymax - ymin) / (xmax - xmin)
    fig_width = 9
    fig_height = fig_width * aspect_ratio * 1.1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    ax.set_facecolor("white")

    # ---------- Base raster ----------
    ax.imshow(
        raster_masked,
        extent=extent,
        origin=origin,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=0,
    )

    # ---------- Overlay excluded (semi-transparent red) ----------
    if use_masking:
        ax.imshow(
            np.where(excluded_mask, 1, np.nan),
            extent=extent,
            origin=origin,
            cmap=ListedColormap([(1, 0, 0, 0.3)]),  # semi-transparent red
            interpolation="nearest",
            zorder=10,
        )

    # ---------- Boundary & labels ----------
    if not boundary_proj.empty:
        boundary_proj.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.2, zorder=3)

    if area_labels and label_column in boundary_proj.columns:
        for _, row in boundary_proj.iterrows():
            c = row.geometry.centroid
            ax.annotate(
                str(row[label_column]),
                (c.x, c.y),
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

    # ---------- VRE plotting ----------
    legend_all: list[Patch] = []
    ax, VRE_legend_handles = get_existing_committed_VRE_plot(
        ax=ax,
        target_crs=target_crs,
        existing_VREs_gdf=existing_VREs_gdf,
        committed_VREs_gdf=committed_VREs_gdf,
        existing_VRE_type_column=vre_type_column,
        existing_marker_col=existing_marker_col,
        committed_marker_col=committed_marker_col,
        marker_scale_existing=marker_scale_existing,
        marker_scale_committed=marker_scale_committed,
        marker_highlight_width=marker_highlight_width,
    )
    legend_all.extend(VRE_legend_handles)

    # ---------- Legend ----------
    handles_classes: list[Patch] = []
    for c in values_to_color:
        base_label = f"{c}: {id_to_name.get(c, f'Class {c}')}"
        tag_note = [tag.capitalize() for tag in include_tags
                    if classes_to_plot and tag in classes_to_plot
                    and c in set(int(v) for v in classes_to_plot[tag])]
        label = f"{base_label} ({' & '.join(tag_note)})" if tag_note else base_label
        handles_classes.append(Patch(facecolor=id_to_hex.get(c, "#888888"),
                                     edgecolor="none", label=label))
    legend_all.extend(handles_classes)

    if use_masking:
        legend_all.append(Patch(facecolor=(1, 0, 0, 0.3),
                                edgecolor=None,
                                label="Excluded Land (outside selected classes)"))

    if legend_all:
        ax.legend(
            handles=legend_all,
            loc="upper left",
            bbox_to_anchor=legend_anchor if legend_anchor else (0.98, 0.98),
            fontsize=legend_fontsize, frameon=True, facecolor="white", edgecolor="none",
            framealpha=0.9, labelspacing=0.4, handlelength=1.4,
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    saved: Path | None = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=dpi, bbox_inches="tight")
        utils.print_update(level=1, message=f"Saved map with excluded overlay → {saved}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    return fig, ax, saved


def plot_vre_sites_by_landcover(
    df: pd.DataFrame,
    class_col: str = None,
    count_prefix: str = "SiteCount_",
    title: str = "Existing VRE Sites by Land-Cover Class and Technology",
    figsize=(8, 6),
    wrap_width: int = 25,
    fontsize: int = 8,
    colors: list = None,
    save_to: str = None,
    show=True
):
    """
    Plots a horizontal stacked bar chart of VRE site counts by land-cover class and technology.
    Wraps long class labels and annotates total counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing land-cover description and site count columns.
    class_col : str, default "CLC_landcover_description"
        Column containing the land-cover class names.
    count_prefix : str, default "SiteCount_"
        Prefix used to identify count columns (e.g. "SiteCount_Solar", "SiteCount_Wind").
    title : str
        Plot title.
    figsize : tuple, default (8, 6)
        Figure size in inches.
    wrap_width : int, default 25
        Maximum character width before wrapping y-axis labels.
    fontsize : int, default 8
        Font size for y-axis labels and annotations.
    colors : list, optional
        List of color hex codes for stacked bars. Defaults to Matplotlib’s default palette.
    """
    if class_col is None:
         utils.print_error("'class_col' must be defined. Check the dataframe for this column")
    
    # --- Identify count columns ---
    count_cols = [c for c in df.columns if c.startswith(count_prefix)]
    if not count_cols:
        raise ValueError(f"No columns found starting with '{count_prefix}'.")

    # --- Compute totals and sort ---
    df = df.copy()
    df["Total"] = df[count_cols].sum(axis=1)
    df = df.sort_values("Total", ascending=True)

    # --- Use landcover as index ---
    df = df.set_index(class_col)

    # --- Create wrapped labels ---
    wrapped_labels = ["\n".join(textwrap.wrap(lbl, width=wrap_width)) for lbl in df.index]

    # --- Plot ---
    ax = df[count_cols].plot(
        kind="barh",
        stacked=True,
        figsize=figsize,
        color=colors or ["#1f77b4", "#ff7f0e"]
    )

    # --- Apply wrapped y-labels ---
    ax.set_yticks(range(len(wrapped_labels)))
    ax.set_yticklabels(wrapped_labels, fontsize=fontsize)

    # --- Titles and labels ---
    ax.set_title(title, pad=12)
    ax.set_xlabel("Number of Sites")
    ax.set_ylabel("Raster Class")

    # --- Annotate total counts ---
    for i, total in enumerate(df["Total"].values):
        ax.text(total + 0.3, i, f"{int(total)}", va="center", fontsize=fontsize)

    # --- Layout ---
    plt.tight_layout()

    
    if save_to is not None:
         if isinstance(save_to, str):
             save_to = Path(save_to)
         save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
