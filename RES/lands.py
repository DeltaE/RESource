"""
lands.py
This module provides classes and functions for processing, analyzing, and visualizing land use and conservation data,
with a focus on Canadian regions. It integrates vector and raster geospatial data sources, including protected and
conserved areas, OSM features, and GAEZ raster datasets, to support land exclusion/inclusion analysis for resource
planning (e.g., renewable energy siting).

Key Components:
---------------
- ConservationLands:
    Handles downloading, extracting, and processing of conservation lands data (e.g., Canadian Protected and Conserved Areas Database).
    Provides methods for loading, simplifying, and mapping conserved land geometries at the provincial level.
- LandContainer:
    Combines multiple geospatial data sources (ERA5 cutouts, GAEZ rasters, OSM data, conservation lands) to manage
    inclusion/exclusion of lands for spatial analysis. Supports loading, buffering, and plotting of raster and vector layers.
- Utility Functions:
    - add_and_plot_exclusion_layer: Adds raster/vector layers to an exclusion container and visualizes eligible areas.
    - load_layers_to_excluder: Loads and visualizes all configured raster and vector exclusion layers for a region.
    - apply_buffer_to_vector: Buffers vector geometries by configurable distances and compares area changes.
    - get_eligible_share: Computes the share of eligible (non-excluded) area within a region.

Dependencies:
-------------
- geopandas, pandas, numpy, matplotlib, rasterio, fiona, atlite.gis, folium
- Custom modules: RES.utility, RES.boundaries, RES.era5_cutout, RES.gaez, RES.osm

Intended Use:
-------------
This module is intended for use in spatial resource assessment workflows, particularly for renewable energy planning
where land exclusions (e.g., protected areas, infrastructure buffers) must be considered. It is designed to be
configurable and extensible for different regions and data sources.


"""

from pathlib import Path
from zipfile import ZipFile

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from atlite.gis import ExclusionContainer
from matplotlib.colors import ListedColormap
from rasterio.plot import show

from RES import utility as utils
from RES.boundaries import GADMBoundaries
from RES.era5_cutout import ERA5Cutout
from RES.gaez import GAEZRasterProcessor
from RES.osm import OSMData

PRINT_LEVEL_BASE: int = 2  # handles the print level for the utils.print_update function


class ConservationLands(GADMBoundaries):
    """
    ConservationLands class
    """

    def __post_init__(self):
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()

        # Set the Class specific attributes
        self.conserved_lands_cfg = self.config["Gov"]["conservation_lands"]

        self.source_url = self.conserved_lands_cfg["url"]
        self.data_root = self.conserved_lands_cfg["root"]
        self.zip_file_name = f"{self.conserved_lands_cfg['data_name']}.zip"
        self.zip_file_path = Path(self.data_root) / self.zip_file_name
        self.extraction_dir = Path(self.data_root) / self.zip_file_path.stem
        self.extraction_dir.parent.mkdir(parents=True, exist_ok=True)

        # Initialize region_boundary attribute
        self.region_boundary = self.get_region_boundary()
        self.region_shape = self.region_boundary.dissolve(
            by=self.gadm_config["datafield_mapping"]["NAME_1"]
        )  # Get the geometry of the region boundary
        self.region_name = self.get_region_name()

        # Set up resource disaggregation configurations
        self.resource_disaggregation_config: dict = (
            self.get_resource_disaggregation_config()
        )

        self.aeroway_gdf: gpd.GeoDataFrame = None  # Initialize aeroway_gdf attribute
        self.raster_configs: list = []  # Initialize raster_configs attribute

    def get_provincial_conserved_lands(
        self, geom_simplification_tolerance=0.005
    ) -> gpd.GeoDataFrame:
        """
        Load provincial conserved lands from the .gdb file.

        ### Args:
            geom_simplification_tolerance (default to _.005_)
            - geometry simplification to avoid unnecessary granular level geometries.
            - This tool is configured to geom in degrees, e.g tolerance of 0.005 corresponds to approximately 500m (at the equator) geoms will be simplified.
        """

        utils.print_update(
            level=PRINT_LEVEL_BASE + 3,
            message=f"{__name__}| Processing Conserved areas for {self.region_name}",
        )

        file_name_prefix: str = self.conserved_lands_cfg.get(
            "data_name", "ProtectedConservedArea"
        )
        gdb_layer: str = self.conserved_lands_cfg.get(
            "gdb_layer", "ProtectedConservedArea_2023"
        )

        provincial_file_path = (
            Path("data/downloaded_data/lands")
            / f"{file_name_prefix}_{self.region_short_code}.pickle"
        )
        provincial_file_path.parent.mkdir(parents=True, exist_ok=True)

        if provincial_file_path.exists():
            utils.print_update(
                level=PRINT_LEVEL_BASE,
                message=f"{__name__}| Loading regional data from Canadian Protected and Conserved Areas Database (CPCAD) from locally stored datafile - {provincial_file_path}",
            )
            gdf = gpd.GeoDataFrame(pd.read_pickle(provincial_file_path))

        else:
            gdb_file_path: Path = self.__get_conserved_lands__()

            # Get Region Boundaries
            self.region_boundary: gpd.GeoDataFrame = self.get_region_boundary()

            layers: list = fiona.listlayers(gdb_file_path)

            try:
                assert gdb_layer in layers, (
                    f"Layer '{gdb_layer}' not found in the GDB file. Please configure the valid 'gdb_layer' key in config file."
                )
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__}| Loading  {gdb_layer} Layer from the GDB file.",
                )

                # Load the .gdb file as a GeoDataFrame
                gdf = gpd.read_file(
                    gdb_file_path, mask=self.region_boundary, layer=gdb_layer
                )  # Specifying layer and mask to load only the relevant region, faster loading
                gdf.to_crs(self.region_boundary.crs, inplace=True)

                gdf["geometry"] = gdf["geometry"].simplify(
                    geom_simplification_tolerance
                )  # Simplify geometries to reduce complexity that are not relevant at ERA5 resolution and faster processing

                # Map IUCN categories to descriptions
                IUCN_CAT = self.conserved_lands_cfg["IUCN_CAT_mapping"]
                gdf["IUCN_CAT_desc"] = gdf["IUCN_CAT"].map(IUCN_CAT)
                gdf.to_pickle(provincial_file_path)
            except AssertionError as e:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1, message=f"{__name__}| {e}", alert=True
                )

        return gdf

    def __get_conserved_lands__(self) -> Path:
        """Download the source ZIP file, extract contents, and return the .gdb file path."""
        # Check if the extraction directory exists
        if self.extraction_dir.exists():
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"Extraction directory {self.extraction_dir} already exists, skipping download and extraction.",
            )
        else:
            if self.zip_file_path.exists():
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message=f"ZIP file {self.zip_file_path} already exists, skipping download.",
                )
            else:
                # Download the ZIP file
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message="Downloading Canadian Protected and Conserved Areas Database (CPCAD)",
                )
                self.zip_file_path.parent.mkdir(parents=True, exist_ok=True)
                utils.download_data(self.source_url, self.zip_file_path)
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message=f"Downloaded ZIP file to {self.zip_file_path}",
                )

            # Create the extraction directory and extract ZIP contents
            self.extraction_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(self.zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.extraction_dir)
            # print(f"Extracted files to {self.extraction_dir}")

        # Load the first .gdb file found in the extraction directory
        gdb_file_path = next(self.extraction_dir.rglob("*.gdb"), None)
        if gdb_file_path is None:
            raise FileNotFoundError(
                ">> !! No .gdb file found in the extracted contents."
            )

        return gdb_file_path

    def show_lands(
        self,
        basemap: str = "CartoDB positron",
        save_path: str = None,
        save: bool = False,
    ):
        """
        Create and save an interactive map for the specified region.

        Args:
            basemap (str): The basemap to use (default is 'CartoDB positron').
            save_path (str): The path to save the HTML map. If None, default is used.
            save (bool): If True, saves the map as a local HTML file.

        Returns:
            folium.Map: The interactive map object.
        """
        conserved_lands = self.get_provincial_conserved_lands()
        self.region_boundary = self.get_region_boundary()

        if self.region_boundary is not None:
            m = self.region_boundary.explore(
                color="grey", linecolor="grey", legend=True, tiles=basemap, alpha=0.4
            )
            conserved_lands.explore("IUCN_CAT_desc", m=m, legend=True, tiles=basemap)

            if save:
                if save_path is None:
                    save_path = f"vis/lands/{self.region_short_code}.html"
                else:
                    save_path = Path(save_path) / f"{self.region_short_code}.html"

                # Ensure the directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the map as an HTML file
                m.save(save_path)
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message="Interactive map for '{self.region_short_code}' saved to {save_path}.",
                )
            else:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 1,
                    message="Skipping save, 'save' is set to False.",
                )

        return m


class LandContainer(ERA5Cutout, GAEZRasterProcessor, ConservationLands, OSMData):
    """
    Handles the inclusion/exclusion of lands from raster/vector data.

    """

    def __post_init__(self):
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()

        self.excluder_crs = self.get_excluder_crs(country="Canada")

        # Initiate Exclusion Container
        self.excluder = ExclusionContainer(
            crs=self.excluder_crs
        )  # CRS 3347 fit for Canada

        # Initialize resource_disaggregation_config attribute
        self.resource_disaggregation_config = self.get_resource_disaggregation_config()

        # Initialize conservation_lands_region_gdf attribute
        self.conservation_lands_region_gdf = None

    def set_excluder(self):
        raster_configs, vector_configs = self.get_layers()

        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__}| Loading layers to Excluder for {self.region_name}. This may take a while to compute and plot...",
        )

        # Load all layers to the excluder
        excluder_with_layers = load_layers_to_excluder(
            self.resource_type,
            self.excluder,
            self.region_shape,
            raster_configs,
            vector_configs,
            font_family=self.default_font_family,
            plot_save_to=f"vis/{self.region_short_code}/lands",
        )
        return excluder_with_layers

    def get_layers(self):
        """Load all raster and vector layers for the specified region.
        Returns:
            tuple: A tuple containing two lists - raster_configs and vector_configs.
        """
        # load Raster Layers
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading GAEZ raster layers for {self.region_name}...",
        )
        self.gaez_config = self.get_gaez_data_config()
        raster_configs: list[dict] = self.gaez_config["raster_types"]
        regional_raster_paths: dict = self.process_all_rasters(show=False)

        for raster_config_item in raster_configs:
            name = raster_config_item.get("name")
            if name and name in regional_raster_paths:
                raster_config_item["filepath"] = str(regional_raster_paths[name])

        # Load Vector layers
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading vector layers for {self.region_name}...",
        )
        vector_configs: list[dict] = self.resource_disaggregation_config[
            "vector_buffers"
        ]

        for vector_config_item in vector_configs:
            # vector_config_item is a dictionary
            vector_name = list(vector_config_item.keys())[0]
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__}| Loading {vector_name} areas for {self.region_name}",
            )

            if vector_name == "conserved_lands":
                # Add local (Canadian) vector geometries to excluder
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__}| Loading Conserved areas for {self.region_name}",
                )
                vector_gdf = self.get_provincial_conserved_lands()
                if vector_gdf.empty:
                    utils.print_update(
                        level=PRINT_LEVEL_BASE + 1,
                        message=f"{__name__}| No {vector_name} data found for {self.region_name}. Skipping.",
                        alert=True,
                    )
                    continue
                vector_config_item[vector_name]["stepwise_plot_title"] = (
                    "Excluding Regional Conservation Areas"
                )

            elif vector_name == "aeroway":
                # Load vector geometries from OSM
                vector_gdf = self.get_osm_layer(vector_name)
                if vector_gdf.empty:
                    utils.print_update(
                        level=PRINT_LEVEL_BASE + 1,
                        message=f"{__name__}| No {vector_name} data found for {self.region_name}. Skipping.",
                        alert=True,
                    )
                    continue
                vector_config_item[vector_name]["stepwise_plot_title"] = (
                    "Excluding Regional Aeroways"
                )

            # Apply buffer to the vector geometries
            vector_gdf_with_buffer, vector_area_comparison = apply_buffer_to_vector(
                vector_gdf,
                vector_config_item[vector_name]["buffer_mapping_key_buffers"],
                vector_config_item[vector_name]["buffer_mapping_key"],
            )
            vector_config_item[vector_name]["gdf"] = vector_gdf_with_buffer

            # Save the area comparison to a CSV file
            area_comparison_save_path = (
                Path("data/processed_data/lands")
                / f"{vector_config_item[vector_name]['buffer_mapping_key']}_area_comparisons_{self.region_name}.csv"
            )
            area_comparison_save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the area comparison DataFrame to CSV
            vector_area_comparison.to_csv(area_comparison_save_path)
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__}| Vector Area comparison for {vector_config_item[vector_name]['buffer_mapping_key']} saved to {area_comparison_save_path}",
            )
            vector_config_item[vector_name]["area_comparison"] = vector_area_comparison

        # We want to flat list of dictionaries without vector_name in the keys
        vector_configs = [list(d.values())[0] for d in vector_configs]

        return raster_configs, vector_configs


@staticmethod
def add_and_plot_exclusion_layer(
    excluder: ExclusionContainer,
    region_shape: gpd.GeoDataFrame,
    ax,
    geometry,
    title,
    invert=False,
    is_raster=False,
    filepath=None,
    codes=None,
):
    """
    Add a layer to the ExclusionContainer and plot the availability of the region shape.

    Args:
        excluder (ExclusionContainer): The ExclusionContainer to add the layer to.
        region_shape (gpd.GeoDataFrame): The region shape GeoDataFrame.
        ax (_type_): The axes to plot on.
        geometry (_type_): The geometry to add to the ExclusionContainer.
        title (_type_): The title for the plot.
        invert (bool, optional): Whether to invert the exclusion. Defaults to False.
        is_raster (bool, optional): Whether the layer is a raster layer. Defaults to False.
        filepath (_type_, optional): The file path for the raster layer. Defaults to None.
        codes (_type_, optional): The codes for the raster layer. Defaults to None.

    Returns:
        _type_: _description_
    """
    if is_raster:
        excluder.add_raster(filepath, codes, invert=invert)
    else:
        excluder.add_geometry(geometry)

    masked, transform, eligible_share = get_eligible_share(region_shape, excluder)

    # Simple solid green for eligible
    cmap = ListedColormap(["#027227"])
    cmap.set_bad(color=(1, 1, 1, 0))  # transparent 0s

    # Keep 1s, mask 0s
    raster_data = masked.astype(float)  # * 100
    masked_data = np.ma.masked_where(raster_data == 0, raster_data)

    # Plot masked raster
    show(
        masked_data,
        transform=transform,
        ax=ax,
        cmap=cmap,
    )

    # Overlay region boundary (no cmap here)
    if region_shape.crs != excluder.crs:
        region_shape = region_shape.to_crs(excluder.crs)
    region_shape.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
    ax.set_title(f"Eligible area (green) {eligible_share:.2%}")

    # Clean aesthetics
    ax.set_axis_off()
    ax.set_title(f"{title} ({eligible_share:.2%})")

    excluder_with_layers: ExclusionContainer = excluder

    return excluder_with_layers


@staticmethod
def load_layers_to_excluder(
    resource_type: str,
    excluder: ExclusionContainer,
    region_shape: gpd.GeoDataFrame,
    raster_configs: list[dict],
    vector_configs: list[dict],
    font_family: str = "serif",
    plot_save_to: str | Path = None,
) -> ExclusionContainer:
    """
    Load raster and vector layers to the ExclusionContainer and plot the availability of the region shape.
    Args:
        excluder (ExclusionContainer): The ExclusionContainer to add the layers to.
        region_shape (gpd.GeoDataFrame): The region shape GeoDataFrame.
        raster_configs (list[dict]): List of raster configurations.
        vector_configs (list[dict]): List of vector configurations.
        plot_save_to (str|Path, optional): Path to save the plot. Defaults to None.
    Returns:
        ExclusionContainer: The ExclusionContainer with the added layers.
    """

    n_rasters = len(raster_configs)
    n_vectors = len(vector_configs)

    # 2. Plot setup
    total_layers = n_rasters + n_vectors

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = 14
    fig, axes = plt.subplots(
        1, total_layers, figsize=(6 * total_layers, total_layers + 4)
    )  # revise this accordingly to make the plot looks nicer
    # 3. Raster layers
    for i, r in enumerate(raster_configs):
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading raster layer '{r.get('name', '')}'...",
        )
        # Handle raster layer inclusion/exclusion logic smartly
        class_inclusion = r.get("class_inclusion")
        class_exclusion = r.get("class_exclusion")
        invert = False
        codes = None

        if class_inclusion and resource_type in class_inclusion:
            codes = class_inclusion[resource_type]
            invert = True
        elif class_exclusion and resource_type in class_exclusion:
            codes = class_exclusion[resource_type]
            invert = False
        else:
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"{__name__}| No valid class_inclusion/class_exclusion for raster '{r.get('name', '')}' and resource '{resource_type}'. Skipping.",
                alert=True,
            )
            continue

        excluder_with_layers = add_and_plot_exclusion_layer(
            excluder,
            region_shape=region_shape,
            ax=axes[i],
            geometry=None,
            title=r.get("stepwise_plot_title", r.get("name", "Raster Layer")),
            invert=invert,
            is_raster=True,
            filepath=r["filepath"],
            codes=codes,
        )

    # 4. Vector layers
    for i, v in enumerate(vector_configs):
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__}| Loading vector layers for '{list(vector_configs[i]['buffer_mapping_key_buffers'].keys())}'...",
        )
        # Assert that the geometries in vector_configs are in the same CRS as excluder
        if v["gdf"].crs != excluder.crs:
            v["gdf"] = v["gdf"].to_crs(excluder.crs)
        excluder_with_layers = add_and_plot_exclusion_layer(
            excluder,
            region_shape=region_shape,
            ax=axes[n_rasters + i],
            geometry=v["gdf"].geometry,
            title=v["stepwise_plot_title"],
            invert=v.get("invert", False),
            is_raster=False,
        )

    plt.tight_layout()
    fig.suptitle(
        "Land Availability for Exclusion/Inclusion Layers for BC", fontsize=24, y=1.05
    )

    # Save the figure
    if isinstance(plot_save_to, str):
        plot_save_to = Path(plot_save_to)
    if not plot_save_to.parent.exists():
        plot_save_to.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        plot_save_to / "stepwise_land_availability_plot.png",
        bbox_inches="tight",
        dpi=300,
    )
    utils.print_update(
        level=3, message=f"Stepwise Availability Plots saved to {plot_save_to}"
    )

    return excluder_with_layers


@staticmethod
def apply_buffer_to_vector(
    gdf: gpd.GeoDataFrame, buffer_mapping: dict, buffer_mapping_key: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Projects the input GeoDataFrame to BC Albers, applies buffer distances from config,
    and reprojects back to EPSG:4326. Returns the buffered GeoDataFrame and area comparison.
    Adds a column 'buffer_applied_m' to show actual buffer distance applied per feature.
    """

    # 1. Project to BC Albers (EPSG:3005)
    gdf_proj = gdf.to_crs(epsg=3005)

    # 2. Assign buffer distances from mapping
    buffer_series = pd.Series(buffer_mapping)
    gdf_proj["buffer_applied_m"] = (
        gdf_proj[buffer_mapping_key].map(buffer_series).fillna(0)
    )

    # 3. Keep unbuffered copy for area comparison
    gdf_unbuffered = gdf_proj.copy()

    # 4. Apply buffer (in meters)
    gdf_buffered = gdf_proj.copy()
    gdf_buffered["geometry"] = gdf_proj.geometry.buffer(gdf_proj["buffer_applied_m"])

    # 5. Area calculations (in km²)
    gdf_unbuffered["area_km2"] = gdf_unbuffered.geometry.area / 1e6
    gdf_buffered["area_km2"] = gdf_buffered.geometry.area / 1e6
    area_original = (
        gdf_unbuffered.groupby(buffer_mapping_key)["area_km2"]
        .sum()
        .rename("original_area_km2")
    )
    area_buffered = (
        gdf_buffered.groupby(buffer_mapping_key)["area_km2"]
        .sum()
        .rename("buffered_area_km2")
    )

    # 6. Area comparison
    area_comparison = pd.concat([area_original, area_buffered], axis=1)
    area_comparison["buffer_applied_m"] = area_comparison.index.map(buffer_mapping)
    area_comparison["delta_km2"] = (
        area_comparison["buffered_area_km2"] - area_comparison["original_area_km2"]
    )
    area_comparison["percent_increase"] = (
        100 * area_comparison["delta_km2"] / area_comparison["original_area_km2"]
    )
    area_comparison = area_comparison.sort_values(
        "original_area_km2", ascending=False
    ).round(4)

    # 7. Reproject back to EPSG:4326
    gdf_buffered = gdf_buffered.to_crs(epsg=4326)

    return gdf_buffered, area_comparison


@staticmethod
def get_eligible_share(region_shape, excluder: ExclusionContainer) -> tuple:
    """
    Calculate the eligible share of the region based on the exclusion container.
    """
    if region_shape.crs != excluder.crs:
        # Reproject region_shape to match the CRS of the excluder
        region_shape = region_shape.to_crs(excluder.crs)
    masked, transform = excluder.compute_shape_availability(region_shape)
    region_area = region_shape.geometry.item().area
    eligible_area = masked.sum() * excluder.res**2
    eligible_share = eligible_area / region_area

    return masked, transform, eligible_share
