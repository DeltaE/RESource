import warnings
from collections import namedtuple
from datetime import datetime
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import Point

from RESource import cluster
from RESource import utility as utils
from RESource import windspeed as wind
from RESource.AttributesParser import AttributesParser
from RESource.boundaries import GADMBoundaries
from RESource.cell import GridCells
from RESource.CellCapacityProcessor import CellCapacityProcessor
from RESource.coders import CODERSData
from RESource.era5_cutout import ERA5Cutout
from RESource.gwa import GWACells
from RESource.hdf5_handler import DataHandler
from RESource.power_nodes import GridNodeLocator
from RESource.score import CellScorer

# RESource local modules
from RESource.step_cache import StepCache
from RESource.timeseries import Timeseries
from RESource.units import Units

current_local_time = datetime.now()
warnings.filterwarnings("ignore")

PRINT_LEVEL_BASE: int = 1


class RESources_builder(AttributesParser):
    """
    Main orchestrator class for renewable energy resource assessment workflows.

    Coordinates the complete workflow for assessing solar and wind potential at
    sub-national scales. Integrates spatial grid cell preparation, land availability
    analysis, weather data processing, economic evaluation, and site clustering into
    a unified, modular framework.

    Parameters
    ----------
    config_file_path : str or Path
        Path to the YAML configuration file containing project settings.
    region_short_code : str
        ISO or custom short code for the target region (e.g. 'BC' for British Columbia).
    resource_type : {'solar', 'wind'}
        Type of renewable energy resource to assess.
    weather_year : int, optional
        Weather year to process. Overrides 'weather_year' key in config if provided.

    Attributes
    ----------
    store : Path
        HDF5 file path for data storage and caching.
    units : Units
        Handler for unit conversions and standardization.
    gridcells : GridCells
        Spatial grid generation and management.
    timeseries : Timeseries
        Climate data processing and capacity factor calculations.
    datahandler : DataHandler
        HDF5-based data storage and retrieval interface.
    cell_processor : CellCapacityProcessor
        Land availability and capacity potential calculations.
    coders : CODERSData
        Canadian power system data integration (Canada only).
    era5_cutout : ERA5Cutout
        ERA5 climate data cutout management.
    scorer : CellScorer
        Economic scoring and LCOE calculations.
    gwa_cells : GWACells
        Global Wind Atlas data integration (wind only).
    results_save_to : Path
        Output directory for assessment results.
    region_name : str
        Full name of the assessed region.

    Methods
    -------
    get_grid_cells()
        Generate spatial grid cells covering the region boundary.
    get_cell_capacity()
        Calculate potential capacity based on land availability constraints.
    extract_weather_data()
        Process climate data for capacity factor calculations.
    update_gwa_scaled_params(memory_resource_limitation=False)
        Integrate Global Wind Atlas wind speed corrections (wind only).
    get_CF_timeseries(cells=None, force_update=False)
        Generate hourly capacity factor time series.
    find_grid_nodes(cells=None, use_grid_lines=False)
        Identify nearest electrical grid connection points.
    score_cells(cells=None)
        Calculate economic scores based on LCOE methodology.
    get_clusters(scored_cells=None, score_tolerance=200, wcss_tolerance=None)
        Perform spatial clustering of viable sites.
    get_cluster_timeseries(clusters=None, dissolved_indices=None, cells_timeseries=None)
        Generate representative time series for each cluster.
    build(select_top_sites=False, use_grid_lines=False,
          get_clusters=False, clean_store=False, memory_resource_limitation=True)
        Execute the full assessment workflow.
    export_results(resource_type, region, weather_year, resource_clusters,
                   cluster_timeseries, save_to)
        Export results as CSV files for downstream models.
    create_summary_info(resource_type, region, sites, timeseries)
        Build a plain-text summary of exported results.
    dump_export_metadata(info, save_to)
        Prepend summary text to the persistent results log file.
    get_top_sites(sites, sites_timeseries, resource_max_capacity)
        Filter clusters to highest-potential sites within a capacity budget.

    Examples
    --------
    Basic wind assessment:

    >>> from RESource.RESources import RESources_builder
    >>> builder = RESources_builder(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind",
    ...     weather_year=2020,
    ... )
    >>> builder.build(select_top_sites=True, use_grid_lines=True)

    Step-by-step with intermediate inspection:

    >>> builder = RESources_builder("config/config.yaml", "AB", "solar", 2020)
    >>> cells          = builder.get_grid_cells()
    >>> cells_cap      = builder.get_cell_capacity()
    >>> scored         = builder.score_cells()
    >>> clusters       = builder.get_clusters()

    Notes
    -----
    - Inherits configuration parsing from AttributesParser.
    - Uses HDF5 storage for efficient handling of large geospatial datasets.
    - Implements caching to avoid redundant computations.
    - Supports both solar PV and onshore wind technologies.
    - Economic calculations follow the NREL LCOE methodology.
    - Clustering uses k-means with automatic cluster-count optimisation.
    """

    def __post_init__(self):
        """Initialise all component classes and shared state."""

        super().__post_init__()

        utils.print_module_title(f"Initiating RESource Builder | {__name__}")

        # Shared kwargs forwarded to every component class
        self.required_args = {
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type,
            "weather_year": self.weather_year,
        }

        self.country_name = self.get_country()
        if self.country_name is None:
            utils.print_warning("Country name is not set in the configuration file.")

        # ── Component classes ────────────────────────────────────────────────
        self.units = Units(**self.required_args)
        self.gridcells = GridCells(**self.required_args)
        self.timeseries = Timeseries(**self.required_args)
        self.gadmBoundary = GADMBoundaries(**self.required_args)
        self.gridNodesProcessor = GridNodeLocator(**self.required_args)
        self.datahandler = DataHandler(self.store)
        self.cell_processor = CellCapacityProcessor(**self.required_args)
        self.era5_cutout = ERA5Cutout(**self.required_args)
        self.scorer = CellScorer(**self.required_args)
        self.gwa_cells = GWACells(**self.required_args)

        self.region_name = self.get_region_name()

        # ── Mutable state placeholders ────────────────────────────────────────
        self.store_grid_cells: gpd.GeoDataFrame = None
        self.region_grid_cells: gpd.GeoDataFrame = None
        self.scored_cells: gpd.GeoDataFrame = None
        self.clusters_nt = None

        # ── Temporal snapshot ─────────────────────────────────────────────────
        self.start_date, self.end_date = self.get_snapshot()
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"Snapshot: {self.start_date}  →  {self.end_date}",
        )

        # Persist config alongside results for reproducibility
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__} | Saving configuration to results directory...",
        )
        utils.save_to_yaml(
            self.config,
            self.results_save_to / f"config_{self.region_short_code}_{self.RUN_ID}.yaml",
        )
        utils.save_to_yaml(
            self.config_provenance,
            self.results_save_to / f"config_provenance_{self.region_short_code}_{self.RUN_ID}.yaml",
        )

    # ── Data-store helpers ────────────────────────────────────────────────────

    def clean_data_store(self):
        """Remove all datasets from the HDF5 store for this region/resource."""
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1, message=f"{__name__} | Cleaning data store..."
        )
        self.datahandler.clean_store()
        utils.print_update(level=PRINT_LEVEL_BASE + 2, message=f"{__name__} | Data store cleaned.")

    # ── Step 1 : Grid cells ───────────────────────────────────────────────────

    def get_grid_cells(self) -> gpd.GeoDataFrame:
        """
        Generate a regular spatial grid covering the analysis region.

        Returns
        -------
        gpd.GeoDataFrame
            Grid cells with geometry, centroid coordinates, and unique identifiers.
        """
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1, message=f"{__name__} | Preparing grid cells..."
        )
        self.region_grid_cells = self.gridcells.get_default_grid()
        utils.print_update(level=PRINT_LEVEL_BASE + 2, message=f"{__name__} | Grid cells ready.")
        return self.region_grid_cells

    # ── Step 1b : Grid-node proximity ─────────────────────────────────────────

    def _find_grid_nodes_from_osm(self) -> gpd.GeoDataFrame:
        """Attach nearest OSM transmission-line connection points to stored cells."""
        utils.print_update(
            level=PRINT_LEVEL_BASE + 3,
            message=f"{__name__} | Using OSM grid lines for connection point analysis...",
        )
        self.grid_lines = self.gridNodesProcessor.get_OSM_grid_lines()

        if self.grid_lines is None or len(self.grid_lines) == 0:
            utils.print_update(
                level=PRINT_LEVEL_BASE + 3,
                message=f"{__name__} | Warning: no OSM grid lines found for {self.region_short_code}.",
            )
            return self.store_grid_cells

        self.store_grid_cells["centroid"] = self.store_grid_cells.apply(
            lambda row: Point(row["x"], row["y"]), axis=1
        )
        connection_results = self.store_grid_cells.apply(
            lambda row: self.gridNodesProcessor.find_nearest_connection_point(
                row["centroid"], row["geometry"], self.store_grid_cells, self.grid_lines
            ),
            axis=1,
            result_type="expand",
        )
        self.store_grid_cells[["nearest_connection_point", "nearest_distance"]] = connection_results
        self.datahandler.to_store(self.store_grid_cells, "cells")
        self.datahandler.to_store(self.grid_lines, "lines")
        self.datahandler.refresh()
        self.store_grid_cells = self.datahandler.from_store("cells")
        return self.store_grid_cells

    def _load_uploaded_substations(self, csv_path: Path) -> gpd.GeoDataFrame:
        """Load an uploaded CSV and maintain its processed GeoDataFrame cache."""
        if not csv_path.is_file():
            raise FileNotFoundError(f"Configured substations file not found: {csv_path}")

        processed_path = self.get_processed_substations_path(csv_path)
        cache_is_current = (
            processed_path.is_file() and processed_path.stat().st_mtime >= csv_path.stat().st_mtime
        )
        if cache_is_current:
            substations = pd.read_pickle(processed_path)
            if not isinstance(substations, gpd.GeoDataFrame):
                raise ValueError(
                    f"Processed substations cache is not a GeoDataFrame: {processed_path}"
                )
            return substations

        substations_df = pd.read_csv(csv_path)
        required_columns = {"longitude", "latitude"}
        missing_columns = required_columns.difference(substations_df.columns)
        if missing_columns:
            raise ValueError(
                "Substations data is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        substations = gpd.GeoDataFrame(
            substations_df,
            geometry=gpd.points_from_xy(substations_df["longitude"], substations_df["latitude"]),
            crs="EPSG:4326",
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        substations.to_pickle(processed_path)
        utils.print_update(
            level=PRINT_LEVEL_BASE + 2,
            message=f"{__name__} | Processed substations saved to {processed_path}",
        )
        return substations

    def find_grid_nodes(
        self,
        cells: gpd.GeoDataFrame = None,
        use_grid_lines: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Attach the nearest electrical grid connection point to each cell.

        Parameters
        ----------
        cells : gpd.GeoDataFrame, optional
            Pre-loaded cells; loads from HDF5 store if None.
        use_grid_lines : bool
            Force OSM transmission lines instead of an available substation source.

        Returns
        -------
        gpd.GeoDataFrame
            Cells with 'nearest_connection_point' and distance columns appended.
        """
        self.cutout, self.region_boundary = self.era5_cutout.get_era5_cutout(
            weather_year=self.weather_year
        )

        if cells is None:
            self.datahandler.refresh()
            self.store_grid_cells = self.datahandler.from_store("cells")
        else:
            self.store_grid_cells = cells.copy()

        utils.print_update(
            level=PRINT_LEVEL_BASE + 1, message=f"{__name__} | Grid node location initiated..."
        )

        if self.country_name == "Canada" and not use_grid_lines:
            utils.print_update(
                level=PRINT_LEVEL_BASE + 3,
                message=f"{__name__} | Using CODERS substations for connection point analysis...",
            )
            # CODERS credentials are required only for this connection strategy.
            # Defer initialization so OSM workflows remain credential-free.
            try:
                self.coders = CODERSData(**self.required_args)
                if not self.coders.api_user:
                    raise RuntimeError("CODERS credentials are unavailable")
                self.grid_ss = self.coders.get_table_provincial("substations")
                if self.grid_ss is None or len(self.grid_ss) == 0:
                    raise RuntimeError("CODERS returned no substations for the region")
                connection_filter = self.coders.coders_data_config.get("connection_filter", {})
                transmission_lines = None
                if connection_filter.get("enabled", True) and connection_filter.get(
                    "require_transmission_endpoint", True
                ):
                    transmission_lines = self.coders.get_table_provincial("transmission_lines")
                self.grid_ss = self.coders.filter_connection_substations(
                    self.grid_ss, transmission_lines
                )
                if self.grid_ss.empty:
                    raise RuntimeError("CODERS connection filter returned no eligible substations")
            except Exception as exc:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__} | CODERS unavailable ({exc}); falling back to OSM.",
                    alert=True,
                )
                return self._find_grid_nodes_from_osm()
            self.region_grid_cells_cap_with_nodes = (
                self.gridNodesProcessor.find_grid_nodes_ERA5_cells(
                    self.grid_ss, self.store_grid_cells
                )
            )
            self.datahandler.to_store(self.store_grid_cells, "cells")
            self.datahandler.to_store(self.grid_ss, "substations")

        elif not use_grid_lines and (substations_path := self.get_buses_path()) is not None:
            try:
                self.grid_ss = self._load_uploaded_substations(substations_path)
                self.region_grid_cells_cap_with_nodes = (
                    self.gridNodesProcessor.find_grid_nodes_ERA5_cells(
                        self.grid_ss, self.store_grid_cells
                    )
                )
                self.datahandler.to_store(self.store_grid_cells, "cells")
                self.datahandler.to_store(self.grid_ss, "substations")
            except (FileNotFoundError, OSError, ValueError) as exc:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__} | Uploaded substations unavailable ({exc}); falling back to OSM.",
                    alert=True,
                )
                return self._find_grid_nodes_from_osm()

        else:
            return self._find_grid_nodes_from_osm()

        self.datahandler.refresh()
        self.store_grid_cells = self.datahandler.from_store("cells")
        return self.store_grid_cells

    # ── Step 2 : Land availability & capacity ─────────────────────────────────

    def get_cell_capacity(self) -> gpd.GeoDataFrame:
        """
        Calculate renewable energy capacity potential for each grid cell.

        Applies land exclusion layers, land-use intensity factors, and cell
        geometry to derive developable capacity.

        Returns
        -------
        gpd.GeoDataFrame
            Grid cells with capacity columns appended.
        """
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1, message=f"{__name__} | Calculating cell capacity..."
        )
        self.cells_with_capacity = self.cell_processor.get_capacity()
        utils.print_update(level=PRINT_LEVEL_BASE + 2, message=f"{__name__} | Cell capacity ready.")
        return self.cells_with_capacity

    # ── Step 3 : Weather data ─────────────────────────────────────────────────

    def extract_weather_data(self):
        """
        Extract ERA5 wind speed (or solar irradiance) for each grid cell.

        For wind: imputes ERA5 wind speed at cell centroids from the cutout and
        stores the result. For solar: placeholder — Global Solar Atlas integration
        is not yet implemented.
        """
        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__} | Extracting ERA5 weather data from cutout...",
        )
        self.store_grid_cells = self.datahandler.from_store("cells")
        self.cutout, _ = self.era5_cutout.get_era5_cutout(weather_year=self.weather_year)

        if self.resource_type == "wind":
            if "windspeed_ERA5" in self.store_grid_cells.columns:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__} | 'windspeed_ERA5' already present — skipping extraction.",
                )
            else:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__} | Extracting 'windspeed_ERA5' from cutout...",
                )
                _updated = wind.impute_ERA5_windspeed_to_Cells(self.cutout, self.store_grid_cells)
                self.store_grid_cells_updated = utils.assign_cell_id(
                    _updated, self.sub_national_unit_tag
                )
                self.datahandler.to_store(self.store_grid_cells_updated, "cells")
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 2,
                    message=f"{__name__} | 'windspeed_ERA5' extraction complete.",
                )
                return self.store_grid_cells_updated

        elif self.resource_type == "solar":
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"{__name__} | Global Solar Atlas integration not yet implemented — skipping.",
            )

    def update_gwa_scaled_params(self, memory_resource_limitation: bool | None = False):
        """
        Rescale ERA5 wind speeds using Global Wind Atlas high-resolution data.

        No-op for solar resources. Results are written back to the HDF5 store.

        Parameters
        ----------
        memory_resource_limitation : bool
            Pass True to constrain memory usage during GWA mapping.

        Returns
        -------
        gpd.GeoDataFrame
            Updated cells with GWA-corrected wind speed columns.
        """
        if self.resource_type == "wind":
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__} | Mapping Global Wind Atlas data to ERA5 cells...",
            )
            required_cols = ["CF_IEC2", "CF_IEC3", "windspeed_gwa", "windspeed_ERA5"]
            self.store_grid_cells = self.datahandler.from_store("cells")
            if all(c in self.store_grid_cells.columns for c in required_cols):
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 3,
                    message=f"{__name__} | GWA columns already present — skipping.",
                )
            else:
                utils.print_update(
                    level=PRINT_LEVEL_BASE + 3,
                    message=f"{__name__} | Extracting GWA columns from source...",
                )
                self.gwa_cells.map_GWA_cells_to_ERA5(
                    aggregation_level=self.sub_national_unit_tag,
                    memory_resource_limitation=memory_resource_limitation,
                )
        elif self.resource_type == "solar":
            utils.print_update(
                level=PRINT_LEVEL_BASE + 2,
                message=f"{__name__} | GWA scaling not applicable for solar — skipping.",
            )

        self.datahandler.refresh()
        self.store_grid_cells = self.datahandler.from_store("cells")
        return self.store_grid_cells

    # ── Step 4 : Capacity-factor time series ──────────────────────────────────

    def get_CF_timeseries(
        self,
        cells: gpd.GeoDataFrame = None,
        weather_year: int | None = None,
        force_update: bool = False,
    ) -> tuple:
        """
        Compute hourly capacity-factor time series for all grid cells.

        Parameters
        ----------
        cells : gpd.GeoDataFrame, optional
            Cells to process; loads from HDF5 store if None.
        weather_year : int, optional
            Year of weather data to use; defaults to the instance's weather_year.
        force_update : bool
            Force recomputation even if results are cached.

        Returns
        -------
        tuple
            (cells_with_mean_CF, cells_timeseries_df)
        """
        utils.print_update(
            level=PRINT_LEVEL_BASE + 3,
            message=f"{__name__} | Building capacity-factor time series...",
        )
        if cells is None:
            self.datahandler.refresh()
            cells = self.datahandler.from_store("cells")
        cells_withCF, cells_timeseries = self.timeseries.get_timeseries(
            cells=cells, weather_year=weather_year
        )
        return cells_withCF, cells_timeseries

    # ── Step 5 : Scoring ──────────────────────────────────────────────────────

    def score_cells(self, cells: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """
        Score each grid cell by LCOE ($/MWh).

        Parameters
        ----------
        cells : gpd.GeoDataFrame, optional
            Cells to score; loads from HDF5 store if None.

        Returns
        -------
        gpd.GeoDataFrame
            Cells with LCOE and cost columns appended.
        """
        if cells is None:
            self.datahandler.refresh()
            cells = self.datahandler.from_store("cells")

        self.scored_cells = self.scorer.get_cell_score(
            cells=cells,
            CF_column=f"{self.resource_type}_CF_mean",
        )
        self.datahandler.to_store(self.scored_cells, "cells", force_update=True)
        return self.scored_cells

    # ── Step 6a : Clustering ──────────────────────────────────────────────────

    def get_clusters(
        self,
        scored_cells: gpd.GeoDataFrame = None,
        score_tolerance: float = 200,
        wcss_tolerance=None,
    ):
        """
        Cluster viable grid cells into representative sites using k-means on LCOE.

        Parameters
        ----------
        scored_cells : gpd.GeoDataFrame, optional
            Pre-scored cells; runs score_cells() if None or not yet computed.
        score_tolerance : float
            Maximum LCOE threshold ($/MWh) for cells eligible for clustering.
        wcss_tolerance : float, optional
            Within-cluster sum-of-squares tolerance for auto-selecting k.
            Falls back to config value if None.

        Returns
        -------
        namedtuple
            cluster_data(clusters: GeoDataFrame, dissolved_indices: dict)
        """
        self.resource_disaggregation_config = self.get_resource_disaggregation_config()
        self.wcss_tolerance = wcss_tolerance if wcss_tolerance else self.get_wcss_tolerance()
        self.gadm_config = self.get_gadm_config()

        utils.print_update(
            level=PRINT_LEVEL_BASE + 1,
            message=f"{__name__} | Clustering resources...",
        )

        # Ensure scored cells are available
        if scored_cells is not None:
            self.scored_cells = scored_cells
        elif self.scored_cells is None:
            utils.print_update(
                level=PRINT_LEVEL_BASE + 3,
                message=f"{__name__} | No scored cells found — running score_cells().",
            )
            self.scored_cells = self.score_cells()

        utils.print_warning(
            f"{__name__} | Filtering: LCOE ≤ {score_tolerance} $/MWh  &  "
            f"grid distance ≤ {self.get_grid_proximity_km()} km"
        )

        node_distance_col = utils.get_available_column(
            self.scored_cells, ["nearest_station_distance_km", "nearest_distance"]
        )
        scored_cells_filtered = self.scored_cells[
            (self.scored_cells[f"lcoe_{self.resource_type}"] <= score_tolerance)
            & (
                self.scored_cells[node_distance_col]
                <= self.gridNodesProcessor.grid_proximity_threshold_km
            )
        ]

        self.vis_dir = self.get_vis_dir()

        self.ERA5_cells_cluster_map, self.region_optimal_k_df = cluster.cells_to_cluster_mapping(
            scored_cells_filtered,
            self.vis_dir,
            self.wcss_tolerance,
            self.sub_national_unit_tag,
            self.resource_type,
            [f"lcoe_{self.resource_type}", f"potential_capacity_{self.resource_type}"],
        )

        self.cell_cluster_gdf, self.dissolved_indices = cluster.create_cells_Union_in_clusters(
            self.ERA5_cells_cluster_map,
            self.region_optimal_k_df,
            self.sub_national_unit_tag,
            self.resource_type,
        )

        self.cell_cluster_gdf["Operational_life"] = self.resource_disaggregation_config.get(
            "Operational_life", 20
        )
        self.cell_cluster_gdf.loc[:, "resource_type"] = self.resource_type.lower()

        cluster_data = namedtuple("cluster_data", ["clusters", "dissolved_indices"])
        self.clusters_nt = cluster_data(self.cell_cluster_gdf, self.dissolved_indices)

        self.datahandler.to_store(
            self.cell_cluster_gdf, f"clusters/{self.resource_type}", force_update=True
        )
        self.dissolved_cell_indices_df = pd.DataFrame(self.dissolved_indices).T
        self.dissolved_cell_indices_df.index.name = self.sub_national_unit_tag
        self.datahandler.to_store(
            self.dissolved_cell_indices_df,
            f"dissolved_indices/{self.resource_type}",
            force_update=True,
        )

        return self.clusters_nt

    # ── Step 6b : Cluster time series ─────────────────────────────────────────

    def get_cluster_timeseries(
        self,
        clusters: gpd.GeoDataFrame = None,
        dissolved_indices: pd.DataFrame = None,
        cells_timeseries: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Build representative hourly CF profiles for each cluster.

        Parameters
        ----------
        clusters : gpd.GeoDataFrame, optional
            Cluster GeoDataFrame; loads from store if None.
        dissolved_indices : pd.DataFrame, optional
            Cell-to-cluster mapping; loads from store if None.
        cells_timeseries : pd.DataFrame, optional
            Cell-level time series; loads from store if None.

        Returns
        -------
        pd.DataFrame
            Hourly capacity-factor profiles indexed by time, one column per cluster.
        """
        self.cells_timeseries = cells_timeseries
        self.cell_cluster_gdf = clusters
        self.dissolved_cell_indices_df = dissolved_indices

        if self.cells_timeseries is None:
            self.cells_timeseries = self.datahandler.from_store(f"timeseries/{self.resource_type}")
        if self.cell_cluster_gdf is None:
            self.cell_cluster_gdf = self.datahandler.from_store(f"clusters/{self.resource_type}")
            utils.print_update(
                level=PRINT_LEVEL_BASE + 1,
                message=f"{__name__} | Building representative profiles for "
                f"{len(self.cell_cluster_gdf)} clusters...",
            )
        if self.dissolved_cell_indices_df is None:
            self.dissolved_cell_indices_df = self.datahandler.from_store(
                f"dissolved_indices/{self.resource_type}"
            )

        self.cluster_ts_df = self.timeseries.get_cluster_timeseries(
            self.cell_cluster_gdf,
            self.cells_timeseries,
            self.dissolved_cell_indices_df,
            self.sub_national_unit_tag,
        )
        return self.cluster_ts_df

    # ── Full pipeline orchestrator ─────────────────────────────────────────────

    def build(
        self,
        select_top_sites: bool | None = False,
        use_grid_lines: bool | None = False,
        make_clusters: bool | None = False,
        clean_store: bool | None = False,
        memory_resource_limitation: bool | None = True,
    ):
        """
        Run the full resource assessment pipeline for the configured region and resource type.

        Parameters
        ----------
        select_top_sites : bool
            If True, filter clusters to a capacity-budget subset (Step 7).
            Implies clustering even when get_clusters=False.
        use_grid_lines : bool
            Force OSM transmission lines for connection point analysis.
        get_clusters : bool
            Explicitly run clustering and cluster time series (Steps 6a/6b).
            Set True to inspect cluster outputs without top-site filtering.
        clean_store : bool
            Wipe the HDF5 store before running (fresh start).
        memory_resource_limitation : bool
            Limit memory usage in GWA mapping (passed to update_gwa_scaled_params).
        """
        utils.print_module_title(
            f"Initiating {self.resource_type} pipeline | {self.get_region_name()}"
        )
        self.memory_resource_limitation = memory_resource_limitation

        # ── Step 0 : Optionally wipe store ────────────────────────────────────
        if clean_store:
            self.clean_data_store()

        # ── Config-hash cache ─────────────────────────────────────────────────
        # Skips any step whose relevant config sections are unchanged AND whose
        # output already exists in the HDF5 store.
        cache = StepCache(
            store_path=self.store,
            config=self.config,
            resource_type=self.resource_type,
        )

        # ── Step 1 : Grid cells & grid-node proximity ─────────────────────────
        utils.print_banner("Step 1 : Prepare grid cells and locate nearest grid connection nodes")
        if cache.is_current("grid_cells", "cells") and cache.is_current("grid_nodes", "cells"):
            utils.print_update(
                level=2, message="Step 1: grid config unchanged — loading from store."
            )
        else:
            self.get_grid_cells()
            self.find_grid_nodes(use_grid_lines=use_grid_lines)
            cache.mark_done("grid_cells")
            cache.mark_done("grid_nodes")

        # ── Step 2 : Land availability & capacity ─────────────────────────────
        utils.print_banner("Step 2 : Calculate land availability and capacity potential")
        if cache.is_current("cell_capacity", "cells"):
            utils.print_update(
                level=2, message="Step 2: capacity config unchanged — loading from store."
            )
        else:
            self.get_cell_capacity()
            cache.mark_done("cell_capacity")

        # ── Step 3 : Weather data & GWA rescaling ─────────────────────────────
        utils.print_banner("Step 3 : Extract weather data and apply GWA wind-speed correction")
        if cache.is_current("weather_data", "cells") and cache.is_current("gwa_scaling", "cells"):
            utils.print_update(
                level=2, message="Step 3: weather config unchanged — loading from store."
            )
        else:
            self.extract_weather_data()
            self.update_gwa_scaled_params(self.memory_resource_limitation)
            cache.mark_done("weather_data")
            cache.mark_done("gwa_scaling")

        # ── Step 4 : Capacity-factor time series ──────────────────────────────
        utils.print_banner("Step 4 : Build capacity-factor time series")
        ts_store_key = f"timeseries/{self.resource_type}"
        if cache.is_current("cf_timeseries", ts_store_key):
            utils.print_update(
                level=2, message="Step 4: cutout/turbine config unchanged — loading from store."
            )
        else:
            self.get_CF_timeseries()
            cache.mark_done("cf_timeseries")

        # ── Step 5 : Economic scoring ─────────────────────────────────────────
        utils.print_banner("Step 5 : Score cells by LCOE")
        if cache.is_current("scoring", "cells"):
            utils.print_update(
                level=2, message="Step 5: economic config unchanged — loading from store."
            )
            self.datahandler.refresh()
            self.scored_cells = self.datahandler.from_store("cells")
        else:
            self.score_cells()
            cache.mark_done("scoring")

        # ── Step 6 : Clustering (optional explicit step) ──────────────────────
        cluster_store_key = f"clusters/{self.resource_type}"
        resource_clusters = None
        cluster_timeseries = None
        if make_clusters:
            utils.print_banner("Step 6a : Cluster cells into representative sites")
            if cache.is_current("clustering", cluster_store_key):
                utils.print_update(
                    level=2, message="Step 6: cluster config unchanged — loading from store."
                )
                self.cell_cluster_gdf = self.datahandler.from_store(cluster_store_key)
            else:
                self.get_clusters()
                cache.mark_done("clustering")
            resource_clusters = self.cell_cluster_gdf
            utils.print_banner("Step 6b : Build cluster representative time series")
            cluster_timeseries = self.get_cluster_timeseries()

        # Units dictionary (documentation only — no calculation impact)
        self.units.create_units_dictionary()

        # ── Step 7 : Site selection & export ─────────────────────────────────
        self.clusters_save_to = self.results_save_to / "clusters"

        if select_top_sites:
            utils.print_banner("Step 7 : Select top sites within capacity budget")
            resource_max_capacity = self.resource_disaggregation_config.get("max_capacity", 50)

            # Load or compute clustering once, then reuse it for selection/export.
            if resource_clusters is not None:
                utils.print_update(level=2, message="Step 7: reusing clusters prepared in Step 6.")
            elif cache.is_current("clustering", cluster_store_key):
                resource_clusters = self.datahandler.from_store(cluster_store_key)
                utils.print_update(
                    level=2, message="Step 7: cluster config unchanged — loading from store."
                )
            else:
                resource_clusters = self.get_clusters().clusters
                cache.mark_done("clustering")

            if cluster_timeseries is None:
                cluster_timeseries = self.get_cluster_timeseries(clusters=resource_clusters)
            resource_clusters, cluster_timeseries = self.get_top_sites(
                sites=resource_clusters,
                sites_timeseries=cluster_timeseries,
                resource_max_capacity=resource_max_capacity,
            )
            utils.print_module_title(
                f"Top sites selected | {self.resource_type} | {self.get_region_name()}"
            )

        else:
            # Return all clusters unfiltered
            if resource_clusters is None:
                if cache.is_current("clustering", cluster_store_key):
                    resource_clusters = self.datahandler.from_store(cluster_store_key)
                else:
                    resource_clusters = self.get_clusters().clusters
                    cache.mark_done("clustering")
            if cluster_timeseries is None:
                cluster_timeseries = self.get_cluster_timeseries(clusters=resource_clusters)
            utils.print_module_title(
                f"All sites (clusters) | {self.resource_type} | {self.get_region_name()}"
            )

        # Export runs regardless of select_top_sites
        self.export_results(
            resource_type=self.resource_type,
            region=self.region_name,
            weather_year=self.weather_year,
            resource_clusters=resource_clusters,
            cluster_timeseries=cluster_timeseries,
            save_to=self.clusters_save_to,
        )
        sites_summary = self.create_summary_info(
            self.resource_type, self.region_name, resource_clusters, cluster_timeseries
        )
        self.dump_export_metadata(sites_summary, self.results_save_to)

    # ── Static utility methods ────────────────────────────────────────────────

    @staticmethod
    def export_results(
        resource_type: str,
        region: str,
        weather_year: int,
        resource_clusters: pd.DataFrame,
        cluster_timeseries: pd.DataFrame,
        save_to: Path | None = Path("results"),
    ):
        """
        Write cluster results and time series to CSV files.

        Output filenames embed resource type, region, and weather year so that
        multiple runs never overwrite each other.

        Parameters
        ----------
        resource_type : str
            'wind' or 'solar'.
        region : str
            Region name used in output filenames.
        weather_year : int
            Weather year used in output filenames.
        resource_clusters : DataFrame or GeoDataFrame
            Cluster-level results (geometry column excluded on export).
        cluster_timeseries : DataFrame
            Hourly CF time series, one column per cluster.
        save_to : Path
            Output directory. Created if it does not exist.
        """
        if not isinstance(resource_clusters, (pd.DataFrame, gpd.GeoDataFrame)):
            raise TypeError(
                f"resource_clusters must be a DataFrame or GeoDataFrame, "
                f"got {type(resource_clusters).__name__}."
            )
        if not isinstance(cluster_timeseries, pd.DataFrame):
            raise TypeError(
                f"cluster_timeseries must be a DataFrame, got {type(cluster_timeseries).__name__}."
            )

        save_to = utils.ensure_path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)

        # Drop geometry for tabular export
        clusters_csv = resource_clusters[[c for c in resource_clusters.columns if c != "geometry"]]

        stem = f"resource_options_{resource_type}_{region}_{weather_year}"
        clusters_csv.to_csv(save_to / f"{stem}.csv", index=True)
        cluster_timeseries.to_csv(save_to / f"{stem}_timeseries.csv", index=True)

        utils.print_update(
            level=2,
            message=f"{resource_type} results exported → {save_to / stem}.csv",
        )

    @staticmethod
    def create_summary_info(
        resource_type: str, region: str, sites: pd.DataFrame, timeseries: pd.DataFrame
    ) -> str:
        """
        Build a plain-text summary string for the exported results.

        Parameters
        ----------
        resource_type : str
        region : str
        sites : DataFrame
            Cluster results containing a 'potential_capacity' column (in MW).
        timeseries : DataFrame
            Cluster time series.

        Returns
        -------
        str
            Formatted summary text.
        """
        formatted_time = current_local_time.strftime("%H:%M:%S")
        info = (
            f"{'_' * 25} Latest results summary {'_' * 25}\n"
            f"{'-' * 100}\n"
            f"  {resource_type.upper()} | {region.upper()}\n"
            f"  Total capacity : {sites['potential_capacity'].sum() / 1e3:.2f} GW\n"
            f"  No. of sites   : {len(sites)}\n"
            f"  Snapshot steps : {len(timeseries)}\n"
            f"  Generated at   : {formatted_time}\n"
            f"{'-' * 100}\n"
        )
        return info

    @staticmethod
    def dump_export_metadata(info: str, save_to: Path | None = Path("results/linking")):
        """
        Prepend a summary string to the persistent results log file.

        If the log file does not exist it is created; existing content is preserved
        below the new entry so that the most recent run is always at the top.

        Parameters
        ----------
        info : str
            Summary text from create_summary_info().
        save_to : Path
            Directory in which 'Resource_options_summary.txt' is written.
        """
        save_to = utils.ensure_path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        file_path = save_to / "Resource_options_summary.txt"

        existing = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        file_path.write_text(info + "\n" + existing, encoding="utf-8")

    @staticmethod
    def get_top_sites(
        sites: gpd.GeoDataFrame | pd.DataFrame,
        sites_timeseries: pd.DataFrame,
        resource_max_capacity: float,
    ) -> tuple[gpd.GeoDataFrame | pd.DataFrame, pd.DataFrame]:
        """
        Select the highest-ranked clusters that fit within a capacity budget.

        Iterates over clusters in score order, accumulating capacity until the
        budget is reached. If the last added cluster overshoots, its capacity is
        trimmed to the remaining headroom.

        Parameters
        ----------
        sites : GeoDataFrame or DataFrame
            Cluster results sorted by score (best first), with a
            'potential_capacity' column in MW.
        sites_timeseries : DataFrame
            Hourly CF time series indexed by cluster identifier.
        resource_max_capacity : float
            Maximum investment capacity in GW.

        Returns
        -------
        Tuple[GeoDataFrame, DataFrame]
            (selected_clusters, timeseries_for_selected_clusters)
        """
        budget_mw = resource_max_capacity * 1000
        print(f">>> Selecting top sites for {resource_max_capacity} GW capacity investment...")

        top_sites = sites.copy()

        if top_sites["potential_capacity"].iloc[0] < budget_mw:
            # Greedy selection
            selected_rows: list = []
            total_capacity: float = 0.0

            for index, row in top_sites.iterrows():
                if total_capacity + row["potential_capacity"] <= budget_mw:
                    selected_rows.append(index)
                    total_capacity += row["potential_capacity"]
                else:
                    break

            top_sites = top_sites.loc[selected_rows]

            remaining = budget_mw - top_sites["potential_capacity"].sum()
            if remaining > 0:
                # Attempt to partially fill from the next cluster
                mask = sites.index > top_sites.index.max()
                extra = sites[mask].head(1).copy()
                if len(extra) > 0:
                    print(
                        f"\n!! Note: cluster {extra.index[-1]} capacity trimmed from "
                        f"{extra['potential_capacity'].iloc[0] / 1000:.2f} GW "
                        f"to {remaining / 1000:.2f} GW to fit the budget.\n"
                    )
                    extra["potential_capacity"] = remaining
                    top_sites = pd.concat([top_sites, extra])
                else:
                    print(
                        f"\n!! Note: no additional cluster available; "
                        f"remaining headroom {remaining / 1000:.2f} GW unfilled.\n"
                    )
        else:
            # Single cluster already exceeds budget — trim it
            original_mw = sites["potential_capacity"].iloc[0]
            print(
                f"!! Note: first cluster ({original_mw / 1000:.2f} GW) exceeds budget "
                f"({resource_max_capacity} GW) — trimming to budget.\n"
            )
            top_sites = top_sites.iloc[:1].copy()
            top_sites.at[top_sites.index[0], "potential_capacity"] = budget_mw

        top_sites_ts = sites_timeseries[top_sites.index]
        return top_sites, top_sites_ts


# ── Module-level convenience wrapper ──────────────────────────────────────────


def build_resources(
    regions: list, resource_types: list, config_path: str | Path, weather_year: int | None = None
):
    """
    Run the full pipeline for every (region, resource_type) combination.

    Parameters
    ----------
    regions : list of str
        Region short codes to process (e.g. ['BC', 'AB']).
    resource_types : list of str
        Resource types to process (e.g. ['wind', 'solar']).
    config_path : str or Path
        Path to the YAML configuration file.
    weather_year : int, optional
        Weather year. Uses config 'weather_year' key if None.
    """
    for region, resource in product(regions, resource_types):
        module = RESources_builder(
            config_file_path=config_path,
            region_short_code=region,
            resource_type=resource,
            weather_year=weather_year,
        )
        module.build(select_top_sites=True, use_grid_lines=True)
