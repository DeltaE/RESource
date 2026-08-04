"""Load per-run config, cluster, and timeseries data for a ``ScenarioRun``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from RESource.reporting.discovery import ScenarioRun


def frozen_config_path(run: ScenarioRun) -> Path:
    """Path to the resolved config snapshot the pipeline wrote for this run."""
    return run.run_dir / f"config_{run.region_code}_{run.run_dir.name}.yaml"


def load_frozen_config(run: ScenarioRun) -> dict:
    """Load the resolved config snapshot written alongside a run's results."""
    path = frozen_config_path(run)
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def region_full_name(run: ScenarioRun) -> str:
    """Region display name used in result filenames (e.g. 'British Columbia')."""
    config = load_frozen_config(run)
    return config.get("region_mapping", {}).get(run.region_code, {}).get("name", run.region_code)


def cluster_csv_path(run: ScenarioRun, resource_type: str = "solar") -> Path:
    region_name = region_full_name(run)
    return (
        run.run_dir
        / "clusters"
        / f"resource_options_{resource_type}_{region_name}_{run.weather_year}.csv"
    )


def timeseries_csv_path(run: ScenarioRun, resource_type: str = "solar") -> Path:
    clusters_path = cluster_csv_path(run, resource_type)
    return clusters_path.with_name(f"{clusters_path.stem}_timeseries.csv")


def load_clusters(run: ScenarioRun, resource_type: str = "solar") -> pd.DataFrame:
    """Load the cluster/site table (lcoe, CF_mean, potential_capacity, ...)."""
    path = cluster_csv_path(run, resource_type)
    if not path.is_file():
        raise FileNotFoundError(
            f"No {resource_type} cluster results for scenario '{run.scenario_name}' "
            f"region '{run.region_code}': {path}"
        )
    return pd.read_csv(path)


def load_timeseries(run: ScenarioRun, resource_type: str = "solar") -> pd.DataFrame:
    """Load the hourly per-cluster capacity-factor timeseries, indexed by time."""
    path = timeseries_csv_path(run, resource_type)
    if not path.is_file():
        raise FileNotFoundError(
            f"No {resource_type} timeseries for scenario '{run.scenario_name}' "
            f"region '{run.region_code}': {path}"
        )
    return pd.read_csv(path, parse_dates=["time"], index_col="time")
