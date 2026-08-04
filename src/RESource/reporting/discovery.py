"""Locate the most recent on-disk pipeline run for each configured scenario.

This module never runs the assessment pipeline; it only inspects
``config/<COUNTRY>/scenarios/*.yaml`` and matches each scenario's ``run_id``
against existing ``results/<Country>/<Region>/<RUN_ID>/`` directories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from RESource import utility as utils

CONFIG_ROOT = Path("config")
RESULTS_ROOT = Path("results")

# RUN_ID = f"{Scenario.run_id}_{weather_year}_{today_str}" (AttributesParser.get_RUN_ID)
_RUN_DIR_RE = re.compile(r"^(?P<run_id>.+)_(?P<year>\d{4})_(?P<date>\d{8})$")


@dataclass(frozen=True)
class ScenarioRun:
    """A scenario resolved to a specific run directory already on disk."""

    scenario_name: str
    scenario_config_path: Path
    run_id: str
    region_code: str
    run_dir: Path
    weather_year: int


def list_scenario_configs(country_code: str) -> list[Path]:
    """Return every scenario YAML declared for a country config."""
    scenarios_dir = CONFIG_ROOT / country_code.upper() / "scenarios"
    if not scenarios_dir.is_dir():
        raise FileNotFoundError(
            f"No scenarios directory found for country '{country_code}': {scenarios_dir}"
        )
    paths = sorted(scenarios_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No scenario YAML files found under {scenarios_dir}")
    return paths


def _find_latest_run_dir(region_results_dir: Path, run_id: str) -> Path | None:
    """Return the most recently generated run directory matching ``run_id``."""
    if not region_results_dir.is_dir():
        return None
    candidates = []
    for child in region_results_dir.iterdir():
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if not match or match.group("run_id") != run_id:
            continue
        candidates.append((match.group("date"), match.group("year"), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def find_scenario_runs(
    country_code: str,
    region_codes: list[str] | None = None,
    scenario_names: list[str] | None = None,
) -> list[ScenarioRun]:
    """Resolve every requested scenario/region combination to its latest run on disk.

    Scenario/region combinations without a matching run directory are skipped
    (not every scenario needs to have been executed for every region).
    """
    scenario_paths = list_scenario_configs(country_code)
    if scenario_names:
        wanted = set(scenario_names)
        scenario_paths = [p for p in scenario_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in scenario_paths}
        if missing:
            raise ValueError(f"Unknown scenario(s) for {country_code}: {sorted(missing)}")

    runs: list[ScenarioRun] = []
    for scenario_path in scenario_paths:
        config, _provenance = utils.resolve_config(scenario_path)
        country_name = config.get("country")
        if not country_name:
            raise ValueError(f"'country' missing from resolved config: {scenario_path}")
        country_kwd = country_name.replace(" ", "")
        run_id = config["Scenario"]["run_id"]
        region_mapping = config.get("region_mapping", {})
        wanted_regions = region_codes or list(region_mapping)
        for region_code in wanted_regions:
            if region_code not in region_mapping:
                raise ValueError(
                    f"Unknown region '{region_code}' for {country_code}. "
                    f"Available: {list(region_mapping)}"
                )
            region_results_dir = RESULTS_ROOT / country_kwd / region_code
            run_dir = _find_latest_run_dir(region_results_dir, run_id)
            if run_dir is None:
                continue
            match = _RUN_DIR_RE.match(run_dir.name)
            runs.append(
                ScenarioRun(
                    scenario_name=scenario_path.stem,
                    scenario_config_path=scenario_path,
                    run_id=run_id,
                    region_code=region_code,
                    run_dir=run_dir,
                    weather_year=int(match.group("year")),
                )
            )
    return runs
