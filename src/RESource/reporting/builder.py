"""Assemble the country solar report from on-disk pipeline results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from RESource import utility as utils
from RESource.reporting import config_section, data_loading, discovery, solar_plots
from RESource.reporting.discovery import ScenarioRun

TEMPLATE_DIR = Path(__file__).parent / "templates"
RESOURCE_TYPE = "solar"


def _weighted_lcoe(clusters) -> float:
    weights = clusters["potential_capacity"]
    total_weight = weights.sum()
    if total_weight == 0:
        return float("nan")
    return float((clusters["lcoe"] * weights).sum() / total_weight)


def _build_region_scenario_section(run: ScenarioRun) -> tuple[dict, dict]:
    clusters = data_loading.load_clusters(run, RESOURCE_TYPE)
    timeseries = data_loading.load_timeseries(run, RESOURCE_TYPE)
    label = f"{run.scenario_name} — {run.region_code}"

    summary = {
        "scenario_name": run.scenario_name,
        "region_code": run.region_code,
        "run_id": run.run_dir.name,
        "weather_year": run.weather_year,
        "site_count": int(len(clusters)),
        "total_gw": float(clusters["potential_capacity"].sum() / 1000.0),
        "mean_cf": float(clusters["CF_mean"].mean()),
        "weighted_lcoe": _weighted_lcoe(clusters),
    }
    plots = {
        "cf_vs_lcoe": solar_plots.plot_cf_vs_lcoe(clusters, f"CF vs LCOE — {label}"),
        "supply_curve": solar_plots.plot_supply_curve(clusters, f"Supply curve — {label}"),
        "cf_distribution": solar_plots.plot_cf_distribution(clusters, f"CF distribution — {label}"),
        "cf_timeseries": solar_plots.plot_cf_timeseries_monthly(timeseries, f"Monthly CF — {label}"),
    }
    return summary, plots


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["to_yaml"] = lambda data: yaml.dump(
        data, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    return env


def build_report(
    country_code: str,
    regions: list[str] | None = None,
    scenarios: list[str] | None = None,
    out_dir: str | Path = "reports",
) -> Path:
    """Build a self-contained solar HTML report for a country's scenarios.

    Uses the most recent pipeline run already on disk for each requested
    scenario/region combination — this never (re-)runs the assessment
    pipeline. Raises ``FileNotFoundError`` if no matching runs exist yet.
    """
    country_code = country_code.upper()
    runs = discovery.find_scenario_runs(
        country_code, region_codes=regions, scenario_names=scenarios
    )
    if not runs:
        raise FileNotFoundError(
            f"No existing results found on disk for country '{country_code}' "
            f"(regions={regions}, scenarios={scenarios}). Run the pipeline first, "
            "e.g. `resource config/<COUNTRY>/scenarios/<scenario>.yaml --year YYYY`."
        )

    scenario_paths = {run.scenario_config_path for run in runs}
    scenario_summaries = {
        path.stem: config_section.build_scenario_summary(path) for path in scenario_paths
    }

    regions_seen: dict[str, list[ScenarioRun]] = {}
    for run in runs:
        regions_seen.setdefault(run.region_code, []).append(run)

    region_reports = []
    contrast_rows = []
    for region_code, region_runs in sorted(regions_seen.items()):
        sections = []
        for run in sorted(region_runs, key=lambda r: r.scenario_name):
            summary, plots = _build_region_scenario_section(run)
            sections.append(
                {
                    "run": run,
                    "summary": summary,
                    "plots": plots,
                    "scenario": scenario_summaries[run.scenario_name],
                }
            )
            contrast_rows.append(
                {
                    "label": f"{run.scenario_name} ({region_code})",
                    "total_gw": summary["total_gw"],
                    "weighted_lcoe": summary["weighted_lcoe"],
                }
            )
        region_reports.append({"region_code": region_code, "sections": sections})

    contrast_chart = solar_plots.plot_scenario_contrast(contrast_rows) if contrast_rows else None

    country_name = next(iter(scenario_summaries.values()))["resolved_config"].get(
        "country", country_code
    )

    env = _jinja_env()
    template = env.get_template("report.html.j2")
    html = template.render(
        country_code=country_code,
        country_name=country_name,
        resource_type=RESOURCE_TYPE,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        scenario_summaries=scenario_summaries,
        region_reports=region_reports,
        contrast_chart=contrast_chart,
        contrast_rows=contrast_rows,
    )

    country_kwd = country_name.replace(" ", "")
    out_dir = utils.ensure_path(Path(out_dir) / country_kwd)
    region_tag = "_".join(sorted(regions_seen)) if len(regions_seen) <= 3 else "AllRegions"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{country_kwd}_{region_tag}_report_{timestamp}.html"
    out_path.write_text(html, encoding="utf-8")
    utils.print_update(message=f"Report written to {out_path}")
    return out_path
