"""Self-contained solar plots for HTML reports.

Deliberately independent of ``RESource.visuals`` (which pulls in folium,
plotly, and ``atlite.ExclusionContainer`` and expects pipeline-internal
objects such as ``dissolved_indices``/HDF5-backed GeoDataFrames). These plots
work only from the flat cluster/timeseries CSVs already written to disk, so a
report can be built without re-running the assessment pipeline or installing
the ``viz`` extra.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_BUBBLE_COLOR = "darkorange"


def _fig_to_data_uri(fig, dpi: int = 140) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_cf_vs_lcoe(clusters: pd.DataFrame, title: str) -> str:
    """Bubble scatter of mean CF vs LCOE, bubble size = potential capacity."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        clusters["CF_mean"],
        clusters["lcoe"],
        s=clusters["potential_capacity"] / 5,
        alpha=0.65,
        c=_BUBBLE_COLOR,
        edgecolors="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean capacity factor")
    ax.set_ylabel("LCOE ($/MWh)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0%}"))
    ax.grid(True, ls=":", linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_supply_curve(clusters: pd.DataFrame, title: str) -> str:
    """Cumulative potential capacity (GW), ordered by ascending LCOE."""
    ordered = clusters.sort_values("lcoe")
    cumulative_gw = ordered["potential_capacity"].cumsum() / 1000.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(cumulative_gw, ordered["lcoe"], where="post", color=_BUBBLE_COLOR, linewidth=1.5)
    ax.set_xlabel("Cumulative potential capacity (GW)")
    ax.set_ylabel("LCOE ($/MWh)")
    ax.set_title(title)
    ax.grid(True, ls=":", linewidth=0.4)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_cf_distribution(clusters: pd.DataFrame, title: str) -> str:
    """Capacity-weighted histogram of mean CF across sites/clusters."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        clusters["CF_mean"],
        weights=clusters["potential_capacity"],
        bins=20,
        color=_BUBBLE_COLOR,
        edgecolor="white",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0%}"))
    ax.set_xlabel("Mean capacity factor")
    ax.set_ylabel("Potential capacity (MW)")
    ax.set_title(title)
    ax.grid(True, ls=":", linewidth=0.4)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_cf_timeseries_monthly(timeseries: pd.DataFrame, title: str) -> str:
    """Monthly mean +/- std of aggregate CF across all clusters/sites."""
    monthly = timeseries.resample("MS").mean()
    mean_values = monthly.mean(axis=1)
    std_values = monthly.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(monthly.index, mean_values, color=_BUBBLE_COLOR, label="Mean CF across clusters")
    ax.fill_between(
        monthly.index,
        mean_values - std_values,
        mean_values + std_values,
        color=_BUBBLE_COLOR,
        alpha=0.2,
        label="±1 std across clusters",
    )
    ax.set_ylabel("Capacity factor")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0)
    ax.grid(True, ls=":", linewidth=0.4)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def plot_scenario_contrast(rows: list[dict]) -> str:
    """Side-by-side bar charts contrasting total GW and weighted LCOE per scenario.

    Args:
        rows: Each item needs ``label``, ``total_gw``, ``weighted_lcoe``.
    """
    labels = [row["label"] for row in rows]
    totals = [row["total_gw"] for row in rows]
    lcoes = [row["weighted_lcoe"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(labels, totals, color=_BUBBLE_COLOR)
    ax1.set_ylabel("Total potential capacity (GW)")
    ax1.set_title("Solar potential by scenario")
    ax1.tick_params(axis="x", rotation=30)
    for label in ax1.get_xticklabels():
        label.set_ha("right")
    ax1.grid(True, ls=":", linewidth=0.4, axis="y")

    ax2.bar(labels, lcoes, color="steelblue")
    ax2.set_ylabel("Capacity-weighted LCOE ($/MWh)")
    ax2.set_title("Solar cost by scenario")
    ax2.tick_params(axis="x", rotation=30)
    for label in ax2.get_xticklabels():
        label.set_ha("right")
    ax2.grid(True, ls=":", linewidth=0.4, axis="y")

    fig.tight_layout()
    return _fig_to_data_uri(fig)
