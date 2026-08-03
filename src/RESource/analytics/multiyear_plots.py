"""
multiyear_extended_plots.py
────────────────────────────
Extended interannual variability plots for BC cell-level analysis.

Paste the `make_extended_figures()` call at the end of main() in
multiyear_CF_variability_cells.ipynb, or run this file directly after
the existing notebook has populated `stats`, `long_df`, and `ranks`.

Six new figures
───────────────
1. fig_M4-1  Spearman ρ rank stability — LCOE and CF, both techs
2. fig_M4-2  Year-over-year accessible capacity — bar chart
3. fig_M4-3  CF anomaly heatmap — (year CF - mean CF) / mean CF per cell decile
4. fig_M4-4  Scatter: mean CF vs CV% — productivity-stability tradeoff
5. fig_M4-5  LCOE CV% violin — uncertainty in economic ranking across years
6. fig_M4-6  Pairwise Spearman ρ matrix — all year-pairs for LCOE ranking
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Style constants ───────────────────────────────────────────────────────────
TECH_COLOR = {"wind": "#2B8CBE", "solar": "#FE9929"}
TECH_MARKER = {"wind": "o", "solar": "s"}
TECHS = ("solar", "wind")
CF_COL = {t: f"{t}_CF_mean" for t in TECHS}
LCOE_COL = {t: f"lcoe_{t}" for t in TECHS}
CAP_COL = {t: f"potential_capacity_{t}" for t in TECHS}
FIG_DPI = 300


def _spines(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Spearman ρ rank stability (LCOE + CF)
# ══════════════════════════════════════════════════════════════════════════════


def fig_rank_stability(ranks: pd.DataFrame, out_dir) -> None:
    """
    Line plot of Spearman ρ (vs reference year) for every year.
    One panel per technology; LCOE and CF ρ on the same panel with different markers.
    Communicates: "how much does the economic ordering of cells shift year to year?"
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=FIG_DPI, sharey=True)

    for ax, tech in zip(axes, TECHS, strict=False):
        sub = ranks[ranks["tech"] == tech].sort_values("year")
        ref = sub["ref_year"].iloc[0]
        col = TECH_COLOR[tech]

        if "lcoe_spearman" in sub.columns:
            ax.plot(
                sub["year"],
                sub["lcoe_spearman"],
                color=col,
                lw=2,
                marker="o",
                ms=6,
                label="LCOE proxy ranking",
                zorder=4,
            )
            ax.fill_between(
                sub["year"],
                sub["lcoe_spearman"] - sub.get("lcoe_p", 0) * 0,  # placeholder
                sub["lcoe_spearman"],
                color=col,
                alpha=0.08,
            )

        if "cf_spearman" in sub.columns:
            ax.plot(
                sub["year"],
                sub["cf_spearman"],
                color=col,
                lw=1.5,
                marker="s",
                ms=5,
                ls="--",
                alpha=0.75,
                label="CF ranking",
                zorder=3,
            )

        ax.axhline(1.0, color="#aaa", lw=0.8, ls=":")
        ax.axhline(0.95, color="#e63946", lw=0.9, ls="--", label="ρ = 0.95 reference", zorder=2)
        ax.axvline(ref, color="#555", lw=0.8, ls=":", label=f"Reference year ({ref})")

        ax.set_xlim(sub["year"].min() - 0.5, sub["year"].max() + 0.5)
        ax.set_ylim(0.70, 1.02)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlabel("Weather year", fontsize=10)
        ax.set_title(
            f"{tech.title()} — cell rank stability vs {ref}", fontweight="bold", fontsize=10
        )
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        _spines(ax)
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)

    axes[0].set_ylabel("Spearman ρ  (vs reference year)", fontsize=10)
    fig.suptitle(
        "Interannual rank stability of cell-level LCOE proxy and CF — BC",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_M4-1_rank_stability.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    print("  Saved fig_M4-1_rank_stability.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Accessible capacity per year (bar chart)
# ══════════════════════════════════════════════════════════════════════════════


def fig_accessible_capacity(long_df: pd.DataFrame, lcoe_thresh: dict, out_dir) -> None:
    """
    Bar chart of total accessible capacity (GW, LCOE ≤ threshold) per year.
    Quantifies how much the economically viable pool fluctuates year to year.
    """
    years = sorted(long_df["year"].unique())
    x = np.arange(len(years))
    width = 0.35

    records = {t: [] for t in TECHS}
    for y in years:
        dy = long_df[long_df["year"] == y]
        for t in TECHS:
            lc, cap = LCOE_COL[t], CAP_COL[t]
            if lc not in dy.columns or cap not in dy.columns:
                records[t].append(np.nan)
                continue
            accessible = dy[dy[lc] <= lcoe_thresh[t]][cap].sum() / 1e3  # GW
            records[t].append(accessible)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=FIG_DPI)
    for i, tech in enumerate(TECHS):
        vals = records[tech]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=TECH_COLOR[tech],
            alpha=0.85,
            label=f"{tech.title()}  (≤ {lcoe_thresh[tech]} $/MWh)",
        )
        # Annotate bar tops
        for bar, v in zip(bars, vals, strict=False):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333",
                )

    # Mean lines
    for tech in TECHS:
        mean_v = np.nanmean(records[tech])
        ax.axhline(mean_v, color=TECH_COLOR[tech], lw=1.2, ls="--", alpha=0.7)
        ax.text(
            x[-1] + 0.55,
            mean_v,
            f"μ={mean_v:.1f}",
            color=TECH_COLOR[tech],
            fontsize=7.5,
            va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_xlabel("Weather year", fontsize=10)
    ax.set_ylabel("Accessible capacity  (GW)", fontsize=10)
    ax.set_title(
        "Interannual variation in accessible capacity — BC", fontweight="bold", fontsize=11
    )
    ax.legend(fontsize=9, frameon=False)
    _spines(ax)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(
        out_dir / "fig_M4-2_accessible_capacity_by_year.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.show()
    print("  Saved fig_M4-2_accessible_capacity_by_year.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — CF anomaly heatmap  (cell CF decile × year)
# ══════════════════════════════════════════════════════════════════════════════


def fig_cf_anomaly_heatmap(long_df: pd.DataFrame, out_dir) -> None:
    """
    Heatmap: rows = CF decile group, columns = year.
    Colour = mean (CF_year - CF_mean) / CF_mean  within the decile.
    Shows whether low- or high-CF cells are systematically more volatile.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=FIG_DPI)

    for ax, tech in zip(axes, TECHS, strict=False):
        cf = CF_COL[tech]
        if cf not in long_df.columns:
            ax.set_visible(False)
            continue

        # Cell-level long-run mean CF
        cell_mean = long_df.groupby("cell_id")[cf].mean().rename("cf_mean_all")
        df = long_df.join(cell_mean, on="cell_id")
        df["anomaly"] = (df[cf] - df["cf_mean_all"]) / df["cf_mean_all"]

        # Assign decile based on long-run mean CF
        df["decile"] = pd.qcut(df["cf_mean_all"], 10, labels=[f"D{i}" for i in range(1, 11)])

        pivot = df.pivot_table(index="decile", columns="year", values="anomaly", aggfunc="mean")

        # Symmetric colour scale
        vabs = np.nanmax(np.abs(pivot.values)) * 1.05
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vabs, vmax=vabs)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Mean CF anomaly  (fraction)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Weather year", fontsize=9)
        ax.set_ylabel("Cell CF decile  (D1 = lowest)", fontsize=9)
        ax.set_title(
            f"{tech.title()} — CF anomaly by decile and year", fontweight="bold", fontsize=10
        )

        # Annotate cells with anomaly value
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if abs(v) > vabs * 0.5 else "#333",
                    )

    fig.suptitle(
        "Cell-level CF anomaly by decile group — BC", fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_M4-3_cf_anomaly_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    print("  Saved fig_M4-3_cf_anomaly_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Scatter: mean CF vs CV%  (productivity–stability tradeoff)
# ══════════════════════════════════════════════════════════════════════════════


def fig_cf_mean_vs_cv(stats: pd.DataFrame, out_dir) -> None:
    """
    Scatter of per-cell mean CF vs CV%.
    Reveals whether the best cells are also the most or least variable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=FIG_DPI, sharey=False)

    for ax, tech in zip(axes, TECHS, strict=False):
        mean_col = f"{tech}_CF_mean"
        cv_col = f"{tech}_CF_cv_pct"
        if mean_col not in stats.columns or cv_col not in stats.columns:
            ax.set_visible(False)
            continue

        x = stats[mean_col].dropna()
        y = stats[cv_col].reindex(x.index).dropna()
        x = x.reindex(y.index)

        # Hex-bin density
        hb = ax.hexbin(x, y, gridsize=35, cmap="YlOrRd", mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="Cell count", shrink=0.85)

        # Trend line
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            xs,
            np.polyval(z, xs),
            color=TECH_COLOR[tech],
            lw=1.5,
            ls="--",
            label=f"Trend  (slope {z[0]:+.1f} %/unit CF)",
        )

        ax.set_xlabel("Mean annual CF  (fraction)", fontsize=10)
        ax.set_ylabel("CF coefficient of variation  (%)", fontsize=10)
        ax.set_title(
            f"{tech.title()} — productivity vs stability tradeoff", fontweight="bold", fontsize=10
        )
        ax.legend(fontsize=8, frameon=False)
        _spines(ax)
        ax.grid(ls="--", lw=0.4, alpha=0.4)

    fig.suptitle(
        "Per-cell CF mean vs interannual CV — BC  (n = 2,956 cells)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_M4-4_cf_mean_vs_cv_scatter.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    print("  Saved fig_M4-4_cf_mean_vs_cv_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — LCOE CV% violin
# ══════════════════════════════════════════════════════════════════════════════


def fig_lcoe_cv_violin(stats: pd.DataFrame, out_dir) -> None:
    """
    Violin + box plot of per-cell LCOE CV% for each technology.
    Complements the CF CV% histogram — shows the economic uncertainty distribution.
    """
    data = []
    labels = []
    colors = []
    for tech in TECHS:
        col = f"{tech}_lcoe_cv_pct"
        if col not in stats.columns:
            continue
        vals = stats[col].dropna().values
        vals = vals[np.isfinite(vals) & (vals < np.percentile(vals, 98))]  # clip outliers
        data.append(vals)
        labels.append(tech.title())
        colors.append(TECH_COLOR[tech])

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=FIG_DPI)
    parts = ax.violinplot(
        data, positions=range(len(data)), showmedians=True, showextrema=False, widths=0.55
    )

    for pc, col in zip(parts["bodies"], colors, strict=False):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)

    parts["cmedians"].set_color("#333")
    parts["cmedians"].set_linewidth(1.5)

    # Overlay box
    bp = ax.boxplot(
        data, positions=range(len(data)), widths=0.12, patch_artist=True, showfliers=False, zorder=4
    )
    for patch, col in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for el in ["whiskers", "caps", "medians"]:
        for line in bp[el]:
            line.set_color("#333")
            line.set_linewidth(1.2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("LCOE coefficient of variation  (%)", fontsize=10)
    ax.set_title(
        "Interannual LCOE proxy uncertainty — BC  (per-cell CV)", fontweight="bold", fontsize=11
    )
    _spines(ax)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_M4-5_lcoe_cv_violin.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    print("  Saved fig_M4-5_lcoe_cv_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Pairwise Spearman ρ matrix (all year-pairs, LCOE ranking)
# ══════════════════════════════════════════════════════════════════════════════


def fig_pairwise_spearman(long_df: pd.DataFrame, out_dir) -> None:
    """
    Heatmap of pairwise Spearman ρ between all year-pairs for LCOE ranking.
    Shows which years are most similar to each other — useful for identifying
    whether specific years (e.g. drought, El Niño) are outliers.
    """
    years = sorted(long_df["year"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_DPI)

    for ax, tech in zip(axes, TECHS, strict=False):
        lc = LCOE_COL[tech]
        if lc not in long_df.columns:
            ax.set_visible(False)
            continue

        n = len(years)
        mat = np.ones((n, n))

        for i, ya in enumerate(years):
            da = long_df[long_df["year"] == ya].set_index("cell_id")[lc]
            for j, yb in enumerate(years):
                if j <= i:
                    continue
                db = long_df[long_df["year"] == yb].set_index("cell_id")[lc]
                common = da.index.intersection(db.index)
                if len(common) > 2:
                    rho, _ = spearmanr(da.loc[common], db.loc[common])
                    mat[i, j] = mat[j, i] = rho

        # Mask diagonal for display
        mat_disp = np.where(np.eye(n, dtype=bool), np.nan, mat)

        vmin = np.nanmin(mat_disp) - 0.005
        im = ax.imshow(mat_disp, cmap="RdYlGn", vmin=max(0.80, vmin), vmax=1.0, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Spearman ρ  (LCOE ranking)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_xticks(range(n))
        ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(years, fontsize=8)

        # Annotate each cell
        for i in range(n):
            for j in range(n):
                v = mat_disp[i, j]
                if np.isfinite(v):
                    norm = (v - max(0.80, vmin)) / (1.0 - max(0.80, vmin))
                    r, g, b, _ = plt.cm.RdYlGn(norm)
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if lum < 0.45 else "#222",
                    )

        ax.set_title(
            f"{tech.title()} — pairwise LCOE rank correlation", fontweight="bold", fontsize=10
        )

    fig.suptitle(
        "Pairwise Spearman ρ of cell LCOE ranking — BC  (all year-pairs)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_M4-6_pairwise_spearman_matrix.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    print("  Saved fig_M4-6_pairwise_spearman_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper — call this from main()
# ══════════════════════════════════════════════════════════════════════════════


def make_extended_figures(stats, long_df, ranks, lcoe_thresh, out_dir) -> None:
    """
    Generate all six extended stability figures.

    Parameters
    ----------
    stats       : per-cell statistics DataFrame  (from per_cell_stats())
    long_df     : long-format year × cell DataFrame  (from stack_years())
    ranks       : Spearman ρ DataFrame  (from rank_stability())
    lcoe_thresh : dict {"solar": float, "wind": float}  LCOE ceilings
    out_dir     : Path  output directory
    """
    print("\n── Extended stability figures ───────────────────────────────────────")
    fig_rank_stability(ranks, out_dir)
    fig_accessible_capacity(long_df, lcoe_thresh, out_dir)
    fig_cf_anomaly_heatmap(long_df, out_dir)
    fig_cf_mean_vs_cv(stats, out_dir)
    fig_lcoe_cv_violin(stats, out_dir)
    fig_pairwise_spearman(long_df, out_dir)
    print("── Extended figures done ────────────────────────────────────────────\n")


# ── Standalone test (uses outputs already in memory from main notebook) ────────
if __name__ == "__main__":
    # Quick smoke-test with synthetic data when run outside the notebook
    rng = np.random.default_rng(0)
    n, Y = 200, 5
    years = list(range(2020, 2020 + Y))
    rows = []
    for y in years:
        cf_s = rng.uniform(0.10, 0.25, n)
        cf_w = rng.uniform(0.18, 0.40, n)
        rows.extend(
            [
                {
                    "cell_id": i,
                    "year": y,
                    "solar_CF_mean": cf_s[i],
                    "wind_CF_mean": cf_w[i],
                    "lcoe_solar": 80 + rng.normal(0, 5),
                    "lcoe_wind": 55 + rng.normal(0, 8),
                    "potential_capacity_solar": rng.uniform(50, 300),
                    "potential_capacity_wind": rng.uniform(50, 300),
                }
                for i in range(n)
            ]
        )
    from pathlib import Path

    long_df = pd.DataFrame(rows)

    # Minimal stats
    g = long_df.groupby("cell_id")
    stats = pd.DataFrame(
        {
            "solar_CF_mean": g["solar_CF_mean"].mean(),
            "solar_CF_sd": g["solar_CF_mean"].std(),
            "solar_CF_cv_pct": 100 * g["solar_CF_mean"].std() / g["solar_CF_mean"].mean(),
            "wind_CF_mean": g["wind_CF_mean"].mean(),
            "wind_CF_sd": g["wind_CF_mean"].std(),
            "wind_CF_cv_pct": 100 * g["wind_CF_mean"].std() / g["wind_CF_mean"].mean(),
            "solar_lcoe_cv_pct": rng.uniform(1, 8, n),
            "wind_lcoe_cv_pct": rng.uniform(1, 15, n),
        }
    )
    stats.attrs["n_years"] = Y

    ranks_rows = []
    for tech in TECHS:
        lc = LCOE_COL[tech]
        ref_df = long_df[long_df["year"] == 2024 if 2024 in years else years[-1]].set_index(
            "cell_id"
        )
        for y in years:
            cur = long_df[long_df["year"] == y].set_index("cell_id")
            common = ref_df.index.intersection(cur.index)
            if len(common) > 2 and lc in long_df.columns:
                rho, p = spearmanr(ref_df.loc[common, lc], cur.loc[common, lc])
                ranks_rows.append(
                    {
                        "tech": tech,
                        "ref_year": years[-1],
                        "year": y,
                        "lcoe_spearman": rho,
                        "cf_spearman": rho - 0.01,
                    }
                )
    ranks = pd.DataFrame(ranks_rows)

    out = Path("/tmp/test_figs")
    out.mkdir(exist_ok=True)
    make_extended_figures(stats, long_df, ranks, {"solar": 90, "wind": 130}, out)
