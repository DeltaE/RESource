"""
M2 Sensitivity Analysis: Grid-Connection Cost Uncertainty
=========================================================
Addresses Reviewer 1 (R1.3) and Reviewer 2 Major Comment 1 (R2.M1):

    "The use of transmission cost estimates derived from non-Canadian contexts
     requires further justification, and the sensitivity of results to these
     assumptions should be discussed."

    "The authors should assess the possible magnitude of this simplification,
     for example through distance multipliers, alternative line-cost assumptions,
     or remote-area penalties."

Approach
--------
The screening-level LCOE proxy depends on grid costs through:

    base_grid_cost = (d_km × gcc) + tx_rebuild            [M$]
    scaling        = (capacity / 100 MW) ^ 0.8            [dimensionless; 1.0 if capacity ≤ 100 MW]
    grid_cost      = base_grid_cost × scaling              [M$]

where gcc = grid_connection_cost_per_km (M$/km) and tx_rebuild = fixed rebuild
cost (M$). To capture both cost-estimate uncertainty and terrain routing detour,
we apply a single multiplier κ to the effective spur-line distance (equivalent
to scaling gcc). We test:
    κ ∈ {0.6, 0.8, 1.0, 1.3, 1.6, 2.0}

Outputs
-------
1. Spearman ρ of cluster ranking vs. baseline for each κ             → CSV
2. Supply-curve bands (cumulative capacity vs. LCOE proxy) for solar/wind → CSV
3. Tornado summary table                                               → CSV

Usage
-----
    python sensitivity_M2_grid_cost.py \
        --solar  results/Canada/BC/<RUN_ID>/clusters/resource_options_solar_BritishColumbia.csv \
        --wind   results/Canada/BC/<RUN_ID>/clusters/resource_options_wind_BritishColumbia.csv \
        --gcc    2.6 \
        --tx     0.56 \
        --outdir results/sensitivity/M2

Author: Md Eliasinul Islam
"""

import argparse
import sys
from pathlib import Path
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# LCOE formula — mirrors score.py CellScorer.calculate_score() exactly
# ---------------------------------------------------------------------------

def get_crf(r: float, N: int) -> float:
    """Capital Recovery Factor.  CRF = r(1+r)^N / ((1+r)^N - 1)."""
    if N <= 0:
        return 0.0
    return (r * (1 + r) ** N) / ((1 + r) ** N - 1)


def smooth_scaling(capacity_mw: float,
                   reference_mw: float = 100.0,
                   exponent: float = 0.8) -> float:
    """Power-law economies of scale for grid connection cost."""
    if capacity_mw <= reference_mw:
        return 1.0
    return (capacity_mw / reference_mw) ** exponent


def compute_lcoe(capacity_mw: float,
                 cf_mean: float,
                 capex_musd_per_mw: float,
                 fom_musd_per_mw: float,
                 vom_musd_per_mwh: float,
                 distance_km: float,
                 gcc_musd_per_km: float,
                 tx_rebuild_musd: float,
                 crf: float,
                 reference_mw: float = 100.0,
                 scaling_exp: float = 0.8) -> float:
    """
    Screening-level LCOE proxy ($/MWh), matching CellScorer.calculate_score().

    Parameters
    ----------
    capacity_mw          : Installed potential capacity (MW)
    cf_mean              : Annual mean capacity factor (dimensionless)
    capex_musd_per_mw    : Technology CAPEX (M$/MW)
    fom_musd_per_mw      : Fixed O&M (M$/MW/yr)
    vom_musd_per_mwh     : Variable O&M (M$/MWh)
    distance_km          : Straight-line distance to nearest substation (km)
    gcc_musd_per_km      : Spur-line cost (M$/km)
    tx_rebuild_musd      : Fixed substation upgrade cost (M$)
    crf                  : Capital Recovery Factor (dimensionless)
    reference_mw         : Reference project size for scaling (default 100 MW)
    scaling_exp          : Economies-of-scale exponent (default 0.8)

    Returns
    -------
    float : LCOE in $/MWh; 999_999 if annual energy == 0
    """
    annual_energy_mwh = 8_760 * cf_mean * capacity_mw
    if annual_energy_mwh <= 0:
        return 999_999.0

    # Total capital cost
    tech_capex = capex_musd_per_mw * capacity_mw
    base_grid  = (distance_km * gcc_musd_per_km) + tx_rebuild_musd
    scale      = smooth_scaling(capacity_mw, reference_mw, scaling_exp)
    total_capex = tech_capex + base_grid * scale             # M$

    # O&M
    fom_annual = fom_musd_per_mw * capacity_mw               # M$/yr
    vom_annual = vom_musd_per_mwh * annual_energy_mwh        # M$/yr

    lcoe_musd_per_mwh = ((total_capex * crf) + fom_annual + vom_annual) / annual_energy_mwh
    return lcoe_musd_per_mwh * 1e6                           # $/MWh


# ---------------------------------------------------------------------------
# Core sensitivity function
# ---------------------------------------------------------------------------

def run_grid_cost_sensitivity(df: pd.DataFrame,
                               resource_type: str,
                               baseline_gcc: float,
                               baseline_tx: float,
                               interest_rate: float = 0.07,
                               kappa_values: list = None,
                               lcoe_threshold: float = 150.0) -> dict:
    """
    Sweep κ (terrain-routing detour multiplier) over gcc and return results.

    Parameters
    ----------
    df              : Cluster-level results DataFrame (from export_results CSV)
    resource_type   : 'solar' or 'wind'
    baseline_gcc    : Baseline grid connection cost (M$/km)
    baseline_tx     : Baseline fixed rebuild cost (M$)
    interest_rate   : Discount rate for CRF
    kappa_values    : List of distance/cost multipliers to test
    lcoe_threshold  : Cut-off LCOE for 'economically accessible' capacity ($/MWh)

    Returns
    -------
    dict with keys:
        'spearman'  : DataFrame of Spearman ρ per κ
        'curves'    : dict{κ: (cumulative_gw, lcoe_sorted)}
        'threshold' : dict{κ: total GW with LCOE ≤ threshold}
    """
    if kappa_values is None:
        kappa_values = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]

    # Operational life read from data
    N = int(df['Operational_life'].iloc[0])
    crf = get_crf(interest_rate, N)

    records = []
    curves = {}
    threshold_gw = {}

    baseline_rank = None

    for kappa in kappa_values:
        gcc_variant = baseline_gcc * kappa   # Effective cost per km

        df_v = df.copy()
        df_v['lcoe_variant'] = df_v.apply(
            lambda row: compute_lcoe(
                capacity_mw       = row['potential_capacity'],
                cf_mean           = row['CF_mean'],
                capex_musd_per_mw = row['capex'],
                fom_musd_per_mw   = row['fom'],
                vom_musd_per_mwh  = row['vom'],
                distance_km       = row['nearest_station_distance_km'],
                gcc_musd_per_km   = gcc_variant,
                tx_rebuild_musd   = baseline_tx,
                crf               = crf,
            ),
            axis=1
        )

        df_v_sorted = df_v.sort_values('lcoe_variant').reset_index(drop=True)
        df_v_sorted['rank_variant'] = range(1, len(df_v_sorted) + 1)

        # Baseline ranks (κ=1.0 defines baseline ordering)
        if kappa == 1.0:
            baseline_rank = df_v_sorted[['cluster_id', 'rank_variant']].copy()
            baseline_rank.columns = ['cluster_id', 'rank_baseline']

        # Supply curve
        df_v_sorted_asc = df_v_sorted.sort_values('lcoe_variant')
        cum_cap_gw = df_v_sorted_asc['potential_capacity'].cumsum() / 1_000.0
        curves[kappa] = (cum_cap_gw.values, df_v_sorted_asc['lcoe_variant'].values)

        # Threshold capacity
        below = df_v_sorted_asc[df_v_sorted_asc['lcoe_variant'] <= lcoe_threshold]
        threshold_gw[kappa] = below['potential_capacity'].sum() / 1_000.0

        records.append({
            'kappa': kappa,
            'gcc_musd_per_km': gcc_variant,
            'threshold_gw': threshold_gw[kappa],
            'median_lcoe': df_v_sorted['lcoe_variant'].median(),
            'min_lcoe': df_v_sorted['lcoe_variant'].min(),
            'max_lcoe': df_v_sorted['lcoe_variant'].max(),
        })

    # Compute Spearman ρ once baseline is established
    assert baseline_rank is not None, "κ=1.0 must be in kappa_values"
    spearman_records = []

    for kappa in kappa_values:
        gcc_variant = baseline_gcc * kappa
        df_v = df.copy()
        df_v['lcoe_variant'] = df_v.apply(
            lambda row: compute_lcoe(
                capacity_mw       = row['potential_capacity'],
                cf_mean           = row['CF_mean'],
                capex_musd_per_mw = row['capex'],
                fom_musd_per_mw   = row['fom'],
                vom_musd_per_mwh  = row['vom'],
                distance_km       = row['nearest_station_distance_km'],
                gcc_musd_per_km   = gcc_variant,
                tx_rebuild_musd   = baseline_tx,
                crf               = crf,
            ),
            axis=1
        )
        df_v_sorted = df_v.sort_values('lcoe_variant').reset_index(drop=True)
        df_v_sorted['rank_variant'] = range(1, len(df_v_sorted) + 1)

        merged = pd.merge(baseline_rank, df_v_sorted[['cluster_id', 'rank_variant']],
                          on='cluster_id', how='inner')
        rho, pval = spearmanr(merged['rank_baseline'], merged['rank_variant'])
        spearman_records.append({
            'resource_type': resource_type,
            'kappa': kappa,
            'gcc_musd_per_km': gcc_variant,
            'spearman_rho': round(rho, 4),
            'p_value': round(pval, 4),
        })

    spearman_df = pd.DataFrame(spearman_records)
    summary_df  = pd.DataFrame(records)

    return {
        'spearman': spearman_df,
        'curves': curves,
        'threshold': threshold_gw,
        'summary': summary_df,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_supply_curves(curves: dict,
                       resource_type: str,
                       save_path: Path,
                       lcoe_threshold: float = 150.0) -> None:
    """Plot supply-curve bands across κ scenarios."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.RdYlGn_r
    kappas = sorted(curves.keys())
    colours = cmap(np.linspace(0.1, 0.9, len(kappas)))

    for kappa, colour in zip(kappas, colours):
        cum_gw, lcoe = curves[kappa]
        lw  = 2.5 if kappa == 1.0 else 1.0
        ls  = '-'  if kappa == 1.0 else '--'
        lbl = f'κ = {kappa:.1f} (baseline)' if kappa == 1.0 else f'κ = {kappa:.1f}'
        ax.step(cum_gw, lcoe, where='post', color=colour,
                linewidth=lw, linestyle=ls, label=lbl)

    ax.axhline(lcoe_threshold, color='grey', linestyle=':', linewidth=1.0,
               label=f'Threshold {lcoe_threshold} $/MWh')
    ax.set_xlabel('Cumulative developable potential (GW)', fontsize=11)
    ax.set_ylabel('Screening-level LCOE proxy ($/MWh)', fontsize=11)
    ax.set_title(f'{resource_type.title()} – Grid-cost sensitivity (M2)', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.set_ylim(0, min(600, lcoe_threshold * 3))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_spearman(spearman_df: pd.DataFrame,
                  resource_type: str,
                  save_path: Path) -> None:
    """Bar plot of Spearman ρ vs. κ."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(spearman_df['kappa'].astype(str),
           spearman_df['spearman_rho'],
           color='steelblue', edgecolor='white', width=0.6)
    ax.axhline(0.9, color='red', linestyle='--', linewidth=1.0,
               label='ρ = 0.90 reference')
    ax.set_xlabel('Cost multiplier κ', fontsize=11)
    ax.set_ylabel('Spearman ρ (rank correlation vs. baseline)', fontsize=11)
    ax.set_title(f'{resource_type.title()} – Rank stability across grid-cost scenarios', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='M2 grid-cost sensitivity analysis')
    parser.add_argument('--solar', type=Path, default=None,
                        help='Path to solar cluster CSV')
    parser.add_argument('--wind',  type=Path, default=None,
                        help='Path to wind cluster CSV')
    parser.add_argument('--gcc',   type=float, default=2.6,
                        help='Baseline grid-connection cost (M$/km); default 2.6')
    parser.add_argument('--tx',    type=float, default=0.56,
                        help='Baseline fixed rebuild cost (M$); default 0.56')
    parser.add_argument('--rate',  type=float, default=0.07,
                        help='Discount rate; default 0.07')
    parser.add_argument('--threshold', type=float, default=150.0,
                        help='LCOE cut-off for accessible capacity ($/MWh); default 150')
    parser.add_argument('--outdir', type=Path, default=Path('results/sensitivity/M2'),
                        help='Output directory')
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    kappa_values = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    all_spearman = []

    for resource_type, csv_path in [('solar', args.solar), ('wind', args.wind)]:
        if csv_path is None:
            print(f"  Skipping {resource_type}: no CSV path provided.")
            continue
        if not csv_path.exists():
            print(f"  ERROR: {csv_path} not found.", file=sys.stderr)
            continue

        print(f"\n=== {resource_type.upper()} ===")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} clusters from {csv_path.name}")

        # Validate required columns
        required = ['cluster_id', 'potential_capacity', 'CF_mean', 'capex',
                    'fom', 'vom', 'nearest_station_distance_km', 'Operational_life']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ERROR: Missing columns: {missing}", file=sys.stderr)
            continue

        results = run_grid_cost_sensitivity(
            df             = df,
            resource_type  = resource_type,
            baseline_gcc   = args.gcc,
            baseline_tx    = args.tx,
            interest_rate  = args.rate,
            kappa_values   = kappa_values,
            lcoe_threshold = args.threshold,
        )

        # Save outputs
        results['spearman'].to_csv(
            args.outdir / f'M2_spearman_{resource_type}.csv', index=False)
        results['summary'].to_csv(
            args.outdir / f'M2_summary_{resource_type}.csv', index=False)

        # Save supply-curve data for each kappa
        for kappa in kappa_values:
            cum_gw, lcoe_arr = results['curves'][kappa]
            pd.DataFrame({'cumulative_gw': cum_gw, 'lcoe_usd_per_mwh': lcoe_arr}).to_csv(
                args.outdir / f'M2_curve_{resource_type}_kappa{kappa:.1f}.csv', index=False)

        # Figures
        plot_supply_curves(
            results['curves'], resource_type,
            args.outdir / f'M2_supply_curves_{resource_type}.png',
            args.threshold)
        plot_spearman(
            results['spearman'], resource_type,
            args.outdir / f'M2_spearman_{resource_type}.png')

        all_spearman.append(results['spearman'])
        print(f"\n  Spearman ρ table:\n{results['spearman'].to_string(index=False)}")
        print(f"\n  Developable capacity ≤ {args.threshold} $/MWh by κ:")
        for kappa, gw in results['threshold'].items():
            print(f"    κ = {kappa:.1f}: {gw:.1f} GW")

    # Combined Spearman table
    if all_spearman:
        combined = pd.concat(all_spearman, ignore_index=True)
        combined.to_csv(args.outdir / 'M2_spearman_combined.csv', index=False)
        print(f"\nResults saved to: {args.outdir}")


if __name__ == '__main__':
    main()
