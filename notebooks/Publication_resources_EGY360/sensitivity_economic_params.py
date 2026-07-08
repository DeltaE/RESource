"""
M3 Sensitivity Analysis: Capacity Density and Economic Parameters
=================================================================
Addresses Reviewer 2 Major Comment 2 (R2.M2):

    "The uncertainty treatment is insufficient for the quantitative claims.
     The reported wind and solar reductions depend on buffers, slope thresholds,
     land-cover classes, capacity density, grid cost, reference plant size,
     discount rate, and weather year. These results should be framed as
     scenario-specific outcomes."

Parameters swept (one-at-a-time, OAT)
--------------------------------------
  1. Capacity density (MW/km²)
       Solar: Low=1.1, Mid=1.45 (baseline), High=1.80
       Wind:  Low=2.25, Mid=3.0  (baseline), High=3.75
  2. Discount rate
       Low=5 %, Mid=7 % (baseline), High=10 %
  3. CAPEX scenario (ATB Low/Mid/High approximated as ±20 % of baseline)
       Low=0.8×, Mid=1.0× (baseline), High=1.2×
  4. FOM (same ±20 % relative to baseline)
       Low=0.8×, Mid=1.0× (baseline), High=1.2×

Outputs
-------
1. Tornado chart: change in total developable capacity (GW) at LCOE ≤ threshold
2. CSV table of all scenario combinations
3. Interaction matrix: CAPEX × discount rate for solar and wind

Usage
-----
    python sensitivity_M3_economic_params.py \
        --solar  results/Canada/BC/<RUN_ID>/clusters/resource_options_solar_BritishColumbia.csv \
        --wind   results/Canada/BC/<RUN_ID>/clusters/resource_options_wind_BritishColumbia.csv \
        --gcc    2.6 \
        --tx     0.56 \
        --outdir results/sensitivity/M3

Author: Md Eliasinul Islam
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# LCOE helpers (identical to M2 script — kept here for self-containment)
# ---------------------------------------------------------------------------

def get_crf(r: float, N: int) -> float:
    if N <= 0:
        return 0.0
    return (r * (1 + r) ** N) / ((1 + r) ** N - 1)


def smooth_scaling(capacity_mw: float,
                   reference_mw: float = 100.0,
                   exponent: float = 0.8) -> float:
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
    annual_energy_mwh = 8_760 * cf_mean * capacity_mw
    if annual_energy_mwh <= 0:
        return 999_999.0
    tech_capex  = capex_musd_per_mw * capacity_mw
    base_grid   = (distance_km * gcc_musd_per_km) + tx_rebuild_musd
    scale       = smooth_scaling(capacity_mw, reference_mw, scaling_exp)
    total_capex = tech_capex + base_grid * scale
    fom_annual  = fom_musd_per_mw * capacity_mw
    vom_annual  = vom_musd_per_mwh * annual_energy_mwh
    lcoe_musd   = ((total_capex * crf) + fom_annual + vom_annual) / annual_energy_mwh
    return lcoe_musd * 1e6


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

BASELINE_DENSITY = {'solar': 1.45, 'wind': 3.0}   # MW/km²

DENSITY_SCENARIOS = {
    'solar': {'Low': 1.10, 'Mid': 1.45, 'High': 1.80},
    'wind':  {'Low': 2.25, 'Mid': 3.00, 'High': 3.75},
}

DISCOUNT_RATE_SCENARIOS = {
    'Low':  0.05,
    'Mid':  0.07,
    'High': 0.10,
}

CAPEX_MULTIPLIER_SCENARIOS = {
    'Low':  0.80,   # Optimistic ATB (cost decline)
    'Mid':  1.00,   # Baseline ATB moderate
    'High': 1.20,   # Conservative ATB
}

FOM_MULTIPLIER_SCENARIOS = {
    'Low':  0.80,
    'Mid':  1.00,
    'High': 1.20,
}


# ---------------------------------------------------------------------------
# Core: recompute cluster LCOE under a parameter variant
# ---------------------------------------------------------------------------

def recompute_lcoe_for_variant(df: pd.DataFrame,
                                resource_type: str,
                                gcc_baseline: float,
                                tx_baseline: float,
                                density_multiplier: float = 1.0,
                                capex_multiplier: float = 1.0,
                                fom_multiplier: float = 1.0,
                                interest_rate: float = 0.07) -> pd.Series:
    """
    Recompute LCOE for every cluster given parameter multipliers.

    density_multiplier scales 'potential_capacity' (proportional to MW/km²),
    capex_multiplier   scales 'capex',
    fom_multiplier     scales 'fom'.
    """
    N   = int(df['Operational_life'].iloc[0])
    crf = get_crf(interest_rate, N)

    return df.apply(
        lambda row: compute_lcoe(
            capacity_mw       = row['potential_capacity'] * density_multiplier,
            cf_mean           = row['CF_mean'],
            capex_musd_per_mw = row['capex'] * capex_multiplier,
            fom_musd_per_mw   = row['fom'] * fom_multiplier,
            vom_musd_per_mwh  = row['vom'],
            distance_km       = row['nearest_station_distance_km'],
            gcc_musd_per_km   = gcc_baseline,
            tx_rebuild_musd   = tx_baseline,
            crf               = crf,
        ),
        axis=1,
    )


def accessible_capacity_gw(df: pd.DataFrame,
                             lcoe_series: pd.Series,
                             density_multiplier: float,
                             lcoe_threshold: float) -> float:
    """Total GW with LCOE ≤ threshold, accounting for density scaling."""
    mask = lcoe_series <= lcoe_threshold
    return (df.loc[mask, 'potential_capacity'] * density_multiplier).sum() / 1_000.0


# ---------------------------------------------------------------------------
# OAT sweep
# ---------------------------------------------------------------------------

def run_oat_sensitivity(df: pd.DataFrame,
                         resource_type: str,
                         gcc_baseline: float,
                         tx_baseline: float,
                         lcoe_threshold: float = 150.0,
                         baseline_rate: float = 0.07) -> pd.DataFrame:
    """
    One-at-a-time sensitivity: vary each parameter while holding others at baseline.

    Returns a DataFrame with columns:
        parameter, scenario_label, scenario_value,
        total_gw, delta_gw, delta_pct
    """
    # ---- Baseline ----
    baseline_lcoe = recompute_lcoe_for_variant(
        df, resource_type, gcc_baseline, tx_baseline,
        density_multiplier=1.0, capex_multiplier=1.0,
        fom_multiplier=1.0, interest_rate=baseline_rate)
    gw_baseline = accessible_capacity_gw(df, baseline_lcoe, 1.0, lcoe_threshold)

    records = []

    # 1. Capacity density
    density_baseline = BASELINE_DENSITY[resource_type]
    for label, density_val in DENSITY_SCENARIOS[resource_type].items():
        mult = density_val / density_baseline
        lcoe_s = recompute_lcoe_for_variant(
            df, resource_type, gcc_baseline, tx_baseline,
            density_multiplier=mult, capex_multiplier=1.0,
            fom_multiplier=1.0, interest_rate=baseline_rate)
        gw = accessible_capacity_gw(df, lcoe_s, mult, lcoe_threshold)
        records.append({
            'parameter':       'Capacity density (MW/km²)',
            'scenario_label':  label,
            'scenario_value':  density_val,
            'total_gw':        gw,
            'delta_gw':        gw - gw_baseline,
            'delta_pct':       100 * (gw - gw_baseline) / (gw_baseline + 1e-9),
        })

    # 2. Discount rate
    for label, rate_val in DISCOUNT_RATE_SCENARIOS.items():
        lcoe_s = recompute_lcoe_for_variant(
            df, resource_type, gcc_baseline, tx_baseline,
            density_multiplier=1.0, capex_multiplier=1.0,
            fom_multiplier=1.0, interest_rate=rate_val)
        gw = accessible_capacity_gw(df, lcoe_s, 1.0, lcoe_threshold)
        records.append({
            'parameter':       'Discount rate (%)',
            'scenario_label':  label,
            'scenario_value':  rate_val * 100,
            'total_gw':        gw,
            'delta_gw':        gw - gw_baseline,
            'delta_pct':       100 * (gw - gw_baseline) / (gw_baseline + 1e-9),
        })

    # 3. CAPEX scenario
    for label, capex_mult in CAPEX_MULTIPLIER_SCENARIOS.items():
        lcoe_s = recompute_lcoe_for_variant(
            df, resource_type, gcc_baseline, tx_baseline,
            density_multiplier=1.0, capex_multiplier=capex_mult,
            fom_multiplier=1.0, interest_rate=baseline_rate)
        gw = accessible_capacity_gw(df, lcoe_s, 1.0, lcoe_threshold)
        records.append({
            'parameter':       'CAPEX scenario (ATB)',
            'scenario_label':  label,
            'scenario_value':  capex_mult,
            'total_gw':        gw,
            'delta_gw':        gw - gw_baseline,
            'delta_pct':       100 * (gw - gw_baseline) / (gw_baseline + 1e-9),
        })

    # 4. FOM
    for label, fom_mult in FOM_MULTIPLIER_SCENARIOS.items():
        lcoe_s = recompute_lcoe_for_variant(
            df, resource_type, gcc_baseline, tx_baseline,
            density_multiplier=1.0, capex_multiplier=1.0,
            fom_multiplier=fom_mult, interest_rate=baseline_rate)
        gw = accessible_capacity_gw(df, lcoe_s, 1.0, lcoe_threshold)
        records.append({
            'parameter':       'Fixed O&M (FOM)',
            'scenario_label':  label,
            'scenario_value':  fom_mult,
            'total_gw':        gw,
            'delta_gw':        gw - gw_baseline,
            'delta_pct':       100 * (gw - gw_baseline) / (gw_baseline + 1e-9),
        })

    result = pd.DataFrame(records)
    result['resource_type'] = resource_type
    result['baseline_gw']   = gw_baseline
    result['lcoe_threshold'] = lcoe_threshold
    return result


# ---------------------------------------------------------------------------
# Interaction sweep: CAPEX × discount rate (joint)
# ---------------------------------------------------------------------------

def run_interaction_sweep(df: pd.DataFrame,
                            resource_type: str,
                            gcc_baseline: float,
                            tx_baseline: float,
                            lcoe_threshold: float = 150.0) -> pd.DataFrame:
    """
    Full factorial sweep over CAPEX multiplier × discount rate.
    Returns a DataFrame suitable for heatmap/table display.
    """
    records = []
    for rate_label, rate_val in DISCOUNT_RATE_SCENARIOS.items():
        for capex_label, capex_mult in CAPEX_MULTIPLIER_SCENARIOS.items():
            lcoe_s = recompute_lcoe_for_variant(
                df, resource_type, gcc_baseline, tx_baseline,
                density_multiplier=1.0, capex_multiplier=capex_mult,
                fom_multiplier=1.0, interest_rate=rate_val)
            gw = accessible_capacity_gw(df, lcoe_s, 1.0, lcoe_threshold)
            records.append({
                'resource_type':  resource_type,
                'rate_label':     rate_label,
                'discount_rate':  rate_val,
                'capex_label':    capex_label,
                'capex_mult':     capex_mult,
                'total_gw':       gw,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tornado(oat_df: pd.DataFrame,
                 resource_type: str,
                 save_path: Path,
                 lcoe_threshold: float) -> None:
    """
    Horizontal tornado chart: shows the range in accessible GW for each parameter.
    Only the Low and High extremes are plotted per parameter.
    """
    # Extract the range [Low, High] per parameter
    params = oat_df['parameter'].unique()
    bars = []
    for param in params:
        sub   = oat_df[oat_df['parameter'] == param]
        low_r = sub[sub['scenario_label'] == 'Low']['delta_gw'].values
        high_r = sub[sub['scenario_label'] == 'High']['delta_gw'].values
        if len(low_r) == 0 or len(high_r) == 0:
            continue
        bars.append({'parameter': param, 'low_delta': low_r[0], 'high_delta': high_r[0]})

    bars_df = pd.DataFrame(bars)
    bars_df['range_abs'] = bars_df['high_delta'].abs() + bars_df['low_delta'].abs()
    bars_df = bars_df.sort_values('range_abs', ascending=True)  # Most impactful at top

    fig, ax = plt.subplots(figsize=(9, 0.8 * len(bars_df) + 2))
    y_pos = range(len(bars_df))

    for i, row in enumerate(bars_df.itertuples()):
        left  = min(row.low_delta, 0)
        right = max(row.high_delta, 0)
        ax.barh(i, right, left=0,      color='#2196F3', alpha=0.85, height=0.5)
        ax.barh(i, left,  left=0,      color='#F44336', alpha=0.85, height=0.5)
        ax.text(right + 0.1, i, f'+{right:.1f} GW', va='center', fontsize=9)
        ax.text(left  - 0.1, i, f'{left:.1f} GW',  va='center', ha='right', fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(bars_df['parameter'].tolist(), fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Change in accessible capacity vs. baseline (GW)', fontsize=11)
    ax.set_title(
        f'{resource_type.title()} – Parameter sensitivity tornado\n'
        f'(LCOE ≤ {lcoe_threshold:.0f} $/MWh threshold)', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_interaction_heatmap(interaction_df: pd.DataFrame,
                              resource_type: str,
                              save_path: Path) -> None:
    """Heatmap of accessible GW across CAPEX × discount rate combinations."""
    pivot = interaction_df.pivot(
        index='rate_label', columns='capex_label', values='total_gw')
    # Ensure order
    row_order = ['Low', 'Mid', 'High']
    col_order = ['Low', 'Mid', 'High']
    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=pivot.values.min() * 0.9,
                   vmax=pivot.values.max() * 1.1)
    plt.colorbar(im, ax=ax, label='Accessible capacity (GW)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'CAPEX {c}' for c in col_order])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f'Rate {r}' for r in row_order])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{pivot.values[i, j]:.1f}',
                    ha='center', va='center', fontsize=10,
                    color='black' if pivot.values[i, j] > pivot.values.mean() else 'white')
    ax.set_title(f'{resource_type.title()} – CAPEX × discount rate interaction', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='M3 economic-parameter sensitivity analysis')
    parser.add_argument('--solar',     type=Path, default=None)
    parser.add_argument('--wind',      type=Path, default=None)
    parser.add_argument('--gcc',       type=float, default=2.6,
                        help='Baseline gcc (M$/km); default 2.6')
    parser.add_argument('--tx',        type=float, default=0.56,
                        help='Baseline tx rebuild cost (M$); default 0.56')
    parser.add_argument('--rate',      type=float, default=0.07,
                        help='Baseline discount rate; default 0.07')
    parser.add_argument('--threshold', type=float, default=150.0,
                        help='LCOE cut-off ($/MWh) for accessible capacity; default 150')
    parser.add_argument('--outdir',    type=Path,
                        default=Path('results/sensitivity/M3'))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_oat         = []
    all_interaction = []

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

        required = ['cluster_id', 'potential_capacity', 'CF_mean', 'capex',
                    'fom', 'vom', 'nearest_station_distance_km', 'Operational_life']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ERROR: Missing columns: {missing}", file=sys.stderr)
            continue

        # OAT sweep
        oat_df = run_oat_sensitivity(
            df, resource_type,
            gcc_baseline   = args.gcc,
            tx_baseline    = args.tx,
            lcoe_threshold = args.threshold,
            baseline_rate  = args.rate,
        )

        oat_df.to_csv(
            args.outdir / f'M3_oat_{resource_type}.csv', index=False)

        plot_tornado(
            oat_df, resource_type,
            args.outdir / f'M3_tornado_{resource_type}.png',
            args.threshold)

        # Interaction sweep
        interaction_df = run_interaction_sweep(
            df, resource_type,
            gcc_baseline   = args.gcc,
            tx_baseline    = args.tx,
            lcoe_threshold = args.threshold,
        )
        interaction_df.to_csv(
            args.outdir / f'M3_interaction_{resource_type}.csv', index=False)
        plot_interaction_heatmap(
            interaction_df, resource_type,
            args.outdir / f'M3_interaction_heatmap_{resource_type}.png')

        all_oat.append(oat_df)
        all_interaction.append(interaction_df)

        # Print summary
        print(f"\n  OAT results (baseline = {oat_df['baseline_gw'].iloc[0]:.1f} GW):")
        print(oat_df[['parameter', 'scenario_label', 'total_gw',
                       'delta_gw', 'delta_pct']].to_string(index=False))

    # Save combined outputs
    if all_oat:
        pd.concat(all_oat).to_csv(
            args.outdir / 'M3_oat_combined.csv', index=False)
    if all_interaction:
        pd.concat(all_interaction).to_csv(
            args.outdir / 'M3_interaction_combined.csv', index=False)
    print(f"\nResults saved to: {args.outdir}")


if __name__ == '__main__':
    main()
