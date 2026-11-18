"""Regroup the grid cells of a RESource store into clusters, by cost, location or both.

This is the Python behind the dash's Cluster tab. The dash sends the cell table to
``cluster_cells_json`` and draws whatever comes back, so a change made here shows up
in the browser after ``resource-dash`` rebuilds the HTML. The same functions run
natively:

    from RESource.dash.cluster_explorer import recluster_store, cluster_profiles, cluster_profiles_package
    res = recluster_store("resources_Canada_BC_BASELINE_2024_20261116.h5", "wind",
                          by="both", k=12, cost_weight=0.5, scope="store")
    res["clusters"]   # one row per cluster, cheapest first (pandas DataFrame)
    res["cells"]      # one row per cell with its cluster id
    cluster_profiles(path, "wind", res["cells"]).to_csv("profiles.csv")
    # several regions: cluster_profiles([store_a, store_b], "wind", cells_with_source_column)
    # one ZIP with mean + spread profiles, tables and metadata.json (what the dash exports):
    # data, summary = cluster_profiles_package(path, "wind", {"cells": {...}, "options": {...}})

Scope and limits
----------------
* This is an exploratory regrouping. It does not replace the clusters that the
  RESource pipeline writes to ``clusters/<tech>`` (``RESource/cluster.py``).
* The method is k-means (Lloyd's algorithm, k-means++ start, fixed seed), written
  out in numpy so each step can be read and changed. No scikit-learn.
* Distances are planar. Longitude is scaled by cos(mean latitude). Good for a
  province or a country, not for a continent.
* Cluster LCOE and CF are capacity-weighted means of the member cells. A cluster
  LCOE is therefore a summary, not a recomputed cost.

Units: capacity in MW, LCOE as stored (USD/MWh in the dash), distances in km,
coordinates in degrees (EPSG:4326).
"""

from __future__ import annotations

import base64
import io
import json
import os
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:  # inside the dash every module sits in one flat folder
    from store_reader import _blocks, open_store, read_fixed_frame, time_index
    from store_summary import _capacity_adjustments, _run_meta
except ImportError:  # native use as part of the RESource package
    from .store_reader import _blocks, open_store, read_fixed_frame, time_index
    from .store_summary import _capacity_adjustments, _run_meta

LCOE_SENTINEL = 999_000.0   # RESource writes 999999 where a cell has no viable LCOE
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32     # at the equator; scaled by cos(latitude) below
MODES = ("cost", "geo", "both")
SCOPES = ("store", "region")


# ------------------------------------------------------------------ k-means

def kmeans(X, k, seed=0, n_iter=100):
    """Plain k-means. ``X`` is (n, d). Returns an integer label per row, 0..k-1.

    Start: k-means++ (each new centre is drawn with probability proportional to
    the squared distance from the centres already chosen). Then Lloyd iterations
    until the labels stop changing. The seed makes the result repeatable.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(n)]]
    d2 = np.sum((X - centres[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = d2.sum()
        pick = rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)
        centres.append(X[pick])
        d2 = np.minimum(d2, np.sum((X - X[pick]) ** 2, axis=1))
    C = np.array(centres)
    labels = np.full(n, -1)
    for _ in range(n_iter):
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (n, k)
        new = dist.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
            else:  # an emptied cluster restarts on the point farthest from its centre
                C[j] = X[dist.min(axis=1).argmax()]
    return labels


def features(lon, lat, lcoe, by="both", cost_weight=0.5):
    """Feature matrix for k-means, plus the cost weight actually used.

    Location becomes two columns in km, divided by one common spread so that
    north-south and east-west distances count the same. Cost becomes one column,
    LCOE divided by its own spread. ``cost_weight`` (0..1) sets the balance:
    0 is location only, 1 is cost only. ``by="cost"`` and ``by="geo"`` are
    shortcuts for those two ends.
    """
    if by not in MODES:
        raise ValueError(f"by must be one of {MODES}, got {by!r}")
    w = {"cost": 1.0, "geo": 0.0}.get(by, float(np.clip(cost_weight, 0.0, 1.0)))
    lon, lat, lcoe = (np.asarray(a, dtype=float) for a in (lon, lat, lcoe))
    x_km = lon * KM_PER_DEG_LON * np.cos(np.radians(np.nanmean(lat)))
    y_km = lat * KM_PER_DEG_LAT
    geo = np.column_stack([x_km - x_km.mean(), y_km - y_km.mean()])
    geo_spread = np.sqrt((geo ** 2).sum(axis=1).mean()) or 1.0
    cost = lcoe - lcoe.mean()
    cost_spread = cost.std() or 1.0
    return np.column_stack([(1.0 - w) * geo / geo_spread, w * cost / cost_spread]), w


# ------------------------------------------------------------------ core

def _wmean(v, w):
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def cluster_cells(cells, by="both", k=10, cost_weight=0.5, scope="store", max_lcoe=None, seed=0):
    """Cluster a cell table. Returns ``{"clusters": DataFrame, "cells": DataFrame, "info": dict}``.

    ``cells`` is a DataFrame indexed by cell id with columns
    ``lon, lat, lcoe, capacity`` and optionally ``cf, region, grid_km``.

    A cell takes part only if it has capacity above zero and a valid LCOE (and an
    LCOE at or below ``max_lcoe`` when that is given). The rest are returned with
    an empty cluster id and counted in ``info``; they are never silently dropped.

    ``scope="store"`` clusters the whole table into ``k`` clusters.
    ``scope="region"`` clusters inside each region separately, ``k`` per region
    (fewer where a region has fewer eligible cells).

    Clusters are numbered by capacity-weighted LCOE, cheapest first.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    ids = [str(i) for i in cells.index]
    df = cells.reset_index(drop=True)   # positions, not ids: a cell id may repeat across regions
    for col in ("cf", "grid_km"):
        if col not in df:
            df[col] = np.nan
    if "region" not in df:
        df["region"] = ""
    df["region"] = df["region"].fillna("").astype(str)
    for col in ("lon", "lat", "lcoe", "capacity", "cf", "grid_km"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["lcoe"] >= LCOE_SENTINEL, "lcoe"] = np.nan

    ok = df["capacity"].gt(0) & df["lcoe"].notna() & df["lon"].notna() & df["lat"].notna()
    n_no_value = int((~ok).sum())
    if max_lcoe is not None and np.isfinite(max_lcoe):
        over = ok & df["lcoe"].gt(float(max_lcoe))
        ok &= ~over
    else:
        over = pd.Series(False, index=df.index)
    use = df[ok]
    if use.empty:
        raise ValueError("No cell has capacity above zero and a valid LCOE under the chosen limit.")

    # 1. label every eligible cell: (group, k-means label)
    raw = pd.Series(index=use.index, dtype=object)
    w_used = None
    groups = [("", use)] if scope == "store" else list(use.groupby("region", sort=True))
    for name, part in groups:
        X, w_used = features(part["lon"], part["lat"], part["lcoe"], by, cost_weight)
        labels = kmeans(X, k, seed=seed)
        raw.loc[part.index] = [f"{name}|{j}" for j in labels]

    # 2. one row per cluster
    rows = []
    for key, part in use.groupby(raw, sort=False):
        cap = part["capacity"].to_numpy()
        rows.append({
            "key": key,
            "region": ", ".join(sorted(set(part["region"]) - {""})),
            "n_cells": int(len(part)),
            "capacity_MW": float(cap.sum()),
            "cf_mean": _wmean(part["cf"].to_numpy(), cap),
            "lcoe": _wmean(part["lcoe"].to_numpy(), cap),
            "lcoe_min": float(part["lcoe"].min()),
            "lcoe_max": float(part["lcoe"].max()),
            "grid_km_mean": float(part["grid_km"].mean()) if part["grid_km"].notna().any() else np.nan,
            "lon": _wmean(part["lon"].to_numpy(), cap),
            "lat": _wmean(part["lat"].to_numpy(), cap),
        })
    cl = pd.DataFrame(rows).sort_values("lcoe", kind="stable").reset_index(drop=True)

    # 3. rank and name: C1 is the cheapest cluster; inside a region, <region>_1 is its cheapest
    cl["rank"] = np.arange(1, len(cl) + 1)
    if scope == "store":
        cl["cluster"] = ["C" + str(r) for r in cl["rank"]]
    else:
        cl["cluster"] = cl["region"] + "_" + (cl.groupby("region").cumcount() + 1).astype(str)
    cl["cumulative_MW"] = cl["capacity_MW"].cumsum()
    names = dict(zip(cl["key"], cl["cluster"]))
    ranks = dict(zip(cl["key"], cl["rank"]))

    out_cells = pd.DataFrame({
        "cluster": raw.map(names).reindex(df.index).fillna(""),
        "rank": raw.map(ranks).reindex(df.index),
        "region": df["region"], "lon": df["lon"], "lat": df["lat"],
        "capacity_MW": df["capacity"], "cf_mean": df["cf"], "lcoe": df["lcoe"],
    })
    if "source" in df:   # index of the store file a cell came from (combined views)
        out_cells["source"] = pd.to_numeric(df["source"], errors="coerce").fillna(0).astype(int)
    out_cells.index = pd.Index(ids, name="cell")

    # 4. how much of the cell-to-cell LCOE variation the grouping keeps (0..1)
    total = float(((use["lcoe"] - use["lcoe"].mean()) ** 2).sum())
    within = float(sum(((p["lcoe"] - p["lcoe"].mean()) ** 2).sum() for _, p in use.groupby(raw)))
    info = {
        "by": by, "k": int(k), "cost_weight": w_used, "scope": scope, "seed": int(seed),
        "max_lcoe": None if max_lcoe is None else float(max_lcoe),
        "n_cells": int(len(df)), "n_clustered": int(len(use)), "n_clusters": int(len(cl)),
        "n_excluded_no_value": n_no_value, "n_excluded_over_limit": int(over.sum()),
        "capacity_MW": float(use["capacity"].sum()),
        "lcoe_variance_kept": None if total <= 0 else 1.0 - within / total,
    }
    cols = ["cluster", "rank", "region", "n_cells", "capacity_MW", "cumulative_MW", "cf_mean",
            "lcoe", "lcoe_min", "lcoe_max", "grid_km_mean", "lon", "lat"]
    return {"clusters": cl[cols].set_index("cluster"), "cells": out_cells, "info": info}


# ------------------------------------------------------------------ store access

def cell_table(path, tech):
    """The columns ``cluster_cells`` needs, read from the ``cells`` table of one store."""
    want = ["x", "y", f"lcoe_{tech}", f"potential_capacity_{tech}", f"{tech}_CF_mean",
            "nearest_station_distance_km", "Region", "Municipality", "Province", "Country"]
    with open_store(path) as f:
        df = read_fixed_frame(f, "cells", columns=want)
        adjustments = _capacity_adjustments(f, _run_meta(f) if "_meta" in f else {})
        if adjustments is not None:
            df[f"potential_capacity_{tech}"] = adjustments["cells"][
                f"potential_capacity_{tech}_adjusted"
            ].to_numpy()
    missing = [c for c in want[:4] if c not in df]
    if missing:
        raise KeyError(f"cells table lacks {missing}; has this store run scoring for {tech}?")
    region = next((c for c in ("Region", "Municipality", "Province", "Country") if c in df), None)
    return pd.DataFrame({
        "lon": df["x"], "lat": df["y"], "lcoe": df[f"lcoe_{tech}"],
        "capacity": df[f"potential_capacity_{tech}"],
        "cf": df.get(f"{tech}_CF_mean", np.nan), "grid_km": df.get("nearest_station_distance_km", np.nan),
        "region": df[region].astype(str) if region else "",
    }, index=[str(i) for i in df.index])


def recluster_store(path, tech, **options):
    """Read one store and cluster its cells. ``options`` go to ``cluster_cells``."""
    return cluster_cells(cell_table(path, tech), **options)


def _paths(path):
    """One store path, a list of paths, or a JSON list of paths (from the dash) -> list."""
    if isinstance(path, (list, tuple)):
        return [str(x) for x in path]
    path = str(path)
    return [str(x) for x in json.loads(path)] if path.lstrip().startswith("[") else [path]


def cluster_profiles(path, tech, cells, chunk_rows=1000, return_std=False):
    """Hourly capacity-factor profile of each cluster, as a DataFrame (hours x clusters).

    ``cells`` is the ``cells`` frame returned by ``cluster_cells``. ``path`` is one
    store, or a list of stores when the cells come from several regions; then
    ``cells`` needs a ``source`` column holding each cell's position in that list.

    Aggregation (the standard used to reduce site profiles to a representative
    technology profile, e.g. PyPSA-Eur, GenX, OSeMOSYS time-slice inputs):

        CF_k(t) = sum_i w_i CF_i(t) / sum_i w_i,   i in cluster k,  w_i = capacity_MW of cell i

    This keeps the cluster's energy: CF_k(t) times the cluster capacity equals the
    summed output of its cells. A member with no value at hour t leaves the sum and
    the weight total for that hour only.

    With ``return_std=True`` the capacity-weighted standard deviation across the
    members at each hour is returned as well, ``(mean, std)``:

        SD_k(t) = sqrt( sum_i w_i CF_i(t)^2 / sum_i w_i - CF_k(t)^2 )

    It measures how far the members spread around the cluster profile in that hour.
    The table is read in row chunks, so a store with thousands of cells does not
    have to fit in memory.
    """
    paths = _paths(path)
    members = cells[cells["cluster"].astype(str) != ""].copy()
    members["source"] = (pd.to_numeric(members["source"], errors="coerce").fillna(0).astype(int)
                         if "source" in members else 0)
    members["cell"] = members.index.astype(str)
    members = members.drop_duplicates(["source", "cell"])
    if members["source"].max() >= len(paths) or members["source"].min() < 0:
        raise ValueError(f"cells refer to store {members['source'].max()} but {len(paths)} store(s) were given.")
    order = list(dict.fromkeys(members.sort_values("rank")["cluster"]))
    col_of = {c: j for j, c in enumerate(order)}
    index = num = sq = den = None
    found = 0
    for si, part in members.groupby("source"):
        weight = {c: (col_of[k], w) for c, k, w in zip(part["cell"], part["cluster"], part["capacity_MW"])}
        with open_store(paths[si]) as f:
            key = f"timeseries/{tech}"
            if key not in f:
                raise KeyError(f"{os.path.basename(paths[si])} has no {key} table.")
            group = f[key]
            idx = time_index(group["axis1"][:])
            if index is None:
                index = idx
                num, sq, den = (np.zeros((len(index), len(order))) for _ in range(3))
            elif len(idx) != len(index) or not np.array_equal(idx.values, index.values):
                raise ValueError("The stores hold different hours, so their cells cannot share one profile.")
            for items, raw in _blocks(group):
                if raw.dtype == object or raw.ndim != 2:
                    continue
                W = np.zeros((len(items), len(order)))   # cell x cluster weight matrix
                for i, name in enumerate(items):
                    hit = weight.get(str(name))
                    if hit is not None:
                        W[i, hit[0]] = hit[1]
                        found += 1
                if not W.any():
                    continue
                for r0 in range(0, len(index), chunk_rows):
                    block = raw[r0:r0 + chunk_rows]
                    good = np.isfinite(block)
                    x = np.where(good, block, 0.0)
                    num[r0:r0 + chunk_rows] += x @ W
                    sq[r0:r0 + chunk_rows] += (x * x) @ W
                    den[r0:r0 + chunk_rows] += good @ W
    if found < len(members):
        raise KeyError(f"timeseries/{tech} holds {found} of the {len(members)} clustered cells.")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
        std = np.sqrt(np.clip(np.where(den > 0, sq / den, np.nan) - mean ** 2, 0.0, None))
    out = pd.DataFrame(mean, index=index, columns=order)
    out.index.name = "time"
    if not return_std:
        return out
    sd = pd.DataFrame(std, index=index, columns=order)
    sd.index.name = "time"
    return out, sd


def profile_summary(mean, std=None, clusters=None):
    """One row per cluster describing its hourly profile.

    ``cf_profile_mean`` is the time mean of CF_k(t). ``full_load_hours`` is the sum
    of CF_k(t) over the record (MWh per MW; annual full-load hours when the record
    is one year). ``cf_p10``/``cf_p90`` are percentiles over the hours.
    ``sd_within_mean`` is the time mean of the within-cluster spread SD_k(t).
    When ``clusters`` (from ``cluster_cells``) is given, ``cf_check`` is
    cf_profile_mean minus the cluster's capacity-weighted ``cf_mean`` from the cells
    table. A value far from zero means the stored mean CF and the stored hourly
    series do not describe the same thing (other year, gaps, other weights).
    """
    m = mean.to_numpy()
    out = pd.DataFrame({
        "hours": np.isfinite(m).sum(axis=0),
        "hours_missing": (~np.isfinite(m)).sum(axis=0),
        "cf_profile_mean": np.nanmean(m, axis=0),
        "full_load_hours": np.nansum(m, axis=0),
        "cf_p10": np.nanpercentile(m, 10, axis=0),
        "cf_p90": np.nanpercentile(m, 90, axis=0),
        "cf_max": np.nanmax(m, axis=0),
    }, index=pd.Index(mean.columns, name="cluster"))
    if std is not None:
        out["sd_within_mean"] = np.nanmean(std.to_numpy(), axis=0)
    if clusters is not None and "cf_mean" in clusters:
        out["cf_check"] = out["cf_profile_mean"] - clusters["cf_mean"].reindex(out.index)
    return out


# ------------------------------------------------------------------ k-sweep (how many clusters?)

def cell_shapes(path, tech, keys, chunk_rows=744):
    """Mean CF by month x hour of day (12 x 24 = 288 values) for each ``(source, cell)`` in ``keys``.

    A stand-in for timeslice-averaged CF when no timeslice definition is given:
    most OSeMOSYS timeslice sets are seasons x day-parts, which are unions of these
    288 bins. Rows are NaN for cells missing from ``timeseries/<tech>``.
    """
    paths = _paths(path)
    pos = {}
    for j, (si, cell) in enumerate(keys):
        pos.setdefault((int(si), str(cell)), []).append(j)
    out = np.full((len(keys), 288), np.nan)
    for si in sorted({int(k[0]) for k in keys}):
        with open_store(paths[si]) as f:
            group = f[f"timeseries/{tech}"]
            idx = time_index(group["axis1"][:])
            M = np.zeros((len(idx), 288))
            M[np.arange(len(idx)), (idx.month.to_numpy() - 1) * 24 + idx.hour.to_numpy()] = 1.0
            for items, raw in _blocks(group):
                if raw.dtype == object or raw.ndim != 2:
                    continue
                hit = [(i, pos[(si, str(n))]) for i, n in enumerate(items) if (si, str(n)) in pos]
                if not hit:
                    continue
                ci = np.array([i for i, _ in hit])
                S = np.zeros((len(ci), 288))
                C = np.zeros((len(ci), 288))
                for r0 in range(0, len(idx), chunk_rows):
                    block = raw[r0:r0 + chunk_rows][:, ci]
                    good = np.isfinite(block)
                    Mb = M[r0:r0 + chunk_rows]
                    S += np.where(good, block, 0.0).T @ Mb
                    C += good.T @ Mb
                with np.errstate(invalid="ignore", divide="ignore"):
                    V = np.where(C > 0, S / C, np.nan)
                for row, (_, js) in zip(V, hit):
                    out[js] = row
    return out


def _knee(k, y):
    """k at the largest distance from the chord joining the first and last points (both axes scaled 0..1)."""
    k, y = np.asarray(k, float), np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None
    k, y = k[ok], y[ok]
    kn = (k - k[0]) / ((k[-1] - k[0]) or 1.0)
    yn = (y - y[0]) / ((y[-1] - y[0]) or 1.0)
    return int(k[np.argmax(np.abs(yn - kn))])


def _kept(x, w, labels):
    """Capacity-weighted share of the variation of ``x`` (n, d) that the labels keep, and the
    capacity-weighted RMS distance of a member from its cluster mean (per column)."""
    W = w.sum()
    mean = (w[:, None] * x).sum(axis=0) / W
    total = float((w * ((x - mean) ** 2).sum(axis=1)).sum())
    K = labels.max() + 1
    num = np.zeros((K, x.shape[1]))
    np.add.at(num, labels, w[:, None] * x)
    den = np.bincount(labels, weights=w, minlength=K)
    cm = num / np.where(den > 0, den, 1.0)[:, None]
    within = float((w * ((x - cm[labels]) ** 2).sum(axis=1)).sum())
    kept = None if total <= 0 else 1.0 - within / total
    return kept, float(np.sqrt(within / (W * x.shape[1])))


def k_sweep(cells, shapes=None, k_max=20, cost_weight=0.5, scope="store", max_lcoe=None, seed=0, modes=None):
    """Score groupings for k = 1..k_max, by cost, by location and by both.

    Same eligibility, features, k-means and seed as ``cluster_cells``, so a point
    on the sweep is the clustering the Cluster tab draws for that k. Per k and mode:

    * ``lcoe_kept``: capacity-weighted share of cell-to-cell LCOE variance kept,
      1 - sum_i w_i (LCOE_i - LCOE_k(i))^2 / sum_i w_i (LCOE_i - mean)^2.
    * ``lcoe_mae``: capacity-weighted mean |LCOE_i - LCOE_k(i)|, in LCOE units.
    * ``shape_kept`` / ``shape_rmse``: the same two measures on the 288-value
      month x hour CF vector of each cell (``cell_shapes``), RMSE in CF units.
    * ``min_cluster_MW``: capacity of the smallest cluster.

    ``knee`` per mode and measure is a heuristic (largest distance from the chord),
    not an optimum. The LCOE measures here are capacity-weighted; the
    ``lcoe_variance_kept`` that ``cluster_cells`` reports is unweighted.
    """
    base = cluster_cells(cells, k=1, cost_weight=cost_weight, scope=scope, max_lcoe=max_lcoe, seed=seed)
    df = cells.reset_index(drop=True)
    if "region" not in df:
        df["region"] = ""
    df["region"] = df["region"].fillna("").astype(str)
    elig = (base["cells"]["cluster"].to_numpy() != "")
    use = df[elig].copy()
    for col in ("lon", "lat", "lcoe", "capacity"):
        use[col] = pd.to_numeric(use[col], errors="coerce")
    w = use["capacity"].to_numpy(float)
    lcoe = use["lcoe"].to_numpy(float)[:, None]
    S = None
    if shapes is not None:
        S = np.asarray(shapes, float)[elig]
        s_ok = np.isfinite(S).all(axis=1)
    groups = [("", np.arange(len(use)))] if scope == "store" else [
        (n, np.flatnonzero(use["region"].to_numpy() == n)) for n in sorted(use["region"].unique())]
    modes = modes or [("cost", 1.0), ("geo", 0.0), ("both", float(np.clip(cost_weight, 0, 1)))]
    ks = list(range(1, int(k_max) + 1))
    out = []
    for by, wt in modes:
        rec = {"by": by, "cost_weight": wt, "k": ks, "n_clusters": [], "lcoe_kept": [], "lcoe_mae": [],
               "shape_kept": [], "shape_rmse": [], "min_cluster_MW": []}
        feats = [(pos, features(use["lon"].to_numpy()[pos], use["lat"].to_numpy()[pos], lcoe[pos, 0], by, wt)[0])
                 for _, pos in groups]
        for k in ks:
            labels = np.zeros(len(use), dtype=int)
            off = 0
            for pos, X in feats:
                lab = kmeans(X, k, seed=seed)
                labels[pos] = lab + off
                off += lab.max() + 1
            _, labels = np.unique(labels, return_inverse=True)
            kept, _ = _kept(lcoe, w, labels)
            K = labels.max() + 1
            cm = np.bincount(labels, w * lcoe[:, 0], K) / np.bincount(labels, w, K)
            rec["n_clusters"].append(int(K))
            rec["lcoe_kept"].append(kept)
            rec["lcoe_mae"].append(float((w * np.abs(lcoe[:, 0] - cm[labels])).sum() / w.sum()))
            rec["min_cluster_MW"].append(float(np.bincount(labels, w, K).min()))
            if S is not None and s_ok.sum() > 1:
                sk, sr = _kept(S[s_ok], w[s_ok], np.unique(labels[s_ok], return_inverse=True)[1])
                rec["shape_kept"].append(sk)
                rec["shape_rmse"].append(sr)
            else:
                rec["shape_kept"].append(None)
                rec["shape_rmse"].append(None)
        rec["knee"] = {m: _knee(ks, [np.nan if v is None else v for v in rec[m]]) for m in ("lcoe_kept", "shape_kept")}
        out.append(rec)
    info = {"k_max": int(k_max), "scope": scope, "seed": int(seed), "max_lcoe": max_lcoe,
            "n_clustered": int(elig.sum()), "capacity_MW": float(w.sum()),
            "n_without_shape": None if S is None else int((~s_ok).sum()),
            "shape_basis": "mean CF by month x hour of day (288 bins)"}
    return {"modes": out, "info": info}


_SHAPE_CACHE = {}


def k_sweep_json(path, tech, payload, k_max=20):
    """Dash entry point. Reads cell shapes once per store set and technology, then sweeps k."""
    p = json.loads(payload) if isinstance(payload, str) else payload
    try:
        cells = _frame_from_payload(p)
        opts = dict(p.get("options") or {})
        opts.pop("by", None)
        opts.pop("k", None)
        paths = tuple(_paths(path))
        keys = list(zip(cells["source"].astype(int), cells.index.astype(str)))
        ck = (paths, tech, len(keys), hash(tuple(keys)))
        if ck not in _SHAPE_CACHE:
            _SHAPE_CACHE.clear()
            try:
                _SHAPE_CACHE[ck] = cell_shapes(list(paths), tech, keys)
            except KeyError:
                _SHAPE_CACHE[ck] = None   # no timeseries/<tech>: sweep cost and location only
        res = k_sweep(cells, _SHAPE_CACHE[ck], k_max=int(k_max), **opts)
    except (ValueError, KeyError) as exc:
        return json.dumps({"error": str(exc)})
    clean = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
    for m in res["modes"]:
        for key in ("lcoe_kept", "lcoe_mae", "shape_kept", "shape_rmse", "min_cluster_MW"):
            m[key] = [clean(v) for v in m[key]]
    return json.dumps(res, default=str)


# ------------------------------------------------------------------ dash entry points (JSON in, JSON out)

def _records(df, sig=6):
    df = df.reset_index()
    out = []
    for row in df.to_dict(orient="records"):
        rec = {}
        for key, v in row.items():
            if isinstance(v, (float, np.floating)):
                rec[key] = None if not np.isfinite(v) else float(f"{float(v):.{sig}g}")
            elif isinstance(v, (int, np.integer)):
                rec[key] = int(v)
            else:
                rec[key] = v
        out.append(rec)
    return out


def _frame_from_payload(payload):
    c = payload["cells"]
    n = len(c["ids"])
    return pd.DataFrame({
        "lon": c["lon"], "lat": c["lat"], "lcoe": c["lcoe"], "capacity": c["capacity"],
        "cf": c.get("cf") or [None] * n, "grid_km": c.get("grid_km") or [None] * n,
        "region": c.get("region") or [""] * n,
        "source": c.get("src") or [0] * n,
    }, index=[str(i) for i in c["ids"]])


def cluster_cells_json(payload):
    """Dash entry point. ``payload`` is a JSON string: ``{"cells": {...}, "options": {...}}``."""
    p = json.loads(payload) if isinstance(payload, str) else payload
    try:
        res = cluster_cells(_frame_from_payload(p), **(p.get("options") or {}))
    except (ValueError, KeyError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"clusters": _records(res["clusters"]), "cells": _records(res["cells"]), "info": res["info"]})


def cluster_profiles_csv(path, tech, payload):
    """Dash entry point: cluster the payload again with the same options and return profile CSV text.

    ``path`` is a store path or a JSON list of store paths (combined views).
    """
    p = json.loads(payload) if isinstance(payload, str) else payload
    try:
        res = cluster_cells(_frame_from_payload(p), **(p.get("options") or {}))
        return json.dumps({"csv": cluster_profiles(path, tech, res["cells"]).to_csv(float_format="%.5f")})
    except (ValueError, KeyError) as exc:
        return json.dumps({"error": str(exc)})


METHOD_NOTE = {
    "clustering": "k-means (k-means++ start, Lloyd iterations, fixed seed) on location in km and/or LCOE, "
                  "each scaled by its spread and blended by cost_weight. Exploratory; the pipeline's own "
                  "clusters/<tech> tables are not changed.",
    "profile": "CF_k(t) = sum_i w_i CF_i(t) / sum_i w_i over the member cells i of cluster k; "
               "w_i = potential capacity (MW) of cell i as used for clustering. Energy-preserving: "
               "CF_k(t) * sum_i w_i equals the summed output of the members.",
    "dispersion": "SD_k(t) = sqrt(sum_i w_i CF_i(t)^2 / sum_i w_i - CF_k(t)^2): capacity-weighted "
                  "standard deviation across the members at hour t.",
    "missing_values": "A member with no value at hour t is left out of that hour's sum and weight total.",
    "source_table": "timeseries/<tech> (one column per cell) in each store.",
}


def cluster_profiles_package(path, tech, payload, context=None):
    """ZIP bytes: profiles (mean and spread), cluster and membership tables, metadata.

    ``context`` (dict or JSON) may hold ``stores`` (file names as the user knows
    them) and ``subject`` (labels for the run). Both go to metadata.json only.
    """
    p = json.loads(payload) if isinstance(payload, str) else payload
    ctx = (json.loads(context) if isinstance(context, str) else context) or {}
    paths = _paths(path)
    res = cluster_cells(_frame_from_payload(p), **(p.get("options") or {}))
    mean, std = cluster_profiles(paths, tech, res["cells"], return_std=True)
    summary = profile_summary(mean, std, res["clusters"])
    clusters = res["clusters"].join(summary)
    cap = res["clusters"]["capacity_MW"].reindex(mean.columns)
    meta = {
        "schema": "resource-dash/cluster-profiles/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "technology": tech,
        "stores": ctx.get("stores") or [os.path.basename(x) for x in paths],
        "subject": ctx.get("subject") or [],
        "clustering": res["info"],
        "method": METHOD_NOTE,
        "time": {"start": str(mean.index[0]), "end": str(mean.index[-1]), "hours": int(len(mean)),
                 "timestamps": "as written in the store; no time-zone shift applied"},
        "clusters": {c: {"capacity_MW": float(cap[c])} for c in mean.columns},
        "units": {"cf": "fraction of potential capacity (0-1)", "capacity": "MW",
                  "lcoe": "as stored (USD/MWh in the dash)", "full_load_hours": "MWh per MW over the record",
                  "coordinates": "degrees, EPSG:4326"},
        "files": {
            "cluster_profiles_mean.csv": "CF_k(t), hours x clusters. Model input (e.g. PyPSA p_max_pu, OSeMOSYS capacity factor before time-slicing).",
            "cluster_profiles_sd.csv": "SD_k(t), hours x clusters. Within-cluster spread.",
            "clusters.csv": "One row per cluster: size, cost, location and profile statistics.",
            "cluster_cells.csv": "Every cell with its cluster; an empty cluster means the cell was excluded.",
            "metadata.json": "This file.",
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cluster_profiles_mean.csv", mean.to_csv(float_format="%.5f"))
        z.writestr("cluster_profiles_sd.csv", std.to_csv(float_format="%.5f"))
        z.writestr("clusters.csv", clusters.to_csv(float_format="%.6g"))
        z.writestr("cluster_cells.csv", res["cells"].to_csv(float_format="%.6g"))
        z.writestr("metadata.json", json.dumps(meta, indent=2, default=str))
    return buf.getvalue(), summary


def cluster_profiles_zip(path, tech, payload, context=None):
    """Dash entry point: ``cluster_profiles_package`` as base64 plus the profile summary."""
    try:
        data, summary = cluster_profiles_package(path, tech, payload, context)
    except (ValueError, KeyError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"zip_b64": base64.b64encode(data).decode("ascii"), "summary": _records(summary)})
