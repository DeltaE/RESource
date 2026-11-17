"""Regroup the grid cells of a RESource store into clusters, by cost, location or both.

This is the Python behind the dash's Cluster tab. The dash sends the cell table to
``cluster_cells_json`` and draws whatever comes back, so a change made here shows up
in the browser after ``resource-dash`` rebuilds the HTML. The same functions run
natively:

    from RESource.dash.cluster_explorer import recluster_store, cluster_profiles
    res = recluster_store("resources_Canada_BC_BASELINE_2024_20261116.h5", "wind",
                          by="both", k=12, cost_weight=0.5, scope="store")
    res["clusters"]   # one row per cluster, cheapest first (pandas DataFrame)
    res["cells"]      # one row per cell with its cluster id
    cluster_profiles(path, "wind", res["cells"]).to_csv("profiles.csv")

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

import json

import numpy as np
import pandas as pd

try:  # inside the dash every module sits in one flat folder
    from store_reader import _blocks, open_store, read_fixed_frame, time_index
except ImportError:  # native use as part of the RESource package
    from .store_reader import _blocks, open_store, read_fixed_frame, time_index

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


def cluster_profiles(path, tech, cells, chunk_rows=1000):
    """Hourly capacity-factor profile of each cluster, as a DataFrame (hours x clusters).

    ``cells`` is the ``cells`` frame returned by ``cluster_cells``. A cluster
    profile is the capacity-weighted mean of its member cells' series in
    ``timeseries/<tech>``. Hours where a member has no value leave that member
    out of the mean for that hour. The table is read in row chunks, so a store
    with thousands of cells does not have to fit in memory.
    """
    members = cells[cells["cluster"].astype(str) != ""]
    members = members[~members.index.duplicated()]
    order = list(dict.fromkeys(members.sort_values("rank")["cluster"]))
    col_of = {c: j for j, c in enumerate(order)}
    with open_store(path) as f:
        group = f[f"timeseries/{tech}"]
        index = time_index(group["axis1"][:])
        num = np.zeros((len(index), len(order)))
        den = np.zeros((len(index), len(order)))
        found = 0
        for items, raw in _blocks(group):
            if raw.dtype == object or raw.ndim != 2:
                continue
            W = np.zeros((len(items), len(order)))   # cell x cluster weight matrix
            for i, name in enumerate(items):
                if name in members.index:
                    W[i, col_of[members.at[name, "cluster"]]] = members.at[name, "capacity_MW"]
                    found += 1
            if not W.any():
                continue
            for r0 in range(0, len(index), chunk_rows):
                block = raw[r0:r0 + chunk_rows]
                good = np.isfinite(block)
                num[r0:r0 + chunk_rows] += np.where(good, block, 0.0) @ W
                den[r0:r0 + chunk_rows] += good @ W
    if found < len(members):
        raise KeyError(f"timeseries/{tech} holds {found} of the {len(members)} clustered cells.")
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(den > 0, num / den, np.nan)
    out = pd.DataFrame(prof, index=index, columns=order)
    out.index.name = "time"
    return out


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
    """Dash entry point: cluster the payload again with the same options and return profile CSV text."""
    p = json.loads(payload) if isinstance(payload, str) else payload
    try:
        res = cluster_cells(_frame_from_payload(p), **(p.get("options") or {}))
        return json.dumps({"csv": cluster_profiles(path, tech, res["cells"]).to_csv(float_format="%.5f")})
    except (ValueError, KeyError) as exc:
        return json.dumps({"error": str(exc)})
