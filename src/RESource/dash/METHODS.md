# `RESource.dash` — Module Methods Reference

Standalone, Pyodide-backed dashboard for exploring RESource `.h5` stores in the
browser. This document describes each module in `src/RESource/dash/` in a
standardized format: **purpose**, **inputs/outputs**, **key functions**, and
**where it sits in the pipeline**. It is written for maintainers who need to
change or extend the dash, especially the path that turns a raw store into
clusters, profiles and aggregated summaries.

Source of truth is the code itself — this file should be updated whenever a
function signature or pipeline step changes.

---

## 0. How the pieces fit together

```
cli.py                     (resource-dash entrypoint)
   -> build.py              reads 4 .py modules as text, inlines them into the html template
        -> templates/RESource_dash.html.j2   (single static HTML file, no server)
             -> Web Worker runs a real Pyodide interpreter at runtime
                  -> store_reader.py     (raw h5py access, no RESource dependency)
                  -> store_summary.py    (one store -> JSON payload: cells/clusters/profiles/story)
                  -> config_reader.py    (YAML scenario configs -> resolved JSON)
                  -> cluster_explorer.py (exploratory re-clustering + cluster profiles)
```

Only `store_reader.py`, `store_summary.py`, `config_reader.py` and
`cluster_explorer.py` are inlined into the browser bundle (see
`build.py::_PY_MODULES`). `build.py` and `cli.py` never run inside the
browser — they only exist to produce the HTML file natively.

**Top-to-bottom pipeline for clusters & profiles** (the part the user asked
about specifically):

1. `store_reader.read_fixed_frame` rebuilds a pandas `DataFrame` straight from
   the HDF5 "fixed" layout (no PyTables) for any table the caller asks for:
   `cells`, `clusters/<tech>`, `timeseries/<tech>`, `timeseries/clusters/<tech>`,
   `dissolved_indices/<tech>`, `boundary`, `lines`, land analytics tables, etc.
2. `store_summary._cells` / `store_summary._clusters` turn the `cells` and
   `clusters/<tech>` frames into JSON-safe column blocks, applying any
   `capacity_haircuts/*` adjustments (`_capacity_adjustments`) and masking the
   LCOE sentinel (999000 -> `null`).
3. `store_summary._profiles` -> `store_summary.profile_stats` reduce the
   `timeseries/clusters/<tech>` hourly table to monthly / diurnal / month×hour
   / daily means, capacity-weighted by `clusters/<tech>.potential_capacity`.
4. `store_summary._story` computes headline numbers (`capacity_mw`, `cf_wmean`,
   `lcoe_wmean`, cluster count, etc.) from the `cells` and `clusters/<tech>`
   frames produced in steps 2–3.
5. `store_summary.summarize_store` is the entry point that calls 1–4 for every
   table present in one store and assembles the full dash payload; per-table
   failures are caught per-block so one bad table doesn't hide the rest.
6. `store_summary.combine_summaries` merges several regional payloads (already
   produced by step 5) into one country-level payload: concatenates cell/
   cluster tables, re-derives `story`, and capacity-weights fleet profiles
   across regions.
7. Separately, `cluster_explorer.recluster_store` / `cluster_cells_json` offer
   an **exploratory** re-clustering of the raw `cells` table (k-means by cost,
   location, or both) that does not touch the pipeline's own
   `clusters/<tech>` tables; `cluster_explorer.cluster_profiles` then builds
   hourly capacity-weighted profiles for those exploratory clusters directly
   from `timeseries/<tech>`.

---

## 1. `store_reader.py`

**Role:** Pure `h5py` reader for RESource stores. The only module with zero
RESource dependency — runs identically natively and inside Pyodide (no
PyTables, which has no WebAssembly build).

**Inputs:** path or open `h5py.File` to a `.h5` store written by
`pandas.HDFStore` in "fixed" format.

**Outputs:** `pandas.DataFrame` reconstructed from raw HDF5 blocks, or small
JSON-safe geometry helpers.

| Function | Purpose | Notes |
|---|---|---|
| `open_store(path)` | Open read-only, raising `StoreError` with a human reason (`truncated`, `unreadable`) instead of a raw `OSError`. | First call in every entry point. |
| `list_frame_keys(h5file)` | Enumerate every group that looks like a pandas fixed-format frame (`axis0`/`axis1` present), with row/col counts. | Used to build `out["keys"]` and the `have` set in `summarize_store`. |
| `frame_columns(h5file, key)` | Column names of one frame without reading data. | Used for previews and cheap existence checks. |
| `read_fixed_frame(h5file, key, columns=None, rows=None)` | Rebuild a `DataFrame`, lazily: only HDF5 blocks holding the requested `columns` are read, and `rows` can be a slice. | Core primitive; the 8760×N timeseries tables rely on this to stay cheap. |
| `time_index(index)` | Convert stored int64-nanosecond index to a `DatetimeIndex`. | Store timestamps are always ns since epoch. |
| `wkt_parts(wkt, max_pts, max_parts, ndigits)` | Parse a WKT geometry into `[[lon, lat], ...]` parts, thinned for display. | Display-grade only — not used for area/analytical computation. |
| `wkt_kind(wkt)` | First token of a WKT string (`POLYGON`, `LINESTRING`, ...). | Used to decide line vs. point rendering in `_lines`. |

---

## 2. `store_summary.py`

**Role:** Turns one store into the full JSON payload the dash renders.
Nothing is invented — a missing table yields a missing block, surfaced to the
user rather than silently defaulted. Depends only on `store_reader` (same
flat namespace once inlined).

**Inputs:** store path (+ optional adjustments table), or a list of JSON
payloads (for `combine_summaries`).

**Outputs:** one JSON string per store (or combined), consumed directly by
the dash's JS layer.

### 2.1 Cell / cluster / geometry blocks

| Function | Purpose | Produces |
|---|---|---|
| `_boundary(f)` | Region boundary polygons + a simplified outline (via shapely if available). | `out["boundary"]` |
| `_cells(f, adjustments)` | Cell table -> JSON column block; applies capacity haircuts; masks LCOE sentinel. | `out["cells"]`, and the `cells_df` used downstream by `_story` and land analytics. |
| `_clusters(f, tech, adjustments)` | `clusters/<tech>` table -> JSON column block, same haircut/sentinel handling as `_cells`. | `out["clusters"][tech]`, and `cl_df` used by `_profiles`/`_story`. |
| `_lines(f)` | Transmission line/point features from the `lines` table, voltage-sorted and capped at `MAX_LINE_FEATURES`. | `out["lines"]` |
| `_nodes(f, key)` | Uploaded substation/grid-node markers (used when a run has no OSM lines). | `out["nodes"]` |

### 2.2 Profile / aggregation methods (the core of the question)

| Function | Purpose | Key detail |
|---|---|---|
| `profile_stats(ts, weights=None)` | Reduce an hourly frame (columns = sites) to **monthly**, **diurnal**, **month×hour**, and **daily** means of the capacity-weighted fleet series. | `weights` defaults to equal weight per column; falls back to equal weight if all weights are zero/NaN. Returns `mean`, `hours`, `start`/`end`, plus the raw `hourly` series for the day picker. |
| `_profiles(f, tech, cl_df)` | Calls `profile_stats` on `timeseries/clusters/<tech>`, weighted by each cluster's `potential_capacity` from `cl_df`; also computes **per-site** monthly/diurnal means. | One of the two heaviest I/O paths — reads a full `hours × n_clusters` table. |
| `_story(cells_df, cl, tech, sentinel)` | Headline numbers for one technology: total capacity, developable share, capacity-weighted CF/LCOE, grid-distance stats, cluster count/capacity/LCOE range. | Pure aggregation over the frames produced by `_cells`/`_clusters` — no further store I/O. |
| `combine_summaries(payloads, label)` | Merge N regional `summarize_store` payloads into one country-level payload. | Concatenates cell/cluster/boundary/line/node blocks; **capacity-weights** fleet profiles across regions (falls back to equal weight if any region reports zero capacity); re-derives `story` from the merged frames. |

### 2.3 Land analytics, metadata, and exports

| Function | Purpose |
|---|---|
| `_landanalytics(f, key, cells_df, cells_out, ...)` | Summarize one `LandAnalytics*` table without assuming its schema; if its index matches ≥90% of `cells_df`, attach its numeric columns onto the cells payload (`short:col`) so the map/layer stack can color by them. |
| `_la_media(f, frame_keys)` / `land_image(path, key)` | List/embed rendered PNG figures stored as raw `uint8` datasets under `LandAnalytics/images`; large figures (> `LA_IMAGE_MAX_BYTES`) are listed, not embedded, and loaded on demand via `land_image`. |
| `_run_meta(f)` | Decode the `_meta` group's attributes (run id, runtime, config, provenance) as stored. |
| `_capacity_adjustments(f, run)` | Validate `capacity_haircuts/*` tables against the source `cells`/`clusters` tables and the applied factor recorded in `_meta`; raises rather than silently trusting a mismatched adjustment. |
| `summarize_store(path, filename=None)` | **Top-level entry point.** Calls `open_store` -> `list_frame_keys` -> each block function above, per technology, catching per-block errors so one bad table doesn't blank the whole payload. Returns `{"meta", "keys", "boundary", "cells", "lines", "nodes", "clusters", "profiles", "cost", "story", "landanalytics", "la_media", "units", "missing", "problems", "run", "haircut_status"}`. |
| `cell_profile(path, tech, cell_id)` | Lazy single-cell monthly/diurnal/month×hour profile, via `profile_stats` on one column read from `timeseries/<tech>`. |
| `cluster_compare(path, tech, cluster_id, max_members=40)` | Pairs a cluster's **member cells'** diurnal CF/LCOE (from `timeseries/<tech>` + `dissolved_indices/<tech>`) against the cluster's own **representative** profile/LCOE (from `timeseries/clusters/<tech>` + `clusters/<tech>`) — lets a user see whether a pipeline cluster is actually homogeneous. |
| `table_preview(path, key, n=200)` / `table_csv(path, key, max_values)` | Raw table browsing/export for the Data tab; `table_csv` refuses tables above `max_values` to avoid freezing the browser. |

---

## 3. `cluster_explorer.py`

**Role:** Exploratory re-clustering of a store's `cells` table — by cost,
location, or both — independent of the pipeline's own `clusters/<tech>`
tables written by `RESource/cluster.py`. Backs the dash's Cluster tab; the
same functions also run natively (see module docstring for a copy-paste
example).

**Method:** plain k-means (k-means++ init, Lloyd iterations, fixed seed),
implemented in numpy only — no scikit-learn, so every step is inspectable.
Distances are planar (longitude scaled by `cos(mean latitude)`) — adequate
for a province/country, not continental scale.

| Function | Purpose | Notes |
|---|---|---|
| `kmeans(X, k, seed=0, n_iter=100)` | Core k-means on an `(n, d)` feature matrix. | k-means++ seeding; an emptied cluster is restarted on the point farthest from its centroid. |
| `features(lon, lat, lcoe, by="both", cost_weight=0.5)` | Build the `(n, d)` feature matrix: location in km (normalized by spread) and/or cost (LCOE normalized by its own spread), blended by `cost_weight`. | `by="cost"`/`"geo"` are shortcuts for `cost_weight=1`/`0`. |
| `cluster_cells(cells, by, k, cost_weight, scope, max_lcoe, seed)` | **Main aggregation entry point.** Labels every eligible cell (capacity > 0, valid LCOE, within `max_lcoe`), groups into clusters, then aggregates each cluster: `n_cells`, capacity-weighted `cf_mean`/`lcoe`, `lcoe_min`/`max`, mean grid distance, capacity-weighted centroid. Ranks clusters cheapest-first (`C1`, `C2`, ... for `scope="store"`; `<region>_1`, `<region>_2`, ... for `scope="region"`). Also reports `lcoe_variance_kept` (within/total variance ratio) as a clustering-quality diagnostic. | Ineligible cells are kept in the output with an empty cluster id, never silently dropped (`info["n_excluded_*"]`). |
| `cell_table(path, tech)` | Read the `cells` table columns `cluster_cells` needs (`lon, lat, lcoe_<tech>, potential_capacity_<tech>, <tech>_CF_mean, nearest_station_distance_km, region`), applying capacity haircuts. | Raises `KeyError` if the store hasn't run scoring for `tech`. |
| `recluster_store(path, tech, **options)` | `cell_table` + `cluster_cells` in one call — the natural native entry point. | |
| `cluster_profiles(path, tech, cells, chunk_rows=1000)` | Hourly, **capacity-weighted mean** profile per exploratory cluster, computed directly from `timeseries/<tech>` in row chunks (so it doesn't require the whole table in memory). Hours where a member has no value exclude that member from the mean for that hour only. | |
| `cluster_cells_json(payload)` | Dash entry point: JSON in (`{"cells": {...}, "options": {...}}`) -> JSON out (`{"clusters", "cells", "info"}`), via `_frame_from_payload` + `cluster_cells`. | |
| `cluster_profiles_csv(path, tech, payload)` | Dash entry point: re-run `cluster_cells` with the same options, then export `cluster_profiles` as CSV text. | |

---

## 4. `config_reader.py`

**Role:** Resolve RESource YAML scenario configs supplied by the user (not
read from the store) for the dash's Config tab.

| Function | Purpose |
|---|---|
| `read_configs(texts)` | Parses each YAML file; resolves `extends: ../base.yaml` chains by matching file names/relative paths among the files the user supplied (a parent not supplied is reported as `missing_parents`, never guessed); merges child over parent (`_merge`). Returns `{"configs": [...], "errors": {...}}` as a JSON string, one entry per file with `run_id`, `description`, `country`, `is_scenario`, and the fully merged `config`. |

---

## 5. `build.py` / `cli.py` (native only — never runs in the browser)

| Function | Purpose |
|---|---|
| `build.build_dashboard(output)` | Reads the 4 inlined modules' source as text, renders `templates/RESource_dash.html.j2` with them JSON-embedded, writes one self-contained HTML file. |
| `cli.main()` / `cli.entrypoint()` | `resource-dash [--out FILE]` CLI wrapper around `build_dashboard`. |

---

## Adding a new aggregation method

1. If it reduces a per-cell/per-cluster table to summary numbers, add it next
   to `_story` or `profile_stats` in `store_summary.py` and wire it into
   `summarize_store`'s `block(...)` calls.
2. If it's a new way to **group** cells (not the pipeline's own clustering),
   add it in `cluster_explorer.py` next to `cluster_cells`, following the
   same contract: ineligible rows kept with an empty cluster id, not dropped;
   return an `info` dict describing what was excluded and why.
3. Either way, add the corresponding JSON-in/JSON-out wrapper (see
   `cluster_cells_json` / `cluster_profiles_csv`) and wire a call into the JS
   `S`/`K` dispatch table in `templates/RESource_dash.html.j2`.
4. Update this table and `tests/test_dash_browser_reader.py` (or the
   equivalent native test) so the browser and native code paths stay in
   sync — `build.py`'s docstring explicitly depends on that invariant.
