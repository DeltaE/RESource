# RESource Step Cache

`RESource.step_cache` implements a lightweight, config-hash-based cache that prevents
pipeline steps from re-running when their configuration has not changed. It is not a
full data version control system — it is deliberately minimal: hash a config subset,
check for data in the store, skip or run.

---

## The Problem It Solves

The RESource pipeline has eight sequential steps. Each step is computationally
expensive. On a re-run — whether to change an economic parameter, update turbine
specs, or test a new clustering tolerance — every step was previously re-executed
from scratch regardless of what actually changed.

The step cache breaks that coupling. Only steps whose governing configuration has
changed (or whose output is missing) will execute. Steps whose inputs are unchanged
and whose output is already in the HDF5 store are skipped.

---

## Mechanism

Before each pipeline step, `StepCache.is_current(step, store_key)` checks two
conditions:

1. **Config hash match** — SHA-256 of the config sections relevant to that step
   equals the hash stored from the previous run.
2. **Data present** — the HDF5 store contains the expected output dataset.

Both conditions must hold to skip the step. If either fails, the step runs and its
hash is updated on completion.

Hashes are persisted to a JSON sidecar file next to the HDF5 store:

```
data/store/<Country>/<RUN_ID>/resources_<Country>_<Region>_<RUN_ID>.h5
data/store/<Country>/<RUN_ID>/resources_<Country>_<Region>_<RUN_ID>.checksums.json
```

Wind and solar results share the same HDF5 file but are namespaced independently
in the sidecar (`wind::grid_cells` vs. `solar::grid_cells`), so a turbine config
change only invalidates wind steps — solar cache entries are unaffected.

---

## Config Keys Per Step

Each step monitors a specific subset of config keys. Changing any key in a step's
subset invalidates that step and all downstream steps whose inputs depend on it.

| Step | Config keys monitored |
|---|---|
| `grid_cells` | `grid_cell_resolution`, `region_mapping`, `GADM`, `default_CRS` |
| `grid_nodes` | `OSM_data`, `region_mapping`, `capacity_disaggregation` |
| `cell_capacity` | `custom_land_layers`, `CORINE`, `GAEZ`, `capacity_disaggregation`, `region_mapping` |
| `weather_data` | `cutout`, `weather_year`, `region_mapping` |
| `gwa_scaling` | `GWA`, `region_mapping` |
| `cf_timeseries` | `cutout`, `capacity_disaggregation`, `weather_year` |
| `scoring` | `economic_parameters`, `NREL`, `capacity_disaggregation` |
| `clustering` | `capacity_disaggregation`, `region_mapping` |

> **Note:** Downstream invalidation is not automatic. If you change a grid parameter
> and want scoring to reflect the new geometry, either pass `clean_store=True` to
> `build()` or call `cache.invalidate_all()` before the run. The cache guards
> individual steps, not dependency chains.

---

## Usage

### Normal pipeline run (automatic)

`StepCache` is instantiated inside `RESources_builder.build()` and requires no
direct interaction from the caller. Pass `clean_store=True` to force a full re-run:

```python
builder = RESources_builder(
    config_file_path="config/config_CAN_baseline.yaml",
    region_short_code="BC",
    resource_type="wind",
    weather_year=2024,
)

# Skips unchanged steps automatically
builder.build(select_top_sites=True, use_grid_lines=True)

# Forces all steps to re-run
builder.build(select_top_sites=True, use_grid_lines=True, clean_store=True)
```

### Manual cache control

```python
from RESource.step_cache import StepCache

cache = StepCache(
    store_path=builder.store,
    config=builder.config,
    resource_type="wind",
)

# Check whether a step would be skipped
cache.is_current("scoring", "cells")   # True → skip; False → run

# Invalidate a single step (forces it to re-run next time)
cache.invalidate("scoring")

# Invalidate all steps (equivalent to a fresh start)
cache.invalidate_all()

# Manually mark a step as done after running it outside build()
scored_cells = builder.score_cells()
cache.mark_done("scoring")
```

---

## Worked Examples

### Scenario 1: Change discount rate only

Only the `scoring` step sees a changed hash. Steps 1–4 are skipped; step 5
(scoring), step 6 (clustering), and step 7 (site selection) re-run.

### Scenario 2: Change turbine specification

`capacity_disaggregation` is monitored by `cell_capacity`, `cf_timeseries`,
`scoring`, and `clustering`. All four re-run. Steps 1–2 (`grid_cells`, `grid_nodes`)
are skipped because `grid_cell_resolution`, `GADM`, and `OSM_data` are unchanged.

### Scenario 3: Change weather year

`weather_year` appears in `weather_data` and `cf_timeseries`. Both re-run, along
with all downstream steps (`scoring`, `clustering`). Grid cells and capacity
estimates are unaffected and are skipped.

### Scenario 4: Re-run with identical config

All hashes match and all HDF5 keys exist. Every step is skipped. The run completes
in seconds — only the export and metadata steps execute.

---

## Adding a New Step

To guard a new pipeline step, add its name and governing config keys to
`STEP_CONFIG_KEYS` in `step_cache.py`:

```python
STEP_CONFIG_KEYS: dict[str, list[str]] = {
    ...
    "my_new_step": [
        "my_config_section",
        "region_mapping",
    ],
}
```

Then wrap the step in `build()`:

```python
if cache.is_current("my_new_step", "my_hdf5_key"):
    utils.print_update(level=2, message="my_new_step config unchanged — skipping.")
else:
    self.my_new_step_method()
    cache.mark_done("my_new_step")
```

No other changes are required.

---

## Limitations

- **No automatic downstream propagation.** If you change a grid parameter, the
  cache does not know that scoring depends on grid geometry. Either use
  `clean_store=True` for full re-runs or manually call `cache.invalidate()` on
  each affected step.
- **Config hash only.** The cache detects changes to config values but not to
  changes in raw input data files (e.g., a new GWA raster or a corrected GADM
  boundary). If input files change without a config change, call
  `cache.invalidate("gwa_scaling")` or the relevant step before the run.
- **Single-process only.** The JSON sidecar is not safe for concurrent writes. Do
  not run two `build()` calls for the same region and resource type simultaneously.
