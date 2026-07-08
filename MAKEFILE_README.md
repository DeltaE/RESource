# Running RESource — Makefile Reference

## Prerequisites

| Requirement | Notes |
|---|---|
| [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html) | `conda` must be on `PATH` |
| Python 3.12 | Installed automatically into the `RESource` environment |
| CDS API key | Required for ERA5 downloads — configure `~/.cdsapirc` before running |

---

## Quick Start

```bash
# 1. Set up the conda environment (first time only)
make setupenv

# 2. Activate it
conda activate RESource

# 3. Run the pipeline
make run-can YEAR=2020
```

---

## Environment Setup

| Command | What it does |
|---|---|
| `make setupenv` | Creates the `RESource` conda env from `env/environment.yml` |
| `make setupenv-clean` | Creates a clean env by pinning exact package versions via `pip` (use when `environment.yml` gives solver conflicts) |
| `make updateenv` | Updates an existing env to match `env/environment.yml` |
| `make exportenv` | Exports the active env back to `env/environment.yml` |

> All pipeline targets check that the `RESource` environment exists before running. If it is missing, you will see an error pointing you to `make setupenv`.

---

## Running the Pipeline

The general pattern is:

```bash
make <target> [YEAR=YYYY] [REGIONS='R1 R2 ...']
```

### Named targets

| Target | Config file | Regions |
|---|---|---|
| `run-wb6` | `config/config_WB6.yaml` | All Western Balkans regions |
| `run-wb6-region` | `config/config_WB6.yaml` | `REGIONS=` subset |
| `run-can` | `config/config_CAN_baseline.yaml` | All Canadian provinces |
| `run-can-region` | `config/config_CAN_baseline.yaml` | `REGIONS=` subset |
| `run-can-policy` | `config/config_CAN_policy1.yaml` | All Canadian provinces |
| `run-bgd` | `config/config_BGD.yaml` | All Bangladesh regions |
| `run` | `CONFIG=` (required) | All regions in config, or `REGIONS=` subset |

### Region codes

**Western Balkans:** `AL BA XK ME MK RS`

**Canada:** `AB BC MB NB NL NS ON PE QC SK`

### Variables

| Variable | Default | Description |
|---|---|---|
| `YEAR` | *(none)* | Weather year to process (`--year YYYY`). Overrides `weather_year` in the config YAML. Required unless `weather_year` is set in the config. |
| `REGIONS` | *(all)* | Space-separated region codes. Omit to process all regions defined in the config. |
| `CONFIG` | `config/config_WB6.yaml` | Path to YAML config. Only relevant for the generic `run` target. |

---

## Examples

```bash
# Canada baseline — all provinces, year 2020
make run-can YEAR=2020

# Canada baseline — BC and Alberta only
make run-can-region YEAR=2020 REGIONS='BC AB'

# Western Balkans — all regions, year 2019
make run-wb6 YEAR=2019

# Western Balkans — three regions only
make run-wb6-region YEAR=2019 REGIONS='AL MK RS'

# Bangladesh — all regions, year 2021
make run-bgd YEAR=2021

# Generic entry point with a custom config
make run CONFIG=config/config_BGD.yaml YEAR=2021

# Generic entry point — custom config, subset of regions
make run CONFIG=config/config_WB6.yaml YEAR=2020 REGIONS='BA RS'
```

---

## Documentation

```bash
# Build HTML docs and deploy to GitHub Pages
make docs

# Live-reload docs server on http://127.0.0.1:8000
make autobuild

# Deploy a previously built docs folder to GitHub Pages
make deploy
```

---

## Utilities

```bash
# Remove build artefacts and __pycache__ folders
make clean

# Print all available targets with descriptions
make help
```

---

## Output Locations

| Output | Path |
|---|---|
| Results (HDF5 / CSV / plots) | `results/<Country>/<Region>/<RUN_ID>/` |
| Runtime log | `results/logs/runtime_log.txt` |
| Intermediate store | `data/store/<Country>/<RUN_ID>/` |
| Visualisations | `vis/<Country>/<RUN_ID>/<Region>/<resource>/` |

The runtime log records wall-clock time, per-region success/failure, hardware metrics (CPU, RAM, process RSS), Python version, and platform for every pipeline invocation. It is appended, not overwritten, so historical runs are preserved.

---

## Notes

- `YEAR` is optional only if `weather_year` is present in the config YAML. Passing `--year` on the command line always takes precedence over the config value.
- `REGIONS` values are case-insensitive — the pipeline converts them to uppercase internally.
- A run that completes but fails for one or more regions logs status `PARTIAL`; a clean run logs `SUCCESS`.
- You do not need to activate the conda environment manually before calling `make` — all targets use `conda run -n RESource` internally.
