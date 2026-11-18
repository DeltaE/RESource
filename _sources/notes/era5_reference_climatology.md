# Ten-year ERA5 reference climatology

## Purpose

This procedure prepares the fixed ERA5 reference period required by the proposed
ERA5–GWA wind-speed bias correction. The climatology is the denominator in:

```text
factor(x, y) = mean_GWA_100m(x, y) / climatology_ERA5_100m(x, y)
```

It must be calculated once from a fixed multiyear window and then reused for every
target weather year. Recalculating the denominator independently for each target
year would suppress genuine interannual variability.

This document describes data acquisition and validation. It does not mark the bias
correction itself as production ready.

## Reference period

As of August 2026, use the ten most recent complete calendar years:

```text
2016-01-01 through 2025-12-31
```

Do not include the incomplete 2026 calendar year. ERA5 preliminary updates are
normally available close to real time, while final ERA5 replaces preliminary ERA5T
after approximately two to three months. Record whether each downloaded year is
final ERA5 or contains ERA5T before freezing a scientific reference product.

If the project requires final-quality data only, confirm that December 2025 has
been finalized in CDS before accepting the climatology. Otherwise use 2015–2024
and record that choice explicitly.

## Scope that must be confirmed

Before starting, record:

- configuration file;
- region codes;
- years and whether final ERA5 is required;
- expected variables and spatial resolution;
- output directory and available disk space;
- CDS credential readiness;
- expected runtime and queue limitations;
- whether the complete RESource workflow or only weather acquisition is intended.

The current `resource-multiyear` command launches the complete `resource` CLI once
per year. It is not an ERA5-only downloader. Consequently it can repeat GADM, GWA,
GAEZ, land, capacity, scoring and time-series stages according to step-cache state.

## Preflight checks

From the repository root:

```bash
uv sync --locked --all-extras
uv run resource-multiyear --help
df -h . /tmp
test -f "$HOME/.cdsapirc"
stat -c '%a %n' "$HOME/.cdsapirc"
```

The credential file should have mode `600`. Never print or commit its contents.
Ensure the repository filesystem has sufficient free space. RESource routes CDS and
Python staging to the ignored `data/tmp/resource-cds/` directory on that filesystem,
rather than relying on a potentially small system `/tmp` mount.

Confirm that the configured cutout root is scenario appropriate. Canadian baseline
outputs currently resolve below:

```text
data/downloaded_data/cutout/
```

## Multiyear command

For a confirmed British Columbia reference run using the existing launcher:

```bash
uv run resource-multiyear config/CAN_baseline.yaml \
  --start 2016 \
  --end 2025 \
  --regions BC
```

The launcher runs years sequentially and continues after a failed year. The CDS
fallback submits monthly requests sequentially and retries recognized temporary
queue-capacity failures with bounded exponential backoff.

Do not launch several RESource processes with the same CDS account. CDS explicitly
limits queued requests, and parallel runs increase rejection risk.

## ERA5-only pipeline

Use the dedicated cutout downloader rather than relying on step-cache side effects:

```bash
uv run resource-cutout-multiyear config/CAN_baseline.yaml \
  --start 2016 \
  --end 2025 \
  --region BC
```

This command calls only boundary preparation and `ERA5Cutout.get_era5_cutout()` for
each year. It validates the resulting NetCDF and maintains a resumable,
machine-readable manifest under `results/manifests/`. An internal preflight task
runs Python garbage collection before each annual CDS job so arrays and file-backed
objects from the previous year do not accumulate in process memory. It records the
cleanup counts but never deletes disk caches or completed cutouts.

## Monitoring and recovery

For every year, record:

- command exit status;
- final NetCDF path and byte size;
- first and last timestamps;
- expected versus actual hourly timestep count;
- variables present;
- missing or duplicated timestamps;
- whether CDS returned ERA5 or ERA5T where detectable;
- retry count and final failure reason.

Expected hourly counts are:

- 8,760 for a normal calendar year;
- 8,784 for leap years 2016, 2020 and 2024.

Timezone-aligned cutouts may include shifted UTC boundaries. Validate against the
configured snapshot convention rather than assuming filenames alone prove complete
coverage.

Re-run only failed years:

```bash
uv run resource config/CAN_baseline.yaml --year YEAR --regions BC
```

Existing valid cutouts should be audited and reused. A partial or invalid NetCDF
must not be treated as a cache hit.

## Building the reference climatology

After all ten annual cutouts pass validation:

1. Open only the required 100 m wind-speed field from each annual cutout.
2. Confirm identical `x` and `y` grids, CRS, units and variable definitions.
3. Concatenate along time without loading all values eagerly into memory.
4. Check for duplicate and missing timestamps across year boundaries.
5. Calculate the time mean of `wnd100m` for each unique ERA5 coordinate.
6. Persist the climatology as a labelled NetCDF or Zarr dataset.
7. Store provenance and a cryptographic digest for every annual input.

Recommended output metadata:

```yaml
product: RESource ERA5 100 m wind reference climatology
reference_period: [2016, 2025]
variable: wnd100m
units: m s-1
source: ERA5
spatial_grid: inherited from annual RESource cutouts
aggregation: arithmetic mean over validated hourly observations
created_with: deltae-resource
input_manifest: path to checksummed annual inputs
```

Do not overwrite annual source cutouts after building the climatology. The manifest
must allow the reference product to be reproduced exactly.

## Acceptance checks

The weather acquisition is complete only when:

1. all ten annual cutouts exist and open successfully;
2. all expected years and timestamps are present;
3. spatial coordinates, CRS, units and variables are consistent;
4. no annual file is suspiciously small or partial;
5. ERA5 versus ERA5T status is documented where relevant;
6. retry and failure summaries are retained;
7. the climatology contains finite values for every supported ERA5 cell; and
8. the input manifest and climatology metadata are stored with the result.

## Related design note

See [ERA5–GWA wind-speed bias correction](wind_bias_correction.md) for the
correction algorithm, quality flags, tests and production acceptance criteria.

## Authoritative ERA5 information

- [ERA5 data documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=388500357)
- [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
