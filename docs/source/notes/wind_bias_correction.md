# ERA5–GWA wind-speed bias correction

## Status

```{warning}
The existing wind-speed rescaling code demonstrates a multiplicative mean-bias
correction, but it is not yet sufficiently robust or validated for production
scientific results. Treat corrected wind time series as experimental until the
acceptance criteria in this note are satisfied.
```

This note records the issue and a recommended implementation direction. It is a
design document, not evidence that the correction has been completed or validated.

## Intended correction

RESource currently targets a multiplicative delta correction at 100 m:

```text
factor(x, y) = mean_GWA_100m(x, y) / climatology_ERA5_100m(x, y)

corrected_ERA5_100m(t, x, y) = ERA5_100m(t, x, y) * factor(x, y)
```

This preserves ERA5's temporal profile while adjusting its long-term mean toward
the spatially detailed GWA mean. The general method is established in wind-energy
modelling, but published validation shows that improvement depends on location and
GWA version. Correction must therefore be optional and evaluated against an
uncorrected baseline.

Relevant implementation locations:

- `src/RESource/windspeed.py`: scaling and cutout mutation
- `src/RESource/gwa.py`: GWA raster preparation and ERA5-cell mapping
- `src/RESource/timeseries.py`: correction before turbine conversion
- `src/RESource/RESources.py`: workflow orchestration and stored cell attributes

## Problems in the current implementation

### Duplicate ERA5 coordinates

Assessment geometries can share the same ERA5 coordinate. The current row-wise
algorithm snaps every row to its nearest coordinate and writes each scaled series
back in sequence. Later rows can overwrite earlier rows, making the final result
dependent on input order.

### Target-year normalization

The denominator is calculated from the ERA5 data being processed. In a multiyear
workflow, normalizing each year to the same GWA climatological mean suppresses real
interannual differences. A fixed ERA5 reference climatology must be calculated once
and reused for every target year.

### Spatial aggregation

The current workflow converts every GWA raster pixel into a point and averages the
points assigned to each assessment cell. This is memory intensive and does not
explicitly weight partial pixels by intersected area. GWA should instead be
aggregated directly from the regional raster onto unique ERA5 grid polygons.

### Input mutation and assignment

The scaling function overwrites the caller's `x` and `y` columns and updates the
cutout one grid cell at a time. This obscures provenance, is slow, and makes duplicate
handling difficult to audit.

### Missing safeguards

The current path does not define policy for:

- missing or nodata GWA pixels;
- zero, near-zero or non-finite ERA5 climatological means;
- extreme or scientifically implausible correction factors;
- inconsistent GWA and ERA5 height, CRS, extent or land/sea masks;
- GWA product version and climatological reference period;
- ERA5 cells containing no valid GWA coverage.

## Recommended implementation

### 1. Define provenance and configuration

Record the following with every run:

- GWA version, layer, nominal height and retrieval date;
- ERA5 variables, nominal height and reference-climatology period;
- raster aggregation method and nodata policy;
- factor bounds and missing-value behavior;
- whether correction is enabled.

Bias correction should be opt-in until regional validation supports making it a
default. A future configuration could use:

```yaml
wind_bias_correction:
  enabled: false
  source: GWA
  height_m: 100
  era5_reference_period: [2006, 2018]
  spatial_statistic: area_weighted_mean
  minimum_valid_coverage: 0.8
  factor_bounds: [0.5, 2.0]
  outside_bounds: flag
```

The example bounds are placeholders and require scientific justification before
adoption.

### 2. Aggregate GWA directly to unique ERA5 cells

Use the region-clipped GWA 100 m GeoTIFF as the authoritative spatial input. Build
one table indexed by unique ERA5 `(x, y)` coordinates and calculate an area-weighted
GWA mean for each ERA5 polygon. Do not create a full-resolution point GeoDataFrame.

The aggregation result should contain at least:

```text
x, y, gwa_mean_100m, valid_pixel_fraction, gwa_version
```

### 3. Build a fixed ERA5 reference climatology

Calculate the mean `wnd100m` for the configured multiyear reference period on the
same ERA5 grid. Cache the climatology with a fingerprint of its variables, period,
grid and source metadata. Do not recalculate the denominator independently for each
target weather year. Follow the
[ten-year ERA5 reference climatology procedure](era5_reference_climatology.md) for
acquisition, validation and provenance.

### 4. Calculate and validate one factor per ERA5 coordinate

Join GWA means to the ERA5 climatology using explicit coordinate indexes. Calculate
one factor per unique coordinate and attach quality flags rather than silently
discarding invalid values.

Suggested flags include:

- `missing_gwa`
- `insufficient_gwa_coverage`
- `invalid_era5_mean`
- `factor_outside_review_range`
- `uncorrected_fallback`

Never allow row order to select a factor.

### 5. Apply correction vectorially

Represent factors as an `xarray.DataArray` aligned to the cutout's `x` and `y`
coordinates, then use labelled-array multiplication:

```python
corrected = cutout.data["wnd100m"] * correction_factor
```

Return a new or explicitly copied dataset rather than modifying the caller's cell
table. Preserve the original wind speed as an auditable baseline, or persist enough
metadata to reproduce it exactly.

### 6. Convert corrected wind to generation

Apply the correction to `wnd100m` before the atlite turbine power-curve conversion.
Confirm that atlite does not subsequently apply an incompatible height adjustment.
Report both uncorrected and corrected capacity factors during validation.

## Required tests

### Deterministic unit tests

- A known ERA5 series is rescaled to the requested climatological mean.
- Temporal shape and calm/high-wind ordering are preserved.
- Reordering input assessment cells does not change results.
- Duplicate assessment geometries sharing one ERA5 coordinate receive one factor.
- Multiple target years retain their original relative annual means.
- Coordinate alignment is correct when x or y ordering is reversed.
- Missing, zero and non-finite inputs follow the documented fallback policy.
- Factors outside the review range are flagged and never silently clipped.
- The input cutout and cell table are not unexpectedly mutated.

### Spatial tests

- Area-weighted zonal means match a small synthetic raster/polygon example.
- Partial edge pixels and nodata coverage are handled deterministically.
- GWA and ERA5 CRS, height and regional extent are checked before aggregation.

### Regional validation

For each supported region, compare uncorrected and corrected results against
independent measurements or observed generation where licensing permits. Report at
least mean bias, MAE/RMSE, correlation, annual capacity factor and distributional
errors. A correction should not be adopted merely because it moves the mean toward
GWA.

## Acceptance criteria

The correction is ready for production only when:

1. one deterministic factor exists per unique ERA5 coordinate;
2. target years share a fixed, documented ERA5 reference climatology;
3. GWA aggregation is raster based and coverage weighted;
4. invalid values and extreme factors produce explicit flags;
5. results are independent of input row order;
6. corrected and uncorrected outputs are reproducible and auditable;
7. unit, spatial and multiyear regression tests pass; and
8. regional evidence demonstrates the correction's effect and limitations.

## Scientific references

- Gruber et al. (2022), [Towards global validation of wind power simulations:
  A multi-country assessment of wind power simulation from MERRA-2 and ERA-5
  reanalyses bias-corrected with the Global Wind Atlas](https://doi.org/10.1016/j.energy.2021.121520).
- [Global Wind Atlas GIS files and API access](https://globalwindatlas.info/en/download/gis-files/).
