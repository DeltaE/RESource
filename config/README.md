# Config layout: base files and scenarios

Each country directory (`CAN/`, `BGD/`, `WB6/`, ...) has one `base.yaml` and a
`scenarios/` folder of files that `extends: ../base.yaml`. Resolution logic
lives in `resolve_config()` / `_merge_config()` in
[`src/RESource/utility.py`](../src/RESource/utility.py).

## Merge semantics

- `extends` is resolved relative to the file that declares it, and chains
  (scenario → base → ...) recursively.
- Mappings deep-merge: a scenario only needs to specify the keys it actually
  changes; everything else falls through from base.
- Lists **replace** on override, not merge — to add to a base list without
  rewriting it, use `{$append: [...]}` instead of a plain list.
- Top-level keys are validated against `KNOWN_CONFIG_TOP_LEVEL_KEYS` in
  `utility.py`; a typo'd top-level key raises rather than silently doing
  nothing.

## Top-level category map

Top-level keys are grouped by theme rather than by data provider. When
adding a new data source, put it under the category that matches what the
data *is*, not where it happens to come from:

| Category | Contains | Was (legacy) |
| --- | --- | --- |
| `admin_boundary` | `GADM` | top-level `GADM` |
| `weather` | `cutout`, `GWA` | top-level `cutout`, `GWA` |
| `technology` | `annual_technology_baseline` (NREL ATB source), `resource_specs.<solar\|wind\|bess>` (per-resource cost/sizing model) | top-level `NREL.ATB`, `capacity_disaggregation` |
| `infrastructure` | `OSM`, `CODERS`, `transmission` | top-level `OSM_data`, `CODERS`, `transmission` |
| `lands` | `GAEZ`, `CORINE`, `EU_DEM`, raster-processing defaults | top-level `GAEZ`, `CORINE`, `EU_DEM` |
| `demand_indicators` | `WorldPop`, `Gov.Population`, `Gov.CEEI` | top-level `WorldPop`, `custom_land_layers.Gov` |
| `filters` | `<solar\|wind>.vector_buffers` — siting exclusion buffers | `capacity_disaggregation.<solar\|wind>.vector_buffers` |
| `custom_land_layers` | `rasters`, `vectors` (raster/vector land-cover sources not tied to a named provider above) | unchanged |
| *(top-level, ungrouped)* | `Title`, `Developer`, `Affiliation`, `version`, `Release_Year`, `Scenario`, `country`, `default_CRS`, `weather_year`, `economic_parameters`, `region_mapping`, `multi_country_flag`, `description` | unchanged |

`resource_specs` holds the cost/sizing model for a resource (`landuse_intensity`,
`cost_data`, `operational_life`, turbine specs, ...). `filters` holds the
siting exclusion buffers (`vector_buffers`) for that same resource — they're
split because one describes the technology, the other describes a
regulatory/policy choice, and policy is what scenarios vary.

## What belongs in `base.yaml`

Anything that is a property of the country/dataset itself, or a default that
every scenario should share unless it has a specific reason not to:

- region mappings, CRS, data source URLs, credentials paths, static lookup
  tables (IUCN categories, land-cover classes, etc.)
- default technical/economic assumptions (e.g. `landuse_intensity`,
  `operational_life`, `max_capacity` per cell) — these describe the
  disaggregation/costing model, not a scenario's regulatory or policy
  assumptions.

If a value is identical across all (or nearly all) scenario files, that's a
signal it isn't actually scenario-specific — move it to `base.yaml` and
delete it from the scenarios.

## What belongs in a scenario file

Only the deltas that define what makes the scenario different: the
regulatory/policy assumptions being tested. Typical examples:

- `Scenario.run_id` / `Scenario.Description` (required, always scenario-local)
- `filters.<solar|wind>.vector_buffers` — buffer distances around exclusion
  features
- `infrastructure.transmission.proximity_filter_km` or other siting
  constraints
- weather year, cost-curve selection, or other explicit "what-if" swaps

A scenario file should read as a diff against the baseline assumptions, not
a restatement of them. If you find yourself copying the same block into
every scenario, that block belongs in base instead (see above).

## Parameter units reference

Key names carry their unit as a suffix where practical (`_km`, `_Km`, `_km2`)
— fields without a suffix are documented here. Keep this table in sync when
adding or renaming a config field.

### `technology.resource_specs.<solar|wind>`

| Key | Unit | Notes |
| --- | --- | --- |
| `max_capacity` | MW / cell | Upper bound on disaggregated capacity per grid cell |
| `landuse_intensity` | MW / km² | Installable capacity density used for cell sizing |
| `operational_life` | years | Asset lifetime, feeds LCOE/CRF calculation |
| `cell_static_CF_tolerance` | fraction (0–1) | Solar-only; capacity-factor homogeneity tolerance within a cell |
| `cell_capacity_tolerance` | fraction (0–1) | Allowed deviation in cell capacity during disaggregation |
| `WCSS_tolerance` | fraction (0–1) | Within-cluster sum-of-squares convergence tolerance for clustering |
| `CF_low` / `CF_high` | fraction (0–1) | Wind-only; valid capacity-factor range used to filter turbine curves |

### `filters.<solar|wind>`

| Key | Unit | Notes |
| --- | --- | --- |
| `vector_buffers.*.buffer_mapping_key_buffers.<feature>` | meters | Exclusion buffer radius around the named OSM/land feature |

### `infrastructure.transmission`

| Key | Unit | Notes |
| --- | --- | --- |
| `grid_connection_cost_per_Km` | M$ / km | Marginal cost of new grid connection distance |
| `tx_line_rebuild_cost` | M$ / km | Cost to rebuild/upgrade an existing line |
| `proximity_filter_km` | km | Max distance from a site to a grid node/line to be considered connectable |
| `buses` | path | CSV of candidate substation/bus locations |
| `lines` | path | CSV of candidate transmission line geometries |

### `economic_parameters`

| Key | Unit | Notes |
| --- | --- | --- |
| `discount_rate` | fraction (0–1) | WACC; used to compute the capital recovery factor for the LCOE proxy |

### `weather.cutout`

| Key | Unit | Notes |
| --- | --- | --- |
| `dx` / `dy` | degrees | ERA5 cutout spatial resolution |

### `weather.GWA.filter`

| Key | Unit | Notes |
| --- | --- | --- |
| `windspeed_low` / `windspeed_high` | m/s | Valid wind-speed range used to filter GWA raster cells |

### `region_mapping.<region>`

| Key | Unit | Notes |
| --- | --- | --- |
| `land_area_km2` | km² | Region land area |
| `percentage_national_land_area` | % | Share of total country land area |

### `custom_land_layers.rasters[]`

| Key | Unit | Notes |
| --- | --- | --- |
| `source_res_meters` | meters | Native resolution of the source raster |
| `target_res_meters` | meters | Resolution the raster is resampled to before use |

### Top-level

| Key | Unit | Notes |
| --- | --- | --- |
| `weather_year` | calendar year | Targeted ERA5 cutout year |

## Adding a new scenario

1. Start from `extends: ../base.yaml` and a `Scenario.run_id` /
   `Scenario.Description`.
2. Add only the keys you're deliberately changing.
3. Before committing, check whether any value you added is identical to
   what's already in `base.yaml`, or duplicated across other scenarios —
   if so, it likely belongs in base, not in the scenario.
