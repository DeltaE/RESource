# API reference

RESource currently exposes its established module API while class and module names
are standardized incrementally. New applications should import through the
`RESource` namespace.

## Assessment orchestration

```python
from RESource.RESources import RESources_builder

builder = RESources_builder(
    config_file_path="config/config_BC_baseline.yaml",
    region_short_code="BC",
    resource_type="wind",
    weather_year=2024,
)
builder.build()
```

`RESources_builder` coordinates boundary preparation, grid construction, land
constraints, climate time series, capacity estimation, scoring, and clustering.

## Main classes

| Area | Import | Responsibility |
|---|---|---|
| Configuration | `RESource.AttributesParser.AttributesParser` | Shared YAML configuration and region metadata |
| Boundaries | `RESource.boundaries.GADMBoundaries` | Administrative boundary acquisition and preparation |
| Cells | `RESource.cell.GridCells` | Spatial assessment grid generation |
| Capacity | `RESource.CellCapacityProcessor.CellCapacityProcessor` | Land-constrained capacity and cost processing |
| Climate | `RESource.timeseries.Timeseries` | Weather and capacity-factor time series |
| Economics | `RESource.score.CellScorer` | LCOE scoring and site ranking |
| Wind atlas | `RESource.gwa.GWACells` | Global Wind Atlas integration |
| Land cover | `RESource.gaez.GAEZRasterProcessor` | GAEZ raster processing |
| Storage | `RESource.hdf5_handler.DataHandler` | HDF5 persistence |
| Technology | `RESource.tech.OEDBTurbines` | Wind turbine metadata |
| Units | `RESource.units.Units` | Unit metadata and conversion support |

## Functional modules

- `RESource.cluster`: spatial clustering and representative-site aggregation.
- `RESource.visuals`: static and interactive result visualizations.
- `RESource.utility`: configuration, download, file, and geospatial helpers.
- `RESource.analytics`: sensitivity and multiyear analytical workflows.
- `RESource.lcoe_calculator`: technology-cost extraction and LCOE processing.

## Command-line API

```{eval-rst}
.. autofunction:: RESource.cli.build_parser

.. autofunction:: RESource.cli.main

.. autofunction:: RESource.cli.entrypoint
```

The executable entry point is `resource = RESource.cli:entrypoint`.

## Compatibility

The former `RES` namespace is deprecated but remains importable during the
migration period. New code and documentation must use `RESource`.
