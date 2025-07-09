# RES APIs
```{warning}
This page is under heavy development
```
## RESource Builder

```{eval-rst}
.. autoclass:: RES.RESources.RESources_builder
```

## Annual Technology baseline (ATB)

```{eval-rst}
.. autoclass:: RES.atb.NREL_ATBProcessor
```

> ℹ️ The ATB data source and configuration may change annually. Ensure you are referencing the correct year and dataset for your analysis.
> * Currently configured for 2024 ATB.
> * Please review and update configuration if using for a different year or context.

## Administrative Boundaries

```{eval-rst}
.. autoclass:: RES.boundaries.GADMBoundaries
```
> ℹ️ `RES.boundaries.GADMBoundaries` is to be used for standalone data download/validation purposes.


## Spatial Grid Cell Processor

```{eval-rst}
.. autoclass:: RES.cell.GridCells
```

```{note}
`RES.boundaries.GADMBoundaries.run` is to be used for standalone data download/validation purposes.
```

## Scorer

```{eval-rst}
.. autoclass:: RES.score.CellScorer
```

## Clustering

```{eval-rst}
.. automodule:: RES.cluster
    :members:
```

## Local Data Store with HDF5 file

```{eval-rst}
.. autoclass:: RES.hdf5_handler.DataHandler
```

## Turbine Configuration
```{eval-rst}
.. autoclass:: RES.tech.OEDBTurbines
```
## Units
```{eval-rst}
.. autoclass:: RES.units.Units
```
---
```{warning}
This page is under heavy development
```

