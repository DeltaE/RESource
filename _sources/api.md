# RES APIs
```{warning}
This page is under heavy development
```
## RESource Builder

**Main class for the RESource framework providing renewable energy resource assessment capabilities.**


## Annual Technology Baseline (ATB)

**Processor for NREL's Annual Technology Baseline data.**

```{eval-rst}
.. autoclass:: RES.atb.NREL_ATBProcessor
   :members:
   :show-inheritance:
```

> ℹ️ The ATB data source and configuration may change annually. Ensure you are referencing the correct year and dataset for your analysis.
> * Currently configured for 2024 ATB.
> * Please review and update configuration if using for a different year or context.

## Administrative Boundaries

**Handler for GADM administrative boundary data.**

```{eval-rst}
.. autoclass:: RES.boundaries.GADMBoundaries
   :members:
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render properly, this indicates import issues with heavy geospatial dependencies. The GADMBoundaries class provides:

- **download_boundaries(country_code)**: Downloads boundary data for the specified country
- **get_regions()**: Retrieves available administrative regions  
- **process_boundaries()**: Processes and validates boundary data
```

> ℹ️ `RES.boundaries.GADMBoundaries` is to be used for standalone data download/validation purposes.


## Spatial Grid Cell Processor

```{eval-rst}
.. autoclass:: RES.cell.GridCells
   :members:
   :show-inheritance:
```
```{eval-rst}
.. autoclass:: RES.CellCapacityProcessor.CellCapacityProcessor
   :members:
   :show-inheritance:
```

```{note}
If the above documentation doesn't render, these classes provide grid cell processing capabilities for spatial analysis.
```

## Global Land Cover
```{eval-rst}
.. autoclass:: RES.gaez.GAEZRasterProcessor
   :members:
   :show-inheritance:
   :noindex:
```

## Global Wind Atlas 

**Handler for Global Wind Atlas data processing.**

```{eval-rst}
.. autoclass:: RES.gwa.GWACells
   :members:
   :show-inheritance:
   :noindex:
```

## Scorer

```{eval-rst}
.. autoclass:: RES.score.CellScorer
   :members:
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render, this class provides cell scoring capabilities for renewable energy site assessment.
```

## Clustering

```{eval-rst}
.. automodule:: RES.cluster
   :members:
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render properly, this module provides clustering algorithms for renewable energy resource grouping and analysis.
```

## Local Data Store with HDF5 file

```{eval-rst}
.. autoclass:: RES.hdf5_handler.DataHandler
   :members:
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render, this class provides HDF5-based data storage and retrieval capabilities for the RESource framework.
```

## Turbine Configuration

```{eval-rst}
.. autoclass:: RES.tech.OEDBTurbines
   :members:
   :show-inheritance:
```

## Units

```{eval-rst}
.. autoclass:: RES.units.Units
   :members:
   :show-inheritance:
```

---
```{warning}
This page is under heavy development
```

