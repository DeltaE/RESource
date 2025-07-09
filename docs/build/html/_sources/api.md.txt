# RES APIs
```{warning}
This page is under heavy development
```
## RESource Builder

**Main class for the RESource framework providing renewable energy resource assessment capabilities.**

```{eval-rst}
.. py:class:: RES.RESources.RESources_builder

   Main builder class for RESource framework that integrates geospatial, temporal, economic, and regulatory data to evaluate site suitability for solar and wind energy development.

   .. py:method:: get_grid_cells()

      Retrieves the default grid cells for the region.

      :returns: GeoDataFrame containing grid cells
      :rtype: geopandas.GeoDataFrame

   .. py:method:: get_temporal_data()

      Retrieves temporal data for the specified region and time period.

      :returns: Temporal data for the region
      :rtype: xarray.Dataset

   .. py:method:: run_assessment()

      Runs the complete renewable energy resource assessment.

      :returns: Assessment results
      :rtype: dict
```

## Annual Technology baseline (ATB)

**Processor for NREL's Annual Technology Baseline data.**

```{eval-rst}
.. py:class:: RES.atb.NREL_ATBProcessor

   Class for processing and managing NREL Annual Technology Baseline data for renewable energy cost and performance projections.

   .. py:method:: load_data()

      Loads ATB data for the specified year.

   .. py:method:: get_technology_data(technology)

      Retrieves data for a specific technology.

      :param technology: Technology type (e.g., 'wind', 'solar')
      :type technology: str
```

> ℹ️ The ATB data source and configuration may change annually. Ensure you are referencing the correct year and dataset for your analysis.
> * Currently configured for 2024 ATB.
> * Please review and update configuration if using for a different year or context.

## Administrative Boundaries

**Handler for GADM administrative boundary data.**

```{eval-rst}
.. py:class:: RES.boundaries.GADMBoundaries

   Class for downloading and processing GADM (Global Administrative Areas) boundary data.

   .. py:method:: download_boundaries(country_code)

      Downloads boundary data for the specified country.

      :param country_code: ISO country code
      :type country_code: str

   .. py:method:: get_regions()

      Retrieves available administrative regions.
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

