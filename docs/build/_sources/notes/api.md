# RES API References

```{warning}
This page is under heavy development - Additional modules and methods will be documented as the API stabilizes.
```

RESource provides a comprehensive API for variable renewable energy (VRE) resource assessment through a modular architecture. This reference documents the main classes and methods available for building custom assessment workflows.

## Core Workflow Classes

```{note}
Class documentation shows only docstrings without individual method details for cleaner overview. Individual methods are not displayed (`:members:` directive omitted).
```

### RESource Builder

**Main orchestrator class for renewable energy resource assessments.**

```{eval-rst}  
.. autoclass:: RES.RESources.RESources_builder
   :show-inheritance:
   :noindex:
```

The `RESources_builder` class coordinates the complete assessment workflow including spatial grid generation, land availability analysis, weather data processing, economic evaluation, and site clustering.

**Key Methods:**
- `get_grid_cells()`: Generate spatial grid covering region
- `get_cell_capacity()`: Calculate land-constrained potential capacity  
- `get_CF_timeseries()`: Generate capacity factor time series
- `score_cells()`: Economic evaluation using LCOE methodology
- `get_clusters()`: Spatial clustering of viable sites
- `build()`: Execute complete assessment workflow

### Spatial Grid Management

**Grid cell generation and spatial discretization.**

```{eval-rst}
.. autoclass:: RES.cell.GridCells
   :show-inheritance:
   :noindex:
```

Handles creation of regular spatial grids for renewable energy assessment with configurable resolution and boundary constraints.

### Administrative Boundaries

**GADM boundary processor for regional analysis.**

```{eval-rst}
.. autoclass:: RES.boundaries.GADMBoundaries
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render properly due to geospatial dependency issues, the GADMBoundaries class provides:

- `get_country_boundary(country, force_update=False)`: Download complete country boundaries
- `get_regional_boundary(force_update=False)`: Extract specific regional boundary
- `create_bounding_box(geometry, buffer_degrees=0.1)`: Generate spatial extent calculations
```

Downloads and processes administrative boundaries from the Global Administrative Areas (GADM) dataset for spatial analysis scope definition.

### Climate Data Processing  

**Weather data processing and capacity factor calculations.**

```{eval-rst}
.. autoclass:: RES.timeseries.Timeseries
   :show-inheritance:
   :noindex:
```

Integrates with Atlite library for climate data processing and generates technology-specific capacity factor time series from meteorological data.

### Economic Evaluation

**LCOE-based economic scoring and site ranking.**

```{eval-rst}
.. autoclass:: RES.score.CellScorer
   :show-inheritance:
   :noindex:
```

Implements Levelized Cost of Energy calculations following NREL methodology, incorporating capital costs, grid connection expenses, and capacity factors.

## Specialized Modules

### Annual Technology Baseline (ATB)

**Processor for NREL's Annual Technology Baseline data.**

```{eval-rst}
.. autoclass:: RES.atb.NREL_ATBProcessor
   :show-inheritance:
   :noindex:
```

> ℹ️ **Version Notice**: Currently configured for 2024 ATB data. Review and update configuration when using different years or datasets.

### Clustering and Aggregation

**Spatial clustering utilities for site aggregation.**

Key functions from `RES.cluster`:
- `assign_cluster_id()`: Generate unique cell identifiers
- `determine_elbow_optimal_clusters()`: Automatic cluster number optimization
- `cluster_sites()`: K-means clustering with economic weighting
- `get_representative_timeseries()`: Cluster-representative time series generation

### Visualization Tools

**Comprehensive plotting and mapping utilities.**

The `RES.visuals` module provides:
- Spatial mapping with choropleth visualization
- Time series plotting and seasonal analysis  
- Economic analysis charts and distributions
- Interactive web-based dashboards
- Publication-quality figure export

### Utility Functions

**Common helper functions and data operations.**

The `RES.utility` module includes:
- Configuration file parsing and validation
- Data I/O operations (YAML, JSON, geospatial formats)
- Coordinate transformations and spatial utilities
- Hierarchical logging and progress reporting
- URL downloading and caching mechanisms

## Configuration Management

All classes inherit configuration parsing capabilities from `AttributesParser`, enabling:
- YAML-based configuration management
- Parameter validation and default value handling  
- Environment-specific settings (development, production)
- Technology-specific parameter sets

## Data Storage

The framework uses HDF5-based storage through `DataHandler` for:
- Efficient large dataset management
- Automated caching to avoid redundant computations
- Cross-platform compatibility
- Hierarchical data organization

## Examples

### Basic Assessment Workflow

```python
from RES.RESources import RESources_builder

# Initialize assessment
builder = RESources_builder(
    config_file_path="config/config_BC.yaml",
    region_short_code="BC",
    resource_type="wind"
)

# Execute complete workflow
results = builder.build(
    select_top_sites=True,
    use_pypsa_buses=True, 
    memory_resource_limitation=True
)

# Export results
builder.export_results(*results, save_to="output/BC_wind/")
```

### Step-by-Step Analysis

```python
# Manual workflow control
cells = builder.get_grid_cells()
cells_with_capacity = builder.get_cell_capacity()
cells_with_timeseries = builder.get_CF_timeseries(cells_with_capacity)
scored_cells = builder.score_cells(cells_with_timeseries) 
clusters = builder.get_clusters(scored_cells)
```

## Notes

- All spatial data maintained in WGS84 (EPSG:4326) coordinate system
- Time series generated at hourly resolution for full assessment years
- Economic calculations follow NREL LCOE methodology
- Clustering uses k-means with automatic optimization
- Caching mechanisms minimize redundant computation
- Modular design enables workflow customization
   :show-inheritance:
```
```{eval-rst}
.. autoclass:: RES.CellCapacityProcessor.CellCapacityProcessor
   :show-inheritance:
```

```{note}
If the above documentation doesn't render, these classes provide grid cell processing capabilities for spatial analysis.
```

## Global Land Cover
```{eval-rst}
.. autoclass:: RES.gaez.GAEZRasterProcessor
   :show-inheritance:
   :noindex:
```

## Global Wind Atlas 

**Handler for Global Wind Atlas data processing.**

```{eval-rst}
.. autoclass:: RES.gwa.GWACells
   :show-inheritance:
   :noindex:
```
## Visualization

```{eval-rst}
.. automodule:: RES.visuals
   :show-inheritance:
   :noindex:
```

## Scorer

```{eval-rst}
.. autoclass:: RES.score.CellScorer
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render, this class provides cell scoring capabilities for renewable energy site assessment.
```

## Clustering

```{eval-rst}
.. automodule:: RES.cluster
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render properly, this module provides clustering algorithms for renewable energy resource grouping and analysis.
```

## Local Data Store with HDF5 file

```{eval-rst}
.. autoclass:: RES.hdf5_handler.DataHandler
   :show-inheritance:
   :noindex:
```

```{note}
If the above documentation doesn't render, this class provides HDF5-based data storage and retrieval capabilities for the RESource framework.
```

## Turbine Configuration

```{eval-rst}
.. autoclass:: RES.tech.OEDBTurbines
   :show-inheritance:
```

## Units

```{eval-rst}
.. autoclass:: RES.units.Units
   :show-inheritance:
```

---
```{warning}
This page is under heavy development
```

