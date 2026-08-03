# RESource Modeling Guide: Common Errors When Setting Up New Regions

This document outlines common errors encountered when setting up RESource for a new region (Bangladesh example) and how to resolve them at the module level.

## Overview
RESource successfully processed **solar resources** for Bangladesh but encountered issues with **wind resources**. The main errors are related to data source compatibility and regional naming conventions.

## Critical Errors and Solutions

### 1. **Unicode Character Encoding Error** ⚠️
**Error:**
```
SyntaxError: Non-ASCII character '\xe2' in file run.py on line 148, but no encoding declared
```

**Root Cause:** Unicode symbols (✓, ✗, 💡, etc.) in `run.py` without proper encoding declaration.

**Solution (Module Level):**
- Add UTF-8 encoding declaration at the top of `run.py`:
```python
#!/usr/bin/env python3
# coding: utf-8
```
- OR replace Unicode characters with ASCII equivalents in the codebase.

---

### 2. **GADM Region Name Mismatch** ❌
**Error:**
```
RESource.boundaries | @ LINE 271 | No data found for region 'Barisal Division'.
```

**Root Cause:** The region names in the configuration file don't match the actual GADM administrative boundary names.

**Investigation Steps:**
1. Check downloaded GADM data structure:
```python
import geopandas as gpd
gdf = gpd.read_file('data/downloaded_data/GADM/gadm41_Bangladesh_L1.geojson')
print('Available region names:', sorted(gdf['NAME_1'].tolist()))
```

**Solution (Configuration Level):**
- Update config file region names to match GADM exactly:
```yaml
# WRONG:
'BD-01':
  name: "Barisal Division"

# CORRECT:
'BD-01':
  name: "Barisal"  # Matches GADM NAME_1 field
```

---

### 3. **Global Wind Atlas (GWA) File System Error** ❌
**Error:**
```
Failed processing BD-03 wind: [Errno 21] Is a directory: 'data/downloaded_data/GWA/BGD_wspd_100m.tif'
```

**Root Cause:** The GWA module expects a file but encounters a directory with the same name.

**Investigation Steps:**
```bash
ls -la data/downloaded_data/GWA/
# Check if BGD_wspd_100m.tif is a file or directory
```

**Potential Solutions (Module Level):**
1. **Check file handling in `gwa.py`:** The download/save logic may be creating directories instead of files
2. **Clear corrupted downloads:** Remove the problematic directory and re-download
3. **Verify GWA API response:** Ensure the API returns valid raster data for Bangladesh

---

### 4. **Conservation Lands Warning** ⚠️
**Warning:**
```
Conservation Lands data supply chain is configured for Canada only
RESource.lands| 'conserved_lands_CAN' not initiated
```

**Root Cause:** The conservation lands module is hardcoded for Canadian data sources.

**Solution (Module Level):**
- Make conservation lands module region-agnostic or add support for international protected areas databases
- Consider integrating with WDPA (World Database on Protected Areas) for global coverage

---

## Successful Components ✅

### Working Features:
1. **GADM Boundary Processing** - Successfully downloaded and processed Bangladesh administrative boundaries
2. **ERA5 Climate Data** - Successfully retrieved meteorological data from Copernicus Climate Data Store
3. **OSM Data Integration** - Downloaded aeroway and power infrastructure data
4. **GAEZ Land Suitability** - Processed global agricultural and ecological zone data
5. **NREL ATB Cost Data** - Retrieved and processed technology cost data
6. **Solar Resource Analysis** - Complete pipeline from data to clustered results
7. **Land Availability Assessment** - Successfully processed exclusion layers
8. **Capacity and LCOE Calculations** - Generated economic assessments
9. **Clustering and Site Selection** - Created representative resource profiles

### Generated Outputs:
- HDF5 data store: `data/store/resources_Bangladesh_BD-03_baseline.h5`
- Results CSV: `results/Bangladesh/BD-03/baseline/resource_options_solar_Dhaka_timeseries.csv`
- Visualization plots: `vis/Bangladesh/BD-03/baseline/solar/`

---

## Configuration Recommendations for New Regions

### 1. **Regional Naming Convention**
Always verify GADM administrative boundary names before configuration:
```python
# Verification script
import geopandas as gpd
country_code = "BGD"  # ISO 3166-1 alpha-3 code
gdf = gpd.read_file(f'data/downloaded_data/GADM/gadm41_{country_name}_L1.geojson')
print("Available regions:", sorted(gdf['NAME_1'].tolist()))
```

### 2. **Coordinate Reference System (CRS)**
Select appropriate UTM zone for the region:
- Bangladesh: `EPSG:32646` (UTM Zone 46N)
- Verify with: https://epsg.io/

### 3. **Time Zone Configuration**
Use proper timezone offset:
- Bangladesh Standard Time: `'Etc/GMT-6'` (UTC+6)

### 4. **Resource Capacity Limits**
Set realistic capacity targets based on country size and renewable potential:
```yaml
capacity_disaggregation:
  solar:
    max_capacity: 5  # GW (conservative for Bangladesh)
  wind:
    max_capacity: 3  # GW (limited onshore potential)
```

---

## Module-Level Improvements Needed

### High Priority:
1. **Fix GWA file handling in `gwa.py`**
2. **Make conservation lands module region-agnostic**
3. **Improve error handling for missing/corrupted downloads**

### Medium Priority:
1. **Add automatic GADM name validation**
2. **Enhance CRS detection based on country bounds**
3. **Add support for offshore wind resources**

### Low Priority:
1. **Replace Unicode characters with ASCII in user messages**
2. **Add configuration validation warnings**
3. **Implement automatic optimal parameter detection**

---

## Testing Recommendations

When setting up a new region:

1. **Start with one region:** Test with a single administrative division first
2. **Test solar before wind:** Solar pipeline is more robust
3. **Verify data downloads:** Check all external data sources are accessible
4. **Monitor disk space:** ERA5 cutouts can be large (several GB)
5. **Check memory usage:** Large regions may require more RAM

---

## Summary

RESource successfully processes **solar resources** for new regions with minimal configuration changes. The main bottleneck is **wind resource processing** due to the Global Wind Atlas integration issue. Users can successfully model solar potential for any region worldwide by following the region naming and CRS configuration guidelines above.
