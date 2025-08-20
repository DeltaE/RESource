# How to use the CONFIG

```{warning}
This library is under heavy development
```

```{hint}
An __interactive user interface is under development__ to replace this config and improve the user experience of the input configuration.
```

## Overview

This document provides comprehensive documentation for the RESource tool configuration file (example provided for the Canadian config file [`config_CAN.yaml`](https://github.com/DeltaE/RESource/blob/main/config/config_CAN.yaml)).

**Version:** 1.0  
**Release Year:** 2025

## Table of Contents

1. [General Information](#general-information)
2. [Regional Mapping](#regional-mapping)
3. [Data Sources](#data-sources)
4. [Capacity Disaggregation](#capacity-disaggregation)
5. [Custom Configurations](#custom-configurations)

---

## General Information

### Basic Configuration

- **Title:** User Configuration for RESource Tool
- **Country:** Canada
- **Purpose:** Configuration file containing quantitative parameters that module results rely on

### Scenario Settings

```yaml
Scenario:
  run_id: default
  Description: Baseline scenario; no additional buffer zones around protected areas or aeroways.
```

```{tip}
A copy of the config file will be saved to 'results/RESources/Region <sub-national-unit name>' for each run of the tool. Each config files will have the 'Scenario' name as a suffix. 
```

---

## Regional Mapping

The configuration includes detailed mapping for all Canadian provinces and territories with the following information for each region:

### Province/Territory Details

| Code | Name | Land Area (km²) | Land Area (mi²) | National % | Timezone | GWA Code |
|------|------|----------------|-----------------|------------|----------|----------|
| AB | Alberta | 642,317 | 275,000 | 7.1% | Etc/GMT-7 | CAN |
| BC | British Columbia | 925,186 | 357,216 | 10.4% | Etc/GMT+7 | CAN |
| MB | Manitoba | 553,556 | 213,733 | 6.1% | Etc/GMT-6 | CAN |
| NB | New Brunswick | 71,450 | 27,587 | 0.8% | Etc/GMT-4 | CAN |
| NL | Newfoundland and Labrador | 373,872 | 144,355 | 4.1% | Etc/GMT-3.5 | CAN |
| NS | Nova Scotia | 53,338 | 20,594 | 0.6% | Etc/GMT-4 | CAN |
| ON | Ontario | 917,741 | 354,348 | 10.1% | Etc/GMT-5 | CAN |
| PE | Prince Edward Island | 5,660 | 2,185 | 0.1% | Etc/GMT-4 | CAN |
| QC | Québec | 1,365,128 | 527,088 | 15.0% | Etc/GMT-5 | CAN |
| SK | Saskatchewan | 591,670 | 228,449 | 6.5% | Etc/GMT-6 | CAN |

### Special Configuration for British Columbia

BC includes additional snapshot timezone configuration:

```yaml
snapshots_tz_BC:
  start: ['2021-01-01 00:00:00']
  end: ['2021-12-31 23:00:00']
```

---

## Data Sources

### GADM (Global Administrative Areas)

- **Root Directory:** `data/downloaded_data/GADM`
- **Processed Directory:** `data/processed_data/regions`
- **Field Mapping:**
  - `NAME_0`: Country
  - `NAME_1`: Province  
  - `NAME_2`: Region

### Government of Canada Data Sources

#### Conservation Lands

- **URL:** [Canadian Protected and Conserved Areas Database](https://data-donnees.az.ec.gc.ca/api/file?path=%2Fspecies%2Fprotectrestore%2Fcanadian-protected-conserved-areas-database%2FDatabases%2FProtectedConservedArea_2023.zip)
- **Root Directory:** `data/downloaded_data/Gov/Conservation_Lands`
- **Data Name:** ProtectedConservedArea
- **GDB Layer:** ProtectedConservedArea_2023

##### IUCN Category Mapping

| Code | Description |
|------|-------------|
| 1 | Strict Nature Reserve |
| 2 | Wilderness Area |
| 3 | National Park |
| 4 | Natural Monument or Feature |
| 5 | Habitat/Species Management Area |
| 6 | Protected Landscape/Seascape |
| 7 | Protected Area with Sustainable Use of Natural Resources |
| 8 | Interim Sites (unknown specifics) |
| 9 | OECM areas |

##### Location Mapping

Canadian provinces and territories are mapped to numeric codes (1-21), including marine areas.

##### Land Ownership Mapping

| Code | Owner Type |
|------|------------|
| 1 | Federal Govt. |
| 2 | Provincial / territorial Govt |
| 3 | Municipal government |
| 4 | Indigenous Community/People |
| 5 | Communal ownership |
| 6 | Individual landowners |
| 7 | For-profit organizations |
| 8 | Non-profit organizations |
| 9 | Joint ownership |
| 10 | Multiple ownership |
| 11 | Contested Ownership |
| 12 | not known or reported |

#### Population Data

- **Root Directory:** `data/downloaded_data/Gov/Population`
- **Data File:** Population_Projections.csv
- **Skip Rows:** 6

#### Community Energy and Emissions Inventory (CEEI)

- **Root Directory:** `data/downloaded_data/Gov/CEEI`
- **Data Files:**
  - Buildings: `bc_utilities_energy_and_emissions_data_at_the_community_level.xlsx`
  - Transportation: `bc_on_road_transportation_data_at_the_community_level.xlsx`
  - Waste: `bc_municipal_solid_waste_data_at_the_community_level.xlsx`

### NREL Annual Technology Baseline (ATB)

- **Root Directory:** `data/downloaded_data/NREL/ATB`
- **Sources:**
  - CSV: [ATBe.csv](https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/csv/2024/ATBe.csv)
  - Parquet: [ATBe.parquet](https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/parquet/2024/v3.0.0/ATBe.parquet)
- **About:** [NREL ATB 2024](https://atb.nrel.gov/electricity/2024/technologies)
- **Cost Parameters:**
  - CAPEX: OCC
  - Fixed O&M: Fixed O&M
  - Variable O&M: None

### OpenStreetMap (OSM) Data

- **Root Directory:** `data/downloaded_data/OSM`
- **Key Categories:**
  - **Aeroway:** aerodrome, runway, taxiway, helipad, apron, gate
  - **Power:** line, cable, minor_line, substation, tower, pole, generator, plant, terminal
  - **Substation:** transmission, distribution, minor_distribution, industrial

### CODERS (Canadian Open Data for Electricity Research)

- **URLs:**
  - Primary: <https://sesit.dev/>
  - Secondary: <http://206.12.95.102/>
- **API Documentation:** <https://sesit.dev/api/docs>
- **Data Types:** network (substations and transmission lines)

### GAEZ (Global Agro-Ecological Zones)

- **Root Directory:** `data/downloaded_data/GAEZ`
- **Source:** [LR.zip](https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR.zip)

#### Raster Types

##### 1. Exclusion Areas (`exclusion_2017.tif`)

- **Color Map:** OrRd
- **Title:** Excluding Global Exclusion Areas
- **Class Exclusions:**
  - Solar: Classes 2-7
  - Wind: Classes 2-7

##### 2. Terrain Resources (`slpmed05.tif`)

- **Color Map:** terrain
- **Title:** Excluding Terrain Slope
- **Class Exclusions:**
  - Solar: Class 9
  - Wind: Classes 7-9

##### 3. Land Cover (`faocmb_2010.tif`)

- **Color Map:** YlGn
- **Title:** Excluding not-Suitable Landcovers
- **Class Inclusions:**
  - Solar: Classes 2, 3, 5, 8, 9
  - Wind: Classes 2, 3, 5, 8, 9

### WorldPop

- **Root Directory:** `data/downloaded_data/WorldPop`
- **Data Sources:**
  - Population Density (Canada)
  - Population Count (Canada)
  - Weighted Population Density (Global)

### Climate Data (Cutout)

- **Root Directory:** `data/downloaded_data/cutout/`
- **Module:** era5
- **Resolution:** 0.25° x 0.25°
- **Time Period:** 2023-01-01 07:00:00 to 2024-01-01 06:00:00

### Global Wind Atlas (GWA)

- **Root Directory:** `data/downloaded_data/GWA`
- **Data Fields:**
  - Wind Speed at 100m
  - Capacity Factor IEC Class 2
  - Capacity Factor IEC Class 3
  - IEC Class Extreme Loads
- **Filters:**
  - Low Wind Speed: 7 m/s
  - High Wind Speed: 45 m/s

---

## Capacity Disaggregation

### Solar Configuration

- **Max Capacity:** 5 MW
- **Land Use Intensity:** 1.45 MW/km²
- **Cost Data:** `data/processed_data/solar/utility_PV_Class5_cost_moderate_NREL_ATB2024.csv`
- **Technology:** UtilityPV Class5
- **Panel Type:** CSi (Crystalline Silicon)
- **Tracking:** Dual-axis
- **Tolerances:**
  - Static CF: 0.16
  - Capacity: 1
  - WCSS: 0.01

#### Buffer Zones (Solar)

All aeroway and conserved land features have 0m buffer zones in the baseline scenario.

### Wind Configuration

- **Max Capacity:** 15 MW
- **Land Use Intensity:** 3 MW/km²
- **Cost Data:** `data/processed_data/wind/land_based_wind_T3_cost_moderate_NREL_ATB2024.csv`
- **Technology:** Land-based Wind Turbine Technology 3
- **Capacity Factor Range:** 0.2 - 1.0
- **WCSS Tolerance:** 0.01

#### Turbine Models

##### NREL ATB Turbines

1. **Enercon E82 3000kW** - 3 MW
2. **Vestas V90 3MW** - 3 MW

##### OEDB Turbines

1. **GE2.75_120** (ID: 116) - 2.75 MW, GE Wind
2. **3.2M114_NES** (ID: 93) - 3.2 MW, Senvion/REpower

#### Buffer Zones (Wind)

All aeroway and conserved land features have 0m buffer zones in the baseline scenario.

### Battery Energy Storage System (BESS)

- **Max Capacity:** 10 MW
- **Cost Data:** `data/processed_data/bess/bess_LI_6hr_cost_moderate_NREL_ATB2024.csv`
- **Type:** 6Hr Battery Storage (Lithium-ion)
- **Variable O&M:** 0
- **Unit Size:** 60 MWh
- **Storage Duration:** 5.5 hours

#### Capacity Estimates per Energy Unit

- **Conservative:** 0.01
- **Moderate:** 0.05
- **Optimistic:** 0.1

### Transmission

- **Grid Connection Cost:** $1.616M per km
- **Transmission Line Rebuild Cost:** $0.348M per km
- **Grid Proximity Limit:** 100 km
- **Bus Data:** `data/processed_data/buses.csv`
- **Line Data:** `data/processed_data/lines.csv`

---

## Custom Configurations

### Custom Land Layers

The configuration supports custom raster and vector layers for specialized land use constraints:

```yaml
custom_land_layers:
  rasters:
    raster_1:
      raster: {}
      class_exclusion: {}
      buffer: {}
      invert: 'False'
  vectors:
    vector_1:
      geometry: {}
      buffer: null
      invert: 'False'
```

---

## Usage Notes

1. **Buffer Zones:** In the baseline scenario, all buffer zones are set to 0m around protected areas and aeroways.

2. **Technology Selection:** The configuration uses moderate cost scenarios from NREL ATB 2024.

3. **Data Sources:** All data sources include proper attribution and licensing information.

4. **Regional Customization:** Each province/territory can be configured with specific parameters through the sub_region_mapping field.

5. **Time Series:** Climate data and capacity factor calculations use 2023 data with hourly resolution.

6. **Quality Control:** Multiple tolerance parameters ensure data quality and consistency across different analysis modules.

---

## File Locations

- **Configuration File:** `config/config_CAN.yaml`
- **Data Directory:** `data/`
- **Results Directory:** `results/`
- **Documentation:** `docs/`

For more information about the RESource tool, contact the developer or refer to the project documentation.
