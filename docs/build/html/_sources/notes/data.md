

The data supply chain is  created mostly from global data sources, local data sources (local gov./authorities) have been used in absence of suitable global data sources. A few of the data sources do not have API/static URL and/or requires form inputs from user. This guideline will help the user to have overall idea about the data, sources and download process. The use-case denotes the usage exclusively in this tool and analysis.

# CODERS
> for CANADIAN studies.

- Create `coders_api.yaml` config file
> structure for `coders_api.yaml` below:

```yaml
Default_user: <your_username> or <other username>)>
api_keys:
  <your_username>: <your_api_key>
  <other_user1>: <other_api_key1>
  <other_user2>: <other_api_key2>
  .....
  <other_userN>: <other_api_keyN>
```

- Save it at directory: `data/downloaded_data/CODERS`



# 1. Demographics
## 1.1 - Population
- <U>Tag</U>: Local
- <U>Authority</U>: Statistics Canada
- <U>License</U> : Data obtained through this application is distributed under the [Canadian Open Government License](https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc). 
    - In-short :  worldwide, royalty-free, perpetual, non-exclusive licence to Copy, modify, publish, translate, adapt, distribute or otherwise use the Information in any medium, mode or format for any lawful purpose
- <U>Data</U>: [Population projection 2021-2046](https://bcstats.shinyapps.io/popApp)
    - <U>Resolution</U>: Annual population for regional districts (sub-provincial).
- <U>Description</U>: Historical data up to 2023 and projection for 2024-2046.
- <U>Use-case</U> : To mimic the load-centers in Canada at sub-provincial level (regional districts of province) 
- <U>Supply_chain_mode</U> : Manual Download from the portal
    - Instruction: Manually download from the portal with mentioned steps given in [data_sources.yml](https://github.com/DeltaE/Linking_tool/blob/main/config/data_source.yml)

# 2. Climate and Weather Data
## 2.1  Cutout from ERA5
- <U>Tag</U>: Global
- <U>Authority</U>: Copernicus Climate Change Service (C3S), ECMWF, EU.
- <U>License</U> : free of charge, worldwide, non-exclusive, royalty free and perpetual. 
    - Caution: have to mention the attribution regarding C3S. 
    - [Check Article 4,5 of the license agreement](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf)
- <U>Data</U>: [Complete ERA5 global atmospheric reanalysis](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview)
- <U>Description</U>: Solar influx, wind speed (vertical components at 100m), land elevation (heights) time-series data for weather years.
    - <U>Resolution</U>: hourly time-series for .25 arc degree (~ 30km) grids. 
- <U>Use-case</U> :
    - A cutout is one of the basis for this work and associated calculations. 
    - We are using [atlite](https://atlite.readthedocs.io/en/master/index.html) to create the cutout and also to download the [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5#:~:text=ERA5%20is%20the%20fifth%20generation,Service%20(C3S)%20at%20ECMWF.) data for the cutout. The cutout will be saved as a NetCDF (__.nc__) file. NetCDF is a file format often used for storing large scientific data sets that often involves time-series data, especially in the fields of climate and weather research. Please check this resource for [more about cutout preparation and customization](https://atlite.readthedocs.io/en/latest/examples/create_cutout.html).
    - In this analysis, we are downloading ERA5 data on-demand for a specified region e.g. __BC region cutout__ . But [atlite](https://atlite.readthedocs.io/en/latest/examples/create_cutout.html) does also work with other data sources e.g. [SARAH-2](https://atlite.readthedocs.io/en/latest/examples/create_cutout_SARAH.html) for high resolution solar dataset.
    - NREL has higher spatio-temporal dataset for renewable resources but does not cover complete global regions. Atlite currently does not support  NREL's [NSDRB for solar](https://nsrdb.nrel.gov) or [WRDB for wind](https://wrdb.nrel.gov/). Users can [follow this thread for updates](https://github.com/PyPSA/atlite/issues/213).
    - Atlite does not support ERA5 forecast data yet. Users can [follow this thread for updates](https://github.com/PyPSA/atlite/issues/184)
    
    Please go through [this documentation](https://atlite.readthedocs.io/en/master/examples/create_cutout.html) and example usage of cutout to learn further.

- <U>Supply_chain_mode</U> : Automated via cdsapi (current version is [cds-beta](https://cds-beta.climate.copernicus.eu/))

    ><U>Note</U>: From Sep 26, 2024 onwards the ERA5 dataset will only be supplied via cds-beta or ads-beta ([source](https://confluence.ecmwf.int/display/CKB/Please+read%3A+CDS+and+ADS+migrating+to+new+infrastructure%3A+Common+Data+Store+%28CDS%29+Engine))

    - Before the data can be downloaded from ERA5, it has to be processed by CDS servers, this might take a while depending on the volume of data requested. This only works if you have in before

        - For linux users, please proceed as follows:

        - Steps to install the Copernicus Climate Data Store cdsapi package at your __local Linux/WSL__ (sourced from > [Registered and setup your CDS API key as described](https://cds-beta.climate.copernicus.eu/how-to-api))
        > step1: Setup the CDS API personal access token <br>
        > step2: Install the CDS API client. <br>
        >> Note: atlite currently supports cdsapi <=0.7.2
        
        Now your datapipeline to create the ERA5 Cutout is set.

# 2. Geospatial Raster/Vectors
## 2.1 Boundaries from GADM
- <U>Tag</U>: Global
    - This data could be sourced locally as well e.g for Canada from [Canadian open-dataset](https://open.canada.ca/data/en/dataset/306e5004-534b-4110-9feb-58e3a5c3fd97)
    - Other global data sources :
        - OpenstreetMap via [pyrosm](https://pyrosm.readthedocs.io/en/latest/basics.html#read-boundaries) library.
        - World Administrative Boundaries - Countries and Territories by opendatasoft (https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export)
       
- <U>License</U> : [freely available for academic use and other non-commercial use](https://gadm.org/license.html)
- <U>Authority</U>:  University of Berkeley, Museum of Vertebrate Zoology and the International Rice Research Institute (2012) 
- <U>Data</U>: [Download GADM data (v4.1 | 16 July 2022 )](https://gadm.org/download_country.html)
- <U>Description</U>: [GADM](https://gadm.org/), the Database of Global Administrative Areas, is a high-resolution database of country administrative areas, with a goal of "all countries, at all levels, at any time period.
- <U>Use-case</U> : This boundary has been processed for admin level 2 (i.e. sub-provincial) to extract geospatial boundaries of the Regional Districts (RD) e.g. 28 RDs inside BC, Canada. This boundary is primarily used for spatial-grid cell/point mapping, regional overlay visuals, clipping point of interests in regional level while clustering.
- <U>Supply_chain_mode</U> : Automated via [pygadm](https://pypi.org/project/pygadm) library [supports GADM data V4.1]

## 2.2 Conservation and Protected Lands
- <U>Tag</U>: Local
    - GAEZ also has similar global data under Land Resources (LR) theme, raster data with 7 classes. We are using this data as a mandatory filter in the process. But the local (pan-Canadian) data has more detailed local government and indigenous protected areas' data. The user can control the classes of exclusion and also can use buffer around exclusion for both case.
- <U>License</U> : Data obtained through this application is distributed under the [Canadian Open Government License](https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc). 
    - In-short :  worldwide, royalty-free, perpetual, non-exclusive licence to Copy, modify, publish, translate, adapt, distribute or otherwise use the Information in any medium, mode or format for any lawful purpose
- <U>Authority</U>: Environment and Climate Change Canada (ECCC)
- <U>Data</U>: [Canadian Protected and Conserved Areas Database (CPCAD) | 2023-12-31](https://catalogue.ec.gc.ca/geonetwork/oilsands/api/records/6c343726-1e92-451a-876a-76e17d398a1c)
    - <U>downloadble</U>_source_url: https://data-donnees.az.ec.gc.ca/api/file?path=%2Fspecies%2Fprotectrestore%2Fcanadian-protected-conserved-areas-database%2FDatabases%2FProtectedConservedArea_2022.gdb.zip
    - <U>Resolution</U>: Spatial boundaries vector data
- <U>Description</U>: CPCAD is the authoritative source of data on protected and conserved areas in Canada. The database consists of the most up-to-date spatial and attribute data on marine and terrestrial protected areas in all governance categories recognized by the International Union for Conservation of Nature (IUCN), as well as other effective area-based conservation measures (OECMs, or conserved areas) across the country. Indigenous Protected and Conserved Areas (IPCAs) are also included if they are recognized as protected or conserved areas. CPCAD adheres to national reporting standards and is available to the public.
- <U>Use-case</U> : These specific areas (raster cells/vectors) are excluded in analysis for site considerations. The modeller can also consider buffer around exclusion areas.
- <U>Supply_chain_mode</U> : Automated via specific url download. Has dependency on [source_url](https://data-donnees.az.ec.gc.ca/api/file?path=%2Fspecies%2Fprotectrestore%2Fcanadian-protected-conserved-areas-database%2FDatabases%2FProtectedConservedArea_2022.gdb.zip).

# Energy and Emission (exogenous)
  ## Community Energy and Emissions Inventory(CEEI)
- <U>Tag</U>: Local
- <U>License</U> : Data obtained through this application is distributed under the [Canadian Open Government License](https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc).
- <U>Authority</U>:  [Community Energy and Emissions Inventory(CEEI)]https://www2.gov.bc.ca/gov/content/environment/climate-change/data/ceei
- <U>Data</U>: [CEEI data up to 2021](https://www2.gov.bc.ca/gov/content/environment/climate-change/data/ceei/current-data) 
    - <U>Resolution</U>: Annual total for Regional Districts, for different sectors and different end-use demands.
- <U>Description</U>: The Community Energy and Emissions Inventory (CEEI) provides community-level greenhouse gas (GHG) emissions and energy consumption estimates for communities across BC. The data covers the buildings, municipal solid waste, and on-road transportation sectors for 161 municipalities, 28 regional districts, and 1 region (Stikine).
    - Buildings :The data is provided by utility companies and includes the amount of electricity and natural gas used by residential, commercial and some industrial buildings.
    - Transportation : Community-level data on greenhouse gas emissions from on-road transportation.
    - Waste : Estimates of community greenhouse gas emissions based on historic annual tonnes of waste disposed at regional district landfills.
    > More about [data methods](https://www2.gov.bc.ca/gov/content/environment/climate-change/data/ceei/methodology) and [inputs](https://www2.gov.bc.ca/gov/content/environment/climate-change/data/ceei/current-data)
- <U>Use-case</U> : Used for load-center estimations on regional district level. Further used for Battery Energy Storage (BESS) size and required discharge hour estimation.
- <U>Supply_chain_mode</U> : Automated via specific url download. Check config file for specific url dependencies.

---
# Information Template
- <U>Tag</U>: 
- <U>License</U> : 
- <U>Authority</U>: 
- <U>Data</U>: [title](Url)
    - <U>Resolution</U>:
- <U>Description</U>:
- <U>Use-case</U> :
- <U>Supply_chain_mode</U> : 
    - Instruction: 
