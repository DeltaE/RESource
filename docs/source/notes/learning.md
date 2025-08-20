# Best Practices and Guidelines

## Selecting Wind Turbines

### IEC Class Selection
As recommended in *[The Global Atlas for Siting Parameters (GASP) project: Extreme wind,turbulence, and turbine classes](https://orbit.dtu.dk/en/publications/the-global-atlas-for-siting-parameters-project-extreme-wind-turbu)*, the parameter values applied at hub height 100m for different IEC classes are listed below. Suitability for a wind turbine depends on crucial factors like mean wind speed, turbulence, extreme wind speed, and air density in high winds. It is important to select the site-specific appropriate turbine class.
| Wind Turbine Class | I    | II   | III  |
|--------------------|------|------|------|
| **V<sub>ave</sub> (m/s)**     | 10.0 | 8.5  | 7.5  |
| **V<sub>ref</sub> (m/s)**     | 50.0 | 42.5 | 37.5 |
| **V<sub>ref</sub>,T (m/s)**   | 57.0 | 57.0 | 57.0 |
| **A+ <sub>ref</sub>**         | 0.18 |      |      |
| **A I<sub>ref</sub>**         | 0.16 |      |      |
| **B I<sub>ref</sub>**         | 0.14 |      |      |
| **C I<sub>ref</sub>**         | 0.12 |      |      |

>**Table source:** Table 1 from [The Global Atlas for Siting Parameters (GASP) project: Extreme wind, turbulence, and turbine classes](https://onlinelibrary.wiley.com/doi/epdf/10.1002/we.2771)

> __Note__: V<sub>ave</sub> is the annual average wind speed; V<sub>ref</sub> is the reference wind speed average over 10 min; V<sub>ref</sub>,T is the reference wind speed average over 10 min applicable for areas subject to tropical cyclones. A+ designates the category for remarkably high turbulence characteristics; A for higher turbulence characteristics; B for medium turbulence characteristics; C for lower turbulence characteristics and I<sub>ref</sub> is a reference value of the turbulence intensity.

#### IEC Class Layer Mapping to Select Turbine Technologies for Wind Energy Estimation


| Layer Name                      | Wind Turbine Class Parameters                                    |
|----------------------------------|------------------------------------------------------------------|
| **IEC Class - Fatigue Loads**    | Mean wind speed (Class I, II, III, S); <br> Turbulence category (A+, A, B, C); <br>*Excludes wake effects* |
| **IEC Class - Fatigue Loads incl. Wake** | Mean wind speed (Class I, II, III, S);<br> Turbulence category (A+, A, B, C); <br>*Includes wake effects* |
| **IEC Class - Extreme Loads**    | Extreme wind speed; <br>Air density at high wind speed (Class I, II, III, T, S) |
> Source: [IEC Classes at GWA](https://globalwindatlas.info/en/about/dataset) 

This involves prioritizing the highest wind turbine class from the GWA's IEC Class layers prior to converting the wind resource potential to energy yield parameters. GWA includes three different IEC Class layers (under the Wind Energy Layers), mapping IEC wind turbine classes for 100m wind turbine hub height. Any wind turbine, regardless of its class, usually needs validation with the manufacturer specific to the site. GWA provides rasterized data to map wind turbine class recommendations for different load scenarios. For resource estimation studies, it is typically essential to evaluate the highest wind turbine class from the IEC Class layers in the GWA.

#### Best Practices

1. **Layer Selection:** Choose the most relevant GWA IEC Class layer (Fatigue Loads, Fatigue Loads incl. Wake, or Extreme Loads) based on your scenario/project’s objectives and local wind conditions.
   - **Consider Wake Effects:** For wind farm layouts, include wake effects in fatigue load assessments to ensure accurate turbine class selection.
   - **Extreme Events:** Factor in extreme wind events and air density when evaluating turbine class for long-term reliability.
2. From the layer, **find the most appropriate IEC Class that represents most of your area of interest**.
3. **Documentation:** Reference authoritative sources (e.g., GWA, IEC standards) and document all assumptions and data sources used in the selection process.
##### Example

For British Columbia (BC), the map on the right displays IEC turbine classes for the _IEC Class - Extreme Loads_ layer. The analysis shows that **IEC Class III turbines** are most representative for this region under extreme load conditions. Areas marked in red indicates _IEC Class II_, likely due to significant terrain transitions such as mountainous slopes.

<img src="../_static/Ruggedness_IEC_classmap_BC_GWA.jpg" alt="Ruggedness_IEC_class_map_BC_GWA" width="900"/>

> The _Ruggedness Index_ (as shown in in left map), also known as the _Terrain Ruggedness Index (TRI)_, quantitatively measures terrain heterogeneity by assessing elevation changes across the landscape. This index helps explain IEC Class variations in specific areas and supports informed decisions regarding IEC Class selection.

> **Data Source:** Global Wind Atlas (GWA) <sub>v3.4</sub>

## Selecting Solar PV Panels

We use the *[atlite.pv](https://atlite.readthedocs.io/en/master/ref_api.html#atlite.convert.pv)* conversion functionality to translate surface solar irradiance (direct + diffuse) and ambient temperature into photovoltaic (PV) power output. Internally, this module relies on a detailed panel model that incorporates parameters such as panel efficiency and temperature-dependent performance losses. 

- Users can specify panel orientation (e.g., fixed tilt or azimuth) or tracking configurations (e.g., single-axis tracking). The “optimal” tilt—commonly based on latitude—or active tracking improves alignment with the sun, enhancing overall energy yield. 
    > For Optimal slope of the panels, atlite uses the formula documented in [solarpaneltilt: Optimum Tilt of Solar Panels](http://www.solarpaneltilt.com/#fixed)

- Currently, panel configuration options are limited to crystalline silicon (c-Si), cadmium telluride (CdTe), and Kaneka (amorphous silicon) technologies. The configurations and assumptions underlying the pv panel models can be found at [atlite/resources/solarpanel](https://github.com/PyPSA/atlite/tree/master/atlite/resources/solarpanel)

### Panel Attribute Configurations:
- Higher-efficiency panels and improved orientation directly increase generation under identical irradiance conditions. If you do not want to explicitly define panel orientation, use __tracking: 'dual'__ and we have defaulted the ['orientation': "latitude_optimal"](https://github.com/DeltaE/RESource/blob/1d9c0672bd924d2b7aae6b571709aee2eb1dd6f9/RES/timeseries.py#L253) at [_timeseries_](https://github.com/DeltaE/RESource/blob/main/RES/timeseries.py) module.
    > Examples regarding custom orientation configuration is provided here: [atlite PV examples](https://atlite.readthedocs.io/en/master/examples/historic-comparison-germany.html)
- We configure the pv panels' attributes at ['capacity_disaggregation/solar'](https://github.com/DeltaE/RESource/blob/1d9c0672bd924d2b7aae6b571709aee2eb1dd6f9/config/config_CAN.yaml#L371) key of the config file.

## Land Availability Calculation from Vector vs Raster Data

Vector and raster data are two fundamental ways of representing spatial information on computers. They each have their strengths and weaknesses, so the best choice depends on what you're trying to achieve.

**Vector data** is like a map made with geometric shapes. It uses points, lines, and polygons (areas) defined by mathematical coordinates to represent features. Imagine a map of a city with parks drawn as green polygons, streets as lines, and important buildings as points. This allows for sharp, clean lines and makes it easy to scale the map without losing quality. Vector data is also efficient for storing information about the features, like names, descriptions, or even photos.

**Raster data**, on the other hand, is like a photograph of the real world. It breaks down space into a grid of tiny squares, like pixels in a digital image. Each square holds a value that represents what's there, such as a color or an elevation level. Satellite imagery and scanned maps are common examples of raster data. Raster data excels at capturing continuous variation and is often simpler to process for certain analyses. However, it can become bulky for large areas and lose detail when zoomed in.

Here's a table summarizing the key differences:

| Feature                 | Vector Data                                 | Raster Data                                   |
|--------------------------|----------------------------------------------|-------------------------------------------------|
| Representation           | Points, lines, and polygons                   | Grid of squares (pixels)                         |
| Detail at high zoom      | Crisp and clear                               | Can appear blocky or pixelated                   |
| Scalability              | Excellent, maintains quality when zoomed      | Loses detail when zoomed in                     |
| File size                | Smaller for similar detail                    | Larger for continuous variation                  |
| Feature information       | Can store additional data about features      | Limited to data represented by pixel values     |
| Common uses              | Maps, logos, illustrations                   | Satellite imagery, photographs, elevation data   |

Ultimately, the choice between vector and raster data depends on analysis specific needs. If you need precise shapes and sharp lines, vector data is the way to go. But if you're working with continuous data or imagery, raster data might be a better fit.


## Open Street Map (OSM) Data
### Goal:
* To create __vector data__ with targeted __landuse__.
e.g. we have used 'aeroway' vector data in this analysis.
 > [What is 'aeroway'?](https://wiki.openstreetmap.org/wiki/Aeroways)

### Usage in RESource (Linking Tool): 
* 1. We can filter the type of aeroway landuse that we want to disregard as a potential site.
* 2. We will create a union geometry of all aeroway area, and later can add buffer area around surrounding this geometry. The Buffer radius can be configured via the user configuration file.
* 3. We will exclude this final geometry [aeroway union+buffer] from our Cutout Grid Cells during land availability calculations for potential VRE sites.
 
### Tool :
We used [pyrosm](https://pyrosm.readthedocs.io/en/latest/) to extract OSM data via python API.
 > [why pyrosm?](https://pyrosm.readthedocs.io/en/latest/#when-should-i-use-pyrosm)

### Method: 
* We created an OSM 'object' which has various attributes. One of the attributes is 'point of interests (_get_pois_)'. 
* Each attribute has several 'keys'. We used '_get_pois_' method to extract one of the available 'keys' (e.g. 'aeroway')

* Each OSM key has several tags associated e.g.
```console
from pyrosm.config import Conf
print("All available OSM keys", Conf.tags.available)
print("\n")
print("Typical tags associated with Aeroway:", Conf.tags.aeroway)
```

```
All available OSM keys ['aerialway', 'aeroway', 'amenity', 'boundary', 'building', 'craft', 'emergency', 'geological', 'highway', 'historic', 'landuse', 'leisure', 'natural', 'office', 'power', 'public_transport', 'railway', 'route', 'place', 'shop', 'tourism', 'waterway']

Typical tags associated with Aeroway: ['aerodrome', 'aeroway', 'apron', 'control_tower', 'control_center', 'gate', 'hangar', 'helipad', 'heliport', 'navigationaid', 'beacon', 'runway', 'taxilane', 'taxiway', 'terminal', 'windsock', 'highway_strip']

```
* We used custom filters to extract data for our target key 'aeroway'
 > [How to read and visualize Point of Interests?](https://pyrosm.readthedocs.io/en/latest/basics.html#read-points-of-interest)
 > [How to custom filter OSM data](https://pyrosm.readthedocs.io/en/latest/basics.html#read-osm-data-with-custom-filter)