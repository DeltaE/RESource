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