import geopandas as gpd
from scipy.spatial import cKDTree
import pandas as pd
import logging as log
from dataclasses import dataclass

from RES.AttributesParser import AttributesParser
from RES.osm import OSMData

@dataclass
class GridNodeLocator(AttributesParser):
    
    def __post_init__(self):
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        self.grid_node_proximity_filter = self.disaggregation_config['transmission']['proximity_filter']

    def __find_nearest_station__(self, cell_geometry, buses_gdf, bus_tree):
        """
        Find the nearest grid station for a given geometry.

        Parameters:
            cell_geometry (shapely.geometry): The geometry of the cell (e.g., a polygon or point).
            buses_gdf (GeoDataFrame): GeoDataFrame containing bus stations with geometry and attributes.
            bus_tree (scipy.spatial.KDTree): A spatial index of the bus station geometries.

        Returns:
            tuple: (nearest_station_code, distance_km) where
                nearest_station_code is the name or node code of the nearest station.
                distance_km is the distance to the nearest station in kilometers.
        """
        # Query the KDTree with the centroid of the cell geometry
        _, index = bus_tree.query((cell_geometry.centroid.x, cell_geometry.centroid.y))

        # Retrieve the nearest bus row
        nearest_bus_row = buses_gdf.iloc[index]

        # Compute the distance (convert degrees to kilometers using approximate conversion factor)
        distance_km = cell_geometry.centroid.distance(nearest_bus_row['geometry']) * 111.32

        # Determine the station code based on available columns
        if 'name' in buses_gdf.columns:
            nearest_station_code = nearest_bus_row['name']
        else:
            nearest_station_code = nearest_bus_row['node_code']

        return nearest_station_code, distance_km


    def find_grid_nodes_ERA5_cells(
        self, 
        buses_gdf: gpd.GeoDataFrame, 
        cells_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Find the nearest grid nodes and calculate distances for ERA5 cells.
        Filters cells based on proximity to the nearest node.
        """
        buses_gdf.sindex  # Generate spatial index
        bus_tree = cKDTree(buses_gdf['geometry'].apply(lambda x: (x.x, x.y)).tolist())
        
        log.info(f"> Calculating Nearest Grid Nodes for Grid Cells of {self.region_short_code}")
        

        # Apply the find_nearest_station method using lambda to pass additional arguments
        result = cells_gdf['geometry'].apply(
            lambda geom: self.__find_nearest_station__(geom, buses_gdf=buses_gdf, bus_tree=bus_tree)
        )

        # Unpack the result into two columns
        cells_gdf[['nearest_station', 'nearest_station_distance_km']] = pd.DataFrame(result.tolist(), index=cells_gdf.index)

        # Filter cells based on proximity to grid nodes
        cells_gdf_with_station_data = cells_gdf.copy()
        proximity_to_nodes_mask = cells_gdf_with_station_data['nearest_station_distance_km'] <= self.grid_node_proximity_filter
        cells_within_proximity_gdf = cells_gdf_with_station_data[proximity_to_nodes_mask]

        log.info(f"ERA5 Cells Filtered based on Proximity to Tx Nodes \n"
                f"Size: {len(cells_within_proximity_gdf)}\n")
        
        return cells_gdf_with_station_data # cells_within_proximity_gdf
    
    def get_OSM_grid_nodes(self) -> gpd.GeoDataFrame:
        """
        Retrieve OSM data for grid nodes.
        """
        osm_data = OSMData(region_code=self.region_short_code)
        
        # Extract the relevant GeoDataFrame from OSM data
        buses_gdf = osm_data.get_osm_layer('substation')
        if buses_gdf is None:
            log.error("No OSM data found for grid nodes.")
            return None
        else:
            return buses_gdf


# Example of a specialized class using inheritance
# class AdvancedGridNodeLocator(GridNodeLocator):
#     def __init__(self, region_code: str, grid_node_proximity_filter: float, extra_param: str):
#         super().__init__(region_code, grid_node_proximity_filter)
#         self.extra_param = extra_param
    
#     def some_advanced_method(self):
#         # Extend with more advanced functionalities here
#         pass