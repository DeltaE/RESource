import geopandas as gpd
from scipy.spatial import cKDTree
import pandas as pd
import logging as log
from dataclasses import dataclass

from RES.AttributesParser import AttributesParser
from RES.osm import OSMData
from shapely.ops import nearest_points
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, MultiLineString

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




    def find_nearest_single_connection_point(self,
        cell_centroid, 
        cell_geometry, 
        cell_gdf, 
        line_gdf
    ):
        """
        For a given cell centroid and its geometry:
        - If any lines intersect the cell, return the nearest point on them.
        - Otherwise, find the nearest cell with intersecting lines and return the nearest point on its lines.
        
        Returns: (nearest_point, distance)
        """
        # 1. Try to find lines intersecting this cell
        intersecting_lines = line_gdf[line_gdf.geometry.intersects(cell_geometry)].copy()

        if not intersecting_lines.empty:
            intersecting_lines["geometry"] = intersecting_lines.geometry.intersection(cell_geometry)
            lines_to_search = intersecting_lines
        else:
            # 2. Find nearest cell with intersecting lines
            cell_gdf = cell_gdf.copy()
            cell_gdf["has_intersection"] = cell_gdf.geometry.apply(
                lambda geom: line_gdf.geometry.intersects(geom).any()
            )

            candidate_cells = cell_gdf[cell_gdf["has_intersection"]].copy()
            if candidate_cells.empty:
                raise RuntimeError("No candidate cells with intersecting lines found.")

            # Find nearest such cell
            candidate_cells["distance"] = candidate_cells.geometry.centroid.distance(cell_centroid)
            nearest_geom = candidate_cells.sort_values("distance").geometry.iloc[0]

            if not isinstance(nearest_geom, BaseGeometry):
                raise TypeError(f"Expected shapely geometry, got {type(nearest_geom)}")

            # Get intersecting lines for nearest cell
            intersecting_lines = line_gdf[line_gdf.geometry.intersects(nearest_geom)].copy()
            intersecting_lines["geometry"] = intersecting_lines.geometry.intersection(nearest_geom)

            # Keep only line types
            lines_to_search = intersecting_lines[
                intersecting_lines.geometry.type.isin(['LineString', 'MultiLineString'])
            ]

        # 3. Filter valid and non-empty geometries
        lines_to_search = lines_to_search[
            lines_to_search.geometry.apply(lambda g: isinstance(g, BaseGeometry) and g.is_valid and not g.is_empty)
        ]

        if lines_to_search.empty:
            raise RuntimeError("No valid geometries to search for nearest point.")

        # 4. Find closest geometry
        distances = lines_to_search.geometry.apply(lambda line: cell_centroid.distance(line))
        nearest_geom = lines_to_search.loc[distances.idxmin(), "geometry"]

        # 5. Find nearest point on that geometry
        if nearest_geom.geom_type == "LineString":
            nearest_point = nearest_geom.interpolate(nearest_geom.project(cell_centroid))
        elif nearest_geom.geom_type == "MultiLineString":
            min_dist = float("inf")
            nearest_point = None
            for line in nearest_geom.geoms:
                projected = line.interpolate(line.project(cell_centroid))
                dist = cell_centroid.distance(projected)
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = projected
        else:
            raise ValueError(f"Expected LineString or MultiLineString, got {nearest_geom.geom_type}")

        distance = cell_centroid.distance(nearest_point)
        return nearest_point, distance


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
    
    def get_OSM_grid_lines(self) -> gpd.GeoDataFrame:
        """
        Retrieve OSM data for grid nodes.
        """
        osm_data = OSMData(region_short_code=self.region_short_code)
        
        osm_power_data = osm_data.get_osm_layer('power')
        if "element" not in osm_power_data.columns:
            osm_power_data = osm_power_data.reset_index()
        lines_gdf=osm_power_data[osm_power_data.element=='way']
        lines_gdf_cleaned= lines_gdf[
            lines_gdf.geometry.apply(lambda g: isinstance(g, (LineString, MultiLineString)))
]                                   
        
        if lines_gdf_cleaned is None:
            log.error("No OSM data found for Grid Lines")
            return None
        else:
            return lines_gdf_cleaned


# Example of a specialized class using inheritance
# class AdvancedGridNodeLocator(GridNodeLocator):
#     def __init__(self, region_code: str, grid_node_proximity_filter: float, extra_param: str):
#         super().__init__(region_code, grid_node_proximity_filter)
#         self.extra_param = extra_param
    
#     def some_advanced_method(self):
#         # Extend with more advanced functionalities here
#         pass