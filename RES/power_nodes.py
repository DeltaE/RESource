import geopandas as gpd
from scipy.spatial import cKDTree
import pandas as pd
import logging as log
from dataclasses import dataclass

from RES.AttributesParser import AttributesParser
from RES.osm import OSMData


@dataclass
class GridNodeLocator(AttributesParser):
    """
    Electrical grid connection point locator for renewable energy site assessment.
    
    This class provides advanced geospatial algorithms for identifying optimal
    electrical grid connection points for renewable energy projects. It integrates
    transmission infrastructure data (substations, transmission lines) with 
    renewable energy site locations to calculate grid integration costs and
    constraints for economic feasibility analysis.
    
    Key Functionality:
    - Nearest grid node identification using spatial indexing
    - Distance-based grid connection cost calculations
    - Transmission line intersection analysis for direct connections
    - Proximity filtering for viable grid integration sites
    - Integration with OSM power infrastructure data
    - Support for both point-based and line-based connection strategies
    
    Inherits from:
        AttributesParser: Base class providing configuration management and regional attributes
        
    Attributes:
        grid_node_proximity_filter (float): Maximum distance threshold for viable
                                          grid connections (from configuration)
        
    Grid Connection Strategies:
    1. Direct Substation Connection: Connects to nearest substation point
    2. Transmission Line Connection: Connects to nearest point on transmission lines
    3. Hybrid Approach: Considers both substations and line tap points
    
    Example:
        >>> locator = GridNodeLocator(
        ...     config_file_path="config/config_BC.yaml",
        ...     region_short_code="BC"
        ... )
        >>> 
        >>> # Find grid connections for renewable energy sites
        >>> sites_with_grid = locator.find_grid_nodes_ERA5_cells(
        ...     buses_gdf=substation_data,
        ...     cells_gdf=renewable_sites
        ... )
        >>> 
        >>> # Get transmission line data from OSM
        >>> transmission_lines = locator.get_OSM_grid_lines()
        
    Applications:
        - Renewable energy project feasibility assessment
        - Grid integration cost estimation
        - Transmission capacity constraint analysis
        - Regional grid accessibility mapping
        - Infrastructure planning and optimization
        
    Notes:
        - Uses scipy.spatial.cKDTree for efficient nearest neighbor searches
        - Integrates OpenStreetMap power infrastructure data
        - Supports filtering based on economic viability thresholds
        - Distance calculations account for geographic coordinate projections
    """
    
    def __post_init__(self):
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        self.required_args = {
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
        }
        
        self.osmData=OSMData(**self.required_args)
        self.grid_node_proximity_filter:float = self.disaggregation_config['transmission']['proximity_filter']

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
        
        DEGREES_TO_KM:float = 111.32  # Approximate conversion factor from degrees to kilometers
        # Query the KDTree with the centroid of the cell geometry
        _, index = bus_tree.query((cell_geometry.centroid.x, cell_geometry.centroid.y))

        # Retrieve the nearest bus row
        nearest_bus_row = buses_gdf.iloc[index]

        # Compute the distance (convert degrees to kilometers using approximate conversion factor)
        distance_km = cell_geometry.centroid.distance(nearest_bus_row['geometry']) * DEGREES_TO_KM

        # Determine the station code based on available columns
        if 'name' in buses_gdf.columns:
            nearest_station_code = nearest_bus_row['name']
        else:
            nearest_station_code = nearest_bus_row['node_code']

        return nearest_station_code, distance_km


    from shapely.ops import nearest_points

    def find_nearest_single_connection_point(self,cell_centroid, cell_geometry, cell_gdf, line_gdf):
        """
        For a given cell centroid and its geometry:
        - If any lines intersect the cell, return the nearest point on them.
        - Otherwise, find the nearest cell with intersecting lines and return the nearest point on its lines.
        Returns: (nearest_point, distance)
        """
        # 1. Lines intersecting this cell
        intersecting_lines = line_gdf[
            (line_gdf.geometry.intersects(cell_geometry)) &
            (line_gdf.geometry.type.isin(['LineString', 'MultiLineString']))
        ].copy()

        if not intersecting_lines.empty:
            # Clip to the cell geometry
            intersecting_lines["geometry"] = intersecting_lines.geometry.intersection(cell_geometry)
            lines_to_search = intersecting_lines
        else:
            # 2. Find nearest neighbor with intersecting lines
            # Filter cells that have at least one intersecting line
            candidate_cells = cell_gdf[cell_gdf.geometry.apply(lambda geom: not line_gdf[line_gdf.geometry.intersects(geom)].empty)]

            # Find the closest such cell
            # candidate_cells["distance"] = candidate_cells.geometry.centroid.distance(cell_centroid)
            candidate_cells["centroid"] = candidate_cells.geometry.centroid
            candidate_cells["distance"] = candidate_cells["centroid"].apply(lambda x: x.distance(cell_centroid))

            nearest_cell = candidate_cells.loc[candidate_cells["distance"].idxmin()]
            
            # Get intersecting lines for that cell
            cell_geom = nearest_cell.geometry
            intersecting_lines = line_gdf[line_gdf.geometry.intersects(cell_geom)].copy()
            intersecting_lines["geometry"] = intersecting_lines.geometry.intersection(cell_geom)
            lines_to_search = intersecting_lines
            lines_to_search = lines_to_search[lines_to_search.geometry.type.isin(['LineString', 'MultiLineString'])]


        # 3. Find the nearest point on those lines
        # distances = lines_to_search.geometry.apply(lambda line: cell_centroid.distance(line))
        # nearest_geom = lines_to_search.loc[distances.idxmin(), "geometry"]
        
        # 3. Find the nearest point on those lines
        distances = lines_to_search.geometry.apply(lambda line: cell_centroid.distance(line)).astype(float)

        # Ensure distances is a proper float Series
        if distances.empty or distances.isna().all():
            raise ValueError("No valid distances found to determine nearest connection point.")

        # Safely get geometry with minimum distance
        min_idx = distances.idxmin()
        nearest_geom = lines_to_search.loc[min_idx, "geometry"]

        
        

        # Check type explicitly

        if nearest_geom.geom_type not in ["LineString", "MultiLineString"]:
            raise ValueError(f"Expected LineString/MultiLineString, got {nearest_geom.geom_type}")

        # Handle both cases
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

        distance = cell_centroid.distance(nearest_point)
        
        return nearest_point, distance


    def find_nearest_connection_point(self,
                                    cell_centroid: gpd.GeoSeries,
                                    cell_geometry: gpd.GeoSeries,
                                    cell_gdf: gpd.GeoDataFrame,
                                    line_gdf: gpd.GeoDataFrame):
        """
        For a given row (cell):
        - If any lines intersect the cell, return the nearest point on them.
        - Otherwise, find the nearest cell with intersecting lines and return the nearest point on its lines.
        Returns: (nearest_point, distance) or (None, None) if not found.
        """


        try:
            # 1. Lines intersecting this cell
            intersecting_lines = line_gdf[
                (line_gdf.geometry.intersects(cell_geometry)) &
                (line_gdf.geometry.type.isin(['LineString', 'MultiLineString']))
            ].copy()

            if not intersecting_lines.empty:
                intersecting_lines["geometry"] = intersecting_lines.geometry.intersection(cell_geometry)
                lines_to_search = intersecting_lines
            else:
                # 2. Find nearest neighbor with intersecting lines
                candidate_cells = cell_gdf[cell_gdf.geometry.apply(lambda geom: not line_gdf[line_gdf.geometry.intersects(geom)].empty)]
                candidate_cells["centroid"] = candidate_cells.geometry.centroid
                candidate_cells["distance"] = candidate_cells["centroid"].apply(lambda x: x.distance(cell_centroid))
                nearest_cell = candidate_cells.loc[candidate_cells["distance"].idxmin()]
                cell_geom = nearest_cell.geometry
                intersecting_lines = line_gdf[line_gdf.geometry.intersects(cell_geom)].copy()
                intersecting_lines["geometry"] = intersecting_lines.geometry.intersection(cell_geom)
                lines_to_search = intersecting_lines[intersecting_lines.geometry.type.isin(['LineString', 'MultiLineString'])]

            distances = lines_to_search.geometry.apply(lambda line: cell_centroid.distance(line)).astype(float)
            if distances.empty or distances.isna().all():
                return (None, None)
            min_idx = distances.idxmin()
            nearest_geom = lines_to_search.loc[min_idx, "geometry"]

            if nearest_geom.geom_type not in ["LineString", "MultiLineString"]:
                return (None, None)

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

            distance = cell_centroid.distance(nearest_point)
            return nearest_point, distance

        except Exception:
            return (None, None)


    def find_grid_nodes_ERA5_cells(
        self, 
        buses_gdf: gpd.GeoDataFrame, 
        cells_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Identify nearest grid connection points for renewable energy sites with proximity filtering.
        
        This method performs comprehensive grid connection analysis for renewable energy
        sites by finding the nearest electrical substations or connection points and
        calculating grid integration distances. It includes economic viability filtering
        based on configurable proximity thresholds to exclude sites with prohibitively
        expensive grid connections.
        
        Process:
        1. Builds spatial index (cKDTree) for efficient nearest neighbor queries
        2. For each renewable energy site, identifies closest grid connection point
        3. Calculates grid connection distance in kilometers
        4. Applies proximity filtering based on economic viability thresholds
        5. Returns enhanced dataset with grid connection metadata
        
        Args:
            buses_gdf (gpd.GeoDataFrame): GeoDataFrame containing electrical substations
                                        or grid connection points with geometry column
                                        and optional 'name' or 'node_code' attributes
            cells_gdf (gpd.GeoDataFrame): GeoDataFrame containing renewable energy sites
                                        with geometry column (typically grid cells or
                                        project locations)
                                        
        Returns:
            gpd.GeoDataFrame: Enhanced input cells_gdf with added columns:
                - 'nearest_station': Name or code of nearest grid connection point
                - 'nearest_station_distance_km': Distance to nearest connection (km)
                
        Grid Connection Columns Added:
            - nearest_station (str): Identifier for closest substation/connection point
            - nearest_station_distance_km (float): Grid connection distance in kilometers
            
        Example:
            >>> # Load grid infrastructure and renewable sites
            >>> substations = gpd.read_file('data/substations.geojson')
            >>> wind_sites = gpd.read_file('data/wind_grid_cells.geojson')
            >>> 
            >>> # Find grid connections
            >>> sites_with_grid = locator.find_grid_nodes_ERA5_cells(
            ...     buses_gdf=substations,
            ...     cells_gdf=wind_sites
            ... )
            >>> 
            >>> # Analyze grid connection costs
            >>> viable_sites = sites_with_grid[
            ...     sites_with_grid['nearest_station_distance_km'] <= 50
            ... ]
            
        Notes:
            - Uses approximate conversion factor (111.32 km/degree) for distance calculation
            - Proximity filtering based on self.grid_node_proximity_filter configuration
            - Spatial indexing ensures efficient processing of large datasets
            - Returns all sites with grid connection data, not just viable ones
        """
        buses_gdf.sindex  # Generate spatial index
        bus_tree = cKDTree(buses_gdf['geometry'].apply(lambda x: (x.x, x.y)).tolist())
        
        log.info("> Calculating Nearest Grid Nodes for Grid Cells")
        

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
        Retrieve transmission line infrastructure data from OpenStreetMap.
        
        This method accesses power transmission infrastructure data from OpenStreetMap
        through the OSM data processor, specifically extracting transmission lines
        that can serve as alternative grid connection points for renewable energy
        projects. It filters OSM power data to include only linear features
        (transmission lines) suitable for tap connections.
        
        Data Processing:
        1. Initializes OSM data processor for the specified region
        2. Retrieves power infrastructure data layer from OSM
        3. Filters data to include only transmission line features (ways)
        4. Returns processed transmission line geometries
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing transmission line geometries
                            with OSM attributes and power infrastructure metadata.
                            Returns None if no transmission line data is found.
                            
        OSM Data Attributes:
            - geometry: LineString geometries of transmission lines
            - power: Power infrastructure type (line, cable, etc.)
            - voltage: Operating voltage level (if available)
            - operator: Transmission system operator (if available)
            - Additional OSM tags as available
            
        Example:
            >>> locator = GridNodeLocator(**config)
            >>> transmission_lines = locator.get_OSM_grid_lines()
            >>> 
            >>> if transmission_lines is not None:
            ...     print(f"Found {len(transmission_lines)} transmission lines")
            ...     # Use lines for alternative grid connection analysis
            ... else:
            ...     print("No transmission line data available")
            
        Applications:
            - Alternative grid connection point identification
            - Transmission line tap connection analysis
            - Grid infrastructure density assessment
            - Regional transmission capacity mapping
            
        Notes:
            - Requires valid region_short_code for geographic filtering
            - OSM data quality and completeness varies by region
            - Linear features (ways) represent transmission corridors
            - Consider data vintage and accuracy for project planning
            
        Raises:
            RegionError: If region_short_code is invalid for OSM data retrieval
            DataError: If OSM power infrastructure data cannot be accessed
        """

        osm_power_data = self.osmData.get_osm_layer('power')
        if "element" not in osm_power_data.columns:
            osm_power_data = osm_power_data.reset_index()
        lines_gdf=osm_power_data[osm_power_data.element=='way']
        
        if lines_gdf is None:
            log.error("No OSM data found for Grid Lines")
            return None
        else:
            return lines_gdf
