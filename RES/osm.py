import osmnx as ox
import geopandas as gpd
from pathlib import Path

from RES.AttributesParser import AttributesParser

ox.settings.max_query_area_size =10_000 * 1E6  # 10,000 sq km

class OSMData(AttributesParser):
    def __post_init__(self):
        """
        Initialize inherited attributes and OSM data-specific configuration.
        """
        super().__post_init__()
        
        # Load OSM-specific configurations
        self.osm_data_config = self.get_osm_config()
        
        # Extract data keys and root path from configuration
        self.data_keys = {key: value['tags'] for key, value in self.osm_data_config['data_keys'].items()}
        self.root_path = Path(self.osm_data_config['root'])

        # Create the directory (and any necessary parent directories) if it doesn't exist
        self.root_path.mkdir(parents=True, exist_ok=True)
        # Format area name for OSM queries
        self.area_name = f"{self.get_province_name()}, {self.get_country()}"
        
        # Dictionary to store GeoDataFrames by data_key
        self.gdfs = {}
        
    def get_osm_layer(self, data_key: str) -> gpd.GeoDataFrame:
        """
        Access or load the GeoDataFrame for a specific data key.
        """
        if data_key in self.gdfs:
            self.log.info(f"GeoDataFrame for '{data_key}' already exists, returning it.")
            return self.gdfs[data_key]
        
        # Load the data if it doesn't exist in memory
        if data_key in self.data_keys:
            gdf = self.__load_tagged_data_from_OSM__(self.data_keys[data_key], data_key)
            self.gdfs[data_key] = gdf  # Cache for future use
            return gdf
        else:
            self.log.warning(f"'{data_key}' is not a valid key in the configuration.")
            return None
        
    def run(self) -> dict:
        """
        Run the OSM data retrieval process for all data keys and store results in self.gdfs.
        """
        for data_key in self.data_keys.keys():
            print(f"Processing OSM data for key: {data_key}")
            self.get_osm_layer(data_key)
        return self.gdfs

    def __load_tagged_data_from_OSM__(self, tags: dict, data_key: str) -> gpd.GeoDataFrame:
        """
        Retrieve and cache OSM data for the specified area and tags.
        """
        geojson_path = self.root_path / f"{self.province_short_code}_{data_key}.geojson"
        tags_dict = {data_key: tags}
        
        # Check if data is already stored locally
        if geojson_path.exists():
            self.log.info(f">> Loading locally stored OSM data for '{data_key}' from {geojson_path}")
            return gpd.read_file(geojson_path)
        else:
            print(f">> Downloading data for {self.area_name} with tags {tags} and saving to {geojson_path}")
            gdf = ox.features_from_place(self.area_name, tags_dict)
            self.__save_local_file__(gdf, geojson_path)
            return gdf

    def __save_local_file__(self, gdf: gpd.GeoDataFrame, geojson_path: Path):
        """
        Save the GeoDataFrame to a local GeoJSON file if it doesn't already exist.
        """
        
        if not geojson_path.exists():
            self.log.info(f">> Saving OSM data to {geojson_path}")
            gdf.to_file(geojson_path, driver='GeoJSON')
        else:
            self.log.info(f">> File {geojson_path} already exists, skipping save.")
