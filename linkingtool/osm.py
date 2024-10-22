import osmnx as ox
import geopandas as gpd
from pathlib import Path
from linkingtool.AttributesParser import AttributesParser

class OSMData(AttributesParser):
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        
        # OSM data specific attributes from user config (e.g., root path and data keys)
        self.osm_data_config = self.get_osm_config()
        
        # Accessing the data_keys
        self.data_keys = {key: value['tags'] for key, value in self.osm_data_config['data_keys'].items()}
        self.root_path = Path(self.osm_data_config['root'])  # Define the root path
        
        # Format the area name for OSM queries (e.g., "British Columbia, Canada")
        self.area_name = self.get_province_name()+ ", " + self.get_country()

        # Dictionary to store GeoDataFrames by their corresponding data_key, while all data is loaded
        self.gdfs = {}
        
    def get_osm_layer(self, data_key: str) -> gpd.GeoDataFrame:
        """
        Access the GeoDataFrame for a specific data key (table names of coders) e.g. 'aeroway','power','substation'
        """
        if data_key in self.gdfs:
            self.log.info(f"GeoDataFrame for '{data_key}' already exists, returning it.")
            return self.gdfs[data_key]
        
        # If it doesn't exist, load it using the tags from the configuration
        if data_key in self.data_keys:
            return self.__load_tagged_data_from_OSM__(self.data_keys[data_key], data_key)
        else:
            self.log.warning(f"'{data_key}' is not a valid key in the configuration.")
            return None  # Return None or raise an exception as per your error handling preference
        
    def run(self):
        """
        Run the OSM data retrieval process for all data keys.
        """
        for data_key in self.data_keys.keys():
            # Check if the data_key matches keys defined in the YAML configuration
            if data_key in self.data_keys:
                print(f"Processing OSM data for key: {data_key}")
                gdf = self.__load_tagged_data_from_OSM__(self.data_keys[data_key], data_key)
                self.gdfs[data_key] = gdf  # Store the GeoDataFrame in the dictionary
            else:
                print(f"Warning: '{data_key}' is not a valid key in the configuration.")
        return self.gdfs

    def __load_tagged_data_from_OSM__(self, 
                                  tags: dict, 
                                  data_key: str) -> gpd.GeoDataFrame:
        """
        Retrieve infrastructure-related data from OSM for the specified area and tags.
        """
        # Path to save the data as GeoJSON
        geojson_path = self.root_path / f"{self.province_short_code}_{data_key}.geojson"
        tags_dict = {data_key: tags}
        
        # Check if the file already exists locally
        if geojson_path.exists():
            # Load the data from the GeoJSON file if it exists
            self.log.info(f"Loading OSM data for '{data_key}' locally stored data from {geojson_path}")
            self.loaded_data = gpd.read_file(geojson_path)
        else:
            # Download data from OSM if the file doesn't exist
            print(f"Downloading data for {self.area_name} with tags {tags} and saving to {geojson_path}")
            self.loaded_data = ox.features_from_place(self.area_name, tags_dict)
            self.__save_local_file__(self.loaded_data, geojson_path)
        
        return self.loaded_data
    
    def __save_local_file__(self, 
                        gdf: gpd.GeoDataFrame, 
                        geojson_path: Path):
        """
        Save the retrieved data to a local GeoJSON file, if it does not exist.
        """
        # Create the parent directories if they do not exist
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not geojson_path.exists():
            # Save the GeoDataFrame to a file
            print(f"Saving data to {geojson_path}")
            gdf.to_file(geojson_path, driver='GeoJSON')
        else:
            print(f"File {geojson_path} already exists, skipping save.")