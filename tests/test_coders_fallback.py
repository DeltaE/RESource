from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box

from RESource.coders import CODERSData
from RESource.RESources import RESources_builder


class _DataHandler:
    def __init__(self):
        self.stored = {}

    def to_store(self, data, key):
        self.stored[key] = data.copy()

    def refresh(self):
        pass

    def from_store(self, key):
        return self.stored[key].copy()


def test_coders_connection_filter_uses_node_type_and_transmission_topology():
    substations = gpd.GeoDataFrame(
        {
            "node_code": ["BC_GEN_GSS", "BC_TERM_TSS", "BC_SW_SWS", "BC_LOAD_DSS", "ORPHAN"],
            "node_type": ["Generation", "Terminal", "Switching", "Distribution", "Terminal"],
        },
        geometry=[Point(index, 0) for index in range(5)],
        crs="EPSG:4326",
    )
    transmission_lines = pd.DataFrame(
        {
            "network_node_code_starting": ["BC_GEN_GSS", "BC_TERM_TSS"],
            "network_node_code_ending": ["BC_SW_SWS", "BC_LOAD_DSS"],
        }
    )
    coders = SimpleNamespace(
        coders_data_config={
            "connection_filter": {
                "enabled": True,
                "eligible_node_types": ["Generation", "Terminal"],
                "excluded_node_suffixes": ["INT", "IPT", "SWS", "JCT"],
                "require_transmission_endpoint": True,
            }
        }
    )

    filtered = CODERSData.filter_connection_substations(coders, substations, transmission_lines)

    assert filtered["node_code"].tolist() == ["BC_GEN_GSS", "BC_TERM_TSS"]


def test_missing_coders_credentials_fall_back_to_osm(monkeypatch):
    class UnauthenticatedCODERS:
        def __init__(self, **_kwargs):
            self.api_user = None

        def get_table_provincial(self, _table_name):
            raise AssertionError("CODERS should not be queried without credentials")

    grid_lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
    locator = SimpleNamespace(
        get_OSM_grid_lines=lambda: grid_lines,
        find_nearest_connection_point=lambda *_args: (Point(0, 0), 0.0),
    )
    cells = gpd.GeoDataFrame({"x": [0.5], "y": [0.5]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    datahandler = _DataHandler()
    builder = SimpleNamespace(
        country_name="Canada",
        region_short_code="BC",
        required_args={},
        era5_cutout=SimpleNamespace(get_era5_cutout=lambda **_kwargs: (None, None)),
        weather_year=2024,
        gridNodesProcessor=locator,
        datahandler=datahandler,
    )
    builder._find_grid_nodes_from_osm = lambda: RESources_builder._find_grid_nodes_from_osm(builder)
    monkeypatch.setattr("RESource.RESources.CODERSData", UnauthenticatedCODERS)

    result = RESources_builder.find_grid_nodes(builder, cells=cells)

    assert "nearest_connection_point" in result
    assert "nearest_distance" in result
    assert "lines" in datahandler.stored


def test_general_region_uses_uploaded_substations(tmp_path):
    substations_path = tmp_path / "substations.csv"
    substations_path.write_text("name,longitude,latitude\nstation,20.0,42.0\n", encoding="utf-8")
    cells = gpd.GeoDataFrame(
        {"x": [20.0], "y": [42.0]}, geometry=[box(19, 41, 21, 43)], crs="EPSG:4326"
    )
    datahandler = _DataHandler()
    locator = SimpleNamespace(
        find_grid_nodes_ERA5_cells=lambda substations, stored_cells: stored_cells,
        get_OSM_grid_lines=lambda: (_ for _ in ()).throw(
            AssertionError("OSM should not be queried when uploaded substations are valid")
        ),
    )
    builder = SimpleNamespace(
        country_name="Montenegro",
        region_short_code="ME",
        era5_cutout=SimpleNamespace(get_era5_cutout=lambda **_kwargs: (None, None)),
        weather_year=2024,
        gridNodesProcessor=locator,
        datahandler=datahandler,
        get_buses_path=lambda: substations_path,
        get_processed_substations_path=lambda _source: (
            tmp_path / "processed" / "substations_ME.pkl"
        ),
    )
    builder._load_uploaded_substations = lambda path: RESources_builder._load_uploaded_substations(
        builder, path
    )

    result = RESources_builder.find_grid_nodes(builder, cells=cells)

    assert len(datahandler.stored["substations"]) == 1
    assert (tmp_path / "processed" / "substations_ME.pkl").is_file()
    assert result.equals(cells)


def test_general_region_falls_back_to_osm_when_substations_are_missing(tmp_path):
    grid_lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
    cells = gpd.GeoDataFrame({"x": [0.5], "y": [0.5]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    datahandler = _DataHandler()
    builder = SimpleNamespace(
        country_name="Montenegro",
        region_short_code="ME",
        era5_cutout=SimpleNamespace(get_era5_cutout=lambda **_kwargs: (None, None)),
        weather_year=2024,
        gridNodesProcessor=SimpleNamespace(
            get_OSM_grid_lines=lambda: grid_lines,
            find_nearest_connection_point=lambda *_args: (Point(0, 0), 0.0),
        ),
        datahandler=datahandler,
        get_buses_path=lambda: tmp_path / "missing.csv",
        get_processed_substations_path=lambda _source: (
            tmp_path / "processed" / "substations_ME.pkl"
        ),
    )
    builder._load_uploaded_substations = lambda path: RESources_builder._load_uploaded_substations(
        builder, path
    )
    builder._find_grid_nodes_from_osm = lambda: RESources_builder._find_grid_nodes_from_osm(builder)

    result = RESources_builder.find_grid_nodes(builder, cells=cells)

    assert "nearest_connection_point" in result
    assert "lines" in datahandler.stored
