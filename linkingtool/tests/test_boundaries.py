from linkingtool.boundaries import GADMBoundaries
import geopandas as gpd
import pandas as pd

gadm_boundary=GADMBoundaries()

def test_country_boundary()->None:
    # Fetch the province boundary using the gadm_boundary module
    country_boundary = gadm_boundary.get_country_boundary()
    # Ensure that the returned object is a GeoDataFrame
    assert isinstance(country_boundary, gpd.GeoDataFrame), "Returned boundary is not a GeoDataFrame"

def test_province_boundary()->None:
    # Fetch the province boundary using the gadm_boundary module
    province_boundary = gadm_boundary.get_province_boundary()
    # Ensure that the returned object is a GeoDataFrame
    assert isinstance(province_boundary, gpd.GeoDataFrame), "Returned boundary is not a GeoDataFrame"


def test_boundary_crs() -> None:
    province_boundary = gadm_boundary.get_province_boundary()
    country_boundary = gadm_boundary.get_country_boundary()
    
    # Ensure that the returned objects are in the correct CRS (EPSG:4326)
    assert (
        province_boundary.crs.to_epsg() == 4326 and country_boundary.crs.to_epsg() == 4326
    ), "Returned boundary is not in the correct CRS"
