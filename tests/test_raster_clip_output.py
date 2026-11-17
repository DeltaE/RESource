"""Tests for regional raster output locations."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from RESource.lands import clip_to_boundary_and_resample_raster


def test_clip_can_use_separate_output_directory(tmp_path):
    source = tmp_path / "source.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 200, 100, 100),
        nodata=255,
    ) as dst:
        dst.write(np.arange(8, dtype="uint8").reshape(2, 4), 1)

    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 400, 200)], crs="EPSG:3035")
    output = clip_to_boundary_and_resample_raster(
        {},
        "region",
        boundary,
        CRS_meters="EPSG:3035",
        source_raster_path=source,
        output_dir=tmp_path / "run",
    )
    assert output == tmp_path / "run/source_clipped_region.tif"
    assert output.is_file()
    assert not (tmp_path / "source_clipped_region.tif").exists()
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), np.arange(8).reshape(2, 4))
