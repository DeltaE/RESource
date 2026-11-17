from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import h5py
import numpy as np
import pytest
import rasterio
from matplotlib.axes import Axes
from rasterio.transform import from_origin
from shapely.geometry import box

from RESource.hdf5_handler import DataHandler
from RESource.land_analytics import (
    land_analytics_version,
    land_impact_images_exist,
    run_land_analytics,
    store_land_layer_images,
)


def test_class_images_use_valid_pixels_within_boundary_and_are_reusable(tmp_path, monkeypatch):
    path = tmp_path / "classes.tif"
    values = np.array([[0, 1, 2], [1, 255, 2], [2, 2, 2]], dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 3, 1, 1),
        nodata=255,
    ) as dst:
        dst.write(values, 1)
    boundary = gpd.GeoDataFrame(geometry=[box(0, 1, 2, 3)], crs="EPSG:4326")
    handler = DataHandler(tmp_path / "store.h5")
    layer = {"name": "test cover", "filepath": str(path)}
    original_barh = Axes.barh
    observed = {}

    def capture_barh(self, labels, widths, *args, **kwargs):
        observed.update(zip(labels, widths, strict=True))
        return original_barh(self, labels, widths, *args, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture_barh)
    store_land_layer_images(handler, [layer], boundary, "v1")
    assert observed == pytest.approx({"0: Class 0": 100 / 3, "1: Class 1": 200 / 3})
    with h5py.File(handler.store, "r") as store:
        images = store["LandAnalytics/images"]
        keys = list(images)
        assert len(keys) == 2
        assert all(images[key][:8].tobytes() == b"\x89PNG\r\n\x1a\n" for key in keys)
        assert all(images[key].attrs["content_type"] == "image/png" for key in keys)
        assert list(store["LandAnalytics"]) == ["images"]
        before = {key: images[key][()].tobytes() for key in keys}

    store_land_layer_images(handler, [layer], boundary, "v1")
    with h5py.File(handler.store, "r") as store:
        assert {key: store["LandAnalytics/images"][key][()].tobytes() for key in keys} == before
    store_land_layer_images(handler, [layer], boundary, "v2")
    with h5py.File(handler.store, "r") as store:
        assert len(store["LandAnalytics/images"]) == 4


def test_cached_capacity_regenerates_only_missing_enabled_impact_plots(tmp_path):
    handler = DataHandler(tmp_path / "store.h5")
    directory = tmp_path / "lands"
    directory.mkdir()
    names = (
        "stepwise_land_availability_plot_wind.png",
        "individual_layers_impact_land_availability_plot_wind.png",
    )

    def generate_impact_plots():
        for name in names:
            (directory / name).write_bytes(b"fresh plot")

    container = SimpleNamespace(
        plot_layer_impacts=True,
        vis_root=tmp_path,
        set_excluder=Mock(side_effect=generate_impact_plots),
        get_layers=Mock(return_value=[]),
        region_boundary=gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"),
    )
    run_land_analytics(handler, container, "wind", "v1", capacity_cached=True)
    assert container.set_excluder.call_count == 1
    assert land_impact_images_exist(handler, "wind", "v1")
    run_land_analytics(handler, container, "wind", "v1", capacity_cached=True)
    assert container.set_excluder.call_count == 1
    container.plot_layer_impacts = False
    run_land_analytics(handler, container, "solar", "v1", capacity_cached=True)
    assert container.set_excluder.call_count == 1
    assert not land_impact_images_exist(handler, "solar", "v1")


def test_land_version_changes_with_configuration_and_region():
    config = {"lands": {"GAEZ": {"raster": "first.tif"}}}
    version = land_analytics_version(config, "AL")
    assert version == land_analytics_version(config, "AL")
    assert version != land_analytics_version(config, "BA")
    assert version != land_analytics_version({"lands": {"GAEZ": {"raster": "second.tif"}}}, "AL")
