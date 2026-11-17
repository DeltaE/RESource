"""Round-trip raster data alongside existing HDF5 store contents."""

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS

from RESource.hdf5_handler import DataHandler


def test_raster_roundtrip_preserves_georeferencing_and_values(tmp_path):
    handler = DataHandler(tmp_path / "results.h5")
    handler.to_store(pd.DataFrame({"value": [1]}), "LandAnalytics/example")
    original = np.array([[0, np.nan], [3.25, 4]], dtype=np.float32)
    transform = Affine(0.5, 0, 10, 0, -0.5, 20)
    provenance = {"provider": "WorldPop", "files": ["example.tif"]}

    handler.save_raster(
        original,
        "density",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        units="people/km²",
        provenance=provenance,
    )
    handler.save_image(b"image bytes", "density")

    data, metadata = handler.load_raster("density")
    np.testing.assert_array_equal(data, original)
    assert metadata["crs"] == CRS.from_epsg(4326)
    assert metadata["transform"] == transform
    assert np.isnan(metadata["nodata"])
    assert metadata["units"] == "people/km²"
    assert metadata["provenance"] == provenance
    assert handler.load_image("density") == b"image bytes"
    assert handler.from_store("LandAnalytics/example")["value"].tolist() == [1]

    with pytest.raises(FileExistsError, match="density"):
        handler.save_raster(
            original,
            "density",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
            units="people/km²",
            provenance=provenance,
        )
    np.testing.assert_array_equal(handler.load_raster("density")[0], original)


def test_raster_rejects_non_spatial_dimensions(tmp_path):
    handler = DataHandler(tmp_path / "results.h5")
    with pytest.raises(ValueError, match="two-dimensional"):
        handler.save_raster(
            np.ones((1, 2, 2)),
            "density",
            crs="EPSG:4326",
            transform=Affine.identity(),
            nodata=None,
            units="people/km²",
            provenance={},
        )
