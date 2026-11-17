import base64
import json

import h5py
import numpy as np
import pytest

from RESource.dash import store_summary


def test_land_images_are_discovered_and_large_figures_load_on_demand(tmp_path):
    path = tmp_path / "store.h5"
    small = b"\x89PNG\r\n\x1a\nsmall"
    large = b"\x89PNG\r\n\x1a\n" + b"x" * store_summary.LA_IMAGE_MAX_BYTES
    with h5py.File(path, "w") as store:
        for key, data in (("small", small), ("large", large)):
            image = store.create_dataset(
                f"LandAnalytics/images/{key}", data=np.frombuffer(data, "u1")
            )
            image.attrs["content_type"] = "image/png"
            image.attrs["title"] = f"{key} plot"
        store.create_dataset("LandAnalytics/rasters/classes", data=np.ones((2, 2), dtype="u1"))
        media = store_summary._la_media(store, set())

    images = {image["name"]: image for image in media["images"]}
    assert base64.b64decode(images["small"]["uri"].split(",", 1)[1]) == small
    assert "uri" not in images["large"]
    assert "Load it on demand" in images["large"]["note"]
    assert media["other"][0]["key"] == "LandAnalytics/rasters/classes"
    loaded = json.loads(store_summary.land_image(path, images["large"]["key"]))
    assert base64.b64decode(loaded["uri"].split(",", 1)[1]) == large
    with pytest.raises(ValueError, match="Not an image dataset"):
        store_summary.land_image(path, "LandAnalytics/rasters/classes")
    with pytest.raises(ValueError, match="Not a land analytics dataset"):
        store_summary.land_image(path, "cells")


def test_combined_images_keep_their_source_indices():
    def payload(region):
        return {
            "meta": {"region": region},
            "keys": [],
            "landanalytics": [],
            "la_media": {"images": [{"key": "LandAnalytics/images/plot"}], "other": []},
        }

    result = json.loads(store_summary.combine_summaries([payload("AL"), payload("BA")], "All"))
    assert [(im["source"], im["source_index"]) for im in result["la_media"]["images"]] == [
        ("AL", 0),
        ("BA", 1),
    ]
