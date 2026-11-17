"""Exercise WB6 notebook storage cells without running regional assessments."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest

from RESource.hdf5_handler import DataHandler
from RESource.store_analytics import store_combined_capacity_haircuts

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks" / "studies_WB6"


def notebook_cell(notebook, marker):
    cells = json.loads((NOTEBOOKS / notebook).read_text())["cells"]
    return next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


def regional_store(root, region, haircut):
    path = (
        root
        / "data/store/WesternBalkanRegions"
        / region
        / (f"resources_WesternBalkanRegions_{region}_VRElow_2023_20261117.h5")
    )
    path.parent.mkdir(parents=True)
    handler = DataHandler(path)
    handler.set_metadata({"capacity_haircut": haircut} if haircut is not None else {"old": True})
    for key in ("cells", "boundary", "lines", "timeseries/solar", "timeseries/wind"):
        pd.DataFrame({"value": [1]}).to_hdf(path, key=key)
    return path


def notebook_environment(root, regions):
    cfg = {
        "region_mapping": {region: {"name": region} for region in regions},
        "technology": {"turbines": {1: "example"}},
    }
    return {
        "ROOT": root,
        "Path": Path,
        "DataHandler": DataHandler,
        "utils": SimpleNamespace(print_update=lambda **kwargs: None, get_runtime_info=lambda: {}),
        "country_kwd": "WesternBalkanRegions",
        "country_name": "Western Balkan Regions",
        "RUN_ID": "VRElow",
        "WEATHER_YEAR": "2023",
        "DATESTAMP": "20261117",
        "run_tag": "VRElow_2023_20261117",
        "regions": regions,
        "cfg": cfg,
        "cfg_provenance": {},
        "combined_store_path": root
        / "data/store/WesternBalkanRegions/Combined"
        / "resources_WesternBalkanRegions_Combined_VRElow_2023_20261117.h5",
    }


def test_combined_haircuts_and_images_are_saved_without_replacing_existing(tmp_path):
    regional_store(tmp_path, "AL", {"solar": 0.8, "wind": 0.6})
    regional_store(tmp_path, "BA", None)
    env = notebook_environment(tmp_path, ["AL", "BA"])
    notebook = "Visuals_WB6_CELLS_combined.ipynb"
    exec(notebook_cell(notebook, "#All the regions should"), env)
    exec(notebook_cell(notebook, "fallback_haircut ="), env)

    handler = DataHandler(env["combined_store_path"])
    assert handler.get_metadata("capacity_haircut") == {"solar": 0.8, "wind": 0.6}
    sources = handler.get_metadata("capacity_haircut_source_by_region")
    assert sources["AL"] == "regional store metadata"
    assert sources["BA"] == "notebook fallback (store metadata missing)"
    assert env["combined_store"]["AL"]["cells"]["value"].tolist() == [1]

    image = tmp_path / "figure.png"
    image.write_bytes(b"first image")
    env["save_combined_image"](image, "Figure")
    image.write_bytes(b"replacement")
    with pytest.raises(FileExistsError):
        env["save_combined_image"](image, "Figure")
    assert handler.load_image("figure") == b"first image"


def test_combined_haircut_mismatch_and_missing_regions_fail(tmp_path):
    regional_store(tmp_path, "AL", {"solar": 0.7, "wind": 0.6})
    regional_store(tmp_path, "BA", None)
    env = notebook_environment(tmp_path, ["AL", "BA"])
    notebook = "Visuals_WB6_CELLS_combined.ipynb"
    exec(notebook_cell(notebook, "#All the regions should"), env)
    with pytest.raises(ValueError, match="Regional haircut factors differ"):
        exec(notebook_cell(notebook, "fallback_haircut ="), env)

    env = notebook_environment(tmp_path, ["AL", "BA", "MK"])
    with pytest.raises(FileNotFoundError, match="MK"):
        exec(notebook_cell(notebook, "#All the regions should"), env)


def test_land_notebook_shares_store_and_refuses_existing_land_analytics(tmp_path):
    regional_store(tmp_path, "AL", {"solar": 0.8, "wind": 0.6})
    regional_store(tmp_path, "BA", {"solar": 0.8, "wind": 0.6})
    env = notebook_environment(tmp_path, ["AL", "BA"])
    cell = notebook_cell("Visuals_WB6_LANDs_combined.ipynb", "land_images =")
    exec(cell, env)
    path = env["combined_store_path"]
    assert path == notebook_environment(tmp_path, ["AL", "BA"])["combined_store_path"]
    exec(notebook_cell("Visuals_WB6_LANDs_combined.ipynb", "#All the regions should"), env)
    assert DataHandler(path).get_metadata("capacity_haircut") == {"solar": 0.8, "wind": 0.6}
    with h5py.File(path, "a") as store:
        store.create_dataset("LandAnalytics/images/CLC_class_distribution", data=[1])
    with pytest.raises(FileExistsError, match="CLC_class_distribution"):
        exec(cell, env)
    with h5py.File(path, "r") as store:
        assert store["LandAnalytics/images/CLC_class_distribution"][:].tolist() == [1]


def test_haircut_metadata_conflicts_and_invalid_values_preserve_store(tmp_path):
    handler = DataHandler(tmp_path / "combined.h5")
    handler.set_metadata({"capacity_haircut": {"solar": 0.8, "wind": 0.6}})
    fallback = {"solar": 0.8, "wind": 0.6}
    with pytest.raises(ValueError, match="already records different"):
        store_combined_capacity_haircuts(handler, {"AL": {"solar": 0.7, "wind": 0.6}}, fallback)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        store_combined_capacity_haircuts(
            handler, {"AL": {"solar": float("nan"), "wind": 0.6}}, fallback
        )
    assert handler.get_metadata("capacity_haircut") == fallback


def test_explorer_selects_a_store_from_the_regional_directory(tmp_path):
    path = regional_store(tmp_path, "AL", {"solar": 0.8, "wind": 0.6})
    env = notebook_environment(tmp_path, ["AL"])
    widgets = SimpleNamespace(Dropdown=lambda **kwargs: SimpleNamespace(value=kwargs["value"]))
    env.update({"region_code": "AL", "widgets": widgets, "display": lambda _: None})
    notebook = "explore_Store_n_Visuals_WB6.ipynb"
    exec(notebook_cell(notebook, "data_store_dir ="), env)
    exec(notebook_cell(notebook, "RUN_ID= run_id_dropdown.value"), env)
    assert env["store"] == path
    assert env["res_data"].get_metadata("capacity_haircut") == {"solar": 0.8, "wind": 0.6}
