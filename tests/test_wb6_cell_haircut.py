"""Verify the WB6 cell notebook uses and annotates its regional stores."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from RESource.capacity_haircut import record_regional_capacity_haircuts
from RESource.hdf5_handler import DataHandler

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/studies_WB6/Visuals_WB6_CELLS_combined.ipynb"
)


def cell(marker):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return next(
        "".join(item["source"])
        for item in cells
        if item["cell_type"] == "code" and marker in "".join(item["source"])
    )


def make_store(root, region, metadata, *, legacy=False):
    run_tag = "VRElow_2025_20261117"
    filename = f"resources_WesternBalkanRegions_{region}_{run_tag}.h5"
    directory = (
        root / "data/store/WesternBalkanRegions" / region
        if legacy
        else root / "results/WesternBalkanRegions" / region / run_tag
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    handler = DataHandler(path)
    handler.set_metadata(metadata)
    for key in ("cells", "boundary", "lines", "timeseries/solar", "timeseries/wind"):
        frame = {"value": [1]}
        if key == "cells":
            frame.update({"potential_capacity_solar": [10.0], "potential_capacity_wind": [20.0]})
        pd.DataFrame(frame).to_hdf(path, key=key)
    for tech in ("solar", "wind"):
        pd.DataFrame({"potential_capacity": [10.0]}).to_hdf(path, key=f"clusters/{tech}")
    return path


def environment(root, regions):
    return {
        "ROOT": root,
        "DataHandler": DataHandler,
        "utils": SimpleNamespace(print_update=lambda **kwargs: None),
        "country_kwd": "WesternBalkanRegions",
        "country_name": "Western Balkan Regions",
        "RUN_ID": "VRElow",
        "run_tag": "VRElow_2025_20261117",
        "regions": regions,
        "cfg": {"region_mapping": {region: {"name": region} for region in regions}},
    }


def test_notebook_records_assumptions_in_loaded_stores_only(tmp_path):
    factors = {"solar": 0.8, "wind": 0.6}
    al = make_store(
        tmp_path, "AL", {"capacity_haircut": factors, "capacity_haircut_notes": "Existing"}
    )
    ba = make_store(tmp_path, "BA", {"old": True})
    legacy_al = make_store(tmp_path, "AL", {"old": True}, legacy=True)
    env = environment(tmp_path, ["AL", "BA"])
    exec(cell("#All the regions should"), env)
    exec(cell("solar_lcoe_threshold=70"), env)

    assert env["haircut"] == factors
    assert DataHandler(al).get_metadata("capacity_haircut_notes") == "Existing"
    assert DataHandler(ba).get_metadata("capacity_haircut") == factors
    assert DataHandler(ba).get_metadata("capacity_haircut_source") == "WB6 notebook fallback"
    assert "not adjusted" in DataHandler(ba).get_metadata("capacity_haircut_notes")
    assert pd.read_hdf(ba, "capacity_haircuts/clusters/wind")[
        "potential_capacity_adjusted"
    ].tolist() == [6.0]
    assert DataHandler(legacy_al).get_metadata("capacity_haircut") is None
    assert not (tmp_path / "data/store/WesternBalkanRegions/Combined").exists()
    assert pd.read_hdf(ba, "cells")["value"].tolist() == [1]


@pytest.mark.parametrize("factor", [0.7, float("nan")])
def test_mismatched_or_invalid_factors_do_not_write_any_store(tmp_path, factor):
    al = make_store(tmp_path, "AL", {"old": True})
    ba = make_store(tmp_path, "BA", {"capacity_haircut": {"solar": factor, "wind": 0.6}})
    handlers = {"AL": DataHandler(al), "BA": DataHandler(ba)}
    with pytest.raises(ValueError):
        record_regional_capacity_haircuts(handlers, {"solar": 0.8, "wind": 0.6})
    assert DataHandler(al).get_metadata("capacity_haircut") is None
    assert DataHandler(ba).get_metadata("capacity_haircut_notes") is None


def test_missing_region_does_not_create_store_or_write_metadata(tmp_path):
    al = make_store(tmp_path, "AL", {"old": True})
    with pytest.raises(FileNotFoundError, match="BA"):
        exec(cell("#All the regions should"), environment(tmp_path, ["AL", "BA"]))
    assert DataHandler(al).get_metadata("capacity_haircut") is None
    assert list(tmp_path.rglob("*.h5")) == [al]
