"""Verify stored capacity haircuts and dashboard interpretation."""

import json

import pandas as pd
import pytest

from RESource.capacity_haircut import apply_capacity_haircuts
from RESource.dash import cluster_explorer, store_summary
from RESource.hdf5_handler import DataHandler


def make_store(path):
    cells = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            "potential_capacity_solar": [100.0, 50.0],
            "potential_capacity_wind": [20.0, 30.0],
            "lcoe_solar": [30.0, 40.0],
            "lcoe_wind": [50.0, 60.0],
        },
        index=["a", "b"],
    )
    cells.to_hdf(path, key="cells")
    for tech in ("solar", "wind"):
        pd.DataFrame({"potential_capacity": [100.0], "lcoe": [30.0]}, index=["c"]).to_hdf(
            path, key=f"clusters/{tech}"
        )
    handler = DataHandler(path)
    handler.set_metadata({"capacity_haircut": {"solar": 0.8, "wind": 0.6}})
    return handler


def test_store_and_dashboard_apply_exactly_once(tmp_path):
    handler = make_store(tmp_path / "store.h5")
    raw = pd.read_hdf(handler.store, "cells")
    factors = {"solar": 0.8, "wind": 0.6}
    apply_capacity_haircuts(handler, factors)
    apply_capacity_haircuts(handler, factors)
    pd.testing.assert_frame_equal(pd.read_hdf(handler.store, "cells"), raw)
    adjusted = pd.read_hdf(handler.store, "capacity_haircuts/cells")
    assert adjusted["potential_capacity_solar_adjusted"].tolist() == [80, 40]
    assert adjusted["potential_capacity_wind_adjusted"].tolist() == [12, 18]
    assert pd.read_hdf(handler.store, "capacity_haircuts/clusters/wind")[
        "potential_capacity_adjusted"
    ].tolist() == [60]
    summary = json.loads(store_summary.summarize_store(str(handler.store)))
    assert summary["haircut_status"] == "adjusted"
    assert summary["story"]["solar"]["capacity_mw"] == 120
    assert summary["story"]["wind"]["cluster_capacity_mw"] == 60
    assert summary["cells"]["cols"]["potential_capacity_solar_unadjusted"] == [100, 50]
    assert cluster_explorer.cell_table(str(handler.store), "solar")["capacity"].tolist() == [80, 40]


def test_stale_capacity_is_explicit_and_never_partially_adjusted(tmp_path):
    handler = make_store(tmp_path / "store.h5")
    apply_capacity_haircuts(handler, {"solar": 0.8, "wind": 0.6})
    cells = pd.read_hdf(handler.store, "cells")
    cells.loc["a", "potential_capacity_solar"] = 200
    cells.to_hdf(handler.store, key="cells")
    with pytest.raises(ValueError, match="conflicts"):
        apply_capacity_haircuts(handler, {"solar": 0.8, "wind": 0.6})
    summary = json.loads(store_summary.summarize_store(str(handler.store)))
    assert summary["haircut_status"] == "invalid"
    assert "capacity haircuts" in summary["problems"]
    assert summary["story"]["solar"]["capacity_mw"] == 250
    assert summary["story"]["wind"]["cluster_capacity_mw"] == 100
    with pytest.raises(ValueError, match="does not match"):
        cluster_explorer.cell_table(str(handler.store), "wind")


def test_missing_cluster_does_not_change_store(tmp_path):
    handler = make_store(tmp_path / "store.h5")
    with pd.HDFStore(handler.store, "a") as store:
        store.remove("clusters/wind")
    with pytest.raises(ValueError, match="clusters/wind"):
        apply_capacity_haircuts(handler, {"solar": 0.8, "wind": 0.6})
    assert handler.get_metadata("capacity_haircut_applied") is None
    with pd.HDFStore(handler.store, "r") as store:
        assert "/capacity_haircuts/cells" not in store


def test_combined_summaries_keep_adjusted_and_unadjusted_stores_distinct(tmp_path):
    a = make_store(tmp_path / "resources_Test_A_run_2025_20261117.h5")
    b = make_store(tmp_path / "resources_Test_B_run_2025_20261117.h5")
    apply_capacity_haircuts(a, {"solar": 0.8, "wind": 0.6})
    adjusted = json.loads(store_summary.summarize_store(str(a.store)))
    raw = json.loads(store_summary.summarize_store(str(b.store)))
    assert raw["haircut_status"] == "unadjusted"
    assert raw["story"]["solar"]["capacity_mw"] == 150
    combined = json.loads(store_summary.combine_summaries([adjusted, raw], "Both"))
    assert combined["story"]["solar"]["capacity_mw"] == 270
    assert combined["story"]["wind"]["cluster_capacity_mw"] == 160
