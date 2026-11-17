"""Verify the PyTables-free HDF5 reader matches pandas.read_hdf exactly.

``RESource.dash.store_reader`` reimplements pandas' HDF5 'fixed' format with
only h5py/pickle/numpy (so it can run inside Pyodide, which has no PyTables).
These tests lock that reconstruction against the real reader on actual store
files, skipping cleanly where no store is present (e.g. a fresh checkout).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from RESource.dash import store_reader

STORE_ROOT = Path("results")


def _sample_stores() -> list[Path]:
    if not STORE_ROOT.is_dir():
        return []
    return sorted(STORE_ROOT.glob("*/*/*/*.h5"))


@pytest.mark.parametrize("store_path", _sample_stores(), ids=lambda p: p.name)
def test_read_fixed_frame_matches_pandas(store_path: Path) -> None:
    with h5py.File(store_path, "r") as h5file:
        keys = store_reader.list_frame_keys(h5file)
        assert keys, f"no pandas frames found in {store_path}"
        for entry in keys:
            key = entry["key"]
            raw_df = store_reader.read_fixed_frame(h5file, key)
            ref_df = pd.read_hdf(store_path, key)
            assert list(raw_df.columns) == list(ref_df.columns), key
            assert len(raw_df) == len(ref_df), key
            for col in ref_df.select_dtypes(include="number").columns:
                assert np.allclose(
                    raw_df[col].to_numpy(dtype=float),
                    ref_df[col].to_numpy(dtype=float),
                    equal_nan=True,
                ), f"{key}.{col}"
            for col in ref_df.select_dtypes(exclude="number").columns:
                for raw_v, ref_v in zip(raw_df[col], ref_df[col]):
                    both_nan = (
                        isinstance(raw_v, float)
                        and isinstance(ref_v, float)
                        and np.isnan(raw_v)
                        and np.isnan(ref_v)
                    )
                    assert both_nan or raw_v == ref_v, f"{key}.{col}: {raw_v!r} != {ref_v!r}"


@pytest.mark.parametrize("store_path", _sample_stores(), ids=lambda p: p.name)
def test_read_fixed_frame_lazy_columns_match_full_read(store_path: Path) -> None:
    """``columns=`` must select the same data a full read would, just narrower."""
    with h5py.File(store_path, "r") as h5file:
        keys = store_reader.list_frame_keys(h5file)
        for entry in keys:
            key = entry["key"]
            full_df = store_reader.read_fixed_frame(h5file, key)
            if full_df.shape[1] < 2:
                continue
            subset_cols = list(full_df.columns[: max(1, full_df.shape[1] // 2)])
            lazy_df = store_reader.read_fixed_frame(h5file, key, columns=subset_cols)
            for col in subset_cols:
                if col not in lazy_df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(full_df[col]):
                    assert np.allclose(
                        lazy_df[col].to_numpy(dtype=float),
                        full_df[col].to_numpy(dtype=float),
                        equal_nan=True,
                    ), f"{key}.{col}"
                else:
                    for lazy_v, full_v in zip(lazy_df[col], full_df[col]):
                        both_nan = (
                            isinstance(lazy_v, float)
                            and isinstance(full_v, float)
                            and np.isnan(lazy_v)
                            and np.isnan(full_v)
                        )
                        assert both_nan or lazy_v == full_v or (
                            isinstance(lazy_v, (list, np.ndarray))
                            and isinstance(full_v, (list, np.ndarray))
                            and list(lazy_v) == list(full_v)
                        ), f"{key}.{col}: {lazy_v!r} != {full_v!r}"


def test_wkt_parts_and_kind() -> None:
    assert store_reader.wkt_kind("POINT (10 20)") == "POINT"
    assert store_reader.wkt_kind(None) == ""
    assert store_reader.wkt_kind(42) == ""

    parts = store_reader.wkt_parts("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    assert len(parts) == 1
    assert parts[0][0] == [0.0, 0.0]
    assert len(parts[0]) == 5

    point_parts = store_reader.wkt_parts("POINT (10 20)")
    assert point_parts == [[[10.0, 20.0]]]


def test_open_store_raises_storeerror_for_bad_path() -> None:
    bogus = Path("/nonexistent/path/not-a-real-store.h5")
    with pytest.raises(store_reader.StoreError):
        store_reader.open_store(bogus)
