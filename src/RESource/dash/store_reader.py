"""Pure h5py reader for RESource stores. Runs natively and inside Pyodide.

RESource writes stores with ``pandas.HDFStore`` in the "fixed" format. PyTables has
no WebAssembly build, so this module rebuilds frames from the raw HDF5 layout with
h5py + pickle + numpy only. Reads are lazy: a caller can ask for a subset of
columns, and only the blocks that hold them are touched. That matters for the
per-cell timeseries tables (8760 x thousands of cells).

No dependency on the RESource package. The source is inlined into the dash.
"""

from __future__ import annotations

import pickle
import re

import h5py
import numpy as np
import pandas as pd

_PAREN_RE = re.compile(r"\(([^()]+)\)")
_TRUNC_RE = re.compile(r"eof = (\d+).*stored_eof = (\d+)")


def _dec(values):
    return [v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v) for v in values]


def open_store(path):
    """Open a store read-only. Raises StoreError with a plain reason on failure."""
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        m = _TRUNC_RE.search(str(exc))
        if m:
            have, want = int(m.group(1)), int(m.group(2))
            raise StoreError(
                "truncated",
                f"File is incomplete: {have / 1e6:.0f} MB on disk, header expects {want / 1e6:.0f} MB. "
                "Copy the store again from the source.",
                have=have, want=want,
            ) from exc
        raise StoreError("unreadable", f"Not a readable HDF5 store: {exc}") from exc


class StoreError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


def list_frame_keys(h5file):
    """Every group path that holds a pandas fixed-format frame, with its shape."""
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "axis0" in obj and "axis1" in obj:
            out.append({"key": name, "rows": int(obj["axis1"].shape[0]), "cols": int(obj["axis0"].shape[0])})

    h5file.visititems(visit)
    return out


def frame_columns(h5file, key):
    return _dec(h5file[key]["axis0"][:])


def _blocks(group):
    i = 0
    while f"block{i}_items" in group:
        yield _dec(group[f"block{i}_items"][:]), group[f"block{i}_values"]
        i += 1


def read_fixed_frame(h5file, key, columns=None, rows=None):
    """Rebuild a DataFrame. ``columns`` limits which blocks are read; ``rows`` is a slice."""
    group = h5file[key]
    axis0 = _dec(group["axis0"][:])
    index = group["axis1"][rows] if rows is not None else group["axis1"][:]
    want = None if columns is None else set(columns)
    cols = {}
    for items, raw in _blocks(group):
        if want is not None and not want.intersection(items):
            continue
        if raw.dtype == object:  # pickled object block
            vals = np.asarray(pickle.loads(bytes(raw[0])))
            if rows is not None:
                vals = vals[rows]
        elif want is not None and raw.ndim == 2 and len(want.intersection(items)) < len(items) / 4:
            # few columns wanted from a wide numeric block: slice column by column
            for j, name in enumerate(items):
                if name in want:
                    cols[name] = raw[rows, j] if rows is not None else raw[:, j]
            continue
        else:
            vals = raw[rows] if rows is not None else raw[:]
        if vals.ndim == 1:
            cols[items[0]] = vals
        elif vals.shape[1] == len(items):
            for j, name in enumerate(items):
                if want is None or name in want:
                    cols[name] = vals[:, j]
        else:
            raise ValueError(f"{key}: block shape {vals.shape} does not match {len(items)} items")
    order = [c for c in axis0 if c in cols]
    df = pd.DataFrame({c: cols[c] for c in order})
    df.index = _dec(index) if index.dtype.kind in "SO" else index
    return df


def time_index(index):
    """Store timestamps are int64 nanoseconds. Returned as a DatetimeIndex, as written."""
    arr = np.asarray(index)
    if arr.dtype.kind in "iu":
        return pd.to_datetime(arr.astype("int64"), unit="ns")
    return pd.to_datetime(arr)


# ---------------------------------------------------------------- geometry (WKT)

def _thin(pts, max_pts):
    n = len(pts)
    if n <= max_pts:
        return pts
    idx = np.unique(np.linspace(0, n - 1, max_pts).round().astype(int))
    return pts[idx]


def wkt_parts(wkt, max_pts=400, max_parts=None, ndigits=4):
    """Coordinate parts (rings or linestrings) of a WKT geometry as [[lon, lat], ...] lists.

    Display-grade only: long parts are thinned to ``max_pts`` vertices and, when
    ``max_parts`` is set, only the parts with the most vertices are kept. Exterior
    rings and holes are not distinguished.
    """
    if not isinstance(wkt, str):
        return []
    parts = []
    for chunk in _PAREN_RE.findall(wkt):
        arr = np.array(chunk.replace(",", " ").split(), dtype=float)
        dim = 3 if (" Z" in wkt[:24].upper() and arr.size % 3 == 0) else 2
        if arr.size < 2 * dim and not wkt.upper().startswith("POINT"):
            continue
        if arr.size % dim:
            continue
        parts.append(arr.reshape(-1, dim)[:, :2])
    if max_parts and len(parts) > max_parts:
        parts = sorted(parts, key=len, reverse=True)[:max_parts]
    return [np.round(_thin(p, max_pts), ndigits).tolist() for p in parts]


def wkt_kind(wkt):
    return wkt.split("(", 1)[0].strip().upper() if isinstance(wkt, str) else ""
