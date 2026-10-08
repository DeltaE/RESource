"""Standalone, dependency-light reader for RESource ``.h5`` result stores.

This module intentionally does NOT import the ``RESource`` package. The
scenario ``.h5`` files produced by the full pipeline are plain
``pandas.HDFStore`` files, so reading them back only needs pandas/PyTables
(+ geopandas/shapely if you want geometry columns as real geometries instead
of WKT strings). That keeps the install for "just look at the results"
users tiny compared to the full pipeline environment (no atlite, cdsapi,
cfgrib, rioxarray, osmnx, pygadm, ...).

See ``explore_store.ipynb`` at the repo root for a worked example, and the
"Just want to explore a results store?" section of README.md for the
install instructions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import geopandas as gpd
    from shapely import wkt as _wkt

    _HAS_GEO = True
except ImportError:  # pragma: no cover - geopandas/shapely are optional
    _HAS_GEO = False

_WKT_PREFIXES = (
    "POINT",
    "LINESTRING",
    "POLYGON",
    "MULTIPOINT",
    "MULTILINESTRING",
    "MULTIPOLYGON",
    "GEOMETRYCOLLECTION",
)


def find_stores(root: str | Path = "data/store") -> list[Path]:
    """Recursively list ``.h5`` store files under ``root``, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)


def _looks_like_wkt(series: pd.Series) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    value = sample.iloc[0]
    return isinstance(value, str) and value.strip().upper().startswith(_WKT_PREFIXES)


class StoreViewer:
    """Read-only helper around a single RESource ``.h5`` results store.

    Example:
        >>> sv = StoreViewer("data/store/Canada/BC/resources_Canada_BC_BASELINE_2024_20260804.h5")
        >>> sv.keys()
        >>> cells = sv.load("cells")
    """

    def __init__(self, store_path: str | Path):
        self.path = Path(store_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def keys(self) -> list[str]:
        """List every key stored in the file."""
        with pd.HDFStore(self.path, mode="r") as store:
            return sorted(store.keys())

    def tree(self) -> None:
        """Print keys grouped with the shape of the object at each key."""
        with pd.HDFStore(self.path, mode="r") as store:
            for key in sorted(store.keys()):
                obj = store.get(key)
                shape = getattr(obj, "shape", None)
                print(f"{key:<40} {type(obj).__name__:<12} {shape}")

    def load(self, key: str, decode_geometry: bool = True) -> pd.DataFrame:
        """Load a key as a DataFrame, decoding WKT geometry columns if possible.

        Args:
            key: Store key, e.g. ``"cells"`` or ``"clusters/solar"`` (with or
                without the leading slash).
            decode_geometry: If True and geopandas/shapely are installed,
                any column that looks like WKT text is converted into a real
                geometry column, and the frame is returned as a GeoDataFrame
                (using the first such column found).

        Returns:
            The stored DataFrame, or a GeoDataFrame if geometry was decoded.
        """
        key = key if key.startswith("/") else f"/{key}"
        with pd.HDFStore(self.path, mode="r") as store:
            if key not in store:
                raise KeyError(f"'{key}' not found. Available keys: {sorted(store.keys())}")
            df = store.get(key)

        if not decode_geometry or not _HAS_GEO:
            return df

        geom_col = None
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                if _looks_like_wkt(df[col]):
                    df[col] = df[col].apply(lambda v: _wkt.loads(v) if isinstance(v, str) else v)
                    geom_col = geom_col or col

        if geom_col is not None:
            return gpd.GeoDataFrame(df, geometry=geom_col, crs="EPSG:4326")
        return df

    def load_all(self, decode_geometry: bool = True) -> dict[str, pd.DataFrame]:
        """Load every key in the store into a ``{key: DataFrame}`` dict."""
        return {key.lstrip("/"): self.load(key, decode_geometry) for key in self.keys()}

    def __repr__(self) -> str:
        return f"StoreViewer({self.path})"
