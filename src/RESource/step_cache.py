"""
step_cache.py
─────────────
Lightweight config-hash cache for RESource pipeline steps.

Each pipeline step depends on a specific subset of config keys. This module
hashes those subsets and persists the hashes to a JSON sidecar file next to
the HDF5 store. A step is skipped when:
    1. Its config hash matches the stored hash (config unchanged), AND
    2. Its output key already exists in the HDF5 store (data present).

If either condition fails the step runs normally and the hash is updated.

Usage in RESources_builder.build()
───────────────────────────────────
    cache = StepCache(store_path=self.store, config=self.config,
                      resource_type=self.resource_type)

    if cache.is_current("grid_cells", "cells"):
        utils.print_update(level=2, message="Step 1: grid cells unchanged — skipping.")
    else:
        self.get_grid_cells()
        cache.mark_done("grid_cells")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# ── Step → config-key mapping ─────────────────────────────────────────────────
# Each entry lists the top-level config keys whose values collectively define
# whether that step's output is still valid.  Add or remove keys here as the
# pipeline evolves — no other file needs to change.

STEP_CONFIG_KEYS: dict[str, list[str]] = {
    "grid_cells": [
        "grid_cell_resolution",
        "region_mapping",
        "admin_boundary",  # GADM
        "default_CRS",
    ],
    "grid_nodes": [
        "infrastructure",  # OSM, transmission
        "region_mapping",
    ],
    "cell_capacity": [
        "custom_land_layers",
        "lands",  # CORINE, GAEZ
        "technology",  # resource_specs sub-key
        "filters",  # vector_buffers sub-key
        "region_mapping",
    ],
    "weather_data": [
        "weather",  # cutout
        "weather_year",
        "region_mapping",
    ],
    "gwa_scaling": [
        "weather",  # GWA
        "region_mapping",
    ],
    "cf_timeseries": [
        "weather",  # cutout
        "technology",  # resource_specs.turbines sub-key
        "weather_year",
    ],
    "scoring": [
        "economic_parameters",
        "technology",  # annual_technology_baseline, resource_specs
        "infrastructure",  # transmission
    ],
    "clustering": [
        "technology",  # resource_specs sub-key
        "region_mapping",
    ],
}


class StepCache:
    """
    Config-hash guard for pipeline steps.

    Parameters
    ----------
    store_path : Path
        Path to the HDF5 store file.  The JSON sidecar is written next to it.
    config : dict
        The full config dict loaded by AttributesParser.
    resource_type : str
        'wind' or 'solar' — included in every hash so wind and solar caches
        are independent even when they share the same HDF5 file.
    """

    def __init__(self, store_path: Path, config: dict, resource_type: str):
        self.store_path = Path(store_path)
        self.config = config
        self.resource_type = resource_type
        self._cache_path = self.store_path.with_suffix(".checksums.json")
        self._hashes: dict[str, str] = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_current(self, step: str, store_key: str | None = None) -> bool:
        """
        Return True if the step can be safely skipped.

        A step is current when:
        - Its config hash matches the persisted hash (config unchanged), AND
        - `store_key` is None OR the key exists in the HDF5 store.

        Parameters
        ----------
        step : str
            Step name — must be a key in STEP_CONFIG_KEYS.
        store_key : str, optional
            HDF5 dataset key to check for existence (e.g. 'cells').
            Pass None to skip the data-presence check.

        Returns
        -------
        bool
        """
        if step not in STEP_CONFIG_KEYS:
            return False  # unknown step → always run

        current_hash = self._compute_hash(step)
        stored_hash = self._hashes.get(self._key(step))

        if current_hash != stored_hash:
            return False  # config changed → must rerun

        if store_key is not None and not self._key_in_store(store_key):
            return False  # data missing even though config is the same → rerun

        return True

    def mark_done(self, step: str) -> None:
        """Persist the current config hash for *step* after it has run."""
        self._hashes[self._key(step)] = self._compute_hash(step)
        self._save()

    def invalidate(self, step: str) -> None:
        """Force *step* to rerun on the next call to is_current()."""
        self._hashes.pop(self._key(step), None)
        self._save()

    def invalidate_all(self) -> None:
        """Clear all stored hashes (forces a full pipeline rerun)."""
        self._hashes.clear()
        self._save()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _key(self, step: str) -> str:
        """Namespace step key by resource type so wind/solar don't collide."""
        return f"{self.resource_type}::{step}"

    @staticmethod
    def _stringify_keys(obj: Any) -> Any:
        """Recursively convert all dict keys to str so json.dumps(sort_keys=True) never compares mixed types."""
        if isinstance(obj, dict):
            return {str(k): StepCache._stringify_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [StepCache._stringify_keys(v) for v in obj]
        return obj

    def _compute_hash(self, step: str) -> str:
        """SHA-256 of the JSON-serialised config subset for *step*."""
        keys = STEP_CONFIG_KEYS[step]
        subset = {k: self.config.get(k) for k in keys}
        # Include resource_type so a config change to turbines only invalidates
        # the affected resource, not both wind and solar.
        subset["__resource_type__"] = self.resource_type
        payload = json.dumps(self._stringify_keys(subset), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _key_in_store(self, store_key: str) -> bool:
        """Return True if *store_key* exists as a dataset in the HDF5 file."""
        if not self.store_path.exists():
            return False
        try:
            import h5py

            with h5py.File(self.store_path, "r") as f:
                return store_key in f
        except Exception:
            return False

    def _load(self) -> dict[str, str]:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._hashes, indent=2, sort_keys=True),
            encoding="utf-8",
        )
