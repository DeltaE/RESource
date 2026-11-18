"""Record capacity haircut assumptions used by regional visualizations."""

import math
import os
import shutil
import tempfile
from numbers import Real

import pandas as pd

from RESource.hdf5_handler import DataHandler


def record_regional_capacity_haircuts(
    stores: dict[str, DataHandler], fallback: dict[str, float]
) -> dict[str, float]:
    """Record plot haircut factors in the regional stores used by a combined plot.

    Existing factors take precedence. Missing factors use the explicit notebook
    fallback, and all regions must agree before any metadata is changed.
    Raw capacity data is not modified.
    """
    if not stores:
        raise ValueError("No regional stores were provided.")

    def validate(value: object, label: str) -> dict[str, float]:
        if not isinstance(value, dict) or set(value) != {"solar", "wind"}:
            raise ValueError(f"{label} must contain solar and wind haircut factors.")
        for factor in value.values():
            if (
                isinstance(factor, bool)
                or not isinstance(factor, Real)
                or not math.isfinite(factor)
                or factor < 0
            ):
                raise ValueError(f"{label} haircut factors must be finite and nonnegative.")
        return dict(value)

    fallback = validate(fallback, "Fallback")
    resolved = {}
    existing_metadata = {}
    for region, handler in stores.items():
        metadata = handler.get_metadata()
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid metadata in regional store {handler.store}.")
        existing_metadata[region] = metadata
        stored = metadata.get("capacity_haircut")
        resolved[region] = validate(stored if stored is not None else fallback, region)

    factors = next(iter(resolved.values()))
    if any(value != factors for value in resolved.values()):
        raise ValueError(f"Regional haircut factors differ: {resolved}")

    for region, handler in stores.items():
        metadata = existing_metadata[region]
        missing_factors = metadata.get("capacity_haircut") is None
        updates = {}
        if missing_factors:
            updates["capacity_haircut"] = factors
        if "capacity_haircut_source" not in metadata:
            updates["capacity_haircut_source"] = (
                "WB6 notebook fallback" if missing_factors else "existing store metadata"
            )
        if "capacity_haircut_notes" not in metadata:
            updates["capacity_haircut_notes"] = (
                "Plotting factors for permitting and acceptance constraints; "
                "stored potential capacity columns are not adjusted."
            )
        if updates:
            handler.set_metadata(updates)
        source = metadata.get("capacity_haircut_source", updates.get("capacity_haircut_source"))
        print(f"{region}: capacity haircut {factors} ({source}); store: {handler.store}")
    return factors


def apply_capacity_haircuts(handler: DataHandler, factors: dict[str, float]) -> None:
    """Store separate adjusted cell and cluster capacities without changing originals.

    The derived tables retain the original values and index for validation.
    Updates are staged in a copy and replace the store only after validation.
    Existing, inconsistent derived tables are not overwritten.
    """
    path = handler.store
    if not path.is_file():
        raise FileNotFoundError(path)
    if set(factors) != {"solar", "wind"} or any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v <= 1
        for v in factors.values()
    ):
        raise ValueError("Solar and wind haircut factors must be finite values between 0 and 1.")
    if handler.get_metadata("capacity_haircut") != factors:
        raise ValueError(f"Recorded haircut factors do not match the requested factors in {path}.")

    derived = {}
    original_stat = path.stat()
    with pd.HDFStore(path, mode="r") as store:
        for key, columns in (
            ("cells", {tech: f"potential_capacity_{tech}" for tech in factors}),
            *((f"clusters/{tech}", {tech: "potential_capacity"}) for tech in factors),
        ):
            if f"/{key}" not in store:
                raise ValueError(f"Required table {key!r} is missing from {path}.")
            source = store[key]
            if not source.index.is_unique:
                raise ValueError(f"Table {key!r} has duplicate indexes in {path}.")
            result = pd.DataFrame(index=source.index)
            for tech, column in columns.items():
                if column not in source or not pd.api.types.is_numeric_dtype(source[column]):
                    raise ValueError(f"Table {key!r} lacks numeric {column!r} in {path}.")
                result[f"{column}_original"] = source[column]
                result[f"{column}_adjusted"] = source[column] * factors[tech]
            target = f"capacity_haircuts/{key}"
            if f"/{target}" in store:
                previous = store[target]
                if not previous.equals(result):
                    raise ValueError(f"Existing derived table {target!r} conflicts in {path}.")
            derived[target] = result
        existing = {key for key in derived if f"/{key}" in store}

    if (
        len(existing) == len(derived)
        and handler.get_metadata("capacity_haircut_applied") == factors
    ):
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        shutil.copy2(path, temporary)
        with pd.HDFStore(temporary, mode="a") as store:
            for key, frame in derived.items():
                if key not in existing:
                    store.put(key, frame, format="fixed")
        temp_handler = DataHandler(temporary)
        temp_handler.set_metadata(
            {
                "capacity_haircut_applied": factors,
                "capacity_haircut_applied_notes": (
                    "Adjusted capacities are stored separately under capacity_haircuts; "
                    "original cell and cluster capacity columns remain unchanged. "
                    "LCOE and capacity-factor profiles are not recalculated."
                ),
            }
        )
        current_stat = path.stat()
        if (current_stat.st_mtime_ns, current_stat.st_size) != (
            original_stat.st_mtime_ns,
            original_stat.st_size,
        ):
            raise RuntimeError(f"Store changed while preparing adjusted capacities: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
