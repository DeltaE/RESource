"""Record capacity haircut assumptions used by regional visualizations."""

import math
from numbers import Real

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
