"""Download and validate ERA5 cutouts across complete weather years."""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import xarray as xr

from RESource.era5_cutout import ERA5Cutout
from RESource.utility import load_config


def expected_hours(year: int) -> int:
    """Return the number of hourly observations in a calendar year."""
    return 8784 if calendar.isleap(year) else 8760


def validate_cutout(path: Path, year: int) -> dict[str, Any]:
    """Validate an annual ERA5 cutout without loading its arrays into memory."""
    result: dict[str, Any] = {
        "path": str(path),
        "year": year,
        "valid": False,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.is_file():
        result["error"] = "cutout does not exist"
        return result

    try:
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            time = dataset.coords.get("time")
            if time is None:
                raise ValueError("missing time coordinate")
            count = int(time.size)
            index = dataset.indexes.get("time")
            duplicates = int(index.duplicated().sum()) if index is not None else 0
            variables = sorted(dataset.data_vars)
            result.update(
                {
                    "timesteps": count,
                    "expected_timesteps": expected_hours(year),
                    "duplicate_timesteps": duplicates,
                    "time_start": str(time.values[0]) if count else None,
                    "time_end": str(time.values[-1]) if count else None,
                    "variables": variables,
                    "x_cells": int(dataset.sizes.get("x", 0)),
                    "y_cells": int(dataset.sizes.get("y", 0)),
                }
            )
            errors = []
            if count != expected_hours(year):
                errors.append(f"expected {expected_hours(year)} hours, found {count}")
            if duplicates:
                errors.append(f"found {duplicates} duplicate timestamps")
            if "wnd100m" not in variables:
                errors.append("missing wnd100m")
            if not result["x_cells"] or not result["y_cells"]:
                errors.append("empty spatial grid")
            if errors:
                result["error"] = "; ".join(errors)
            else:
                result["valid"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the resumable run manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the ERA5-only multiyear command parser."""
    parser = argparse.ArgumentParser(
        description="Download and validate ERA5 cutouts only, one year at a time."
    )
    parser.add_argument("config", help="RESource YAML configuration")
    parser.add_argument("--start", "-s", type=int, required=True, help="First year, inclusive")
    parser.add_argument("--end", "-e", type=int, required=True, help="Last year, inclusive")
    parser.add_argument("--region", "-r", required=True, help="One configured region code")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path (default: results/manifests/era5_cutouts_*.json)",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Run sequential annual ERA5 downloads and return a process exit status."""
    config_path = Path(args.config)
    if args.start > args.end:
        raise ValueError("--start must not be later than --end")
    config = load_config(config_path)
    region = args.region.upper()
    regions = config.get("region_mapping", {})
    if region not in regions:
        raise ValueError(f"unknown region {region!r}; available: {', '.join(regions)}")

    manifest_path = args.manifest or Path(
        f"results/manifests/era5_cutouts_{region}_{args.start}_{args.end}.json"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "RESource ERA5-only multiyear cutout download",
        "config": str(config_path),
        "region": region,
        "start_year": args.start,
        "end_year": args.end,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "temporary_directory": str(Path("data/tmp/resource-cds").resolve()),
        "years": {},
    }

    for year in range(args.start, args.end + 1):
        print(f"\nERA5 cutout: {region} / {year}", flush=True)
        processor = ERA5Cutout(config_path, region, "wind", weather_year=year)
        cutout_path = processor.get_cutout_path(weather_year=year)
        existing = validate_cutout(cutout_path, year)
        if existing["valid"]:
            existing["status"] = "reused"
            manifest["years"][str(year)] = existing
            write_manifest(manifest_path, manifest)
            print(f"Reusing validated cutout: {cutout_path}", flush=True)
            del processor
            continue

        if cutout_path.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            quarantined = cutout_path.with_suffix(f"{cutout_path.suffix}.invalid-{timestamp}")
            cutout_path.replace(quarantined)
            print(f"Moved invalid cutout to: {quarantined}", flush=True)

        try:
            processor.get_era5_cutout(weather_year=year)
            memory_cleanup = processor.cds_memory_cleanup
            validation = validate_cutout(cutout_path, year)
            validation["status"] = "downloaded" if validation["valid"] else "invalid"
        except Exception as exc:
            memory_cleanup = getattr(processor, "cds_memory_cleanup", None)
            validation = {
                "path": str(cutout_path),
                "year": year,
                "valid": False,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        manifest["years"][str(year)] = validation
        if memory_cleanup is not None:
            validation["preflight_memory_cleanup"] = memory_cleanup
        write_manifest(manifest_path, manifest)
        print(json.dumps(validation, indent=2), flush=True)
        del processor

    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    manifest["all_valid"] = all(item["valid"] for item in manifest["years"].values())
    write_manifest(manifest_path, manifest)
    print(f"\nManifest: {manifest_path}", flush=True)
    return 0 if manifest["all_valid"] else 1


def entrypoint() -> None:
    """CLI entry point for ERA5-only multiyear acquisition."""
    try:
        status = run(build_parser().parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    entrypoint()
