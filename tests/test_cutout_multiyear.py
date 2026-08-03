"""Tests for the ERA5-only multiyear cutout pipeline."""

import json
import os
import tempfile
import tomllib
from pathlib import Path

from RESource.cutout_multiyear import expected_hours, write_manifest
from RESource.utility import configure_repository_temp, release_process_memory


def test_cutout_command_is_declared() -> None:
    """The distribution exposes the ERA5-only multiyear command after installation."""
    repository = Path(__file__).resolve().parents[1]
    with (repository / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert (
        pyproject["project"]["scripts"]["resource-cutout-multiyear"]
        == "RESource.cutout_multiyear:entrypoint"
    )


def test_expected_hours_handles_leap_years() -> None:
    """Annual validation distinguishes leap and ordinary years."""
    assert expected_hours(2016) == 8784
    assert expected_hours(2017) == 8760
    assert expected_hours(2020) == 8784


def test_memory_cleanup_is_non_destructive_and_reported() -> None:
    """The CDS preflight exposes garbage-collection counts for the manifest."""
    report = release_process_memory()

    assert report["unreachable_objects_collected"] >= 0
    assert report["generation_0_after"] >= 0


def test_cds_temporary_storage_uses_repository_filesystem(tmp_path) -> None:
    """CDS and Python temporary files share the configured repository scratch path."""
    original_tempdir = tempfile.tempdir
    original_environment = os.environ.get("TMPDIR")
    scratch = tmp_path / "data" / "tmp" / "resource-cds"
    try:
        configured = configure_repository_temp(scratch)

        assert configured == scratch.resolve()
        assert configured.is_dir()
        assert os.environ["TMPDIR"] == str(configured)
        assert Path(tempfile.gettempdir()) == configured
    finally:
        tempfile.tempdir = original_tempdir
        if original_environment is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = original_environment


def test_manifest_is_written_atomically(tmp_path) -> None:
    """A completed manifest replaces its temporary file."""
    path = tmp_path / "manifest.json"
    write_manifest(path, {"all_valid": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"all_valid": True}
    assert not path.with_suffix(".json.tmp").exists()
