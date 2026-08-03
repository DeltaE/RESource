"""Tests for filename-based access to a mounted Google Drive."""

from pathlib import Path

import pytest

from RESource.google_drive import (
    AmbiguousDriveFileError,
    GoogleDriveFiles,
    GoogleDriveMount,
    GoogleDriveMountError,
)


def test_resolves_unique_filename_recursively(tmp_path: Path) -> None:
    target = tmp_path / "weather" / "era5.nc"
    target.parent.mkdir()
    target.write_bytes(b"netcdf")

    drive = GoogleDriveFiles(tmp_path)

    assert drive.path("era5.nc") == target
    assert drive.open("weather/era5.nc").read() == b"netcdf"


def test_rejects_ambiguous_filename(tmp_path: Path) -> None:
    for folder in ("scenario-a", "scenario-b"):
        path = tmp_path / folder / "results.csv"
        path.parent.mkdir()
        path.write_text(folder, encoding="utf-8")

    drive = GoogleDriveFiles(tmp_path)

    with pytest.raises(AmbiguousDriveFileError, match="scenario-a/results.csv"):
        drive.path("results.csv")
    assert drive.path("scenario-b/results.csv").read_text(encoding="utf-8") == "scenario-b"


def test_rejects_paths_outside_mount(tmp_path: Path) -> None:
    drive = GoogleDriveFiles(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        drive.path("../secret.txt")
    with pytest.raises(ValueError, match="relative"):
        drive.path("/etc/passwd")


def test_missing_mount_has_clear_error(tmp_path: Path) -> None:
    drive = GoogleDriveFiles(tmp_path / "not-mounted")

    with pytest.raises(GoogleDriveMountError, match="unavailable"):
        drive.path("input.csv")


def test_mount_requires_rclone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("RESource.google_drive.shutil.which", lambda _binary: None)
    mount = GoogleDriveMount("gdrive:", tmp_path / "drive")

    with pytest.raises(GoogleDriveMountError, match="rclone was not found"):
        mount.mount()


def test_mount_rejects_nonempty_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mount_point = tmp_path / "drive"
    mount_point.mkdir()
    (mount_point / "local.txt").write_text("keep me", encoding="utf-8")
    monkeypatch.setattr("RESource.google_drive.shutil.which", lambda _binary: "/usr/bin/rclone")
    mount = GoogleDriveMount("gdrive:", mount_point)

    with pytest.raises(GoogleDriveMountError, match="must be empty"):
        mount.mount()
