"""Tests for public Google Drive file downloads."""

from pathlib import Path

import pytest

from RESource.utility import get_gdrive_public_file


def test_downloads_file_and_creates_parent_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "downloads" / "weather.nc"
    calls = []

    def download(url: str, output: str, *, quiet: bool) -> str:
        calls.append((url, output, quiet))
        Path(output).write_bytes(b"weather")
        return output

    monkeypatch.setattr("gdown.download", download)

    get_gdrive_public_file("drive-file-id", destination)

    assert calls == [
        ("https://drive.google.com/uc?id=drive-file-id", str(destination), False)
    ]
    assert destination.read_bytes() == b"weather"


def test_rejects_empty_file_id_before_creating_parent(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "downloads" / "weather.nc"

    with pytest.raises(ValueError, match="non-empty string"):
        get_gdrive_public_file("  ", destination)

    assert not destination.parent.exists()


def test_raises_when_gdown_reports_download_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("gdown.download", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="Google Drive download failed"):
        get_gdrive_public_file("drive-file-id", tmp_path / "weather.nc")


def test_existing_destination_skips_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "weather.nc"
    destination.write_bytes(b"existing")

    def unexpected_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("existing files must not be downloaded again")

    monkeypatch.setattr("gdown.download", unexpected_download)

    get_gdrive_public_file("drive-file-id", destination)

    assert destination.read_bytes() == b"existing"
