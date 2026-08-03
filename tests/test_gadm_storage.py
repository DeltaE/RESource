"""Tests for lazy GADM storage creation."""

from pathlib import Path

from RESource.boundaries import GADMBoundaries


def test_gadm_initialization_does_not_create_storage(monkeypatch, tmp_path: Path) -> None:
    """Resolving GADM paths must not create download or processed directories."""
    download_root = tmp_path / "downloaded_data" / "GADM"
    processed_root = tmp_path / "processed_data" / "regions"

    def initialize_parent(self) -> None:
        self.crs_d = "EPSG:4326"

    monkeypatch.setattr(
        "RESource.boundaries.AttributesParser.__post_init__", initialize_parent
    )
    monkeypatch.setattr(
        "RESource.boundaries.AttributesParser.get_gadm_config",
        lambda _self: {
            "root": str(download_root),
            "processed": str(processed_root),
            "admin_level": 2,
            "datafield_mapping": {
                "NAME_0": "Country",
                "NAME_1": "Province",
                "NAME_2": "Region",
            },
        },
    )
    monkeypatch.setattr(
        "RESource.boundaries.AttributesParser.get_country", lambda _self: "Canada"
    )

    GADMBoundaries("config.yaml", "BC", "solar")

    assert not download_root.exists()
    assert not processed_root.exists()
