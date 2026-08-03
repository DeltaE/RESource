"""Tests for small data assets shipped with RESource."""

from pathlib import Path

import pytest

from RESource.assets import legend_file, mapping_file


@pytest.mark.parametrize(
    "filename",
    [
        "CLC_2018_legend.csv",
        "CPCAD_legends.csv",
        "LandCover_CANgov_2020_legend.csv",
        "gaez_exclusion_legend.csv",
        "gaez_landcover_legend.csv",
        "gaez_terrains_legend.csv",
    ],
)
def test_packaged_legends_exist(filename: str) -> None:
    """Every documented plotting legend is available through package resources."""
    assert legend_file(filename).is_file()


def test_packaged_bc_mapping_exists() -> None:
    """The BC region mapping is shipped separately from plotting legends."""
    assert mapping_file("region_mapping_BC.csv").is_file()


def test_asset_lookup_rejects_paths() -> None:
    """Asset lookups accept basenames and cannot escape the package directory."""
    with pytest.raises(ValueError):
        legend_file("../secret.csv")


def test_configured_legends_exist() -> None:
    """All non-archived workflow configurations reference tracked assets."""
    repository = Path(__file__).resolve().parents[1]
    for config in (repository / "config").glob("*.yaml"):
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("legends:"):
                configured_path = line.split(":", maxsplit=1)[1].strip()
                assert (repository / configured_path).is_file(), config
