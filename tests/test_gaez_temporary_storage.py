"""Tests for the GAEZ source, processed-output, and temporary lifecycles."""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from RESource import utility as utils
from RESource.gaez import GAEZRasterProcessor


def _processor(tmp_path: Path) -> GAEZRasterProcessor:
    processor = object.__new__(GAEZRasterProcessor)
    processor.gaez_root = tmp_path / "GAEZ"
    processor.gaez_root.mkdir(parents=True)
    processor.processed_region_path = tmp_path / "processed_data/GAEZ/BC"
    processor.processed_region_path.mkdir(parents=True)
    processor._legacy_output_layout = False
    processor.zip_file = Path("LR.zip")
    processor.archive_path = processor.gaez_root / processor.zip_file
    processor.region_short_code = "BC"
    processor.raster_types = [
        {
            "name": "terrain_resources",
            "raster": "slpmed05.tif",
            "zip_extract_direct": "LR/ter",
        }
    ]
    processor.gadmBoundary = SimpleNamespace(
        get_region_boundary=lambda: SimpleNamespace(geometry=[])
    )
    return processor


def test_existing_clip_skips_temporary_download(tmp_path: Path) -> None:
    """A durable regional clip prevents another archive download."""
    processor = _processor(tmp_path)
    clipped = processor._clipped_raster_path(processor.raster_types[0])
    clipped.parent.mkdir(parents=True, exist_ok=True)
    clipped.touch()

    def unexpected_download(_archive_path: Path) -> None:
        raise AssertionError("existing regional clips must skip the GAEZ download")

    processor.__download_resources_zip_file__ = unexpected_download

    assert processor.process_all_rasters()["terrain_resources"] == clipped


def test_archive_persists_but_extracted_workspace_is_removed(tmp_path: Path, monkeypatch) -> None:
    """Downloaded sources persist while extracted global inputs are temporary."""
    processor = _processor(tmp_path)
    monkeypatch.chdir(tmp_path)
    staged_paths: list[Path] = []

    def download(archive_path: Path) -> None:
        archive_path.touch()
        staged_paths.append(archive_path)

    def extract(archive_path: Path, extraction_root: Path, raster_types: list[dict]) -> None:
        assert archive_path.is_file()
        source = extraction_root / raster_types[0]["zip_extract_direct"] / raster_types[0]["raster"]
        source.parent.mkdir(parents=True)
        source.touch()
        staged_paths.append(source)

    def clip(raster_type: dict, _geometry: list, _show: bool, *, input_root: Path) -> Path:
        assert (input_root / raster_type["zip_extract_direct"] / raster_type["raster"]).is_file()
        output = processor._clipped_raster_path(raster_type)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        return output

    processor.__download_resources_zip_file__ = download
    processor.__extract_rasters__ = extract
    processor.__clip_to_boundary_n_plot__ = clip

    output = processor.process_all_rasters()["terrain_resources"]

    assert output.is_file()
    assert staged_paths[0] == processor.archive_path
    assert staged_paths[0].is_file()
    assert all(not path.exists() for path in staged_paths[1:])
    assert (tmp_path / "data/tmp/resource-gaez").is_dir()


def test_archive_members_are_validated_before_extraction(tmp_path: Path) -> None:
    """A bad config cannot leave a partially extracted temporary workspace."""
    processor = _processor(tmp_path)
    archive = tmp_path / "LR.zip"
    extraction_root = tmp_path / "extracted"
    raster_types = [
        {
            "name": "exclusion_areas",
            "raster": "exclusion_2017.tif",
            "zip_extract_direct": "LR/excl",
        },
        {
            "name": "terrain_resources",
            "raster": "missing.tif",
            "zip_extract_direct": "LR/ter",
        },
    ]
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("LR/excl/exclusion_2017.tif", b"raster")

    with pytest.raises(FileNotFoundError, match=r"LR/ter/missing\.tif"):
        processor.__extract_rasters__(archive, extraction_root, raster_types)

    assert not extraction_root.exists()


def test_active_configs_separate_gaez_source_and_processed_roots() -> None:
    """Direct downloads and derived regional rasters use distinct lifecycles."""
    for config_path in Path("config").glob("*.yaml"):
        config = utils.load_config(config_path)
        gaez = config.get("GAEZ")
        if not gaez:
            continue
        assert gaez["root"] == "data/downloaded_data/GAEZ"
        assert gaez["processed_root"] == "data/processed_data/GAEZ"
        assert "Rasters_in_use_direct" not in gaez
