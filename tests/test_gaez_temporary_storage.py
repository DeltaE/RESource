"""Tests for the GAEZ source, processed-output, and temporary lifecycles."""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from RESource import utility as utils
from RESource.gaez import GAEZRasterProcessor, load_gaez_raster


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


def test_direct_and_archive_rasters_are_clipped_at_source_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    processor = _processor(tmp_path)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.tif"
    transform = from_origin(0, 2 / 120, 1 / 120, 1 / 120)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        nodata=0,
        transform=transform,
    ) as dst:
        dst.write(np.array([[1, 7], [8, 9]], dtype="uint8"), 1)

    processor.raster_types = [
        {"name": "exclusion", "raster": "exclusion.tif", "zip_extract_direct": "LR/excl"},
        {
            "name": "slope",
            "raster": "slpmed30s.tif",
            "zip_extract_direct": "LR/ter",
            "source": "https://example.org/slpmed30s.tif",
        },
    ]
    with ZipFile(processor.archive_path, "w") as archive:
        archive.write(source, "LR/excl/exclusion.tif")
    processor.gadmBoundary = SimpleNamespace(
        get_region_boundary=lambda: SimpleNamespace(geometry=[box(0, 0, 2 / 120, 2 / 120)])
    )
    calls = []

    def fetch(url, destination, *, description):
        calls.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(utils, "fetch_file_if_missing", fetch)
    processor.__download_resources_zip_file__ = lambda path: None
    old_clip = processor.processed_region_path / "slpmed05.tif"
    old_clip.touch()

    outputs = processor.process_all_rasters()

    assert len(calls) == 1
    assert calls[0][1] == processor.gaez_root / "LR/ter/slpmed30s.tif"
    assert old_clip.exists()
    for path in outputs.values():
        with rasterio.open(path) as clipped:
            assert clipped.res == (1 / 120, 1 / 120)
            np.testing.assert_array_equal(clipped.read(1), [[1, 7], [8, 9]])
    assert processor.process_all_rasters() == outputs
    assert len(calls) == 1


def test_direct_only_does_not_download_archive(tmp_path: Path, monkeypatch) -> None:
    processor = _processor(tmp_path)
    monkeypatch.chdir(tmp_path)
    raster = processor.raster_types[0]
    raster["source"] = "https://example.org/slpmed30s.tif"

    def unexpected_archive(_path):
        raise AssertionError("direct sources must not download the archive")

    def fetch(_url, destination, *, description):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()

    def clip(item, _geometry, _show, *, input_root):
        assert (input_root / item["zip_extract_direct"] / item["raster"]).exists()
        return processor._clipped_raster_path(item)

    monkeypatch.setattr(utils, "fetch_file_if_missing", fetch)
    processor.__download_resources_zip_file__ = unexpected_archive
    processor.__clip_to_boundary_n_plot__ = clip
    assert processor.process_all_rasters()["terrain_resources"] == processor._clipped_raster_path(
        raster
    )


def test_wb6_scenarios_default_to_fao_30_second_slope() -> None:
    for scenario in Path("config/WB6/scenarios").glob("*.yaml"):
        config = utils.load_config(scenario)
        slope = next(
            item
            for item in config["lands"]["GAEZ"]["raster_types"]
            if item["name"] == "terrain_resources"
        )
        assert slope["raster"] == "slpmed30s.tif"
        assert slope["source"].endswith("/LR/ter/slpmed30s.tif")
        assert slope["class_exclusion"]["solar"] == [8, 9]
        assert slope["class_exclusion"]["wind"] == [7, 8, 9]


@pytest.mark.parametrize("nodata", [None, 99])
def test_load_archive_raster_clips_all_boundaries_and_masks_outside(
    tmp_path: Path, nodata: int | None
) -> None:
    """Source values from multiple regions survive, including a valid zero."""
    raster = tmp_path / "source.tif"
    values = np.array([[0, 99, 2, 3], [4, 5, 6, 7]], dtype="uint8")
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=4,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        nodata=nodata,
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(values, 1)
    archive = tmp_path / "LR.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.write(raster, "LR/excl/source.tif")
    boundary = gpd.GeoDataFrame(
        geometry=[box(0, 1, 2, 2), box(3, 0, 4, 1)], crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    result = load_gaez_raster(archive, "LR/excl/source.tif", boundary)

    np.testing.assert_equal(
        result.values,
        [[0, np.nan if nodata is not None else 99, np.nan, np.nan], [np.nan, np.nan, np.nan, 7]],
    )
    assert str(result.rio.crs) == "EPSG:4326"
    assert result.rio.transform() == from_origin(0, 2, 1, 1)
    assert np.isnan(result.rio.nodata)


def test_load_archive_raster_reports_missing_member_and_boundary_crs(tmp_path: Path) -> None:
    archive = tmp_path / "LR.zip"
    with ZipFile(archive, "w"):
        pass
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(FileNotFoundError, match="missing.tif"):
        load_gaez_raster(archive, "LR/ter/missing.tif", boundary)
    with pytest.raises(ValueError, match="boundary must have a CRS"):
        load_gaez_raster(archive, "LR/ter/missing.tif", boundary.set_crs(None, allow_override=True))


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
