"""Tests for temporary Global Wind Atlas source storage."""

from pathlib import Path
from types import SimpleNamespace

from RESource.gwa import GWACells


class _FakeRaster:
    """Minimal rioxarray-like raster used to test the storage lifecycle."""

    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.rio = self

    def clip(self, _geometry, _crs, *, drop: bool):
        assert drop is True
        assert self.source_path.is_file()
        return self

    def to_raster(self, destination: Path, *, driver: str, compress: str):
        assert driver == "GTiff"
        assert compress == "deflate"
        destination.touch()

    def close(self):
        return None


def test_country_raster_is_temporary_and_regional_clip_persists(
    monkeypatch, tmp_path: Path
) -> None:
    """Only the region-clipped GWA raster survives processing."""
    processor = object.__new__(GWACells)
    monkeypatch.chdir(tmp_path)
    processor.gwa_root = tmp_path / "GWA"
    processor.region_short_code = "BC"
    staged_sources: list[Path] = []

    def download(_url: str, destination: Path) -> None:
        destination.touch()
        staged_sources.append(destination)

    processor.download_file = download
    monkeypatch.setattr(
        "RESource.gwa.rxr.open_rasterio",
        lambda source, masked: _FakeRaster(source),
    )
    boundary = SimpleNamespace(geometry=[], crs="EPSG:4326")
    output = processor._regional_raster_path("CAN_wspd_100m.tif")

    processor._create_regional_raster(
        "https://example.test/CAN.tif",
        "CAN_wspd_100m.tif",
        output,
        boundary,
    )

    assert output.is_file()
    assert all(not source.exists() for source in staged_sources)
    assert not (processor.gwa_root / "CAN_wspd_100m.tif").exists()
    assert (tmp_path / "data/tmp/resource-gwa").is_dir()
