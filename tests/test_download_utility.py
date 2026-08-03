"""Tests for configuration-driven file downloads."""

from pathlib import Path

from RESource import utility as utils


class _FakeResponse:
    """Minimal streaming requests response."""

    headers = {"content-length": "11"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):
        assert chunk_size > 0
        yield b"landcover-"
        yield b"x"


def test_missing_file_is_streamed_to_configured_location(tmp_path: Path, monkeypatch) -> None:
    """A missing configured file is atomically installed at its destination."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils.requests, "get", lambda *_args, **_kwargs: _FakeResponse())
    destination = tmp_path / "data/downloaded/landcover.tif"

    result = utils.fetch_file_if_missing(
        "https://example.test/landcover.tif",
        destination,
        description="land cover",
    )

    assert result == destination
    assert destination.read_bytes() == b"landcover-x"
    assert not list((tmp_path / "data/tmp/resource-downloads").glob("run-*"))


def test_existing_file_skips_remote_request(tmp_path: Path, monkeypatch) -> None:
    """A non-empty existing dataset is reused without contacting its source."""
    destination = tmp_path / "landcover.tif"
    destination.write_bytes(b"existing")

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("existing files must not be downloaded again")

    monkeypatch.setattr(utils.requests, "get", unexpected_request)

    assert utils.fetch_file_if_missing("https://example.test/file.tif", destination) == destination


def test_active_custom_land_rasters_use_standard_download_root() -> None:
    """Active configs store custom layers outside legacy provider directories."""
    expected_prefix = "data/downloaded_data/custom_land_layers/"
    configured_roots = []
    for config_path in Path("config").glob("*.yaml"):
        config = utils.load_config(config_path)
        rasters = config.get("custom_land_layers", {}).get("rasters", []) or []
        raster_entries = rasters.values() if isinstance(rasters, dict) else rasters
        configured_roots.extend(
            raster["root"]
            for raster in raster_entries
            if isinstance(raster, dict) and raster.get("root")
        )

    assert configured_roots
    assert all(root.startswith(expected_prefix) for root in configured_roots)


def test_active_canadian_configs_nest_government_custom_layers() -> None:
    """Government datasets live below custom_land_layers in active configs."""
    for config_path in Path("config").glob("*.yaml"):
        config = utils.load_config(config_path)
        if config.get("country") != "Canada":
            continue
        assert "Gov" not in config
        custom_layers = config.get("custom_land_layers", {})
        government = custom_layers.get("Gov", {})
        assert "conservation_lands" not in government
        for dataset in government.values():
            root = dataset.get("root")
            if root:
                assert root.startswith("data/downloaded_data/custom_land_layers/")
        conservation = [
            vector
            for vector in custom_layers.get("vectors", [])
            if vector.get("name") == "conservation_lands"
        ]
        assert len(conservation) == 1
        assert conservation[0]["provider"] == "government_conservation"
        assert conservation[0]["root"].startswith("data/downloaded_data/custom_land_layers/")
