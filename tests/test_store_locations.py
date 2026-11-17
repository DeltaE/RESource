from pathlib import Path

from RESource.AttributesParser import AttributesParser
from RESource.store_viewer import find_stores


def test_pipeline_store_is_shared_by_technologies_in_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "country": "Test Country",
        "Scenario": {"run_id": "baseline"},
        "region_mapping": {"AA": {}},
    }
    monkeypatch.setattr("RESource.AttributesParser.utils.resolve_config", lambda _: (config, {}))
    legacy = tmp_path / "data/store/TestCountry/AA/old.h5"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"existing")

    wind = AttributesParser("config.yaml", "aa", "wind", 2025)
    solar = AttributesParser("config.yaml", "AA", "solar", 2025)
    assert wind.store == solar.store
    assert wind.store.parent == wind.results_save_to
    assert wind.store.name == f"resources_TestCountry_AA_{wind.RUN_ID}.h5"
    assert not wind.store.exists()
    assert legacy.read_bytes() == b"existing"


def test_find_stores_searches_results_and_legacy_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    new = Path("results/Test/AA/run/new.h5")
    legacy = Path("data/store/Test/AA/old.h5")
    for path in (new, legacy):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert set(find_stores()) == {new, legacy}
    assert find_stores("data/store") == [legacy]
