"""Tests for the RESource.reporting submodule."""

from pathlib import Path

import pytest

from RESource.reporting import build_report, config_section, discovery


def _write_fixture_config(tmp_path: Path) -> Path:
    """Minimal base + scenario config tree mirroring config/<COUNTRY>/ layout."""
    country_dir = tmp_path / "config" / "ZZ"
    (country_dir / "scenarios").mkdir(parents=True)
    (country_dir / "base.yaml").write_text(
        "country: Testland\n"
        "weather_year: 2024\n"
        "Scenario:\n  run_id: BASE\n  Description: base scenario\n"
        "region_mapping:\n  TL:\n    name: Test Land\n",
        encoding="utf-8",
    )
    scenario_path = country_dir / "scenarios" / "baseline.yaml"
    scenario_path.write_text(
        "extends: ../base.yaml\n"
        "Scenario:\n  run_id: BASELINE\n  Description: baseline scenario\n",
        encoding="utf-8",
    )
    return scenario_path


class TestDiscovery:
    def test_find_scenario_runs_picks_latest_run_dir(self, tmp_path, monkeypatch):
        scenario_path = _write_fixture_config(tmp_path)
        monkeypatch.setattr(discovery, "CONFIG_ROOT", tmp_path / "config")
        monkeypatch.setattr(discovery, "RESULTS_ROOT", tmp_path / "results")

        region_results = tmp_path / "results" / "Testland" / "TL"
        for run_dir_name in ["BASELINE_2024_20240101", "BASELINE_2024_20240315", "BASELINE_2024_20240201"]:
            (region_results / run_dir_name).mkdir(parents=True)
        # Unrelated run_id must never be picked.
        (region_results / "OTHER_2024_20240401").mkdir(parents=True)

        runs = discovery.find_scenario_runs("ZZ")

        assert len(runs) == 1
        run = runs[0]
        assert run.scenario_name == "baseline"
        assert run.scenario_config_path == scenario_path
        assert run.run_id == "BASELINE"
        assert run.region_code == "TL"
        assert run.run_dir.name == "BASELINE_2024_20240315"
        assert run.weather_year == 2024

    def test_find_scenario_runs_skips_region_without_a_run(self, tmp_path, monkeypatch):
        _write_fixture_config(tmp_path)
        monkeypatch.setattr(discovery, "CONFIG_ROOT", tmp_path / "config")
        monkeypatch.setattr(discovery, "RESULTS_ROOT", tmp_path / "results")

        runs = discovery.find_scenario_runs("ZZ")

        assert runs == []

    def test_find_scenario_runs_rejects_unknown_scenario_name(self, tmp_path, monkeypatch):
        _write_fixture_config(tmp_path)
        monkeypatch.setattr(discovery, "CONFIG_ROOT", tmp_path / "config")
        monkeypatch.setattr(discovery, "RESULTS_ROOT", tmp_path / "results")

        with pytest.raises(ValueError, match="Unknown scenario"):
            discovery.find_scenario_runs("ZZ", scenario_names=["does-not-exist"])

    def test_list_scenario_configs_missing_country_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(discovery, "CONFIG_ROOT", tmp_path / "config")

        with pytest.raises(FileNotFoundError):
            discovery.list_scenario_configs("NOPE")


class TestConfigSection:
    def test_build_scenario_summary_reports_override_diff(self, tmp_path):
        scenario_path = _write_fixture_config(tmp_path)

        summary = config_section.build_scenario_summary(scenario_path)

        assert summary["run_id"] == "BASELINE"
        assert summary["description"] == "baseline scenario"
        override_paths = {o["path"] for o in summary["overrides"]}
        assert "Scenario.run_id" in override_paths
        assert "Scenario.Description" in override_paths
        run_id_override = next(o for o in summary["overrides"] if o["path"] == "Scenario.run_id")
        assert run_id_override["base_value"] == "BASE"
        assert run_id_override["scenario_value"] == "BASELINE"


@pytest.mark.skipif(
    not Path("results/Canada/BC").is_dir(),
    reason="requires existing results/Canada/BC pipeline outputs on disk",
)
class TestBuildReportSmoke:
    def test_build_report_for_can_bc(self, tmp_path):
        out_path = build_report("CAN", regions=["BC"], out_dir=tmp_path)

        assert out_path.is_file()
        html = out_path.read_text(encoding="utf-8")
        assert out_path.stat().st_size > 10_000
        assert "region-BC" in html
        assert "Supply curve" in html
        assert "baseline" in html
