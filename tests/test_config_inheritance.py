"""Tests for deterministic YAML configuration inheritance."""

import hashlib
import json
from pathlib import Path

import pytest

from RESource.utility import load_config, resolve_config


def test_inherited_config_deep_merges_and_replaces_lists(tmp_path: Path) -> None:
    """Mappings merge recursively while lists are intentional replacements."""
    base = tmp_path / "base.yaml"
    base.write_text(
        "region_mapping:\n  BC:\n    timezone: PST\n    tags: [base]\n"
        "Scenario:\n  run_id: base\n  Description: baseline\n"
        "custom_land_layers:\n  vectors: [conservation]\n",
        encoding="utf-8",
    )
    scenario = tmp_path / "scenarios/policy.yaml"
    scenario.parent.mkdir()
    scenario.write_text(
        "extends: ../base.yaml\n"
        "region_mapping:\n  BC:\n    tags: [policy]\n"
        "Scenario:\n  run_id: policy\n"
        "custom_land_layers:\n  vectors:\n    $append: [agriculture]\n",
        encoding="utf-8",
    )

    resolved, provenance = resolve_config(scenario)

    assert resolved["region_mapping"]["BC"] == {
        "timezone": "PST",
        "tags": ["policy"],
    }
    assert resolved["Scenario"] == {
        "run_id": "policy",
        "Description": "baseline",
    }
    assert resolved["custom_land_layers"]["vectors"] == ["conservation", "agriculture"]
    assert provenance["override_paths"] == [
        "region_mapping.BC.tags",
        "Scenario.run_id",
        "custom_land_layers.vectors.$append",
    ]
    assert len(provenance["sources"]) == 2
    assert load_config(scenario) == resolved


def test_inheritance_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    """A scenario cannot silently introduce a new configuration section."""
    base = tmp_path / "base.yaml"
    base.write_text("Scenario:\n  run_id: base\n", encoding="utf-8")
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("extends: base.yaml\nScenairo:\n  run_id: typo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Scenairo"):
        load_config(scenario)


def test_inheritance_rejects_cycles(tmp_path: Path) -> None:
    """Configuration inheritance cycles report the involved files."""
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular configuration inheritance"):
        load_config(first)


@pytest.mark.parametrize(
    ("scenario", "expected_hash"),
    [
        ("baseline.yaml", "4afb1470f918d5e47b1a50f8d8ac1781eace06c5f7d3823b888e06caf5dcd15a"),
        (
            "grid_restricted.yaml",
            "310a1fed13510126022f5d7b4de218289702a4ff6c1974a67e984fd7b411bb5e",
        ),
        (
            "no_buffers.yaml",
            "504911d25b4a2257b9554894f14d3583250ab2b6d4cfc3d82487e9b940e6fcd6",
        ),
        ("policy_1.yaml", "868a1780f9f97fa2b5b0703a8184296e4a34153f4a7c9db643562e8d374b2ccc"),
    ],
)
def test_can_scenarios_preserve_legacy_resolved_contract(scenario: str, expected_hash: str) -> None:
    """Inherited Canadian scenarios remain equivalent to reviewed full configs."""
    config = load_config(Path("config/CAN/scenarios") / scenario)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash


@pytest.mark.parametrize(
    ("scenario", "expected_hash"),
    [
        (
            "config/WB6/scenarios/baseline.yaml",
            "269f398a8414d0f96a6d26778bf0c66158434511cb0f203b1d7e6b8b973ca1f2",
        ),
        (
            "config/WB6/scenarios/legacy_2023.yaml",
            "5c91f882576c134bd967be1abc62ad174c5a34a6488527e0fd8f2beab498f44c",
        ),
        (
            "config/BGD/scenarios/baseline.yaml",
            "16f522144d8694d432c8ec533e982d25ba5a71ead17601f3dd7610a0b8d15643",
        ),
    ],
)
def test_other_regional_scenarios_preserve_legacy_resolved_contract(
    scenario: str, expected_hash: str
) -> None:
    """WB6 and Bangladesh inheritance preserves reviewed full configurations."""
    config = load_config(scenario)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
