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
        ("baseline.yaml", "6b60ac5a95a69e10c36383fcd4a88ee159229879300551fa04315d8ce5c9ca84"),
        (
            "grid_restricted.yaml",
            "adf4fd35d5380fae42569fd608a8d0dcfee3b9c25df10a7a9fc8e91b7807278f",
        ),
        (
            "no_buffers.yaml",
            "64ad6626f10ed0192c852a0440ecabd3402cb13a3a77db510456d9297433f1f0",
        ),
        ("policy_1.yaml", "dabbdefc9af5015ed0963593893506ef113fbd69b15efa9678e9101c58236308"),
        (
            "bc_baseline_2020.yaml",
            "fb820e9855eb4904ba2c524d3e0cbc43a1c5267d018c74e327d86c538aadcd88",
        ),
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
            "a66e5526d7ff354e251e4c552fe64c56fafb37db038cac0a79b5b90845310000",
        ),
        (
            "config/WB6/scenarios/legacy_2023.yaml",
            "527b1e0fa569d571b75a4745695162438cc3ab902f57a46454f5e1e4be365379",
        ),
        (
            "config/BGD/scenarios/baseline.yaml",
            "c74b2fe6bd799c347052e8d3754adf188261d42e27c840f385f55f4e5d6b98f1",
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
