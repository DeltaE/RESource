"""Resolved config + override-diff + provenance for the report's config section."""

from __future__ import annotations

from pathlib import Path

from RESource import utility as utils


def _get_by_path(config: dict, dotted_path: str):
    """Read a dotted config path, ignoring the synthetic '$append' leaf marker."""
    node = config
    for part in dotted_path.split("."):
        if part == "$append":
            return node
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def build_scenario_summary(scenario_config_path: Path) -> dict:
    """Resolve a scenario config and diff it against its base config.

    Reuses ``RESource.utility.resolve_config`` so the diff is derived from the
    same inheritance/provenance logic the pipeline itself uses, rather than a
    parallel implementation.
    """
    resolved, provenance = utils.resolve_config(scenario_config_path)
    base_path = Path(provenance["sources"][0]["path"])
    base_resolved, _base_provenance = utils.resolve_config(base_path)

    overrides = [
        {
            "path": dotted_path,
            "base_value": _get_by_path(base_resolved, dotted_path),
            "scenario_value": _get_by_path(resolved, dotted_path),
        }
        for dotted_path in provenance["override_paths"]
    ]

    scenario_block = resolved.get("Scenario", {})
    return {
        "scenario_name": scenario_config_path.stem,
        "run_id": scenario_block.get("run_id"),
        "description": scenario_block.get("Description"),
        "resolved_config": resolved,
        "base_config_path": str(base_path),
        "provenance": provenance,
        "overrides": overrides,
    }
