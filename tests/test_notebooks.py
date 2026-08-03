"""Structural checks for repository notebooks."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPOSITORY_ROOT / "notebooks"
WORKFLOW_ROOT = NOTEBOOK_ROOT / "workflows"


def test_all_notebooks_are_in_notebook_directory():
    notebooks = {
        path
        for path in REPOSITORY_ROOT.rglob("*.ipynb")
        if not {".git", ".venv"}.intersection(path.parts)
    }

    outside = sorted(path for path in notebooks if NOTEBOOK_ROOT not in path.parents)

    assert not outside, f"Notebooks outside notebooks/: {outside}"


def test_notebooks_are_valid_and_nonempty():
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    assert notebooks, "No notebooks found"

    invalid = []
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not notebook.get("cells"):
                invalid.append(f"{path}: contains no cells")
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(f"{path}: {exc}")

    assert not invalid, "Invalid notebooks:\n" + "\n".join(invalid)


def test_workflow_notebooks_use_installed_package():
    """Keep maintained workflows independent of repository-relative setup."""
    forbidden = ("from RES ", "import RES.", "sys.path", "os.chdir", "!pip", "!conda", "%cd")
    violations = []

    for path in sorted(WORKFLOW_ROOT.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        for pattern in forbidden:
            if pattern in source:
                violations.append(f"{path}: contains {pattern!r}")

    assert not violations, "Workflow setup violations:\n" + "\n".join(violations)
