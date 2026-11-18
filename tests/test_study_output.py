"""Tests for reusable notebook study outputs."""

import ast
import json
from pathlib import Path

import pytest

from RESource.study import create_study_output

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/studies_WB6"


def test_study_runs_are_isolated_and_copy_descriptions(tmp_path):
    template = tmp_path / "descriptions.md"
    template.write_text("# Study\n\n## ${country}_map.png\nMap of ${country}.\n")
    first = create_study_output(
        tmp_path, "results_Publication", "cells", template, {"country": "Example"}
    )
    (first / "figures" / "Example_map.png").write_bytes(b"first")
    second = create_study_output(
        tmp_path, "results_Publication", "cells", template, {"country": "Example"}
    )
    assert first != second
    assert first.relative_to(tmp_path).parts[:4] == (
        "results",
        "Studies",
        "results_Publication",
        "cells",
    )
    assert (first / "images.md").read_text() == "# Study\n\n## Example_map.png\nMap of Example.\n"
    assert (first / "figures" / "Example_map.png").read_bytes() == b"first"
    assert (second / "figures").is_dir()


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", r"a\b"])
def test_study_names_cannot_escape_output_directory(tmp_path, name):
    template = tmp_path / "descriptions.md"
    template.write_text("# Study")
    with pytest.raises(ValueError):
        create_study_output(tmp_path, name, "notebook", template)


def test_missing_template_variable_does_not_create_output(tmp_path):
    template = tmp_path / "descriptions.md"
    template.write_text("## ${missing}.png")
    with pytest.raises(KeyError):
        create_study_output(tmp_path, "study", "notebook", template)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("name", "output_name", "template"),
    [
        ("Visuals_WB6_CELLS_combined.ipynb", "cells", "publication_cells_images.md"),
        ("Visuals_WB6_LANDs_combined.ipynb", "lands", "publication_lands_images.md"),
        ("wb6_population_notebook.ipynb", "population", "publication_population_images.md"),
    ],
)
def test_publication_notebooks_use_study_output(tmp_path, name, output_name, template):
    notebook = json.loads((NOTEBOOKS / name).read_text())
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert f"'results_Publication', '{output_name}'" in code
    assert template in code
    assert "combined_datahandler" not in code
    template_path = tmp_path / "notebooks/studies_WB6" / template
    template_path.parent.mkdir(parents=True)
    template_path.write_text((NOTEBOOKS / template).read_text())

    start = code.index("from RESource.study import create_study_output")
    stop_marker = (
        "MAPS_DIR = STUDY_DIR / 'figures'"
        if output_name == "population"
        else ("vis_save_to_root = results_save_to_root / 'figures'")
    )
    end = code.index(stop_marker, start) + len(stop_marker)
    env = {
        "ROOT": tmp_path,
        "BASE_DIR": tmp_path,
        "country_name": "Western Balkan Regions",
        "country_kwd": "WesternBalkanRegions",
        "run_tag": "VRElow_2025_20261117",
        "SCENARIO_CFG_PATH": "config/WB6/scenarios/baseline.yaml",
    }
    exec(code[start:end], env)
    run = env["STUDY_DIR"] if output_name == "population" else env["results_save_to_root"]
    assert (run / "figures").is_dir()
    descriptions = (run / "images.md").read_text()
    assert "## " in descriptions
    if output_name != "population":
        assert "VRElow_2025_20261117" in descriptions
        assert "WesternBalkanRegions" in descriptions
