"""Create isolated, browser-discoverable notebook study outputs."""

from datetime import datetime
from pathlib import Path
from string import Template


def create_study_output(
    root: Path,
    study: str,
    notebook: str,
    descriptions: Path,
    variables: dict[str, str] | None = None,
) -> Path:
    """Create a fresh study run with a figure directory and image descriptions.

    Args:
        root: Repository root containing the results directory.
        study: Study folder name under ``results/Studies``.
        notebook: Identifier for the notebook producing this run.
        descriptions: Markdown template, with ``$name`` substitutions if needed.
        variables: Values to substitute into the template.

    Returns:
        Path to the new run directory. Figures belong in its ``figures`` directory.
    """
    for label, name in (("study", study), ("notebook", notebook)):
        if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
            raise ValueError(f"{label} must be a single directory name: {name!r}")
    text = Template(Path(descriptions).read_text(encoding="utf-8")).substitute(variables or {})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(root) / "results" / "Studies" / study / notebook / stamp
    (output / "figures").mkdir(parents=True, exist_ok=False)
    (output / "images.md").write_text(text, encoding="utf-8")
    return output
