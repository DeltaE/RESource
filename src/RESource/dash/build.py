"""Render the standalone dashboard template.

The HTML has no server-side data baked in — it's a static file plus whatever
``.h5`` store(s) the user picks in their own browser. What's generated here is
the inlined copy of four Python modules' source (``store_reader.py``,
``store_summary.py``, ``config_reader.py``, ``cluster_explorer.py``) that the page's Web Worker writes
to a virtual filesystem and imports into a real Pyodide interpreter at
runtime — so the in-browser HDF5 reader and summarizer never drift from the
versions tested in ``tests/test_dash_browser_reader.py``.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from RESource import utility as utils

DEFAULT_OUTPUT = utils.get_root() / "RESource_dash.html"

_PY_MODULES = ("store_reader.py", "store_summary.py", "config_reader.py", "cluster_explorer.py")


def _template_env() -> Environment:
    templates_dir = resources.files("RESource.dash") / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
    )


def build_dashboard(output: Path = DEFAULT_OUTPUT) -> Path:
    """Render the standalone, store-file-picking dashboard to a single HTML file."""
    modules = {
        name: resources.files("RESource.dash").joinpath(name).read_text(encoding="utf-8")
        for name in _PY_MODULES
    }
    template = _template_env().get_template("RESource_dash.html.j2")
    html = template.render(py_modules_json=json.dumps(modules))
    output = Path(output)
    output.write_text(html, encoding="utf-8")
    return output
