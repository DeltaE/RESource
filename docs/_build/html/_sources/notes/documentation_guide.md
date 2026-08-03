# Documentation guide

RESource documentation is built with Sphinx, MyST Markdown, Napoleon, autodoc,
and nbsphinx. Sources live under `docs/source/`.

## Build locally

```bash
uv sync --locked --extra docs
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Open `docs/_build/html/index.html` after a successful build. Generated files in
`docs/_build/` are ignored by Git.

## Source organization

```text
docs/source/index.md          Landing page and toctrees
docs/source/notes/            Guides and API reference
notebooks/                    Research notebooks and maintained workflow examples
docs/source/_static/          Images and other static assets
docs/source/conf.py           Sphinx configuration
```

## Writing conventions

- Write guides in MyST Markdown.
- Use sentence-case headings.
- Prefer runnable `uv run ...` examples for repository workflows.
- Use `RESource` in imports; mention `RES` only when documenting compatibility.
- Use `deltae-resource` in pip installation commands.
- Link methodological claims to the peer-reviewed publication at
  <https://doi.org/10.1016/j.energ.2026.100077>.
- Keep data paths configurable and avoid machine-specific absolute paths.
- Use Google-style Python docstrings so Napoleon can render them.
- Put reusable implementation in `src/RESource`, not in documentation notebooks.

## Python docstrings

Public modules, classes, functions, and methods should explain behavior and
contracts—not repeat the identifier. Use this shape:

```python
def load_config(path: str | Path) -> dict:
    """Load and validate a workflow configuration.

    Args:
        path: YAML configuration path.

    Returns:
        Parsed configuration values.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required configuration fields are missing.
    """
```

Include `Notes`, `Examples`, or `Warnings` only when they add useful operational
context. Do not preserve commented-out historical implementations in docstrings.

## Notebook policy

Notebooks are narrative examples and exploratory analysis. Every notebook lives
under the repository's `notebooks/` directory and imports the installed package
without changing `sys.path` or the working directory. Move reusable functions
into `src/RESource` and cover them with tests. See `notebooks/README.md` for the
organization and retention policy.

## Before submitting documentation changes

```bash
uv run ruff check src tests docs/source/conf.py
uv run ruff format --check src tests docs/source/conf.py
uv run pytest
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```
