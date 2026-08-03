# Quick start

RESource supports Python 3.11 and 3.12. End users can install it with `pip`;
contributors should use `uv` with the committed lockfile.

## Install a PyPI release

```bash
python -m pip install deltae-resource
```

Confirm the package and command are available:

```bash
python -c "import RESource; print(RESource.__version__)"
resource --help
```

The distribution is called `deltae-resource`, while Python code imports
`RESource`. Until the first PyPI release is available, use the source-checkout
workflow below.

## Set up a development checkout

```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
uv sync --locked
uv run pytest
```

For notebooks and documentation:

```bash
uv sync --locked --extra notebooks --extra docs
uv run jupyter lab notebooks/
```

`uv sync --locked` creates `.venv`, installs RESource in editable mode, and
reproduces the versions recorded in `uv.lock`. Do not install project
dependencies manually into `.venv`; use `uv add`, `uv remove`, or dependency
groups in `pyproject.toml`.

## Run an assessment

```bash
uv run resource config/config_BC_baseline.yaml --year 2024 -r BC
```

Other examples:

```bash
uv run resource config/CAN_baseline.yaml --year 2024
uv run resource config/config_WB6_2023.yaml --year 2023 -r AL BA
```

Paths inside configuration files are resolved by the running workflow. Large
downloaded and processed datasets are not bundled in the Python wheel.

## Import the API

```python
from RESource.RESources import RESources_builder

builder = RESources_builder(
    config_file_path="config/config_BC_baseline.yaml",
    region_short_code="BC",
    resource_type="wind",
    weather_year=2024,
)
```

The former `RES` import namespace remains available temporarily for existing
notebooks, but new code must use `RESource`.

## Common checks

```bash
uv lock --check
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
uv build
```
