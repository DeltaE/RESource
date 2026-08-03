# Installation and environment setup

## Choosing an installation method

Use `pip` when consuming a published release:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install deltae-resource
```

Use `uv` when developing RESource or reproducing an analysis from a repository
checkout:

```bash
uv sync --locked
```

The package-index distribution name is `deltae-resource`; its import package
and project name are `RESource`. Before the first PyPI release, use the locked
source-checkout workflow instead.

The project metadata lives in `pyproject.toml`; exact development versions live
in `uv.lock`. The legacy files under `env/` are retained only as references for
platforms where native geospatial wheels are unavailable.

## Optional features

```bash
uv sync --locked --extra notebooks  # JupyterLab and widgets
uv sync --locked --extra viz        # Additional interactive plotting tools
uv sync --locked --extra docs       # Sphinx documentation toolchain
```

Development tools such as pytest and Ruff are in the `dev` dependency group and
are installed by default during `uv sync`.

## Repository layout

```text
src/RESource/   Installable implementation
tests/          Fast automated tests
config/         Example workflow configuration
notebooks/      Exploratory and case-study notebooks
docs/source/    Sphinx documentation sources
data/           Local/downloaded data, mostly excluded from packages
workflow/       Snakemake workflow assets
```

Because the package uses a `src` layout, importing directly from a fresh checkout
without installing it is intentionally unsupported. Run commands with `uv run`
or activate `.venv` first.

## Dependency changes

Add runtime dependencies with:

```bash
uv add PACKAGE
```

Add development-only dependencies with:

```bash
uv add --group dev PACKAGE
```

Add an optional feature dependency with:

```bash
uv add --optional notebooks PACKAGE
```

Commit both `pyproject.toml` and `uv.lock` after a dependency change.

## Validation

```bash
uv lock --check
uv run resource --help
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

The build should produce a wheel and source archive in `dist/`. Test the wheel
before publishing to ensure the result does not depend on repository-relative
imports.

## Native geospatial dependencies

Most supported platforms receive wheels for Rasterio, Fiona, PyProj, Shapely,
and GeoPandas. On an HPC or unusual platform without compatible wheels, install
the required GDAL/PROJ system libraries first or use a Conda/micromamba base
environment, then install RESource into that environment.

## Data location

Large assessment data is intentionally outside the wheel. Configuration files
normally provide data paths. The EU DEM pipeline additionally recognizes
`RESOURCE_DATA_DIR`:

```bash
export RESOURCE_DATA_DIR=/path/to/resource-data
uv run python -m RESource.eu_dem_pipeline --help
```
