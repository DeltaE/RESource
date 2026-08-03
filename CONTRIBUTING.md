# Contributing to RESource

Thank you for improving RESource. Contributions should preserve reproducibility,
methodological traceability, and the distinction between reusable package code
and region-specific inputs.

## Set up a development checkout

RESource uses `uv` as its single development interface:

```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
uv sync --locked --all-extras
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push
```

Create a focused branch from the development branch. Do not commit downloaded
rasters, generated results, credentials, API keys, or a local virtual environment.

## Repository boundaries

- Put importable code in `src/RESource/` and tests in `tests/`.
- Put every notebook under `notebooks/`; reusable logic must move into the package.
- Put regional contracts in `config/REGION/base.yaml`, decision-specific overrides
  in `config/REGION/scenarios/`, and explain their data sources. Scenario files
  should contain only intentional departures from the base.
- Put durable documentation in `docs/source/notes/` and link it from the docs index.
- Use `RESource` for all imports. The former `RES` namespace is no longer distributed.

## Standards

Public functions and classes need type annotations and Google-style docstrings.
Prefer small, composable functions and explicit paths. A scientific-method change
must state its assumptions, units, CRS, input provenance, and effect on outputs.
Tests should cover the smallest deterministic unit possible; large downloads and
external APIs should not be required by the default test suite.

Use `uv add`, `uv remove`, or their `--optional` variants for dependencies and
commit `pyproject.toml` together with `uv.lock`.

## Validate a change

The installed hooks apply safe formatting and lint fixes on commit, and run the
test suite before push. Run the full checks explicitly before a pull request:

```bash
uv run pre-commit run --all-files
uv run pre-commit run --all-files --hook-stage pre-push
uv run --locked pytest
uv build
uvx twine check dist/*
```

If a hook edits files, review and stage those changes before committing again.
Do not bypass hooks without explaining the exceptional reason in the pull request.

## Pull requests

A pull request should:

1. Describe the problem, approach, and user-visible result.
2. Identify affected regions, technologies, metrics, and configuration keys.
3. Link an issue and include tests or explain why automated testing is impractical.
4. Document new data sources, licenses, attribution, spatial/temporal resolution,
   and transformations.
5. Update examples and documentation when the public workflow changes.
6. Call out backward incompatibilities and migration steps.

Small reviewable pull requests are preferred. Generated figures may be supplied
as review evidence, but should be committed only when they are documentation assets.

## Research and licensing

Methodological work should cite the RESource publication:
<https://doi.org/10.1016/j.energy.2026.100077>. Contributors must have the right to
submit their code and data. Preserve third-party copyright and license notices and
do not copy code from a source whose license is unknown or incompatible.

The project is currently MIT-licensed. See the
[license transition assessment](docs/source/notes/licensing.md) before making any
license or provenance change.

Maintainers preparing public artifacts must follow the
[developer deployment guide](docs/source/notes/deployment.md) and the focused
[PyPI publishing guide](docs/source/notes/publishing.md).
