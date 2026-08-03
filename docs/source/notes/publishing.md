# Publishing to PyPI

This guide is for maintainers. End users install releases with:

```bash
python -m pip install deltae-resource
```

The distribution must be published as `deltae-resource`. PyPI normalizes
`RESource` to `resource`, which is already an unrelated registered distribution.
The product and import package remain `RESource`.

## Pre-release checklist

1. Choose and set a PEP 440 version in `pyproject.toml` and
   `src/RESource/__init__.py`.
2. Update release notes and user-facing compatibility notices.
3. Confirm `pyproject.toml` and `uv.lock` are committed together.
4. Run the complete validation suite:

```bash
uv lock --check
uv run ruff check src tests docs/source/conf.py
uv run ruff format --check src tests docs/source/conf.py
uv run pytest
uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/_build/html
uv build
uv run twine check dist/*
```

## Test the wheel locally

Create a clean environment outside the repository and install the wheel:

```bash
uv venv /tmp/deltae-resource-wheel-test
uv pip install --python /tmp/deltae-resource-wheel-test/bin/python \
  dist/deltae_resource-*.whl
/tmp/deltae-resource-wheel-test/bin/python -c \
  "import RESource; print(RESource.__version__)"
/tmp/deltae-resource-wheel-test/bin/resource --help
```

## Publish

Configure the PyPI project to use a trusted publisher from the repository's
release workflow. Prefer trusted publishing over long-lived API tokens. Validate
the first release on TestPyPI before publishing the same tested commit to PyPI.

For a manual token-based fallback:

```bash
export UV_PUBLISH_TOKEN='pypi-token'
uv publish
```

Never commit credentials. Tags, release artifacts, documentation, and the package
version should all identify the same source commit.

## Verify the release

```bash
uvx --from deltae-resource resource --help
python -m pip index versions deltae-resource
```

Then verify that the PyPI project description renders correctly and that its
project links resolve to the repository, documentation, publication, and issue
tracker.
