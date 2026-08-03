# Developer deployment guide

This guide describes how maintainers promote a validated RESource commit into a
Python package release and public documentation. Deployment is a maintainer action;
ordinary contributors should stop after opening a reviewed pull request.

## Deployment targets

RESource has two independent artifacts:

1. The `deltae-resource` distribution published to PyPI. It provides the
   `RESource` import package and the `resource` and `resource-multiyear` commands.
2. The Sphinx HTML site published at <https://deltae.github.io/RESource/>.

Both artifacts must originate from the same reviewed commit and identify the same
release version. Data products, credentials, caches, notebooks, and generated
research results are not deployment artifacts.

## 1. Prepare the release commit

Start from a clean, reviewed development branch and synchronize the locked
environment:

```bash
uv sync --locked --all-extras
uv lock --check
uv run pre-commit run --all-files
uv run pre-commit run --all-files --hook-stage pre-push
```

Set the same PEP 440 version in `pyproject.toml` and
`src/RESource/__init__.py`. Update release notes, compatibility guidance, and
citations. Confirm that `credentials/coders_api.yaml`, downloaded data, results,
and local caches are absent from `git status` and release artifacts.

## 2. Validate package and documentation

```bash
uv run --locked pytest
uv run --locked ruff check src tests run.py
uv run --locked ruff format --check src tests run.py
uv run sphinx-build -E -a -W --keep-going \
  -b html docs/source docs/_build/html
uv build
uvx twine check dist/*
```

Inspect the rendered documentation locally:

```bash
uv run python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000> and check installation commands, navigation, API
pages, DOI links, and the release version.

## 3. Test the built wheel

Install the wheel into a clean environment outside the repository:

```bash
uv venv /tmp/deltae-resource-release-test
uv pip install --python /tmp/deltae-resource-release-test/bin/python \
  dist/deltae_resource-*.whl
/tmp/deltae-resource-release-test/bin/python -c \
  "import RESource; print(RESource.__version__)"
/tmp/deltae-resource-release-test/bin/resource --help
/tmp/deltae-resource-release-test/bin/resource-multiyear --help
```

Do not run a complete regional assessment as a release smoke test because it may
download large datasets and require personal credentials. Use deterministic test
fixtures for automated validation.

## 4. Publish the Python package

Prefer PyPI Trusted Publishing from a protected GitHub release workflow. Validate
the first release candidate on TestPyPI, then publish the exact same commit to PyPI.
For the documented manual-token fallback, see the
[PyPI publishing guide](publishing.md). Never place a PyPI token in the repository,
shell history, logs, or workflow YAML.

After publishing, verify from a clean context:

```bash
uvx --from deltae-resource resource --help
python -m pip index versions deltae-resource
```

## 5. Deploy the documentation

RESource historically deploys the generated site to the `gh-pages` branch with
`ghp-import`. The former Make target ran:

```text
ghp-import -n -p -f docs/_build/html
```

The maintained uv equivalent is:

```bash
# Build and validate first.
uv run sphinx-build -E -a -W --keep-going \
  -b html docs/source docs/_build/html

# Publish only after reviewing the rendered site and current commit.
uvx ghp-import --no-jekyll --push --force \
  --branch gh-pages --remote origin docs/_build/html
```

`--push` changes the remote repository and `--force` replaces the generated branch
contents, so this command is for authorized maintainers only. Before running it:

```bash
git status --short
git remote get-url origin
git rev-parse HEAD
```

Deploy only a reviewed, committed source state and record that commit SHA in the
release. In the GitHub repository settings, Pages must publish from the `gh-pages`
branch (root), or an existing action must be configured to react to that branch.
The generated branch contains the site; `docs/_build/` remains ignored on the
development branch.

The repository currently contains no checked-in GitHub Actions workflow that builds
or deploys Pages. A future protected Pages workflow could replace the direct push by:

1. checks out the tagged commit;
2. installs uv;
3. runs `uv sync --locked --extra docs`;
4. builds Sphinx with warnings treated as errors;
5. uploads only `docs/_build/html`; and
6. deploys that artifact to GitHub Pages after required checks pass.

Until such a workflow is added and reviewed, use the established `ghp-import`
branch deployment above. Never commit `docs/_build/` to the development branch.

## 6. Tag and verify

Create the release tag only after validation, following the repository's chosen tag
format. The GitHub release, PyPI version, documentation, DOI citation, and artifact
checksums should all point to the same commit.

Final verification includes:

- PyPI installation and both CLI help commands;
- public documentation availability and navigation;
- correct project and DOI links;
- absence of credentials and local paths in artifacts;
- wheel and source-distribution license files; and
- a rollback note describing how to yank a broken PyPI release and redeploy the
  previous documentation artifact without reusing a published version number.

## Rollback principles

PyPI releases are immutable: publish a corrected version rather than overwriting a
file. A seriously broken release may be yanked while preserving its history.
Documentation may be redeployed from the last known-good tag. Never delete tags or
rewrite release history merely to make artifacts appear consistent.
