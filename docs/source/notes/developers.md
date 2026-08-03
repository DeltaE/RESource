# Contributing Developers

The canonical contribution policy is
[`CONTRIBUTING.md`](https://github.com/DeltaE/RESource/blob/main/CONTRIBUTING.md).
It defines repository boundaries, hooks, scientific-change requirements, validation,
and pull-request expectations. AI coding agents must also follow the root
[`AGENTS.md`](https://github.com/DeltaE/RESource/blob/main/AGENTS.md).

RESource is a published research tool. Changes should preserve traceability to
the methodology described at
[https://doi.org/10.1016/j.energ.2026.100077](https://doi.org/10.1016/j.energ.2026.100077)
and clearly document intentional methodological changes.

## Development workflow

Create focused branches from the development line and reproduce the locked
environment before editing:

```bash
uv sync --locked --all-extras
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push
uv run pytest
```

Code belongs under `src/RESource`; tests belong under `tests`. New public APIs use
Google-style docstrings and type annotations. Before opening a pull request, run:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
uv build
```

When dependencies change, use `uv add` or `uv remove` and commit both
`pyproject.toml` and `uv.lock`.

Maintainers should use the [developer deployment guide](deployment.md) for release
validation, PyPI publication, documentation deployment, verification, and rollback.

## Maintainer


Md Eliasinul Islam (__Elias__)<br>
PhD Researcher,
[Delta E+ Research Lab](https://www.sfu.ca/fas/research/fas-research-labs/delta-e.html)<br>
Simon Fraser University, Canada

---
Feel free to connect 😊 <br>
[![Email: elias_islam@sfu.ca](https://img.shields.io/badge/Email-l%20icon?logo=gmail&logoColor=white&style=flat-square)](mailto:elias_islam@sfu.ca)
&nbsp;|&nbsp;
[![GitHub: eliasinul](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white&style=flat-square)](https://github.com/eliasinul)
&nbsp;|&nbsp;
[![LinkedIn: eliasinul](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white&style=flat-square)](https://www.linkedin.com/in/eliasinul/)
&nbsp;|&nbsp;
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-yellow?logo=buy-me-a-coffee&logoColor=white&style=flat-square)](https://coff.ee/eliasinul)
