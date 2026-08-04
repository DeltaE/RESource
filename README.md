<img src="docs/source/_static/Issue_msg_box.png" alt="Issue" width="600"/>


__One of the many solutions ?__

<img src="docs/source/_static/graphic_RES_logo_202508.jpg" alt="RESource logo" width="250"/>

__A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.__

> RESource is described and applied in the peer-reviewed publication
> [Mapping feasible renewable transition space: Land-use, conservation, and grid-access constraints on wind and solar in British Columbia](https://doi.org/10.1016/j.energ.2026.100077).

RESource is developed to enable reproducible, adaptable assessments of VRE potential that are sensitive to local constraints and planning priorities. We developed a structured, modular workflow that integrates geospatial, temporal, economic, and regulatory data to evaluate site suitability for solar and wind energy development. This structured methodology ensures transparency and transferability, allowing RESource to be adapted for different regions and scaled for long-term strategic energy planning.


## Workflow overview
<img src="docs/source/_static/workflow.jpg" alt="high_level_workflow" width="1000"/>

## 🚀 Quick Start

**New to RESource?** Get started with

📖 **[Full Quickstart Guide](https://deltae.github.io/RESource/#quick-start)** | 📚 **[Complete Documentation](https://deltae.github.io/RESource/)**

### Installation

After the first PyPI release, install the `deltae-resource` distribution.
The Python import name remains `RESource`:

```bash
pip install deltae-resource
```

Until that release is published, clone the repository and use the locked `uv`
development workflow below.

For local development, `uv` creates the environment from the committed lockfile
and installs RESource in editable mode:

```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
uv sync --locked
uv run pytest
uv run resource --help
```

Notebook users can install the notebook extra:

```bash
uv sync --locked --extra notebooks
uv run jupyter lab notebooks/
```

See the [notebook index](notebooks/README.md) for maintained workflows and the
notebook retention policy.

New code should import the package as `RESource`. The former `RES` namespace is
temporarily retained as a compatibility layer for existing notebooks.

#### Just want to explore a results store?

If you only care about the output of a scenario run — the `.h5` store
under `data/store/` — you don't need to install the full RESource pipeline
(no `atlite`, `cdsapi`, `cfgrib`, `rioxarray`, `osmnx`, `pygadm`, ...).
Every store is a plain `pandas.HDFStore` file, so a much lighter
environment is enough to open it and do post-processing:

```bash
python -m venv .venv-viewer
source .venv-viewer/bin/activate   # .venv-viewer\Scripts\activate on Windows
pip install -r requirements-viewer.txt
jupyter lab explore_store.ipynb
```

[`explore_store.ipynb`](explore_store.ipynb) at the repo root walks through
listing available store files, inspecting what keys/tables they contain,
loading a table (with geometry columns auto-decoded if `geopandas` is
installed), and a few basic post-processing/plotting examples. It's built
on [`store_viewer.py`](store_viewer.py), a small standalone module with no
dependency on the `RESource` package itself.

```python
from RESource.RESources import RESources_builder
```

### Enhanced Analysis Pipeline

The installed `resource` command provides flexible region selection with colored output:

| Command | Description |
|---------|-------------|
| `resource config/CAN/scenarios/baseline.yaml --year 2024` | Canadian analysis (all provinces) |
| `resource config/WB6/scenarios/baseline.yaml --year 2023` | Western Balkans analysis (all countries) |
| `resource config/WB6/scenarios/baseline.yaml --year 2023 -r AL BA` | Specific regions only |
| `resource-multiyear config/CAN/scenarios/baseline.yaml --start 2014 --end 2024 -r BC` | Sequential multi-year assessment |
| `resource --help` | Show all available options |

**Features:** Smart region detection • Input validation • Colored error messages • Flexible region selection

### Country Reports

The `resource-report` command builds a self-contained HTML report for a country — resolved input config per scenario (with diffs against the base config), a scenario-vs-scenario contrast, and solar plots — directly from existing `resource` pipeline outputs on disk. It never re-runs the assessment; it only reads whatever is already in `results/<Country>/<Region>/<RUN_ID>/`.

**1. Install the `reporting` extra** (pulls in `jinja2`):

```
uv sync --extra reporting
```

**2. Make sure the pipeline has run at least once** for the scenarios/regions you want to report on, e.g.:

```
resource config/CAN/scenarios/baseline.yaml --year 2024 -r BC
```

**3. Build the report:**

```
resource-report CAN --regions BC
```

| Command | Description |
|---------|-------------|
| `resource-report CAN --regions BC` | Report for one region, all scenarios found under `config/CAN/scenarios/` |
| `resource-report CAN` | Report for every region declared in the scenario configs |
| `resource-report CAN --scenarios baseline no_buffers` | Restrict to specific scenarios |
| `resource-report CAN --regions BC AB --out /tmp/my_reports` | Custom regions and output directory |
| `resource-report --help` | Show all available options |

Output is written to `reports/<Country>/<Country>_<Regions>_report_<timestamp>.html` — a single portable file (images are embedded as base64) that opens directly in a browser, offline.

See [`docs/examples/CAN_BC_solar_report.html`](docs/examples/CAN_BC_solar_report.html) for a sample report generated from the Canada/BC scenarios.

------

## 📋 Key Features

- **🌍 Multi-Regional**: Canada, Western Balkans, and custom regions
- **⚡ Multi-Technology**: Wind and solar resource assessment
- **🔧 Modular Design**: Configurable exclusions, constraints, and parameters
- **📊 Rich Outputs**: Time series, capacity maps, and interactive visualizations
- **🔄 Reproducible**: Locked environments and standardized workflows

------

## Build the documentation

Documentation is built with Sphinx through the locked uv environment:

```bash
uv sync --locked --extra docs
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Preview the generated site locally:

```bash
uv run python -m http.server 8000 --directory docs/_build/html
```

Then open <http://localhost:8000>. For a complete clean rebuild, use:

```bash
uv run sphinx-build -E -a -W --keep-going \
  -b html docs/source docs/_build/html
```

Documentation pages live in `docs/source/notes/`, navigation is maintained in
`docs/source/index.md`, and images belong in `docs/source/_static/`. Add every new
page to an appropriate `toctree` in `index.md` and resolve all strict-build warnings.
The generated `docs/_build/` directory is ignored; commit documentation sources,
not generated HTML.

Authorized maintainers can deploy a reviewed build through the established
`gh-pages` branch workflow:

```bash
uvx ghp-import --no-jekyll --push --force \
  --branch gh-pages --remote origin docs/_build/html
```

This force-pushes generated documentation to the remote `gh-pages` branch. Review
the local site and follow the [developer deployment guide](docs/source/notes/deployment.md)
before running it.

------

## 📚 Resources

- **[Complete Setup Guide](docs/source/notes/setup_guide.md)** - Installation and environment setup
- **[Quickstart Guide](docs/source/notes/quickstart.md)** - Get running in 5 minutes
- **[🏔️ BC Case Study](https://deltae.github.io/RESource/notes/case_BC.html)** - Detailed regional analysis
- **[📘 Full Documentation](https://deltae.github.io/RESource/)** - Complete reference
- **[📄 Peer-reviewed publication](https://doi.org/10.1016/j.energ.2026.100077)** - Methodology and British Columbia application
- **[Contributing](CONTRIBUTING.md)** - Development setup, standards, and pull-request expectations
- **[Developer deployment](docs/source/notes/deployment.md)** - Package and documentation release procedure
- **[Development pipeline](docs/source/notes/development_pipeline.md)** - Active and planned methodological work
- **[AI agent agreement](AGENTS.md)** - Safe regional-adaptation workflow for coding agents
