# RESource Project Makefile

.PHONY: help setupenv setupenv-clean updateenv exportenv \
        run run-wb6 run-wb6-region run-can run-can-region run-can-policy run-bgd \
        docs autobuild deploy clean

# ── Defaults (override on the command line) ───────────────────────────────────
CONFIG  ?= config/config_WB6.yaml
YEAR    ?=
REGIONS ?=

# ── Internal helpers ──────────────────────────────────────────────────────────
_YEAR_FLAG   = $(if $(YEAR),--year $(YEAR))
_REGION_FLAG = $(if $(REGIONS),-r $(REGIONS))
_ENV_CHECK   = @conda env list | grep -q "^RESource " || \
               { echo "❌  Environment 'RESource' not found. Run 'make setupenv' first."; exit 1; }

# Default target
help:
	@echo "RESource — Available Commands"
	@echo ""
	@echo "Environment:"
	@echo "  setupenv        Set up conda environment from env/environment.yml"
	@echo "  setupenv-clean  Set up tested clean environment (manual pin)"
	@echo "  updateenv       Update existing environment"
	@echo "  exportenv       Export environment to env/environment.yml"
	@echo ""
	@echo "Run pipeline:  make <target> [YEAR=YYYY] [REGIONS='R1 R2']"
	@echo "  run             Generic entry point (requires CONFIG=)"
	@echo "  run-wb6         Western Balkans, all regions"
	@echo "  run-wb6-region  Western Balkans, REGIONS= required"
	@echo "  run-can         Canada baseline, all provinces"
	@echo "  run-can-region  Canada baseline, REGIONS= required"
	@echo "  run-can-policy  Canada policy1, all provinces"
	@echo "  run-bgd         Bangladesh, all regions"
	@echo ""
	@echo "  Western Balkans : AL BA XK ME MK RS"
	@echo "  Canada          : AB BC MB NB NL NS ON PE QC SK"
	@echo ""
	@echo "Examples:"
	@echo "  make run-can YEAR=2020"
	@echo "  make run-can-region YEAR=2020 REGIONS='BC AB'"
	@echo "  make run-wb6 YEAR=2019"
	@echo "  make run-wb6-region YEAR=2019 REGIONS='AL MK RS'"
	@echo "  make run CONFIG=config/config_BGD.yaml YEAR=2021"
	@echo "  make run CONFIG=config/config_WB6.yaml YEAR=2020 REGIONS='BA RS'"
	@echo ""
	@echo "Utilities:"
	@echo "  docs        Build and deploy documentation"
	@echo "  autobuild   Live documentation rebuild (port 8000)"
	@echo "  clean       Remove build files and cache"

# Environment Management
setupenv-clean:
	@echo "🚀 Setting up clean, tested RESource environment..."
	@if conda env list | grep -q "^RESource "; then \
		echo "⚠️  Environment 'RESource' already exists."; \
		echo "To replace with clean version, run: conda env remove -n RESource && make setupenv-clean"; \
	else \
		echo "📦 Creating conda environment with Python 3.12..."; \
		conda create -n RESource python=3.12 -y; \
		echo "📦 Installing core geospatial packages..."; \
		conda run -n RESource pip install numpy pandas geopandas==1.0.1 shapely==2.0.6 \
			dask-geopandas==0.4.2 fiona==1.10.1 pyproj==3.6.1 rasterio==1.4.3; \
		echo "📦 Installing additional packages..."; \
		conda run -n RESource pip install atlite xarray netcdf4 matplotlib seaborn jupyter \
			ipywidgets h5py scikit-learn requests pyyaml tqdm geojson rioxarray colorama \
			pygadm osmnx plotly tables progressbar memory-profiler configparser lxml \
			pyogrio openpyxl; \
		echo "✅ Clean RESource environment setup completed!"; \
		echo "💡 Activate with: conda activate RESource"; \
	fi

setupenv:
	@echo "Setting up conda environment 'RESource'..."
	@if conda env list | grep -q "^RESource "; then \
		echo "Environment 'RESource' already exists. Use 'make updateenv' to update."; \
	else \
		conda env create -f env/environment.yml; \
		conda run -n RESource pip install -e .; \
		echo "✅ Environment 'RESource' setup completed!"; \
	fi

updateenv:
	$(_ENV_CHECK)
	conda env update -f env/environment.yml
	conda run -n RESource pip install -e .
	@echo "✅ Environment updated."

exportenv:
	$(_ENV_CHECK)
	@mkdir -p env
	conda env export -n RESource > env/environment.yml
	@echo "✅ Exported to env/environment.yml"

# ── Pipeline ──────────────────────────────────────────────────────────────────
run:
	$(_ENV_CHECK)
	conda run -n RESource python run.py $(CONFIG) $(_YEAR_FLAG) $(_REGION_FLAG)

run-wb6:
	$(MAKE) run CONFIG=config/WB6_baseline.yaml

run-wb6-region:
ifndef REGIONS
	$(error REGIONS is required. Example: make run-wb6-region YEAR=2019 REGIONS='AL MK RS')
endif
	$(MAKE) run CONFIG=config/WB6_baseline.yaml

run-can:
	$(MAKE) run CONFIG=config/CAN_baseline.yaml

run-can-region:
ifndef REGIONS
	$(error REGIONS is required. Example: make run-can-region YEAR=2020 REGIONS='BC AB')
endif
	$(MAKE) run CONFIG=config/CAN_baseline.yaml

run-can-policy:
	$(MAKE) run CONFIG=config/CAN_policy1.yaml

run-bgd:
	$(MAKE) run CONFIG=config/config_BGD.yaml

# ── Documentation ─────────────────────────────────────────────────────────────
docs:
	$(_ENV_CHECK)
	@mkdir -p docs/_build/html docs/source/notebooks
	@cp notebooks/*.ipynb docs/source/notebooks/ 2>/dev/null || true
	conda run -n RESource sphinx-build -b html docs/source docs/_build/html
	@echo "" > docs/_build/html/.nojekyll
	conda run -n RESource ghp-import -n -p -f docs/_build/html
	@echo "✅ Documentation deployed."

autobuild:
	$(_ENV_CHECK)
	@mkdir -p docs/source/notebooks
	@cp notebooks/*.ipynb docs/source/notebooks/ 2>/dev/null || true
	conda run -n RESource sphinx-autobuild docs/source docs/_build/html \
		--host 127.0.0.1 --port 8000

deploy:
	$(_ENV_CHECK)
	conda run -n RESource ghp-import -n -p -f docs/_build/html
	@echo "✅ Deployed to GitHub Pages."

# Cleanup
clean:
	@echo "Cleaning build files and cache..."
	@rm -rf docs/build/
	@rm -rf cache/*.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned build files and cache!"

