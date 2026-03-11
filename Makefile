# RESource Project Makefile

.PHONY: help setupenv updateenv exportenv run run-wb6 run-wb6-region run-can run-can-region run-can-policy run-bgd docs autobuild deploy jupyter clean

# Default config / regions (overridable on the command line)
CONFIG    ?= config/config_WB6.yaml
REGIONS   ?=

# Default target
help:
	@echo "RESource Project - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  📚 See SETUP.md for complete setup guide"
	@echo ""
	@echo "Environment:"
	@echo "  setupenv       - Setup conda environment from env/environment.yml (RECOMMENDED)"
	@echo "  setupenv-clean - Setup tested working environment (manual method)"
	@echo "  updateenv      - Update existing conda environment"
	@echo "  exportenv      - Export current environment to env/environment.yml"
	@echo ""
	@echo "Run Pipeline:"
	@echo "  run              - Run with custom CONFIG= and optional REGIONS= (see examples below)"
	@echo "  run-wb6          - Run Western Balkans config (all 6 countries: AL BA XK ME MK RS)"
	@echo "  run-wb6-region   - Run Western Balkans config for specific REGIONS="
	@echo "  run-can          - Run Canada baseline config (all provinces)"
	@echo "  run-can-region   - Run Canada baseline config for specific REGIONS="
	@echo "  run-can-policy   - Run Canada policy1 config (all provinces)"
	@echo "  run-bgd          - Run Bangladesh config (all regions)"
	@echo ""
	@echo "  Examples:"
	@echo "    make run CONFIG=config/config_WB6.yaml                    # WB6, all regions"
	@echo "    make run CONFIG=config/config_WB6.yaml REGIONS='AL MK'    # WB6, Albania + N. Macedonia only"
	@echo "    make run CONFIG=config/config_CAN_baseline.yaml           # Canada baseline, all provinces"
	@echo "    make run CONFIG=config/config_CAN_baseline.yaml REGIONS='BC QC AB'  # Canada, 3 provinces"
	@echo "    make run CONFIG=config/config_CAN_policy1.yaml REGIONS=ON # Canada policy1, Ontario only"
	@echo "    make run CONFIG=config/config_BGD.yaml                    # Bangladesh, all regions"
	@echo "    make run-wb6-region REGIONS='BA XK RS'                    # WB6 shortcut with regions"
	@echo "    make run-can-region REGIONS='BC AB SK MB ON QC'           # CAN shortcut with regions"
	@echo ""
	@echo "  Western Balkans regions : AL (Albania) BA (Bosnia) XK (Kosovo)"
	@echo "                            ME (Montenegro) MK (N. Macedonia) RS (Serbia)"
	@echo "  Canada provinces        : AB BC MB NB NL NS ON PE QC SK"
	@echo ""
	@echo "Documentation:"
	@echo "  docs        - Build and deploy documentation"
	@echo "  autobuild   - Live rebuild documentation (port 8000)"
	@echo "  deploy      - Deploy documentation to GitHub Pages"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       - Clean build files and cache"

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
	@echo "Updating conda environment 'RESource'..."
	@if conda env list | grep -q "^RESource "; then \
		conda env update -f env/environment.yml; \
		conda run -n RESource pip install -e .; \
		echo "✅ Environment updated!"; \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
	fi

exportenv:
	@echo "Exporting conda environment 'RESource' to env/environment.yml..."
	@if conda env list | grep -q "^RESource "; then \
		mkdir -p env; \
		conda env export -n RESource > env/environment.yml; \
		echo "✅ Environment exported to env/environment.yml"; \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

# Running Code
# ─────────────────────────────────────────────────────────────────────────────
# Generic entry point – accepts any CONFIG= and optional REGIONS= on the CLI.
#   make run CONFIG=config/config_WB6.yaml
#   make run CONFIG=config/config_WB6.yaml REGIONS='AL MK RS'
#   make run CONFIG=config/config_CAN_baseline.yaml REGIONS='BC QC AB'
# ─────────────────────────────────────────────────────────────────────────────
run:
	@echo "Running RESource pipeline with CONFIG=$(CONFIG)$(if $(REGIONS), regions: $(REGIONS))..."
	@if conda env list | grep -q "^RESource "; then \
		conda run -n RESource python run.py $(CONFIG) $(if $(REGIONS),-r $(REGIONS)); \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# Western Balkans shortcuts
#   Regions: AL (Albania) BA (Bosnia & Herzegovina) XK (Kosovo)
#            ME (Montenegro) MK (North Macedonia) RS (Serbia)
# ─────────────────────────────────────────────────────────────────────────────
run-wb6:
	$(MAKE) run CONFIG=config/config_WB6.yaml

# Usage: make run-wb6-region REGIONS='AL MK'
run-wb6-region:
	$(MAKE) run CONFIG=config/config_WB6.yaml

# ─────────────────────────────────────────────────────────────────────────────
# Canada shortcuts
#   Provinces: AB BC MB NB NL NS ON PE QC SK
# ─────────────────────────────────────────────────────────────────────────────
run-can:
	$(MAKE) run CONFIG=config/config_CAN_baseline.yaml

# Usage: make run-can-region REGIONS='BC QC AB'
run-can-region:
	$(MAKE) run CONFIG=config/config_CAN_baseline.yaml

run-can-policy:
	$(MAKE) run CONFIG=config/config_CAN_policy1.yaml

# ─────────────────────────────────────────────────────────────────────────────
# Bangladesh shortcut
# ─────────────────────────────────────────────────────────────────────────────
run-bgd:
	$(MAKE) run CONFIG=config/config_BGD.yaml

# Documentation
docs:
	@echo "Building and deploying documentation..."
	@if conda env list | grep -q "^RESource "; then \
		mkdir -p docs/_build/html; \
		mkdir -p docs/source/notebooks; \
		cp notebooks/*.ipynb docs/source/notebooks/ 2>/dev/null || true; \
		conda run -n RESource sphinx-build -b html docs/source docs/_build/html; \
		echo "" > docs/_build/html/.nojekyll; \
		conda run -n RESource ghp-import -n -p -f docs/_build/html; \
		echo "✅ Documentation deployed to GitHub Pages!"; \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

autobuild:
	@echo "Starting live documentation rebuild on port 8000..."
	@if conda env list | grep -q "^RESource "; then \
		mkdir -p docs/source/notebooks; \
		cp notebooks/*.ipynb docs/source/notebooks/ 2>/dev/null || true; \
		echo "🔄 Server: http://127.0.0.1:8000"; \
		conda run -n RESource sphinx-autobuild docs/source docs/_build/html --host 127.0.0.1 --port 8000; \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

deploy:
	@echo "Deploying documentation to GitHub Pages..."
	@if conda env list | grep -q "^RESource "; then \
		conda run -n RESource ghp-import -n -p -f docs/_build/html; \
		echo "✅ Documentation deployed!"; \
		echo "🌐 Visit: https://deltae.github.io/RESource/"; \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

# Cleanup
clean:
	@echo "Cleaning build files and cache..."
	@rm -rf docs/build/
	@rm -rf cache/*.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned build files and cache!"

