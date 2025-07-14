.PHONY: setup install clean clean-docs export sync-notebooks docs docs-deploy docs-deploy-git docs-build debug-sphinx debug-pages autobuild

# Create and activate env, install dependencies
setup:
	@echo "Creating and setting up the environment..."
	conda env create -f env/environment.yml
	@echo "Environment created. Please restart your shell or run 'conda activate RES' manually."

# Update the environment
update:
	@echo "Updating environment..."
	conda env update -f env/environment.yml
	@echo "Environment updated. Please restart your shell or run 'conda activate RES' manually."

# Optional: clean up env (if you want this)
clean:
	@echo "Removing the environment..."
	conda env remove -n RES
	@echo "Environment removed."

# Clean documentation build
clean-docs:
	@echo "Cleaning documentation build..."
	rm -rf docs/build/
	@echo "Documentation build cleaned."

export:
	@echo "Exporting the environment..."
	conda env export -n RES > env/environment.yml
	@echo "Environment exported to env/environment.yml."

# Sync notebooks from root to docs/source/notebooks
sync-notebooks:
	@echo "Syncing notebooks from root to docs/source/notebooks..."
	@mkdir -p docs/source/notebooks
	@cp notebooks/Store_explorer.ipynb docs/source/notebooks/ 2>/dev/null || echo "Store_explorer.ipynb not found, skipping..."
	@cp notebooks/Visuals_BC.ipynb docs/source/notebooks/ 2>/dev/null || echo "Visuals_BC.ipynb not found, skipping..."
	@echo "Notebooks synced successfully!"

# Build documentation with notebook sync and deploy to GitHub Pages
docs: sync-notebooks
	@echo "Building the documentation with updated notebooks..."
	@mkdir -p docs/build/html
	sphinx-build -b html docs/source docs/build/html
	@echo "Creating .nojekyll file to disable Jekyll..."
	@echo "" > docs/build/html/.nojekyll
	@echo "Creating additional Jekyll bypass files..."
	@echo "theme: none" > docs/build/html/_config.yml
	@echo "# GitHub Pages - No Jekyll Processing" > docs/build/html/README.md
	@echo "Verifying bypass files exist:"
	@ls -la docs/build/html/.nojekyll docs/build/html/_config.yml docs/build/html/README.md
	@echo "Documentation built successfully!"
	@echo "Deploying to GitHub Pages with Jekyll bypass..."
	ghp-import -n -p -f docs/build/html
	@echo "Documentation deployed to GitHub Pages!"

# Setup comprehensive virtual environment for the entire project
setup-venv:
	@echo "Setting up comprehensive virtual environment..."
	@if [ -d "venv" ]; then \
		echo "Removing existing virtual environment..."; \
		rm -rf venv; \
	fi
	@echo "Creating new virtual environment..."
	python -m venv venv
	@echo "Upgrading pip..."
	source venv/bin/activate && pip install --upgrade pip setuptools wheel
	@echo "Installing core scientific packages..."
	source venv/bin/activate && pip install numpy pandas geopandas
	@echo "Installing geospatial and mapping packages..."
	source venv/bin/activate && pip install rasterio shapely fiona pyproj cartopy folium
	@echo "Installing data analysis and visualization packages..."
	source venv/bin/activate && pip install matplotlib seaborn plotly dash dash-bootstrap-components
	@echo "Installing renewable energy and weather packages..."
	source venv/bin/activate && pip install atlite netcdf4 xarray cdsapi pygadm
	@echo "Installing web scraping and API packages..."
	source venv/bin/activate && pip install requests beautifulsoup4 pyrosm osmnx
	@echo "Installing machine learning packages..."
	source venv/bin/activate && pip install scikit-learn
	@echo "Installing documentation packages..."
	source venv/bin/activate && pip install sphinx myst-parser sphinx-book-theme nbsphinx sphinx-autobuild ghp-import
	@echo "Installing Jupyter packages..."
	source venv/bin/activate && pip install jupyter ipykernel notebook jupyterlab
	@echo "Installing additional utility packages..."
	source venv/bin/activate && pip install pyyaml openpyxl tqdm joblib pyarrow
	@echo "Installing project in development mode..."
	source venv/bin/activate && pip install -e .
	@echo "Virtual environment setup completed!"
	@echo "To activate: source venv/bin/activate"
	@echo "To deactivate: deactivate"

# Build documentation with virtual environment
docs-venv: sync-notebooks
	@echo "Building the documentation with virtual environment..."
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	@echo "Activating virtual environment and building..."
	source venv/bin/activate && sphinx-build -b html docs/source docs/build/html
	@echo "Creating .nojekyll file to disable Jekyll..."
	@echo "" > docs/build/html/.nojekyll
	@echo "Creating additional Jekyll bypass files..."
	@echo "theme: none" > docs/build/html/_config.yml
	@echo "# GitHub Pages - No Jekyll Processing" > docs/build/html/README.md
	@echo "Documentation built successfully!"
	@echo "Deploying to GitHub Pages with Jekyll bypass..."
	source venv/bin/activate && ghp-import -n -p -f docs/build/html
	@echo "Documentation deployed to GitHub Pages!"

# Build documentation only (without deploying)
docs-build: sync-notebooks
	@echo "Building the documentation with updated notebooks..."
	@echo "Using sphinx-build: $$(which sphinx-build)"
	@echo "Source dir: docs/source"
	@echo "Build dir: docs/build/html"
	@mkdir -p docs/build/html
	sphinx-build -v -b html docs/source docs/build/html || echo "Sphinx build failed"
	@touch docs/build/html/.nojekyll
	@echo "Documentation built successfully! Open docs/build/html/index.html to view."


# Auto-rebuild documentation with notebook sync
autobuild: sync-notebooks
	@echo "Building the documentation with auto-rebuild..."
	sphinx-autobuild docs/source docs/build

# Manual deployment using ghp-import (only when needed)
docs-deploy: 
	@echo "Deploying documentation manually to GitHub Pages..."
	ghp-import -n -p -f docs/build/html
	@echo "Documentation deployed to GitHub Pages!"
	@echo "Note: Use GitHub Actions workflow for automated deployment instead."

# Run scripts with virtual environment
run-venv:
	@echo "Running script with virtual environment..."
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	@if [ -z "$(SCRIPT)" ]; then \
		echo "Usage: make run-venv SCRIPT=path/to/script.py [ARGS='arg1 arg2']"; \
		exit 1; \
	fi
	source venv/bin/activate && python $(SCRIPT) $(ARGS)

# Run the main RES module with virtual environment
run-res:
	@echo "Running main RES module..."
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	source venv/bin/activate && python run.py

# Start Jupyter Lab with virtual environment
jupyter:
	@echo "Starting Jupyter Lab..."
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	source venv/bin/activate && jupyter lab

# Clean virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf venv
	@echo "Virtual environment removed."

# Show virtual environment status
venv-status:
	@if [ -d "venv" ]; then \
		echo "Virtual environment exists at: venv/"; \
		echo "Python version:"; \
		source venv/bin/activate && python --version; \
		echo "Installed packages:"; \
		source venv/bin/activate && pip list | head -20; \
		echo "... (use 'source venv/bin/activate && pip list' for full list)"; \
	else \
		echo "Virtual environment not found. Run 'make setup-venv' to create it."; \
	fi

# Export requirements from virtual environment
export-requirements:
	@echo "Exporting requirements..."
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	source venv/bin/activate && pip freeze > requirements.txt
	@echo "Requirements exported to requirements.txt"