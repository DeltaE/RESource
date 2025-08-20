# RESource Project Makefile - Conda Environment Management

.PHONY: help setup-conda clean-conda conda-status run-res run-conda jupyter docs-conda docs-build deploy sync-notebooks export-env autobuild

# Default target
help:
	@echo "RESource Project - Available commands:"
	@echo ""
	@echo "Environment Management:"
	@echo "  setup-conda          - Create conda environment 'RES' from env/environment.yml"
	@echo "  clean-conda          - Remove conda environment 'RES'"
	@echo "  conda-status         - Show conda environment status"
	@echo "  export-env           - Export environment to env/environment_exported.yml"
	@echo ""
	@echo "Running Code:"
	@echo "  run-res              - Run main RESource module"
	@echo "  run-conda SCRIPT=... - Run any script with conda environment"
	@echo "  jupyter              - Start Jupyter Lab with conda environment"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-conda           - Build and deploy documentation"
	@echo "  docs-build           - Build documentation only"
	@echo "  deploy               - Deploy documentation to GitHub Pages"
	@echo "  sync-notebooks       - Sync notebooks to docs"
	@echo "  autobuild            - Live rebuild documentation with auto-reload (port 8000)"
	@echo "  autobuild-alt        - Live rebuild documentation on alternative port (8001)"
	@echo "  autobuild-remote     - Live rebuild with remote development instructions"

# Environment Management
setup-conda:
	@echo "Creating conda environment 'RES'..."
	@if conda env list | grep -q "^RES "; then \
		echo "Environment 'RES' already exists. Updating..."; \
		conda env update -f env/environment.yml; \
	else \
		echo "Creating new environment 'RES'..."; \
		conda env create -f env/environment.yml; \
	fi
	@echo "Installing RESource package in development mode..."
	conda run -n RES pip install -e .
	@echo "✅ Conda environment 'RES' setup completed!"
	@echo "To activate: conda activate RES"
	@echo "To deactivate: conda deactivate"

clean-conda:
	@echo "Removing conda environment 'RES'..."
	@conda env remove -n RES -y || echo "Environment 'RES' not found"
	@echo "Environment removed."

conda-status:
	@echo "=== Conda Environment Status ==="
	@if conda env list | grep -q "^RES "; then \
		echo "✅ Environment 'RES' exists"; \
		echo ""; \
		echo "Environment details:"; \
		conda env list | grep RES; \
		echo ""; \
		echo "Python version:"; \
		conda run -n RES python --version; \
		echo ""; \
		echo "Key packages installed:"; \
		conda run -n RES conda list | grep -E "(numpy|pandas|geopandas|atlite|sphinx)" || echo "Package list unavailable"; \
		echo ""; \
		echo "RESource module:"; \
		conda run -n RES python -c "import RES; print('✅ RES module available')" 2>/dev/null || echo "❌ RES module not available"; \
	else \
		echo "❌ Environment 'RES' not found"; \
		echo "Run 'make setup-conda' to create it"; \
	fi

export-env:
	@echo "Exporting current environment to env/environment_exported.yml..."
	conda env export -n RES > env/environment_exported.yml
	@echo "Environment exported to env/environment_exported.yml"

# Running Code
run-res:
	@echo "Running main RESource module..."
	@if conda env list | grep -q "^RES "; then \
		conda run -n RES python run.py; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

run-conda:
	@if [ -z "$(SCRIPT)" ]; then \
		echo "Usage: make run-conda SCRIPT=path/to/script.py [ARGS='arg1 arg2']"; \
		echo "Example: make run-conda SCRIPT=examples/analysis.py"; \
		echo "Example: make run-conda SCRIPT=workflow/scripts/bess_module_v2.py ARGS='config/config_CAN.yaml bess'"; \
		exit 1; \
	fi
	@echo "Running $(SCRIPT) with conda environment 'RES'..."
	@if conda env list | grep -q "^RES "; then \
		conda run -n RES python $(SCRIPT) $(ARGS); \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

jupyter:
	@echo "Starting Jupyter Lab with conda environment 'RES'..."
	@if conda env list | grep -q "^RES "; then \
		conda run -n RES jupyter lab; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

# Documentation
sync-notebooks:
	@echo "Syncing notebooks from root to docs/source/notebooks..."
	@mkdir -p docs/source/notebooks
	@cp notebooks/Store_explorer.ipynb docs/source/notebooks/ 2>/dev/null || echo "Store_explorer.ipynb not found at 'notebooks', skipping..."
	@cp notebooks/Visuals_BC.ipynb docs/source/notebooks/ 2>/dev/null || echo "Visuals_BC.ipynb not found at 'notebooks', skipping..."
	@cp resource_module_runner.ipynb docs/source/notebooks/ 2>/dev/null || echo "resources_playground.ipynb not found at root, skipping..."
	@echo "Notebooks synced successfully!"

docs-build: sync-notebooks
	@echo "Building documentation with conda environment 'RES'..."
	@if conda env list | grep -q "^RES "; then \
		mkdir -p docs/build/html; \
		conda run -n RES sphinx-build -b html docs/source docs/build/html; \
		echo "Creating .nojekyll file to disable Jekyll..."; \
		echo "" > docs/build/html/.nojekyll; \
		echo "Documentation built successfully in docs/build/html/"; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

docs-conda: docs-build
	@echo "Deploying documentation to GitHub Pages..."
	@echo "Creating additional Jekyll bypass files..."
	@echo "theme: none" > docs/build/html/_config.yml
	@echo "# GitHub Pages - No Jekyll Processing" > docs/build/html/README.md
	@echo "Verifying bypass files exist:"
	@ls -la docs/build/html/.nojekyll docs/build/html/_config.yml docs/build/html/README.md
	@echo "Deploying with ghp-import..."
	@if conda env list | grep -q "^RES "; then \
		conda run -n RES ghp-import -n -p -f docs/build/html; \
		echo "✅ Documentation deployed to GitHub Pages!"; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

# Auto-rebuild documentation with live reload
autobuild: sync-notebooks
	@echo "Starting live documentation rebuild with conda environment 'RES'..."
	@if conda env list | grep -q "^RES "; then \
		echo "🔄 Auto-building documentation with live reload..."; \
		echo "📂 Source: docs/source"; \
		echo "🌐 Build: docs/build"; \
		echo "🔗 Server starting on http://127.0.0.1:8000"; \
		echo "🌐 For remote access: VS Code will auto-forward port 8000"; \
		echo "📝 Check VS Code's 'PORTS' tab to see the forwarded URL"; \
		echo "⚡ Live reload enabled - changes will auto-refresh"; \
		conda run -n RES sphinx-autobuild docs/source docs/build --host 127.0.0.1 --port 8000; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

# Auto-rebuild on alternative port (if 8000 is busy)
autobuild-alt: sync-notebooks
	@echo "Starting live documentation rebuild on alternative port..."
	@if conda env list | grep -q "^RES "; then \
		echo "🔄 Auto-building documentation with live reload on port 8001..."; \
		echo "📂 Source: docs/source"; \
		echo "🌐 Build: docs/build"; \
		echo "🔗 Server starting on http://127.0.0.1:8001"; \
		echo "🌐 For remote access: VS Code will auto-forward port 8001"; \
		echo "📝 Check VS Code's 'PORTS' tab to see the forwarded URL"; \
		conda run -n RES sphinx-autobuild docs/source docs/build --host 127.0.0.1 --port 8001; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

# Auto-rebuild with explicit remote setup instructions
autobuild-remote: sync-notebooks
	@echo "🌐 Starting documentation server for REMOTE development..."
	@if conda env list | grep -q "^RES "; then \
		echo "📡 Remote Setup Instructions:"; \
		echo "   1. VS Code will automatically forward port 8000"; \
		echo "   2. Look for 'PORTS' tab in VS Code terminal panel"; \
		echo "   3. Click on the forwarded URL to open in browser"; \
		echo "   4. Or use Command Palette: 'Ports: Focus on Ports View'"; \
		echo ""; \
		echo "🔄 Starting sphinx-autobuild on 127.0.0.1:8000..."; \
		echo "📂 Source: docs/source → Build: docs/build"; \
		echo "⚡ Live reload enabled - edit files and see changes instantly!"; \
		echo ""; \
		conda run -n RES sphinx-autobuild docs/source docs/build --host 127.0.0.1 --port 8000; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

# Deployment
deploy: docs-build
	@echo "Deploying documentation to GitHub Pages..."
	@echo "Creating additional Jekyll bypass files..."
	@echo "theme: none" > docs/build/html/_config.yml
	@echo "# GitHub Pages - No Jekyll Processing" > docs/build/html/README.md
	@echo "Verifying bypass files exist:"
	@ls -la docs/build/html/.nojekyll docs/build/html/_config.yml docs/build/html/README.md
	@echo "Deploying with ghp-import..."
	@if conda env list | grep -q "^RES "; then \
		conda run -n RES ghp-import -n -p -f docs/build/html; \
		echo "✅ Documentation deployed to GitHub Pages!"; \
		echo "🌐 Visit: https://deltae.github.io/RESource/"; \
	else \
		echo "❌ Conda environment 'RES' not found. Run 'make setup-conda' first."; \
		exit 1; \
	fi

# Legacy aliases (for backward compatibility)
docs: docs-conda
setup-venv: setup-conda
	@echo "Note: setup-venv is deprecated. Use 'make setup-conda' instead."

# Cleanup
clean-docs:
	@echo "Cleaning documentation build files..."
	@rm -rf docs/build/
	@echo "Documentation build files removed."

clean-all: clean-conda clean-docs
	@echo "All build files and environments cleaned."