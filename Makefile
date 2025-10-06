# RESource Project Makefile

.PHONY: help setupenv updateenv exportenv run docs autobuild deploy jupyter clean

# Default target
help:
	@echo "RESource Project - Available Commands:"
	@echo ""
	@echo "Environment:"
	@echo "  setupenv    - Setup conda environment from env/environment.yml"
	@echo "  updateenv   - Update existing conda environment"
	@echo "  exportenv   - Export current environment to env/environment.yml"
	@echo ""
	@echo "Documentation:"
	@echo "  docs        - Build and deploy documentation"
	@echo "  autobuild   - Live rebuild documentation (port 8000)"
	@echo "  deploy      - Deploy documentation to GitHub Pages"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       - Clean build files and cache"

# Environment Management
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
run:
	@echo "Running main RESource script..."
	@if conda env list | grep -q "^RESource "; then \
		conda run -n RESource python run.py $(ARGS); \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

# Documentation
docs:
	@echo "Building and deploying documentation..."
	@if conda env list | grep -q "^RESource "; then \
		mkdir -p docs/build/html; \
		mkdir -p docs/source/notebooks; \
		cp notebooks/*.ipynb docs/source/notebooks/ 2>/dev/null || true; \
		conda run -n RESource sphinx-build -b html docs/source docs/build/html; \
		echo "" > docs/build/html/.nojekyll; \
		conda run -n RESource ghp-import -n -p -f docs/build/html; \
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
		conda run -n RESource sphinx-autobuild docs/source docs/build --host 127.0.0.1 --port 8000; \
	else \
		echo "❌ Environment 'RESource' not found. Run 'make setupenv' first."; \
		exit 1; \
	fi

deploy:
	@echo "Deploying documentation to GitHub Pages..."
	@if conda env list | grep -q "^RESource "; then \
		conda run -n RESource ghp-import -n -p -f docs/build/html; \
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

