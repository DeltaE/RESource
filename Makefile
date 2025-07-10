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
	sphinx-autobuild docs/source docs/build/html

# Manual deployment using ghp-import (only when needed)
docs-deploy-manual: docs
	@echo "Deploying documentation manually to GitHub Pages..."
	ghp-import -n -p -f docs/build/html
	@echo "Documentation deployed to GitHub Pages!"
	@echo "Note: Use GitHub Actions workflow for automated deployment instead."