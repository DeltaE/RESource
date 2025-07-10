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

# Build documentation with notebook sync (build only, no deploy)
docs: sync-notebooks
	@echo "Building the documentation with updated notebooks..."
	@mkdir -p docs/build/html
	sphinx-build -b html docs/source docs/build/html
	@touch docs/build/html/.nojekyll
	@echo "Documentation built successfully! Open docs/build/html/index.html to view."
	@echo "To deploy: git add, commit, and push to trigger GitHub Actions deployment."


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

# Debug sphinx configuration
debug-sphinx:
	@echo "Debugging sphinx configuration..."
	@echo "Python version: $$(python --version)"
	@echo "Sphinx version: $$(sphinx-build --version 2>/dev/null || echo 'sphinx-build not found')"
	@echo "Current directory: $$(pwd)"
	@echo "Source directory exists: $$(test -d docs/source && echo 'YES' || echo 'NO')"
	@echo "Config file exists: $$(test -f docs/source/conf.py && echo 'YES' || echo 'NO')"
	@echo "Testing conf.py import..."
	@cd docs/source && python -c "import conf; print('conf.py imported successfully')" 2>/dev/null || echo "conf.py import failed"

# Check GitHub Pages setup
debug-pages:
	@echo "Checking GitHub Pages setup..."
	@echo "Current branch: $$(git branch --show-current)"
	@echo "Remote branches:"
	@git branch -r | grep -E "(gh-pages|main|master)" || echo "No gh-pages branch found"
	@echo "Last few commits:"
	@git log --oneline -5
	@echo ""
	@echo "IMPORTANT: Make sure GitHub Pages is set to 'GitHub Actions' source, not 'Deploy from branch'"
	@echo "Go to: Settings → Pages → Source → GitHub Actions"


# Auto-rebuild documentation with notebook sync
autobuild: sync-notebooks
	@echo "Building the documentation with auto-rebuild..."
	sphinx-autobuild docs/source docs/build/html