.PHONY: setup install clean export sync-notebooks docs autobuild

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

# Build documentation with notebook sync
docs: sync-notebooks
	@echo "Building the documentation with updated notebooks..."
	cd docs && make html
	@echo "Documentation built successfully! Open docs/build/html/index.html to view."

# Auto-rebuild documentation with notebook sync
autobuild: sync-notebooks
	@echo "Building the documentation with auto-rebuild..."
	sphinx-autobuild docs/source docs/build/html