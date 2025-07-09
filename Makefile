.PHONY: setup install clean vscode

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

autobuild:
	@echo "Building the documentation..."
	sphinx-autobuild docs/source docs/build/html