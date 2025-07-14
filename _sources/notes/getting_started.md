# Getting Started

```{warning}
This library is under heavy development
```

This guide will help you install and set up RESource on your system.

## Prerequisites

- **Anaconda or Miniconda**: RESource uses conda for environment management
- **Git**: For cloning the repository
- **Python 3.8+**: Included in the conda environment

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DeltaE/RESource
cd RESource
```

### 2. Set Up the Environment

RESource provides a convenient Makefile for environment setup:

```bash
# Create and set up the conda environment
make setup
```

This command will:
- Create a new conda environment called `RES` (or as specified in `env/environment.yml`)
- Install all required dependencies
- Set up the development environment

### 3. Activate the Environment

After the setup completes, activate the environment:

```bash
conda activate RES
```

## Alternative Installation Methods

### Manual Environment Setup

If you prefer to set up the environment manually:

```bash
# Create environment from the environment file
conda env create -f env/environment.yml

# Activate the environment
conda activate RES
```

### Update Existing Environment

To update your existing environment with new dependencies:

```bash
make update
```

## Verify Installation

To verify that RESource is installed correctly, try importing the main module:

```python
import RES
print("RESource installed successfully!")
```

## Quick Start

### 1. Basic Usage

```python
from RES import RESources_builder

# Create a RESource builder instance
builder = RESources_builder(
    region_short_code='BC',  # British Columbia
    resource_type='solar'    # or 'wind'
)

# Access configuration
print(builder.config)
```

### 2. Process ATB Data

```python
from RES.atb import NREL_ATBProcessor

# Initialize ATB processor
atb_processor = NREL_ATBProcessor(
    region_short_code='BC',
    resource_type='solar'
)

# Pull and process data
solar_cost, wind_cost, bess_cost = atb_processor.pull_data()
```

## Working with Notebooks

RESource includes several Jupyter notebooks for demonstration and analysis:

- **Store_explorer.ipynb**: Explore the HDF5 data storage
- **Visuals_BC.ipynb**: Visualization examples for British Columbia
- **RESources_report_builder.ipynb**: Generate assessment reports

To start Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

## Documentation

### Build Documentation Locally

To build the documentation locally:

```bash
# Build documentation with updated notebooks (recommended)
make docs

# Auto-rebuild documentation (watches for changes and syncs notebooks)
make autobuild

# Sync notebooks only (without building docs)
make sync-notebooks

# Or build manually
cd docs
make html
```

The `make docs` and `make autobuild` commands automatically sync the latest notebooks from the root `notebooks/` directory to `docs/source/notebooks/` before building the documentation.

The documentation will be available at `docs/build/html/index.html`.

## Automated Documentation Deployment

RESource includes automated GitHub Pages deployment for documentation:

### GitHub Actions Workflows

Two workflows are available for documentation deployment:

1. **`deploy-docs.yml`** (Primary)
   - Automatically triggers on pushes to `main` or `master` branch
   - Uses pip installation for faster builds
   - Syncs notebooks from root directory
   - Deploys to GitHub Pages

2. **`deploy-docs-conda.yml`** (Backup)
   - Manual trigger only (workflow_dispatch)
   - Uses conda environment for complex dependencies
   - Fallback option if pip installation fails

### Setup GitHub Pages

To enable GitHub Pages for your repository:

1. Go to your repository settings
2. Navigate to "Pages" section
3. Set source to "GitHub Actions"
4. The documentation will be available at: `https://yourusername.github.io/RESource`

### Workflow Features

- **Automatic notebook sync**: Copies notebooks from root to docs directory
- **Dependency management**: Installs project with docs dependencies
- **Error handling**: Graceful handling of missing notebooks
- **Branch protection**: Only deploys from main/master branches
- **Manual triggers**: Allows manual deployment when needed

### Local Testing Before Deployment

Always test your documentation locally before pushing:

```bash
# Test the complete build process
make docs

# Verify notebooks are included
ls docs/source/notebooks/

# Check for build errors
cat docs/build/html/index.html
```

## Configuration

RESource uses YAML configuration files located in the `config/` directory. Key configuration files:

- Regional settings and data sources
- Technology parameters and assumptions

## Troubleshooting

### Common Issues

1. **Environment Creation Fails**
   ```bash
   # Clean up and try again
   make clean
   make setup
   ```

2. **Import Errors**
   - Ensure the environment is activated: `conda activate RES`
   - Check that all dependencies are installed

3. **Missing Data**
   - Some workflows require downloading external datasets
   - Check the logs for download status and errors

### Getting Help

- Check the [API documentation](api.md) for detailed class and method documentation
- Explore the example notebooks in the `notebooks/` directory
- Review the case studies in `CASE_studies/`

## Development

### Export Environment

If you've added new dependencies, export the updated environment:

```bash
make export
```

### Clean Environment

To remove the conda environment:

```bash
make clean
```

## Next Steps

- Explore the [API Reference](api.md) for detailed documentation
- Try the example notebooks to understand the workflow
- Check out the case studies for real-world applications
- Read about the methodology in the main documentation

