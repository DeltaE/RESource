# RESource Project Complete Setup & Usage Guide

## 🎯 Quick Start (New Users)

### Prerequisites
- Python 3.11+
- Git
- **Conda or Miniconda** (recommended for better package management)

### 1. Clone and Navigate
```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
```

### 2. Environment Setup (Conda - Recommended)

```bash
# Create conda environment with all dependencies
make setup-conda

# Activate environment
conda activate RES

# Test installation
python -c "import RES; print('✅ RESource is ready!')"
```

### 3. Optional: Fix Conda Conflicts (if you see double environments)
```bash
# If you see (RES) (base) instead of just (RES)
conda config --set auto_activate_base false
source ~/.bashrc
```

## 🛠️ Development Environment

### Conda Environment Commands

#### Setup & Management
```bash
make setup-conda       # Create/update conda environment 'RES'
make conda-status      # Check environment health and packages
make clean-conda       # Remove conda environment
make export-env        # Export environment to env/environment_exported.yml
```

#### Daily Usage
```bash
conda activate RES     # Activate environment (shows: (RES) $)
conda deactivate      # Deactivate when done
```

### Running Code

#### Main RESource Module
```bash
make run-res
# Or manually:
conda activate RES
python run.py
```

#### Custom Scripts
```bash
# Run any script with the conda environment
make run-conda SCRIPT=path/to/script.py

# Run with arguments
make run-conda SCRIPT=workflow/scripts/bess_module_v2.py ARGS="config/config_CAN.yaml bess"

# Examples:
make run-conda SCRIPT=examples/quick_start.py
make run-conda SCRIPT=RES/cli.py ARGS="--help"
```

#### Jupyter Notebooks
```bash
make jupyter
# Or manually:
conda activate RES
jupyter lab
```

## 📚 Documentation

### Building Documentation
```bash
# Build and deploy to GitHub Pages
make docs-conda

# Build locally only
make docs-build
```

### Documentation Features
- **Auto-generated API docs** from code docstrings
- **Jupyter notebook integration** from `/notebooks` folder
- **Automatic deployment** to GitHub Pages
- **NODOC support** - add `NODOC` or `:nodoc:` to docstrings to exclude from docs

### Working with Notebooks
- Notebooks in `/notebooks` are automatically synced to documentation
- Use the conda environment kernel for consistency
- Examples: `Store_explorer.ipynb`, `Visuals_BC.ipynb`

## 🏗️ Project Structure

```
RESource/
├── RES/                    # Main Python package
│   ├── __init__.py
│   ├── atb.py             # NREL ATB processor
│   ├── tech.py            # Technology modules
│   ├── boundaries.py      # Geographic boundaries
│   └── ...
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation source
│   ├── source/
│   └── build/
├── env/                   # Environment definitions
│   ├── environment.yml       # Main conda environment definition
│   └── environment_exported.yml # Exported snapshot
├── config/               # Configuration files
├── Makefile             # Build automation
└── pyproject.toml       # Project configuration
```

## 🔧 VS Code Integration

### Automatic Setup
The project includes VS Code configurations for:
- **Auto conda environment activation**
- **Python interpreter selection** (uses conda RES environment)
- **Jupyter kernel configuration**
- **Integrated terminal with conda activation**
- **Task runner integration**

### Recommended Extensions
- Python Extension Pack
- Jupyter
- Makefile Tools
- MyST-Markdown
- Black Formatter

### Quick Actions
- `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose conda RES environment
- `Ctrl+Shift+P` → "Tasks: Run Task" → Choose from available Make commands
- New terminals automatically activate the conda environment

## 📦 Dependencies Included

### Core Scientific Computing (from conda-forge)
- **python=3.11**: Python interpreter
- **numpy**: Numerical computing with optimized BLAS
- **pandas**: Data manipulation
- **scipy**: Scientific computing
- **xarray**: Multi-dimensional data

### Geospatial Stack (optimized conda versions)
- **geopandas**: Geographic data analysis
- **rasterio**: Raster data I/O
- **shapely**: Geometric operations
- **fiona**: Vector data I/O
- **pyproj**: Coordinate transformations
- **cartopy**: Cartographic projections

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots

### Renewable Energy Specific (via pip)
- **atlite**: Renewable energy modeling
- **cdsapi**: Climate Data Store API
- **pygadm**: Administrative boundaries

### Web & Development
- **folium**: Interactive maps
- **dash**: Web applications
- **pyrosm**: OpenStreetMap data
- **osmnx**: Street network analysis
- **requests**: HTTP requests
- **beautifulsoup4**: Web scraping

### Documentation & Notebooks
- **sphinx**: Documentation generator
- **myst-parser**: Markdown parser for Sphinx
- **sphinx-book-theme**: Documentation theme
- **nbsphinx**: Jupyter notebook integration
- **jupyter**: Notebook ecosystem
- **jupyterlab**: Modern notebook interface
- **ghp-import**: GitHub Pages deployment

## 🚨 Troubleshooting

### Environment Issues

#### Double Environment `(RES) (base)`
```bash
# Fix conda auto-activation
conda config --set auto_activate_base false
source ~/.bashrc
```

#### Conda Environment Not Found
```bash
# Recreate environment
make clean-conda
make setup-conda
```

#### Package Import Errors
```bash
# Check environment status
make conda-status

# Reinstall dependencies
conda activate RES
conda env update -f environment.yml
```

### Documentation Issues

#### Sphinx Build Errors
```bash
# Clean and rebuild
make clean-docs
make docs-build

# Check specific errors
conda activate RES
cd docs
sphinx-build -v source build/html
```

#### Missing Images in Docs
```bash
# Check static files
ls docs/source/_static/

# Rebuild with image sync
make sync-notebooks
make docs-build
```

### Jupyter Issues

#### Kernel Not Found
```bash
# Reinstall kernel
conda activate RES
python -m ipykernel install --user --name=RES --display-name="Python (RES)"
```

#### Package Not Available in Notebook
```bash
# Verify environment
make jupyter
# In notebook: !which python
# Should show: ~/miniconda3/envs/RES/bin/python
```

## 🔄 Common Workflows

### Daily Development
```bash
# Start working
cd RESource
conda activate RES

# Make changes to code
# ... edit files ...

# Test changes
python -m pytest  # if tests exist
python run.py

# Update documentation
make docs-build

# Commit changes
git add .
git commit -m "Your changes"
git push origin main  # Auto-deploys docs
```

### Adding New Dependencies
```bash
# For conda packages
conda activate RES
conda install new-package

# For pip packages
conda activate RES
pip install new-package

# Update environment file
make export-env
```

### Sharing Your Work
```bash
# Export clean environment
make export-env

# Share environment.yml and pyproject.toml
# Others can recreate with: make setup-conda
```

## 🌐 Deployment & Sharing

### GitHub Pages (Automatic)
- Documentation auto-deploys on push to main branch
- Available at: `https://yourusername.github.io/RESource`
- Uses GitHub Actions with Jekyll bypass

### Manual Deployment
```bash
make docs-conda  # Builds and deploys manually
```

### Environment Sharing
```bash
# Export current environment
make export-env

# Others can recreate with:
# make setup-conda
```

## 📋 Checklist for New Users

- [ ] Install conda/miniconda
- [ ] Clone repository
- [ ] Run `make setup-conda`
- [ ] Activate with `conda activate RES`
- [ ] Test with `python -c "import RES; print('Success!')"`
- [ ] Open VS Code and select conda RES interpreter
- [ ] Try running `make jupyter`
- [ ] Build docs with `make docs-build`
- [ ] Check environment with `make conda-status`

## 🆘 Getting Help

### Check Status
```bash
make conda-status  # Environment diagnostics
```

### Common Commands Summary
```bash
# Environment
make setup-conda
conda activate RES

# Development
make run-res
make jupyter
make run-conda SCRIPT=path/to/file.py

# Documentation
make docs-conda
make sync-notebooks

# Maintenance
make clean-conda
make export-env
make conda-status
```

## 🎉 Why Conda vs Virtual Environment?

### Performance Benefits
- **Optimized packages**: NumPy/SciPy with MKL/OpenBLAS
- **Better geospatial support**: Conda-forge has optimized GDAL, PROJ packages
- **Faster installs**: Binary packages instead of compilation
- **Cross-platform**: Consistent across Windows, macOS, Linux

### Reliability Benefits
- **Dependency resolution**: Better handling of complex scientific packages
- **Native libraries**: GDAL, PROJ, GEOS compiled optimally
- **Reproducibility**: More consistent environments across systems

---

## 📄 License & Contributing

This project is licensed under MIT License. Contributions welcome!

For issues or questions, please check the troubleshooting section above or open an issue on GitHub.
