# RESource Setup Guide

**Simple, clean setup for RESource - Renewable Energy Resource Assessment framework**

## 🚀 Quick Start

### Option 1/3: Interactive Setup (Recommended for New Users)
```bash
./setup_environment_clean.sh
```
**Features:** Environment selection menu • Colored output • Auto-detects mamba • Verification

### Option 2/3: Quick Setup (For Developers)
```bash
# Standard environment (most users)
make setupenv

# Or manual setup
conda env create -f env/environment.yml
conda activate RESource
```

### Option 3/3: Manual Setup
```bash
# Create and activate environment
conda env create -f env/environment.yml
conda activate RESource

# Verify installation
python -c "import RES; print('✅ RESource ready')"
```

## 📋 Environment Options

Choose the environment that fits your use case:

| Environment | Use Case | Command |
|-------------|----------|---------|
| **Standard** | Research, analysis, most users | `conda env create -f env/environment.yml` |
| **Development** | Contributing, testing, code quality | `conda env create -f env/environment_development.yml` |
| **Production** | Deployment, minimal footprint | `conda env create -f env/environment_production.yml` |

## 🔧 What's Included

### Core Scientific Computing
- **Python 3.12** - Latest stable version
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation  
- **SciPy** - Scientific algorithms
- **Matplotlib** - Plotting

### Geospatial & GIS
- **GeoPandas** - Geospatial data analysis
- **Rasterio** - Raster data processing
- **Shapely** - Geometric operations
- **PyProj** - Coordinate transformations
- **Fiona** - Vector data I/O

### Climate & Energy Data
- **Xarray** - Multi-dimensional data
- **NetCDF4** - Climate data format
- **Atlite** - Renewable energy modeling
- **CDSAPI** - Climate Data Store access

### Visualization & Analysis
- **Plotly** - Interactive plots
- **Folium** - Interactive maps
- **Seaborn** - Statistical visualization
- **Jupyter** - Interactive computing

## ✅ Verification

After setup, test your installation:

```bash
# Activate environment
conda activate RESource

# Quick test
python -c "
import numpy as np
import pandas as pd
import geopandas as gpd
import RES
print('🎉 All systems go!')
"

# Full validation
python validate_environment.py
```

## 🛠️ Management

### Update Environment
```bash
conda env update -f env/environment.yml --prune
```

### Remove Environment
```bash
conda env remove -n RESource
```

### Export for Sharing
```bash
conda env export > my_environment.yml
```

## 🐛 Troubleshooting

### Environment Creation Fails
```bash
# Clear conda cache
conda clean --all

# Try with mamba (faster)
conda install mamba -n base -c conda-forge
mamba env create -f env/environment.yml
```

### RESource Import Fails
```bash
# Ensure RESource is installed in editable mode
conda activate RESource
pip install -e .
```

### Package Conflicts
```bash
# Update conda
conda update conda

# Use strict channel priority
conda config --set channel_priority strict
```

## 💡 Tips

- **Use Standard Environment** for most work
- **Use Development Environment** when contributing
- **Use Production Environment** for deployment
- **Export environments** for reproducibility
- **Consider mamba** for faster installs

## 📊 System Requirements

- **OS**: Linux, macOS, Windows
- **Python**: 3.12+ (managed by conda)
- **RAM**: 4GB+ (8GB+ recommended)
- **Storage**: 2GB+ for environment
- **Internet**: Required for downloads

---

**Need help?** Run `python validate_environment.py` to diagnose issues or check the [main README](README.md).