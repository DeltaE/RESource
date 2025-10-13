# RESource - Complete Setup Guide

**The definitive guide to setting up the RESource (Renewable Energy Resource Assessment) framework**

## 🎯 Quick Start

### Prerequisites
- **Python 3.12** (recommended)
- **Git** 
- **Conda or Miniconda** (required for environment management)
- **Linux/macOS/WSL2** environment

### 1. Clone Repository
```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
```

### 2. Environment Setup (Recommended)
```bash
# Create environment from specification - ONE COMMAND!
conda env create -f env/environment.yml
conda activate RESource

# Verify installation
python run.py --help
```

**That's it!** 🎉 The `env/environment.yml` file contains all exact package versions tested and verified to work.

---

## 📦 Environment Details

### Reproducibility & Cross-Platform Support

The `env/environment.yml` file uses **exact version pinning** (`==`) for all packages, providing:

✅ **Reproducible builds** - Same versions across all installations  
✅ **Cross-platform compatibility** - Works on Linux, macOS, and Windows/WSL2  
✅ **No hash conflicts** - Avoids platform-specific hash mismatches  
✅ **Tested stability** - All 176+ packages verified working together  

### Key Package Versions
- **Python**: 3.12.11
- **GeoPandas**: 1.0.1 (with exact spatial dependencies)
- **Shapely**: 2.0.6
- **Rasterio**: 1.4.3
- **PyProj**: 3.6.1
- **Dask-GeoPandas**: 0.4.2
- **Atlite**: 0.4.1 (renewable energy modeling)

---

## 🛠️ Alternative Setup Methods

### Method 1: Using Makefile
```bash
# Automated setup with verification
make setupenv
```

### Method 2: Manual Setup (Advanced Users)
```bash
# Create environment manually
conda create -n RESource python=3.12 -y
conda activate RESource

# Install core geospatial packages
pip install geopandas==1.0.1 shapely==2.0.6 fiona==1.10.1 \
           pyproj==3.6.1 rasterio==1.4.3 dask-geopandas==0.4.2

# Install additional scientific packages
pip install atlite xarray netcdf4 matplotlib seaborn jupyter \
           scikit-learn pyyaml tqdm h5py tables plotly
```

### Method 3: Interactive Setup
```bash
# Guided setup with menu options
./setup_environment_clean.sh
```

---

## 🚀 Getting Started

### Basic Usage
```bash
# Activate environment
conda activate RESource

# Run Canadian analysis
python run.py config/config_CAN_baseline.yaml

# Run specific region analysis
python run.py config/config_WB6.yaml -r AL BA

# Get help
python run.py --help
```

### Available Configurations
- **`config_CAN_baseline.yaml`** - Canadian provinces analysis
- **`config_WB6.yaml`** - Western Balkans countries
- **Custom configurations** - Create your own analysis regions

---

## 🔧 Development Setup

### For Contributors & Developers

```bash
# Clone and setup
git clone https://github.com/DeltaE/RESource.git
cd RESource

# Create development environment
conda env create -f env/environment.yml
conda activate RESource

# Install in development mode
pip install -e .

# Run tests
make test  # or pytest tests/

# Build documentation
make docs
```

### Development Tools Included
- **Jupyter Lab** - Interactive development
- **Testing framework** - pytest with coverage
- **Documentation** - Sphinx with MkDocs
- **Code quality** - Pre-commit hooks available

---

## 📁 Project Structure

```
RESource/
├── env/
│   └── environment.yml          # Complete environment specification
├── config/
│   ├── config_CAN_baseline.yaml # Canadian analysis config
│   └── config_WB6.yaml         # Western Balkans config
├── RES/                        # Core RESource module
├── data/                       # Input datasets
├── results/                    # Analysis outputs
├── notebooks/                  # Jupyter notebooks
├── docs/                      # Documentation source
├── run.py                     # Main analysis script
└── Makefile                   # Automated commands
```

---

## 🐛 Troubleshooting

### Common Issues

**Environment conflicts:**
```bash
# Remove existing environment and recreate
conda env remove -n RESource
conda env create -f env/environment.yml
```

**Import errors:**
```bash
# Verify environment activation
conda info --envs
conda activate RESource
python -c "import geopandas; print('✅ GeoPandas working')"
```

**Permission issues:**
```bash
# Ensure proper conda installation
conda update conda
conda clean --all
```

### Getting Help

- **Documentation**: [https://deltae.github.io/RESource/](https://deltae.github.io/RESource/)
- **Issues**: [GitHub Issues](https://github.com/DeltaE/RESource/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DeltaE/RESource/discussions)

---

## 📊 What RESource Does

RESource is a comprehensive framework for **renewable energy resource assessment**:

- **🌬️ Wind Energy**: Analyze wind patterns and capacity factors
- **☀️ Solar Energy**: Assess photovoltaic and concentrated solar potential  
- **⚡ Grid Integration**: Model transmission constraints and capacity
- **🗺️ Spatial Analysis**: High-resolution geospatial resource mapping
- **📈 Optimization**: Economic and technical potential assessment
- **🌍 Multi-Regional**: Support for countries and custom regions

### Key Features
- **Reproducible workflows** with locked dependencies
- **Flexible configuration** for different regions/scenarios
- **High-performance computing** with Dask parallelization
- **Rich visualization** with interactive maps and plots
- **Comprehensive documentation** and examples

---

## 🏆 Why This Setup Works

### Tested & Verified
- ✅ **Cross-platform**: Linux, macOS, Windows/WSL2
- ✅ **Geospatial stack**: All GDAL/GEOS/PROJ dependencies resolved
- ✅ **Scientific computing**: NumPy, SciPy, Pandas optimized versions
- ✅ **Renewable energy**: Atlite and related packages integrated
- ✅ **Performance**: Dask for parallel processing included

### Production Ready
The environment specification has been tested with:
- Multiple operating systems and architectures
- Real-world renewable energy datasets
- Complex geospatial processing workflows
- Large-scale parallel computations

---

*Need help? Check the [documentation](https://deltae.github.io/RESource/) or open an [issue](https://github.com/DeltaE/RESource/issues).*