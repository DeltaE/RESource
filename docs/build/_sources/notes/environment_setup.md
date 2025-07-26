# Conda Environment Setup for RESource

The RESource project uses a conda environment called `RES` for reproducible analysis workflows. The environment specification is located in `env/environment.yml`.

## Quick Start

### 1. Prerequisites
- Conda or Miniconda installed
- Git

### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/DeltaE/RESource.git
cd RESource

# Create conda environment from env/environment.yml
make setup-conda

# Activate environment
conda activate RES

# Test installation
python -c "import RES; print('✅ RESource is ready!')"
```

## Environment Management Commands

### Setup and Management
```bash
make setup-conda       # Create/update conda environment 'RES' from env/environment.yml
make conda-status      # Check environment status and packages
make clean-conda       # Remove conda environment
make export-env        # Export current environment to env/environment_exported.yml
```

### Daily Usage
```bash
conda activate RES     # Activate environment
conda deactivate      # Deactivate environment
```

## Running Code

### Main RESource Module
```bash
make run-res
```

### Custom Scripts
```bash
# Basic usage
make run-conda SCRIPT=path/to/script.py

# With arguments
make run-conda SCRIPT=workflow/scripts/bess_module_v2.py ARGS="config/config_CAN.yaml bess"

# Examples
make run-conda SCRIPT=examples/analysis.py
make run-conda SCRIPT=RES/cli.py ARGS="--help"
```


## Documentation

### Build and Deploy
```bash
# Build and deploy to GitHub Pages
make docs-conda

# Build locally only
make docs-build
```

## Environment Structure

The `RES` conda environment includes:

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

### Climate and Weather
- **netcdf4**: Climate data handling
- **h5py**: HDF5 file format

### Development Tools
- **jupyter**: Notebook ecosystem
- **jupyterlab**: Modern notebook interface
- **ipykernel**: Jupyter kernel
- **sphinx**: Documentation

### Specialized Packages (via pip)
- **atlite**: Renewable energy modeling
- **cdsapi**: Climate Data Store API
- **pygadm**: Administrative boundaries
- **folium**: Interactive maps
- **dash**: Web applications
- **pyrosm**: OpenStreetMap data
- **osmnx**: Street network analysis

## Environment File Location

The environment definition is located at `env/environment.yml`:

```yaml
# env/environment.yml
name: RES
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - geopandas
  # ... other conda packages
  - pip
  - pip:
    - atlite
    - cdsapi
    # ... pip-only packages
```

## File Structure

The environment files are organized in the `env/` directory:

```
RESource/
├── env/
│   ├── environment.yml          # Main conda environment definition
│   └── environment_exported.yml # Exported snapshot (created by make export-env)
├── docs/
├── RES/
├── Makefile                     # References env/environment.yml
└── README.md
```

### Why `env/` Directory?

- **Organization**: Keeps environment files separate from code
- **Multiple environments**: Can store different environment variants
- **Clear purpose**: Makes it obvious what the directory contains
- **Standard practice**: Common pattern in Python projects

### Working with Environment Files

```bash
# Create environment from env/environment.yml
make setup-conda

# Export current environment to env/environment_exported.yml  
make export-env

# Manually update environment
conda env update -f env/environment.yml

# View environment definition
cat env/environment.yml
```

## Environment File

The `environment.yml` file defines the conda environment:

```yaml
name: RES
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - geopandas
  # ... other conda packages
  - pip
  - pip:
    - atlite
    - cdsapi
    # ... pip-only packages
```

## Sharing and Collaboration

### Export Current Environment
```bash
make export-env
# Creates environment_exported.yml with exact versions
```

### Reproduce Environment
```bash
# On any machine with conda
git clone <repository>
cd RESource
make setup-conda
```

## Troubleshooting

### Environment Creation Issues
```bash
# Clean and recreate
make clean-conda
make setup-conda
```

### Package Conflicts
```bash
# Check environment status
make conda-status

# Update environment
conda env update -f environment.yml
```

### Missing Packages
```bash
# Activate environment and install
conda activate RES
conda install package-name
# or
pip install package-name

# Export updated environment
make export-env
```

### Jupyter Kernel Issues
```bash
conda activate RES
python -m ipykernel install --user --name=RES --display-name="Python (RES)"
```

## Best Practices
## Daily Workflow, Maintenance, and Collaboration

| Task Category            | Step/Action                                                                                  | Command/Description                                 |
|-------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------|
| **Daily Workflow**      | Activate environment                                                                         | `conda activate RES`                                |
|                         | Work on analysis/code                                                                        | *(your usual workflow)*                             |
|                         | Test code                                                                                    | `make run-res` or `make jupyter`                    |
|                         | Update documentation                                                                         | `make docs-build`                                   |
|                         | Deactivate environment                                                                       | `conda deactivate`                                  |
| **Environment Maintenance** | Regular updates                                                                          | `conda env update -f environment.yml`               |
|                         | Export changes after adding packages                                                         | `make export-env`                                   |
|                         | Clean rebuild if issues arise                                                                | `make clean-conda && make setup-conda`              |
| **Team Collaboration**  | Commit environment file to version control                                                   | `git add environment.yml && git commit`             |
|                         | Team members set up environment                                                              | `make setup-conda`                                  |
|                         | Update environment as needed                                                                 | `conda env update -f environment.yml`               |

## Performance Benefits

| Benefit Area        | Conda Advantages                                                                                  |
|---------------------|--------------------------------------------------------------------------------------------------|
| Optimized Packages  | NumPy/SciPy with MKL/OpenBLAS for improved performance                                           |
| Native Dependencies | GDAL, PROJ, GEOS compiled for your platform                                                      |
| Faster Installs     | Pre-built binary packages reduce installation time                                                |
| Dependency Handling | Superior conflict resolution compared to pip                                                     |
| Memory & Speed      | Faster imports and better runtime performance via optimized binaries                             |
| Reduced Conflicts   | Compatible package versions minimize dependency issues                                           |

## See Also

- [Conda Documentation](https://docs.conda.io/)
- [Conda-Forge](https://conda-forge.org/)
- {doc}`getting_started` - Project setup
- {doc}`api` - Python API documentation
