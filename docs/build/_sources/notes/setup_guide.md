# RESource Project - Complete Setup & Development Guide

## 🎯 Quick Start (New Users)

### Prerequisites
- **Python 3.11+**
- **Git**
- **Conda or Miniconda** (strongly recommended for better package management)
- **Linux environment** (native Linux, WSL2 on Windows, or macOS)

### 1. Clone and Navigate
```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
```

### 2. Environment Setup (One Command)
```bash
# Create conda environment with all dependencies
make setup-conda

# Activate environment
conda activate RES

# Test installation
python -c "import RES; print('✅ RESource is ready!')"
```

---

## 🛠️ Environment Management

### Available Make Commands

#### Environment Setup & Management
```bash
make setup-conda       # Create/update conda environment 'RES'
make conda-status      # Check environment health and packages
make clean-conda       # Remove conda environment
make export-env        # Export environment to env/environment_exported.yml
```

#### Running Code
```bash
make run-res           # Run main RESource module
make run-conda SCRIPT=path/to/script.py [ARGS='arg1 arg2']  # Run custom scripts
make jupyter           # Start Jupyter Lab with conda environment
```

#### Documentation
```bash
make docs-build        # Build documentation only
make docs-conda        # Build and deploy documentation
make deploy            # Deploy documentation to GitHub Pages
make sync-notebooks    # Sync notebooks to docs
make autobuild         # Live rebuild documentation with auto-reload
```

#### Utilities
```bash
make help             # Show all available commands
make clean-docs       # Clean documentation build files
make clean-all        # Clean all build files and environments
```

---

## 🐧 Platform-Specific Setup

### Windows Users (WSL2 Required)
RESource is designed for Linux environments. Windows users must use WSL2:

1. **Install WSL2:**
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install Ubuntu distribution:**
   ```powershell
   # Check available distributions
   wsl --list --online
   
   # Install Ubuntu (recommended)
   wsl --install -d Ubuntu-22.04
   ```

3. **Install Miniconda in WSL2:**
   ```bash
   # In WSL2 terminal
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/.bashrc
   ```

### Linux/macOS Users
Install Miniconda directly:
```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

---

## 📦 Environment Details

### Core Dependencies
The `env/environment.yml` includes:

**Scientific Computing:**
- Python 3.11, NumPy, Pandas, SciPy, XArray

**Geospatial:**
- GeoPandas, Rasterio, Shapely, Fiona, PyProj, Cartopy

**Energy Modeling:**
- Atlite (wind/solar resource assessment)
- CDS API (climate data)

**Visualization:**
- Matplotlib, Seaborn, Plotly, Folium

**Development:**
- Jupyter Lab, Sphinx (documentation), pytest

**Full list:** See `env/environment.yml` for complete dependency specification.

---

## 🚀 Daily Development Workflow

### 1. Activate Environment
```bash
conda activate RES
# Your prompt should show: (RES) $
```

### 2. Run RESource
```bash
# Main module
make run-res

# Custom scripts
make run-conda SCRIPT=workflow/scripts/your_script.py

# Interactive development
make jupyter
```

### 3. Documentation Development
```bash
# Live rebuild (recommended for doc development)
make autobuild
# Opens http://localhost:8000 with auto-reload

# Build only
make docs-build
```

---

## 🔧 Troubleshooting

### Environment Issues

**Problem:** Double environment prompts `(RES) (base) $`
```bash
# Solution: Disable auto-activation of base
conda config --set auto_activate_base false
source ~/.bashrc
```

**Problem:** "Environment 'RES' not found"
```bash
# Solution: Recreate environment
make clean-conda
make setup-conda
```

**Problem:** Package conflicts or broken environment
```bash
# Solution: Clean rebuild
make clean-conda
conda clean --all
make setup-conda
```

**Problem:** Import errors for pandas/geopandas after atlite installation
```bash
# Solution: Reinstall conflicting packages through conda
conda activate RES
conda update pandas geopandas
# Or clean reinstall:
make clean-conda && make setup-conda
```

**Problem:** Local package not updating after code changes
```bash
# Solution: Reinstall in development mode
conda activate RES
pip install -e .
```

### Check Environment Health
```bash
make conda-status
# Shows: environment status, Python version, key packages, RESource module availability
```

### Export Current Environment
```bash
make export-env
# Creates env/environment_exported.yml with exact versions
```

### WSL2 Specific Issues

**Problem:** Conda commands not found in WSL2
```bash
# Solution: Ensure conda is in PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Problem:** Permission issues in WSL2
```bash
# Solution: Fix file permissions
sudo chown -R $USER:$USER /path/to/RESource
```

---

## 📋 Development Checklist

### New Developer Setup
- [ ] Clone repository: `git clone https://github.com/DeltaE/RESource.git`
- [ ] Install Conda/Miniconda
- [ ] Run `make setup-conda`
- [ ] Test: `python -c "import RES; print('✅ RESource is ready!')"`
- [ ] Test Jupyter: `make jupyter`

### Before Committing Code
- [ ] Environment is clean: `make conda-status`
- [ ] Code runs: `make run-res`
- [ ] Documentation builds: `make docs-build`
- [ ] Export environment if dependencies changed: `make export-env`

### Before Deployment
- [ ] Documentation is current: `make docs-conda`
- [ ] All notebooks sync: `make sync-notebooks`
- [ ] Environment is exportable: `make export-env`

---

## 🌐 Documentation Deployment

### Local Development
```bash
# Live reload during development
make autobuild
# Visit: http://localhost:8000
```

### GitHub Pages Deployment
```bash
# Build and deploy to GitHub Pages
make deploy
# Visit: https://deltae.github.io/RESource/
```

### Documentation Structure
```
docs/
├── source/           # Source files
│   ├── index.md     # Main page
│   ├── notes/       # Documentation pages
│   └── notebooks/   # Jupyter notebooks
└── build/           # Generated HTML (auto-created)
```

---

## 💡 Tips & Best Practices

### Environment Management
- Always use `make setup-conda` instead of manual conda commands
- Keep `env/environment.yml` updated when adding dependencies
- Use `make export-env` to document exact working environment
- Use `make conda-status` to check environment health regularly

### Development
- Use `make run-conda SCRIPT=...` for testing scripts with proper environment
- Use `make jupyter` for interactive development (ensures correct kernel)
- Use `make autobuild` for documentation development with live reload

### Collaboration
- Always commit updated `env/environment.yml` when dependencies change
- Use `make export-env` to share exact working environment
- Document any platform-specific setup requirements

---

## 🆘 Getting Help

### Check Status
```bash
make conda-status    # Environment health check
make help           # Show all available commands
```

### Common Issues
1. **Import errors:** Usually environment not activated or packages missing
2. **Jupyter kernel issues:** Use `make jupyter` instead of direct jupyter commands
3. **Documentation build fails:** Check Sphinx dependencies with `make conda-status`
4. **Platform issues on Windows:** Ensure using WSL2, not native Windows

### Resources
- [RESource GitHub Repository](https://github.com/DeltaE/RESource)
- [Documentation](https://deltae.github.io/RESource/)
- [Issue Tracker](https://github.com/DeltaE/RESource/issues)

---

*This guide covers the complete setup and development workflow for RESource. All commands are tested and maintained through the project Makefile.*
