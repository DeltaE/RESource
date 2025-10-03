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
make setupenv

# Activate environment
conda activate RES

# Test installation
python -c "import RES; print('✅ RESource is ready!')"
```

---

## � Environment Reproducibility

RESource ensures reproducible research through:

- **Locked Dependencies**: `env/environment.yml` contains pinned versions for all packages
- **Automated Setup**: Single command (`make setupenv`) creates identical environments
- **Environment Export**: `make exportenv` captures exact package versions
- **Cross-Platform Support**: Tested on Linux, macOS, and Windows (WSL2)

### Reproducing Exact Environment

```bash
# Create environment from lockfile
make setupenv

# Verify environment matches expectations
conda activate RES
python -c "import RES; print('✅ RESource is ready!')"

# Export current state for sharing
make exportenv
```

---

## �🛠️ Environment Management

### Available Make Commands

#### Environment Setup & Management
```bash
make setupenv          # Create conda environment from env/environment.yml
make updateenv         # Update existing conda environment
make exportenv         # Export current environment to env/environment.yml
make clean            # Clean build files and cache
```


#### Documentation
```bash
make docs              # Build and deploy documentation
make autobuild         # Live rebuild documentation (port 8000)
make deploy            # Deploy documentation to GitHub Pages
```

#### Utilities
```bash
make help              # Show all available commands
make clean             # Clean build files and cache
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
make run

# Interactive development
make jupyter
```

### 3. Documentation Development
```bash
# Live rebuild (recommended for doc development)
make autobuild
# Opens http://127.0.0.1:8000 with auto-reload

# Build and deploy
make docs
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
conda env remove -n RES -y
make setupenv
```

**Problem:** Package conflicts or broken environment
```bash
# Solution: Clean rebuild
conda env remove -n RES -y
conda clean --all
make setupenv
```

**Problem:** Import errors for pandas/geopandas after atlite installation
```bash
# Solution: Update environment
make updateenv
# Or clean reinstall:
conda env remove -n RES -y && make setupenv
```

**Problem:** Local package not updating after code changes
```bash
# Solution: Reinstall in development mode
conda activate RES
pip install -e .
```

### Check Environment Health
```bash
# Check if environment exists and is activated
conda env list | grep RES
conda activate RES
python -c "import RES; print('✅ RESource is ready!')"
```

### Export Current Environment
```bash
make exportenv
# Creates env/environment.yml with current versions
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
- [ ] Run `make setupenv`
- [ ] Test: `python -c "import RES; print('✅ RESource is ready!')"`
- [ ] Test Jupyter: `make jupyter`

### Before Committing Code

- [ ] Environment is working: `conda activate RES && python -c "import RES"`
- [ ] Code runs: `make run`
- [ ] Documentation builds: `make docs`
- [ ] Export environment if dependencies changed: `make exportenv`

### Before Deployment

- [ ] Documentation is current: `make docs`
- [ ] Environment is exportable: `make exportenv`

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

- Always use `make setupenv` instead of manual conda commands
- Keep `env/environment.yml` updated when adding dependencies
- Use `make exportenv` to document exact working environment
- Check environment health with `conda activate RES && python -c "import RES"`

### Development

- Use `make run` for running the main RESource script
- Use `make jupyter` for interactive development (ensures correct kernel)
- Use `make autobuild` for documentation development with live reload

### Collaboration

- Always commit updated `env/environment.yml` when dependencies change
- Use `make exportenv` to share exact working environment
- Document any platform-specific setup requirements

---

## 🆘 Getting Help

### Check Status

```bash
conda activate RES && python -c "import RES"  # Environment health check
make help                                     # Show all available commands
```

### Common Issues

1. **Import errors:** Usually environment not activated or packages missing
2. **Jupyter kernel issues:** Use `make jupyter` instead of direct jupyter commands
3. **Documentation build fails:** Check Sphinx dependencies are installed
4. **Platform issues on Windows:** Ensure using WSL2, not native Windows

### Resources
- [RESource GitHub Repository](https://github.com/DeltaE/RESource)
- [Documentation](https://deltae.github.io/RESource/)
- [Issue Tracker](https://github.com/DeltaE/RESource/issues)

---

*This guide covers the complete setup and development workflow for RESource. All commands are tested and maintained through the project Makefile.*
