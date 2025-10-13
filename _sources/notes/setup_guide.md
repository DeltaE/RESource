# RESource Setup Guide (Deprecated)

> **🚨 This guide has been moved and consolidated!**
>
> **📚 Please use the new complete setup guide: [SETUP.md](../../../SETUP.md) in the repository root.**
>
> This page is kept for backward compatibility but will be removed in a future version.

---

## Quick Migration to New Setup

The setup process is now simplified to a single command:

```bash
# Clone repository
git clone https://github.com/DeltaE/RESource.git
cd RESource

# Create environment (ONE COMMAND!)
conda env create -f env/environment.yml
conda activate RESource

# Verify installation
python run.py --help
```

**For complete instructions, troubleshooting, and all setup options, please see [SETUP.md](../../../SETUP.md).**

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
conda activate RESource
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
conda activate RESource
# Your prompt should show: (RESource) $
```

### 2. Run RESource

#### Enhanced Analysis Pipeline (`run.py`)

**Smart Region Selection & Colored Output** - The enhanced `run.py` script provides flexible region selection with validation and colored terminal output.

| Feature | Description |
|---------|-------------|
| **Smart Detection** | Automatically reads available regions from config file |
| **Validation** | Invalid regions trigger helpful error messages with suggestions |
| **Colored Output** | Errors (red), warnings (yellow), success (green), info (cyan) |
| **Flexible Selection** | Process all regions or specify subset via command line |

#### Command Reference

| Command | Description | Output Colors |
|---------|-------------|---------------|
| `python3 run.py` | Default config (Canadian provinces) | 🟢 Success messages |
| `python3 run.py -c config/config_WB6.yaml` | Western Balkans (all regions) | 🔵 Info messages |
| `python3 run.py -c config/config_WB6.yaml -r AL BA` | Specific regions only | 🟡 Warnings |
| `python3 run.py -c invalid.yaml` | Shows available configs | 🔴 Error messages |
| `python3 run.py --help` | Display all options | - |

#### Regional Configurations

| Config File | Regions Available | Example Usage |
|-------------|-------------------|---------------|
| `config_CAN_baseline.yaml` | AB, BC, MB, NB, NL, NS, ON, PE, QC, SK | `python3 run.py --regions BC QC` |
| `config_CAN_policy1.yaml` | AB, BC, MB, NB, NL, NS, ON, PE, QC, SK | `python3 run.py -c config/config_CAN_policy1.yaml -r BC ON` |
| `config_WB6.yaml` | AL, BA, XK, ME, MK, RS | `python3 run.py -c config/config_WB6.yaml -r AL BA` |


#### Error Handling Examples

```bash
# Invalid config file - shows available options
python3 run.py -c nonexistent.yaml
# Output: ✗ Configuration file 'nonexistent.yaml' not found.
#         💡 Available config files:
#            • config/config_CAN_baseline.yaml (Canadian provinces - baseline)
#            • config/config_CAN_policy1.yaml (Canadian provinces - policy scenario)
#            • config/config_WB6.yaml (Western Balkans)

# Invalid regions - shows valid options  
python3 run.py -c config/config_WB6.yaml -r INVALID
# Output: ✗ Invalid region(s): ['INVALID']
#         ⚠️  Available regions in config: ['AL', 'BA', 'XK', 'ME', 'MK', 'RS']
#         💡 Examples of valid commands:
#            • python3 run.py -c config/config_WB6.yaml --regions AL BA XK
```

#### Alternative Development Methods

```bash
# Legacy make command (still supported)
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
conda activate RESource
pip install -e .
```

### Check Environment Health
```bash
# Check if environment exists and is activated
conda env list | grep RESource
conda activate RESource
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
