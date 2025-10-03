# RESource Quickstart Guide

**Get up and running with RESource** - A modular framework for renewable energy resource assessment.

---

## 🚀 Quick Setup

### Prerequisites

- **Linux environment** (Linux, macOS, or Windows WSL2)  
- **Git** and **Conda/Miniconda** installed
- **Python 3.12+** support

### 1️⃣ Clone & Navigate

```bash
git clone https://github.com/DeltaE/RESource.git
cd RESource
```

### 2️⃣ One-Command Setup

```bash
# Create complete conda environment
make setupenv
```

### 3️⃣ Activate & Test

```bash
# Activate environment
conda activate RES

# Quick test
python -c "import RES; print('✅ RESource ready!')"
```

**🎉 You're ready to go!**

### 4️⃣ Download Case Study Data

```bash
# Download required data for Canadian analysis (~2GB)
# Visit and download: https://zenodo.org/records/16658067
# Extract to data/ directory

wget https://zenodo.org/record/16658067/files/RESource_data.zip
unzip RESource_data.zip -d data/
```

```{note}
**Case Study Data**: Contains processed geospatial data for Canadian provinces including ERA5 weather data, exclusion zones, and land cover classifications needed for renewable energy resource assessment.
```

### 5️⃣ Setup API Access (For Custom Analysis)

```{warning}
**Required for Custom Regions**: Skip this step if only using the provided case study data.
```

For running custom analyses beyond the provided case study data, some data sources require API registration:

- **ERA5 Climate Data**: Requires Copernicus Climate Data Store registration for global weather data
- **CODERS API**: Requires registration for Canadian-specific data sources

```{note}
**Complete Setup Instructions**: For detailed API registration steps, configuration files, and data source information, see [data.md](data.md).
```

---

## 🔥 First Run

### Run Canadian Wind & Solar Analysis

```bash
# Run with default Canadian configuration
make run

# Or run specific region/config
python run.py --config config/config_CAN_baseline.yaml
```

### Start Interactive Environment

```bash
# Launch Jupyter Lab
make jupyter

# Open notebooks/
# Try: resources_playground_CAN.ipynb
```

---

## 📊 What RESource Does

RESource analyzes **renewable energy potential** by:

1. **📍 Site Selection**: Filters suitable land using exclusion zones, slopes, land cover
2. **🌤️ Weather Processing**: Converts ERA5 climate data to energy capacity factors  
3. **⚡ Capacity Estimation**: Calculates wind/solar potential per grid cell
4. **📈 Time Series Generation**: Creates hourly profiles for energy system modeling
5. **🎯 Optimization**: Selects optimal sites based on capacity and cost criteria

**Output Organization:**
- **`data/store/`**: Primary resource data (HDF5 format) - detailed analysis results
- **`results/Country/Region/Scenario/`**: Organized CSV files - resource clusters, timeseries, and cost-filtered sites
- **`vis/Country/Region/`**: SVG/PNG visualizations - maps, supply curves, clustering analysis, and policy scenarios

### Example Output

```text
data/store/
├── resources_Canada_BC_BASELINE.h5     # resource data store for BASELINE

results/Canada/BC/BASELINE/
├── resource_options_wind_British Columbia.csv         # Wind resource clusters with LCOE
├── resource_options_wind_British Columbia_timeseries.csv  # Hourly capacity factors
├── resource_options_solar_British Columbia.csv        # Solar resource clusters with LCOE  
├── resource_options_solar_British Columbia_timeseries.csv # Hourly capacity factors
├── cells_aggregated_by_Region_BC_BASELINE.csv         # Regional capacity summaries
├── wind_cells_below_50_$pMWh_BC_BASELINE.csv         # Cost-filtered wind sites
├── solar_cells_below_57_$pMWh_BC_BASELINE.csv        # Cost-filtered solar sites
└── Resource_options_summary.txt                       # Analysis summary report

vis/Canada/BC/
├── BC_gridcells_outline.svg                      # Grid cell boundaries
├── BC_regions.png                                # Administrative regions
├── supply_curve_baseline_vs_policy_BC.svg        # Supply curve comparisons
├── Resources_proximity_to_grid_BC.svg            # Grid connectivity analysis
├── BASELINE/
│   ├── Resources_combined_CAPACITY.svg           # Combined capacity maps
│   ├── Resources_combined_CF.svg                 # Capacity factor maps
│   ├── Resources_combined_SCORE.svg              # Resource quality scores
│   ├── wind/
│   │   ├── Regional_cluster_Elbow_Plots/         # Clustering analysis plots
│   │   └── lands/                                # Land availability analysis
│   └── solar/
│       ├── Regional_cluster_Elbow_Plots/         # Clustering analysis plots
│       └── lands/                                # Land availability analysis
└── strict_policy_aeroway_CPCAD_buffer/           # Alternative policy scenario
    ├── cost_map_wind.svg                         # Cost mapping visualizations
    ├── cost_map_solar.svg
    └── potential_capacity_lost_*.svg             # Policy impact analysis
```

---

## 📁 Key Files & Structure

```text
RESource/
├── config/                      # Configuration files
│   ├── config_CAN.yaml         # Canadian analysis setup
│   └── config_WB6.yaml         # Western Balkans setup
├── RES/                        # Core analysis modules
├── notebooks/                  # Interactive examples
├── run.py                      # Main analysis script
├── data/
│   └── store/                  # Main resource data storage (HDF5)
├── results/                    # Downstream modeling exports (CSV)
└── vis/                        # Visualizations & plots
```

---

## 🛠️ Common Commands

```bash
make help              # Show all available commands
make setupenv          # Create/setup conda environment  
make updateenv         # Update existing environment
make run              # Run main analysis
make jupyter          # Start Jupyter Lab
make docs             # Build documentation
make clean            # Clean cache & build files
```

---

## 🌍 Supported Regions

- **🇨🇦 Canada**: Provincial analysis (BC, AB, SK, ON, QC, NS, MB)
- **🌍 Western Balkans**: Regional analysis (6 countries)
- **🔧 Custom**: Configure any region with your own data

---

## 📚 Next Steps

### Explore Examples

```bash
# Try the playground notebooks
jupyter lab notebooks/resources_playground_CAN.ipynb
```

### Build Documentation

```bash
# Generate full documentation
make docs

# Live development server
make autobuild  # Visit http://127.0.0.1:8000
```

## 🆘 Need Help?

### Quick Debugging

```bash
# Check environment
conda activate RES && python -c "import RES"

# Test all dependencies  
python workflow/scripts/test_venv.py

# Check data store (look for resources_Canada_*_BASELINE.h5)
ls -la data/store/resources_Canada_*

# View recent outputs - organized by Country/Region/Scenario
ls -la results/Canada/BC/BASELINE/
ls -la vis/Canada/BC/BASELINE/

# Check visualization files
find vis/Canada/BC/ -name "*.svg" -o -name "*.png" | head -10
```

### Resources

- 📖 **Full Documentation**: [https://deltae.github.io/RESource/](https://deltae.github.io/RESource/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/DeltaE/RESource/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/DeltaE/RESource/discussions)

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Environment 'RES' not found` | Run `make setupenv` |
| `ImportError: No module named RES` | Run `conda activate RES` |
| `FileNotFoundError: data/...` | Download case study data from [Zenodo](https://zenodo.org/records/16658067) |
| `Permission denied` on WSL2 | Run `sudo chown -R $USER:$USER .` |
| Slow downloads | Check internet connection, try different mirror |

### Still Need Help?

If you encounter issues not covered above, you can reach out to the development team:

```{tip}
**Contact the Developer**: For direct assistance, see contact information in [developers.md](developers.md) or open an issue on GitHub.
```

---

**🎯 Ready to analyze renewable energy resources? Run `make run` and explore the results!**

```{tip}
For more detailed setup and development information, see the [Complete Setup Guide](setup_guide.md).
```