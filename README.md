<img src="docs/source/_static/Issue_msg_box.png" alt="Issue" width="600"/>


__One of the many solutions ?__

<img src="docs/source/_static/graphic_RES_logo_202508.jpg" alt="RESource logo" width="250"/>

__A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.__

> ⚠️ **Note: This library is under heavy development**

RESource is developed to enable reproducible, adaptable assessments of VRE potential that are sensitive to local constraints and planning priorities. We developed a structured, modular workflow that integrates geospatial, temporal, economic, and regulatory data to evaluate site suitability for solar and wind energy development. This structured methodology ensures transparency and transferability, allowing RESource to be adapted for different regions and scaled for long-term strategic energy planning.


## Workflow overview
<img src="docs/source/_static/workflow.jpg" alt="high_level_workflow" width="1000"/>

## 🚀 Quick Start

**New to RESource?** Get started with 

📖 **[Full Quickstart Guide](https://deltae.github.io/RESource/#quick-start))** | 📚 **[Complete Documentation](https://deltae.github.io/RESource/)**

### Enhanced Analysis Pipeline

The enhanced `run.py` script provides flexible region selection with colored output:

| Command | Description |
|---------|-------------|
| `python3 run.py config/config_CAN_baseline.yaml` | Canadian analysis (all provinces) |
| `python3 run.py config/config_WB6.yaml` | Western Balkans analysis (all countries) |
| `python3 run.py config/config_WB6.yaml -r AL BA` | Specific regions only |
| `python3 run.py --help` | Show all available options |

**Features:** Smart region detection • Input validation • Colored error messages • Flexible region selection

------

## 📋 Key Features

- **🌍 Multi-Regional**: Canada, Western Balkans, and custom regions
- **⚡ Multi-Technology**: Wind and solar resource assessment
- **🔧 Modular Design**: Configurable exclusions, constraints, and parameters
- **📊 Rich Outputs**: Time series, capacity maps, and interactive visualizations
- **🔄 Reproducible**: Locked environments and standardized workflows

------

## 📚 Resources

- **[� Complete Setup Guide](SETUP.md)** - Definitive installation & setup guide
- **[📖 Quickstart Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[🏔️ BC Case Study](https://deltae.github.io/RESource/notes/case_BC.html)** - Detailed regional analysis
- **[📘 Full Documentation](https://deltae.github.io/RESource/)** - Complete reference
