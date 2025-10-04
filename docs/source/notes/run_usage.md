# Enhanced run.py Usage Guide

The enhanced `run.py` script provides a user-friendly interface for renewable energy resource assessment with smart region detection, input validation, and colored terminal output.

## 🚀 Quick Reference

### Command Syntax

```bash
python3 run.py [--config CONFIG_FILE] [--regions REGION1 REGION2 ...]
```

### Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Smart Detection** | Reads available regions from config `region_mapping` | No hardcoded region lists |
| **Input Validation** | Validates regions against config file | Prevents invalid runs |
| **Colored Output** | Color-coded messages and progress | Better user experience |
| **Error Guidance** | Helpful suggestions when errors occur | Faster troubleshooting |
| **Flexible Selection** | Process all regions or custom subset | Efficient partial runs |

---

## 📋 Command Reference Table

| Command Pattern | Example | Description |
|-----------------|---------|-------------|
| **Default Config** | `python3 run.py` | Canadian provinces (all) |
| **Custom Config** | `python3 run.py -c config/config_WB6.yaml` | Western Balkans (all) |
| **Specific Regions** | `python3 run.py -c config/config_WB6.yaml -r AL BA` | Selected regions only |
| **Multiple Regions** | `python3 run.py --regions BC QC AB` | Multiple Canadian provinces |
| **Single Region** | `python3 run.py -c config/config_WB6.yaml -r AL` | Single region analysis |
| **Help** | `python3 run.py --help` | Show all options |

---

## 🌍 Regional Configurations

### Available Region Sets

| Config File | Description | Available Regions |
|-------------|-------------|-------------------|
| `config/config_CAN_baseline.yaml` | Canadian provinces (baseline) | AB, BC, MB, NB, NL, NS, ON, PE, QC, SK |
| `config/config_CAN_policy1.yaml` | Canadian provinces (policy scenario) | AB, BC, MB, NB, NL, NS, ON, PE, QC, SK |
| `config/config_WB6.yaml` | Western Balkan countries | AL, BA, XK, ME, MK, RS |

### Region Code Reference

#### Canadian Provinces
| Code | Full Name | Code | Full Name |
|------|-----------|------|-----------|
| AB | Alberta | NB | New Brunswick |
| BC | British Columbia | NL | Newfoundland and Labrador |
| MB | Manitoba | NS | Nova Scotia |
| ON | Ontario | PE | Prince Edward Island |
| QC | Quebec | SK | Saskatchewan |

#### Western Balkan Countries
| Code | Full Name | Code | Full Name |
|------|-----------|------|-----------|
| AL | Albania | ME | Montenegro |
| BA | Bosnia and Herzegovina | MK | North Macedonia |
| XK | Kosovo | RS | Serbia |

---

## 🎨 Colored Output Guide

### Color Coding System

| Color | Type | When Used | Example |
|-------|------|-----------|---------|
| 🔴 **Red** | Errors | File not found, invalid regions | `✗ Configuration file not found` |
| 🟡 **Yellow** | Warnings | Process issues, alerts | `⚠️ Available regions in config` |
| 🟢 **Green** | Success | Completed operations | `✅ Completed AL wind processing` |
| 🔵 **Cyan** | Information | Status updates, progress | `📍 Available regions in config` |
| 🟣 **Magenta** | Suggestions | Helpful tips, examples | `💡 Examples of valid commands` |

### Terminal Output Examples

#### Successful Configuration Load

```bash
$ python3 run.py -c config/config_WB6.yaml
✓ Configuration file loaded: config/config_WB6.yaml
📍 Available regions in config: ['AL', 'BA', 'XK', 'ME', 'MK', 'RS']
🌍 Processing all regions from config: ['AL', 'BA', 'XK', 'ME', 'MK', 'RS']
```

#### Invalid Configuration File

```bash
$ python3 run.py -c nonexistent.yaml
✗ Configuration file 'nonexistent.yaml' not found.
💡 Available config files:
   • config/config_CAN_baseline.yaml (Canadian provinces)
   • config/config_WB6.yaml (Western Balkans)
   • config/config_CAN.yaml (Canadian default)
💡 Example: python3 run.py -c config/config_WB6.yaml
```

#### Invalid Region Selection

```bash
$ python3 run.py -c config/config_WB6.yaml -r INVALID TEST
✓ Configuration file loaded: config/config_WB6.yaml
📍 Available regions in config: ['AL', 'BA', 'XK', 'ME', 'MK', 'RS']  
✗ Invalid region(s): ['INVALID', 'TEST']
⚠️ Available regions in config: ['AL', 'BA', 'XK', 'ME', 'MK', 'RS']
💡 Examples of valid commands:
   • python3 run.py -c config/config_WB6.yaml --regions AL BA XK
   • python3 run.py -c config/config_WB6.yaml --regions AL
   • python3 run.py -c config/config_WB6.yaml (process all regions)
```

---

## 🛠️ Advanced Usage Patterns

### Partial Region Processing

```bash
# Process first 3 Western Balkan countries
python3 run.py -c config/config_WB6.yaml -r AL BA XK

# Process specific Canadian provinces
python3 run.py --regions BC QC AB

# Single region development/testing
python3 run.py -c config/config_WB6.yaml -r AL
```

### Development Workflows

```bash
# Test configuration without full run
python3 run.py -c config/config_WB6.yaml -r AL | head -20

# Check available regions in new config
python3 run.py -c path/to/new_config.yaml -r INVALID

# Quick validation of region codes
python3 run.py -c config/config_WB6.yaml --help
```

### Error Recovery

```bash
# If colorama is missing (fallback mode)
pip install colorama
python3 run.py -c config/config_WB6.yaml

# Process interrupted - resume with specific regions
python3 run.py -c config/config_WB6.yaml -r ME MK RS
```

---

## 🔧 Installation Requirements

### Required Dependencies

| Package | Purpose | Installation |
|---------|---------|-------------|
| **colorama** | Colored terminal output | `pip install colorama` |
| **RES modules** | Core analysis functionality | Included in environment |
| **PyYAML** | Configuration file parsing | Included in environment |

### Optional Enhancements

```bash
# Install colorama for enhanced output (if missing)
pip install colorama

# Verify installation
python3 -c "import colorama; print('✅ Colorama available')"
```

---

## 📖 Integration with Documentation

This enhanced `run.py` functionality is documented across:

- **[Quickstart Guide](quickstart.md)** - Basic usage examples
- **[Setup Guide](setup_guide.md)** - Comprehensive command reference  
- **[Resource Builder](resource_builder.md)** - Integration with analysis pipeline
- **[README.md](../../README.md)** - Project overview and quick reference

For detailed API documentation and advanced configuration options, see the [Complete Documentation](https://deltae.github.io/RESource/).