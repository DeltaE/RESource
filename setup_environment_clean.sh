#!/bin/bash

# RESource Environment Setup Script
# Automated setup for RESource package with multiple environment options
# Created: 2025-10-05

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Display banner
show_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "       RESource Environment Setup Script"
    echo "=================================================="
    echo -e "${NC}"
    echo "This script will help you set up a clean, reproducible"
    echo "environment for the RESource package."
    echo ""
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_status "Please install Miniconda or Anaconda first"
        print_status "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_status "Found conda: $(conda --version)"
}

# Check if mamba is available (faster than conda)
check_mamba() {
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        print_status "Using mamba for faster package resolution"
    else
        CONDA_CMD="conda"
        print_warning "Mamba not found, using conda"
        print_status "Tip: Install mamba for faster setup: conda install mamba -n base -c conda-forge"
    fi
}

# Environment selection menu
select_environment() {
    echo ""
    print_header "Select Environment Type:"
    echo "1) Full Development Environment (recommended for developers)"
    echo "   - All packages + development tools + documentation + testing"
    echo "   - Environment name: RESource-dev"
    echo ""
    echo "2) Standard Environment (recommended for most users)"
    echo "   - All packages needed for RESource functionality"
    echo "   - Environment name: RESource"
    echo ""
    echo "3) Production Environment (minimal, for deployment)"
    echo "   - Only essential packages"
    echo "   - Environment name: RESource-prod"
    echo ""
    read -p "Enter your choice (1-3): " env_choice

    case $env_choice in
        1)
            ENV_FILE="env/environment_development.yml"
            ENV_NAME="RESource-dev"
            print_status "Selected: Full Development Environment"
            ;;
        2)
            ENV_FILE="env/environment.yml"
            ENV_NAME="RESource"
            print_status "Selected: Standard Environment"
            ;;
        3)
            ENV_FILE="env/environment_production.yml"
            ENV_NAME="RESource-prod"
            print_status "Selected: Production Environment"
            ;;
        *)
            print_error "Invalid choice. Defaulting to Standard Environment."
            ENV_FILE="env/environment.yml"
            ENV_NAME="RESource"
            ;;
    esac
}

# Check if environment file exists
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file $ENV_FILE not found!"
        print_status "Please make sure you're running this script from the RESource root directory"
        exit 1
    fi
    print_status "Found environment file: $ENV_FILE"
}

# Check if environment already exists
check_existing_env() {
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to update it? (y/n): " update_choice
        if [[ $update_choice =~ ^[Yy]$ ]]; then
            UPDATE_ENV=true
        else
            print_status "Skipping environment creation"
            exit 0
        fi
    else
        UPDATE_ENV=false
    fi
}

# Create or update environment
setup_environment() {
    print_header "Setting up environment: $ENV_NAME"
    
    if [ "$UPDATE_ENV" = true ]; then
        print_status "Updating existing environment..."
        $CONDA_CMD env update -f "$ENV_FILE" --prune
    else
        print_status "Creating new environment..."
        $CONDA_CMD env create -f "$ENV_FILE"
    fi
    
    if [ $? -eq 0 ]; then
        print_status "Environment setup completed successfully!"
    else
        print_error "Environment setup failed!"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying installation..."
    
    # Activate environment and test import
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    # Test Python and basic imports
    python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    import matplotlib.pyplot as plt
    print('✓ Core scientific packages imported successfully')
except ImportError as e:
    print(f'✗ Error importing core packages: {e}')
    sys.exit(1)

# Test RESource import
try:
    import RES
    print('✓ RESource package imported successfully')
except ImportError as e:
    print(f'✗ Error importing RESource: {e}')
    print('  This is normal if you haven\\'t installed RESource in editable mode yet')

print('\\n🎉 Environment setup verification completed!')
"
}

# Display final instructions
show_instructions() {
    echo ""
    print_header "Setup Complete! 🎉"
    echo ""
    print_status "To activate your environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    print_status "To deactivate the environment, run:"
    echo "  conda deactivate"
    echo ""
    print_status "To update the environment in the future:"
    echo "  conda env update -f $ENV_FILE --prune"
    echo ""
    print_status "To export exact package versions (for reproducibility):"
    echo "  conda env export > environment_locked.yml"
    echo ""
    
    if [ "$ENV_NAME" = "RESource-dev" ]; then
        print_status "Development tools included:"
        echo "  - pytest (testing): pytest"
        echo "  - black (formatting): black ."
        echo "  - isort (import sorting): isort ."
        echo "  - flake8 (linting): flake8 ."
        echo "  - sphinx (docs): sphinx-build docs/source docs/build"
        echo ""
    fi
    
    print_status "Happy coding! 🚀"
}

# Main execution
main() {
    show_banner
    check_conda
    check_mamba
    select_environment
    check_env_file
    check_existing_env
    setup_environment
    verify_installation
    show_instructions
}

# Run main function
main "$@"