#!/usr/bin/env python3
"""
RESource Environment Validation Script
Tests that all required packages can be imported successfully
"""

import sys
import importlib
from typing import List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        return True, f"✅ {package_name or module_name}"
    except ImportError as e:
        return False, f"❌ {package_name or module_name}: {str(e)}"

def main():
    """Main validation function."""
    print("🔍 RESource Environment Validation")
    print("=" * 50)
    
    # Core scientific packages
    core_packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("matplotlib.pyplot", "Matplotlib.pyplot"),
    ]
    
    # Geospatial packages
    geo_packages = [
        ("geopandas", "GeoPandas"),
        ("rasterio", "Rasterio"),
        ("shapely", "Shapely"),
        ("pyproj", "PyProj"),
        ("fiona", "Fiona"),
        ("rioxarray", "RioXarray"),
    ]
    
    # Climate data packages
    climate_packages = [
        ("xarray", "Xarray"),
        ("netcdf4", "NetCDF4"),
        ("cftime", "CfTime"),
    ]
    
    # Optional packages that might not be available in all environments
    optional_packages = [
        ("atlite", "Atlite"),
        ("cdsapi", "CDS API"),
        ("plotly", "Plotly"),
        ("folium", "Folium"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn"),
        ("dask", "Dask"),
        ("h5py", "H5Py"),
        ("bokeh", "Bokeh"),
        ("holoviews", "HoloViews"),
        ("hvplot", "HvPlot"),
    ]
    
    # Development packages (only in dev environment)
    dev_packages = [
        ("pytest", "Pytest"),
        ("black", "Black"),
        ("isort", "isort"),
        ("flake8", "Flake8"),
        ("sphinx", "Sphinx"),
    ]
    
    # RESource package
    res_packages = [
        ("RES", "RESource"),
    ]
    
    all_passed = True
    
    def test_category(packages: List[Tuple[str, str]], category: str, required: bool = True):
        nonlocal all_passed
        print(f"\n📦 {category}:")
        category_passed = True
        for module, name in packages:
            success, message = test_import(module, name)
            print(f"  {message}")
            if not success and required:
                category_passed = False
                all_passed = False
        
        if not required and not category_passed:
            print(f"  ℹ️  Some {category.lower()} packages missing (optional)")
        
        return category_passed
    
    # Test required packages
    test_category(core_packages, "Core Scientific Packages", required=True)
    test_category(geo_packages, "Geospatial Packages", required=True)
    test_category(climate_packages, "Climate Data Packages", required=True)
    
    # Test optional packages
    test_category(optional_packages, "Optional Packages", required=False)
    test_category(dev_packages, "Development Packages", required=False)
    
    # Test RESource
    print(f"\n🎯 RESource Package:")
    res_success, res_message = test_import("RES", "RESource")
    print(f"  {res_message}")
    
    if not res_success:
        print("  💡 Tip: Install RESource in editable mode with 'pip install -e .'")
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed and res_success:
        print("🎉 All required packages imported successfully!")
        print("✅ Environment is ready for RESource!")
    elif all_passed:
        print("⚠️  Core packages OK, but RESource package not found")
        print("   Install with: pip install -e .")
    else:
        print("❌ Some required packages failed to import")
        print("   Please check your environment setup")
        sys.exit(1)
    
    # Python version info
    print(f"\n🐍 Python version: {sys.version}")
    print(f"📍 Python executable: {sys.executable}")

if __name__ == "__main__":
    main()