#!/usr/bin/env python3
"""
Test script to validate the virtual environment setup
"""

def test_imports():
    """Test if all major dependencies can be imported"""
    import sys
    print(f"Python version: {sys.version}")
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
        
        import pandas as pd
        print("✓ pandas")
        
        import geopandas as gpd
        print("✓ geopandas")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
        
        import rasterio
        print("✓ rasterio")
        
        import cartopy
        print("✓ cartopy")
        
        import plotly
        print("✓ plotly")
        
        import dash
        print("✓ dash")
        
        import xarray
        print("✓ xarray")
        
        import atlite
        print("✓ atlite")
        
        import netCDF4
        print("✓ netCDF4")
        
        import sphinx
        print("✓ sphinx")
        
        import jupyter
        print("✓ jupyter")
        
        # Test RES module import
        import RES
        print("✓ RES module")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    import numpy as np
    import pandas as pd
    
    # Test numpy
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Numpy array: {arr}")
    
    # Test pandas
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    print(f"Pandas DataFrame:\n{df}")
    
    print("✓ Basic functionality test passed!")

if __name__ == "__main__":
    print("RESource Virtual Environment Test")
    print("=" * 40)
    
    success = test_imports()
    if success:
        test_basic_functionality()
        print("\n✅ Virtual environment is working correctly!")
    else:
        print("\n❌ Virtual environment has issues!")
        exit(1)
