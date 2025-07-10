#!/usr/bin/env python3
"""
Test script to diagnose Sphinx/myst_parser issues
"""

import sys
import os

# Add docs/source to path
sys.path.insert(0, os.path.join(os.getcwd(), 'docs', 'source'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    modules_to_test = [
        'sphinx',
        'myst_parser', 
        'nbsphinx',
        'furo'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}: OK")
        except ImportError as e:
            print(f"✗ {module}: {e}")

def test_conf():
    """Test if conf.py can be loaded"""
    print("\nTesting conf.py...")
    try:
        import conf
        print("✓ conf.py: Loaded successfully")
        
        # Test specific configurations
        if hasattr(conf, 'extensions'):
            print(f"  Extensions: {conf.extensions}")
        if hasattr(conf, 'myst_enable_extensions'):
            print(f"  MyST extensions: {conf.myst_enable_extensions}")
            
    except Exception as e:
        print(f"✗ conf.py: {e}")
        import traceback
        traceback.print_exc()

def test_myst_config():
    """Test MyST parser configuration"""
    print("\nTesting MyST configuration...")
    try:
        from myst_parser import __version__
        print(f"✓ MyST parser version: {__version__}")
        
        # Test if myst_fence_as_directive is supported
        from myst_parser.config import MdParserConfig
        print("✓ MyST configuration classes available")
        
    except Exception as e:
        print(f"✗ MyST configuration test: {e}")

def main():
    print("RESource Documentation Diagnostic Tool")
    print("=" * 40)
    
    test_imports()
    test_conf()
    test_myst_config()
    
    print("\n" + "=" * 40)
    print("Diagnostic complete!")

if __name__ == "__main__":
    main()
