#!/usr/bin/env python3
"""
Test script to validate which classes can be documented successfully.
This simulates Read the Docs environment issues.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set Read the Docs environment variable to simulate their environment
os.environ['READTHEDOCS'] = 'True'

def test_class_import(class_path):
    """Test if a class can be imported without errors."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return True, f"✓ {class_path}: Successfully imported"
    except Exception as e:
        return False, f"✗ {class_path}: {str(e)}"

def test_sphinx_autodoc(class_path):
    """Test if Sphinx autodoc can process a class."""
    try:
        from sphinx.ext.autodoc import ClassDocumenter
        from sphinx.util.docutils import docutils_namespace
        
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Try to create a documenter
        with docutils_namespace():
            documenter = ClassDocumenter(None, class_path)
            return True, f"✓ {class_path}: Autodoc can process"
    except Exception as e:
        return False, f"✗ {class_path}: Autodoc failed - {str(e)}"

def main():
    """Test all classes mentioned in the API documentation."""
    print("Testing RESource classes for Read the Docs compatibility...")
    print("=" * 60)
    
    # Classes from api.md
    test_classes = [
        'RES.atb.NREL_ATBProcessor',
        'RES.boundaries.GADMBoundaries', 
        'RES.cell.GridCells',
        'RES.score.CellScorer',
        'RES.hdf5_handler.DataHandler',
        'RES.tech.OEDBTurbines',
        'RES.units.Units'
    ]
    
    print("Testing direct imports...")
    print("-" * 30)
    
    import_results = []
    for class_path in test_classes:
        success, message = test_class_import(class_path)
        print(message)
        import_results.append((class_path, success))
    
    print("\nSummary:")
    print("-" * 30)
    working_classes = [cp for cp, success in import_results if success]
    failing_classes = [cp for cp, success in import_results if not success]
    
    print(f"Working classes ({len(working_classes)}):")
    for cp in working_classes:
        print(f"  ✓ {cp}")
    
    if failing_classes:
        print(f"\nFailing classes ({len(failing_classes)}):")
        for cp in failing_classes:
            print(f"  ✗ {cp}")
        
        print(f"\nRecommendations for failing classes:")
        print("1. Add their dependencies to autodoc_mock_imports in conf.py")
        print("2. Use manual documentation instead of autodoc")
        print("3. Refactor to reduce complex dependencies (like we did with NREL_ATBProcessor)")
    
    print(f"\nOverall: {len(working_classes)}/{len(test_classes)} classes can be documented")
    return len(failing_classes) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
