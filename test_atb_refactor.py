#!/usr/bin/env python3
"""
Test script to verify that the refactored NREL_ATBProcessor works correctly.
"""

import sys
from pathlib import Path

# Add the current directory to sys.path to import RES
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test that we can import the refactored class."""
    try:
        from RES.atb import NREL_ATBProcessor
        print("✓ Successfully imported NREL_ATBProcessor")
        return True
    except Exception as e:
        print(f"✗ Failed to import NREL_ATBProcessor: {e}")
        return False

def test_instantiation():
    """Test that we can instantiate the class."""
    try:
        from RES.atb import NREL_ATBProcessor
        
        # Test with default parameters
        processor = NREL_ATBProcessor()
        print("✓ Successfully instantiated NREL_ATBProcessor with defaults")
        
        # Check that key attributes are set
        assert hasattr(processor, 'config'), "Config attribute missing"
        assert hasattr(processor, 'store'), "Store attribute missing"
        assert hasattr(processor, 'atb_config'), "ATB config attribute missing"
        assert hasattr(processor, 'attributes_parser'), "AttributesParser instance missing"
        
        print("✓ All expected attributes are present")
        return True
        
    except Exception as e:
        print(f"✗ Failed to instantiate NREL_ATBProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_custom_params():
    """Test that we can instantiate with custom parameters."""
    try:
        from RES.atb import NREL_ATBProcessor
        
        processor = NREL_ATBProcessor(
            region_short_code='AB',
            resource_type='solar'
        )
        print("✓ Successfully instantiated with custom parameters")
        
        # Verify the parameters were set correctly
        assert processor.region_short_code == 'AB', "Region code not set correctly"
        assert processor.resource_type == 'solar', "Resource type not set correctly"
        
        print("✓ Custom parameters set correctly")
        return True
        
    except Exception as e:
        print(f"✗ Failed to instantiate with custom parameters: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing refactored NREL_ATBProcessor...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_instantiation,
        test_with_custom_params
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The refactoring was successful.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
