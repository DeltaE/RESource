"""
Example file demonstrating NODOC functionality.

This file shows how to use NODOC or :nodoc: markers to exclude
items from documentation.
"""

def documented_function():
    """
    This function will appear in the documentation.
    
    Returns:
        str: A simple message
    """
    return "This is documented"

def nodoc_function():
    """
    NODOC: This function will be skipped from documentation.
    
    This function contains the NODOC marker in its docstring,
    so it will be excluded from the generated documentation.
    """
    return "This won't be documented"

def another_nodoc_function():
    """
    This function also won't be documented.
    
    :nodoc: This marker will also exclude it from documentation.
    """
    return "Also not documented"

class DocumentedClass:
    """
    This class will appear in documentation.
    """
    
    def public_method(self):
        """A public method that will be documented."""
        pass
    
    def nodoc_method(self):
        """
        NODOC: This method will be excluded from docs.
        """
        pass

class NODOCClass:
    """
    This entire class will be skipped because its name contains NODOC.
    """
    
    def some_method(self):
        """This won't be documented because the whole class is skipped."""
        pass
