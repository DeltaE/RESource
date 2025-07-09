# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('../..'))

# For Read the Docs compatibility
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # Set up minimal environment for Read the Docs
    import warnings
    warnings.filterwarnings("ignore")
    
    # Add additional mock imports for Read the Docs
    additional_mocks = [
        'numba', 'dask', 'distributed', 'lxml', 'openpyxl', 'xlrd',
        'pyarrow', 'fastparquet', 'tables', 'bottleneck', 'numexpr'
    ]

project = 'RESource'
copyright = '2025, Md Eliasinul Islam'
author = 'Md Eliasinul Islam'
release = '2025.07'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For Markdown support
    "sphinx.ext.duration",  # For generating documentation
    "sphinx.ext.autodoc",  # For automatic documentation generation from docstrings
    "sphinx.ext.napoleon",  # For Google-style docstrings
    "nbsphinx",  # For Jupyter Notebook support in Sphinx
    "sphinx.ext.autosummary", # For generating summary tables from docstrings
    "sphinx.ext.viewcode",  # For linking to source code
    "sphinx.ext.autosectionlabel",  # For automatic section labels
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "html_admonition",
]

# Enable MyST to handle eval-rst directive
myst_fence_as_directive = ["eval-rst"]

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'  # Furo is a modern, responsive theme
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'ignore-module-all': True,
}

# Suppress warnings and errors for Read the Docs compatibility
suppress_warnings = ['autodoc.import_error', 'autodoc', 'app.add_autodoc_attrgetter']
autodoc_inherit_docstrings = False

# Suppress warnings for missing imports
suppress_warnings = ['autodoc.import_error']

html_context = {
    "default_mode": "light"
}

html_logo = "_static/RESource_logo_2025.07.jpg"

# Mock imports for Read the Docs compatibility
# Only mock heavy dependencies that cause import issues
autodoc_mock_imports = [
    # Scientific computing libraries
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
    # Geospatial libraries
    'geopandas', 'shapely', 'pyproj', 'rasterio', 'fiona', 'cartopy',
    # Data formats
    'xarray', 'netcdf4', 'h5py', 'tables', 'zarr',
    # Other heavy dependencies
    'requests', 'urllib3', 'pyyaml', 'jinja2', 'bokeh', 'plotly',
    # Specific libraries that might cause issues
    'atlite', 'pypsa', 'powerplantmatching', 'country_converter',
    # System/OS specific
    'psutil', 'tqdm', 'progressbar2'
]

# Auto-generate mocks for submodules to reduce import issues
autodoc_mock_imports.extend([
    'numpy.random', 'pandas.core', 'geopandas.tools', 
    'matplotlib.pyplot', 'matplotlib.patches', 'matplotlib.colors',
    'scipy.spatial', 'scipy.stats', 'scipy.optimize',
    'sklearn.cluster', 'sklearn.preprocessing', 'sklearn.metrics'
])

# Add Read the Docs specific mocks
if on_rtd:
    autodoc_mock_imports.extend([
        'numba', 'dask', 'distributed', 'lxml', 'openpyxl', 'xlrd',
        'pyarrow', 'fastparquet', 'tables', 'bottleneck', 'numexpr'
    ])

# More robust function for Read the Docs compatibility
def skip_private_members(app, what, name, obj, skip, options):
    # Skip private and dunder methods
    if name.startswith('_'):
        return True
    
    # Skip methods with NODOC marker
    if hasattr(obj, '__doc__') and obj.__doc__ and 'NODOC' in obj.__doc__:
        return True
    
    return skip

# Handle autodoc import failures gracefully
def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    return skip_private_members(app, what, name, obj, skip, options)

def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)
    # Continue on import errors
    try:
        app.config.autodoc_mock_imports_fallback = True
    except Exception:
        pass
    
    # Add custom CSS for better rendering
    try:
        app.add_css_file('custom.css')
    except Exception:
        pass
    
    # Handle import errors gracefully
    def missing_reference_handler(app, env, node, contnode):
        # Return None to let Sphinx handle it normally
        return None
    
    app.connect('missing-reference', missing_reference_handler)
