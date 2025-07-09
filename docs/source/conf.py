
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Point to the root of the project



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
    "deflist",
    "tasklist", 
    "fieldlist",
    "colon_fence",
    "dollarmath",
    "amsmath",
    "html_admonition",
    "html_image",
    "attrs_inline",
    "attrs_block",
]

# Enable MyST to handle eval-rst directive
myst_fence_as_directive = ["eval-rst"]

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Some popular Sphinx HTML themes:
html_theme = 'furo' # Furo is a modern, responsive theme

# Other popular Sphinx HTML themes:
# html_theme = 'alabaster'  # Clean, default Sphinx theme
# html_theme = 'sphinx_rtd_theme'  # Read the Docs theme, widely used for documentation
# html_theme = 'classic'  # Classic Sphinx look
# html_theme = 'pydata_sphinx_theme'  # PyData community theme, good for data science projects
# html_theme = 'sphinx_book_theme'  # Book-style theme, great for tutorials and guides
# html_theme = 'nature'  # Green, nature-inspired theme



html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # Include undocumented members
    'show-inheritance': True,
    'inherited-members': True,
    'special-members': '__init__',
}

# Handle import errors gracefully
autodoc_mock_imports = [
    'numpy', 'pandas', 'geopandas', 'matplotlib', 'sklearn', 'scipy', 'xarray', 
    'netcdf4', 'h5py', 'rasterio', 'shapely', 'pyproj', 'cartopy', 'seaborn', 
    'plotly', 'folium', 'osmnx', 'requests', 'beautifulsoup4', 'lxml', 'openpyxl', 
    'xlrd', 'psutil', 'tqdm', 'joblib', 'dask', 'zarr', 'bottleneck', 'numexpr', 
    'cython', 'numba', 'pyyaml', 'toml', 'configparser'
]

html_context = {
    "default_mode": "light"
}

html_logo = "_static/RESource_logo.png"