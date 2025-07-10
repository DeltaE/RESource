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
    "myst_parser",  # For Markdown support - must be first
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

# MyST configuration for eval-rst
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

# NBSphinx configuration for Jupyter notebooks
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Allow notebooks with errors to be included
nbsphinx_timeout = 60  # Timeout for notebook execution

# Additional NBSphinx settings for better compatibility
nbsphinx_kernel_name = 'python3'
nbsphinx_codecell_lexer = 'none'
