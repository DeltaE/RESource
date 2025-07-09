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

# More robust function for Read the Docs compatibility
def skip_private_members(app, what, name, obj, skip, options):
    # Skip private and dunder methods
    if name.startswith('_'):
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_private_members)

# Suppress warnings and errors for Read the Docs compatibility
suppress_warnings = ['autodoc.import_error', 'autodoc', 'app.add_autodoc_attrgetter']
autodoc_inherit_docstrings = False

# Suppress warnings for missing imports
suppress_warnings = ['autodoc.import_error']

html_context = {
    "default_mode": "light"
}

html_logo = "_static/RESource_logo.png"
