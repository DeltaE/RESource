
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))



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
    'undoc-members': False,  # include undocumented just in case
    # 'private-members': True,
}
html_context = {
    "default_mode": "light"
}

html_logo = "_static/logo.png"

html_theme_options = {
    "rightsidebar": True,  # enables the right sidebar
    "stickysidebar": True,  # sidebar scrolls with page
    "collapsiblesidebar": True,  # sidebar can be collapsed
    "externalrefs": False,  # don't style external links
}
