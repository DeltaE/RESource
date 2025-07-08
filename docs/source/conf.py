
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../RES'))



project = 'RESource'
copyright = '2025, Md Eliasinul Islam'
author = 'Md Eliasinul Islam'
release = '2025.07'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser"  # For Markdown support
]

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Some popular Sphinx HTML themes:
html_theme = 'sphinx_rtd_theme'  # 'sphinx_rtd_theme', 'alabaster', 'furo', 'pydata_sphinx_theme', 'sphinx_book_theme', 'classic', 'bizstyle', 'nature', 'scrolls', 'agogo', 'haiku', 'pyramid', 'traditional'


html_static_path = ['_static']
