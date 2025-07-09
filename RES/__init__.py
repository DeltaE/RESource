"""
RESource - A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.
"""

__version__ = "2025.07"
__author__ = "Md Eliasinul Islam"


import os
import importlib

# Dynamically import all modules in the current directory except __init__.py
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        modulename = filename[:-3]
        try:
            globals()[modulename] = importlib.import_module(f".{modulename}", __package__)
        except ImportError:
            globals()[modulename] = None
