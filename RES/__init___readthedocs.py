"""
RESource - A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.
"""

__version__ = "2025.07"
__author__ = "Md Eliasinul Islam"

import sys
import warnings

# Suppress warnings during documentation build
if 'sphinx' in sys.modules:
    warnings.filterwarnings("ignore")

# Import main classes with robust error handling for Read the Docs
try:
    from .RESources import RESources_builder
except Exception:
    RESources_builder = None

try:
    from .atb import NREL_ATBProcessor
except Exception:
    NREL_ATBProcessor = None

try:
    from .boundaries import GADMBoundaries
except Exception:
    GADMBoundaries = None

try:
    from .cell import GridCells
except Exception:
    GridCells = None

try:
    from .score import CellScorer
except Exception:
    CellScorer = None

try:
    from .hdf5_handler import DataHandler
except Exception:
    DataHandler = None

try:
    from .tech import OEDBTurbines
except Exception:
    OEDBTurbines = None

try:
    from . import cluster
except Exception:
    cluster = None

try:
    from .units import Units
except Exception:
    Units = None

__all__ = [
    "RESources_builder",
    "NREL_ATBProcessor", 
    "GADMBoundaries",
    "GridCells",
    "CellScorer",
    "DataHandler",
    "OEDBTurbines",
    "cluster",
    "Units",
]
