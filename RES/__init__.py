"""
RESource - A Modular and Transparent Open-Source Framework for Sub-National Assessment of Solar and Land-based Wind Potential.
"""

# Import main classes and modules for easier access
try:
    from .RESources import RESources_builder
except ImportError:
    RESources_builder = None

try:
    from .atb import NREL_ATBProcessor
except ImportError:
    NREL_ATBProcessor = None

try:
    from .boundaries import GADMBoundaries
except ImportError:
    GADMBoundaries = None

try:
    from .cell import GridCells
except ImportError:
    GridCells = None

try:
    from .score import CellScorer
except ImportError:
    CellScorer = None

try:
    from .hdf5_handler import DataHandler
except ImportError:
    DataHandler = None

try:
    from .tech import OEDBTurbines
except ImportError:
    OEDBTurbines = None

# Make cluster module available
try:
    from . import cluster
except ImportError:
    cluster = None

__version__ = "2025.07"
__author__ = "Md Eliasinul Islam"

__all__ = [
    "RESources_builder",
    "NREL_ATBProcessor", 
    "GADMBoundaries",
    "GridCells",
    "CellScorer",
    "DataHandler",
    "OEDBTurbines",
    "cluster",
]