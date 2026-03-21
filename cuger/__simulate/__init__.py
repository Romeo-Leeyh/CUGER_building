"""
simulation package
------------------
EnergyPlus simulation utilities for building energy analysis.

This package provides:
- IDFEdit: IDF file editing and graph conversion
- SQLReader: SQL output reading and analysis
- simulate module: Single and batch simulation utilities
"""

from .idfedit import IDFEdit
from .sqlread import SQLReader
from . import simulate

__all__ = [
    'IDFEdit',
    'SQLReader',
    'simulate',
]

__version__ = '0.2.0'
