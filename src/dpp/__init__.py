# src/datapreprocessor/__init__.py

"""
datapreprocessor - A modular data preprocessing package.

Example usage:
import datapreprocessor as dpp
obj = dpp(dataset, "target")
"""

from .DPP import DataPreprocessor

__all__ = ["DataPreprocessor"]
import sys
sys.modules["datapreprocessor"] = DataPreprocessor
