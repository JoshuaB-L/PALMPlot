# ============================================================================
# File: palmplot/core/__init__.py
# Core modules package initialization
# ============================================================================

"""
Core modules for PALMPlot

This subpackage contains the core functionality for data loading and
configuration management.

Classes:
    PALMDataLoader: Handles loading and preprocessing of PALM simulation data
    ConfigHandler: Manages configuration loading, validation, and access
"""

from .data_loader import PALMDataLoader
from .config_handler import ConfigHandler

# Define public API for core subpackage
__all__ = [
    "PALMDataLoader",
    "ConfigHandler"
]