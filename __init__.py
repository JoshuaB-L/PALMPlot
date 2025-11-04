# ============================================================================
# File: palmplot/__init__.py
# Main package initialization file
# ============================================================================

"""
PALMPlot - PALM Simulation Data Visualization Package

A comprehensive Python package for visualizing PALM-LES simulation data,
specifically designed for analyzing the cooling effects of urban trees
during heatwaves.

Main Features:
- Modular architecture with separate modules for each plot type
- Publication-quality plots designed for high-end journals
- Flexible YAML-based configuration
- Support for multiple output formats (PNG, PDF, SVG)
- Parallel processing for efficient data loading
- Comprehensive logging and error handling

Usage:
    from palmplot import PALMPlot
    
    # Initialize with configuration file
    palmplot = PALMPlot('config.yaml')
    
    # Run all analyses
    palmplot.run()

Command Line Usage:
    palmplot config.yaml

Author: Joshua Brook-Lawson
Institution: University of Bath
Email: josh.brooklawson@gmail.com
"""

__version__ = "1.0.0"
__author__ = "Joshua Brook-Lawson"
__email__ = "josh.brooklawson@gmail.com"
__license__ = "MIT"

# Import main class for convenience
from .__main__ import PALMPlot

# Define public API
__all__ = [
    "PALMPlot",
    "__version__",
    "__author__",
    "__email__",
    "__license__"
]

# Package metadata
PACKAGE_NAME = "palmplot"
PACKAGE_DESCRIPTION = "PALM Simulation Data Visualization Package"
PACKAGE_URL = "https://github.com/yourusername/palmplot"