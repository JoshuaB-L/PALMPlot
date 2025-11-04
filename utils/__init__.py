# ============================================================================
# File: palmplot/utils/__init__.py
# Utility modules package initialization
# ============================================================================

"""
Utility modules for PALMPlot

This subpackage contains utility functions and classes for logging,
output management, and other helper functionality.

Functions:
    setup_logger: Configure logging for PALMPlot
    log_execution_time: Decorator to log function execution time
    log_memory_usage: Decorator to log memory usage

Classes:
    LoggerContext: Context manager for temporary logging configuration
    OutputManager: Manages output directory structure and file saving
"""

from .logger import (
    setup_logger,
    LoggerContext,
    log_execution_time,
    log_memory_usage
)
from .output_manager import OutputManager
from .figure_mapper import FigureMapper

# Define public API for utils subpackage
__all__ = [
    "setup_logger",
    "LoggerContext",
    "log_execution_time",
    "log_memory_usage",
    "OutputManager",
    "FigureMapper"
]

# Utility functions that might be useful across the package
def ensure_list(value):
    """
    Ensure that a value is a list.
    
    Args:
        value: Value to convert to list
        
    Returns:
        List containing the value(s)
    """
    if value is None:
        return []
    elif isinstance(value, list):
        return value
    else:
        return [value]

def format_case_name(spacing, age):
    """
    Format a standard case name from spacing and age.
    
    Args:
        spacing: Tree spacing in meters
        age: Tree age in years
        
    Returns:
        Formatted case name string
    """
    return f"thf_forest_spacing_{spacing}m_age_{age}yrs"