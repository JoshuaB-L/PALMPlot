# ============================================================================
# File: palmplot/plots/__init__.py
# Plotting modules package initialization
# ============================================================================

"""
Plotting modules for PALMPlot

This subpackage contains all visualization modules for different plot types
corresponding to the presentation slides.

Classes:
    BasePlotter: Base class providing common functionality for all plotters
    TreeDensityPlotter: Creates visualizations for tree density scenarios (Slide 6)
    TemperatureDynamicsPlotter: Visualizes temperature dynamics (Slide 7)
    SpatialCoolingPlotter: Creates spatial cooling pattern plots (Slide 8)
    VerticalProfilePlotter: Visualizes vertical cooling profiles (Slide 9)
    CoolingRelationshipPlotter: Quantifies age-density-cooling relationships (Slide 10)
"""

from .base_plotter import BasePlotter
from .tree_density import TreeDensityPlotter
from .temperature_dynamics import TemperatureDynamicsPlotter
from .spatial_cooling import SpatialCoolingPlotter
from .vertical_profile import VerticalProfilePlotter
from .cooling_relationship import CoolingRelationshipPlotter
from .terrain_transect import TerrainTransectPlotter

# Define public API for plots subpackage
__all__ = [
    "BasePlotter",
    "TreeDensityPlotter",
    "TemperatureDynamicsPlotter",
    "SpatialCoolingPlotter",
    "VerticalProfilePlotter",
    "CoolingRelationshipPlotter",
    "TerrainTransectPlotter"
]

# Dictionary mapping slide IDs to plotter classes for dynamic instantiation
# Supports both legacy slide-based and new figure-based naming
PLOTTER_REGISTRY = {
    # Legacy slide-based naming
    "slide_6": TreeDensityPlotter,
    "slide_7": TemperatureDynamicsPlotter,
    "slide_8": SpatialCoolingPlotter,
    "slide_9": VerticalProfilePlotter,
    "slide_10": CoolingRelationshipPlotter,
    # New figure-based naming
    "fig_1": TreeDensityPlotter,
    "fig_2": TemperatureDynamicsPlotter,
    "fig_3": SpatialCoolingPlotter,
    "fig_4": VerticalProfilePlotter,
    "fig_5": CoolingRelationshipPlotter,
    "fig_6": TerrainTransectPlotter
}

# Mapping between slide IDs and figure IDs
SLIDE_TO_FIGURE_MAP = {
    "slide_6": "fig_1",
    "slide_7": "fig_2",
    "slide_8": "fig_3",
    "slide_9": "fig_4",
    "slide_10": "fig_5"
}

FIGURE_TO_SLIDE_MAP = {v: k for k, v in SLIDE_TO_FIGURE_MAP.items()}