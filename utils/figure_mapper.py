"""
Figure Mapper Utility for PALMPlot
Handles mapping between slide numbers and figure numbers with subfigure lettering.

This module provides a centralized way to manage the naming convention
transformation from presentation slide numbers (slide_6, slide_7, etc.) to
publication figure numbers (fig_1, fig_2, etc.) with automatic subfigure
letter assignment (a, b, c, etc.).

Author: Joshua Brook-Lawson
Institution: University of Bath
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class FigureMapper:
    """
    Manages mapping between slide IDs and figure IDs with subfigure lettering.

    This class provides a clean interface for converting between old slide-based
    naming (slide_6, slide_7, etc.) and new figure-based naming (fig_1, fig_2, etc.)
    It also handles automatic assignment of subfigure letters (a, b, c, etc.) to
    individual plot types within each figure.

    Attributes:
        SLIDE_TO_FIGURE_MAP: Static mapping from slide IDs to figure numbers
        FIGURE_TO_SLIDE_MAP: Reverse mapping from figure IDs to slide IDs
    """

    # Static mapping: slide_X -> fig_Y
    SLIDE_TO_FIGURE_MAP: Dict[str, str] = {
        'slide_6': 'fig_1',
        'slide_7': 'fig_2',
        'slide_8': 'fig_3',
        'slide_9': 'fig_4',
        'slide_10': 'fig_5'
    }

    # Reverse mapping for backward compatibility
    FIGURE_TO_SLIDE_MAP: Dict[str, str] = {
        v: k for k, v in SLIDE_TO_FIGURE_MAP.items()
    }

    # Figure number extraction
    FIGURE_NUMBERS: Dict[str, int] = {
        'fig_1': 1,
        'fig_2': 2,
        'fig_3': 3,
        'fig_4': 4,
        'fig_5': 5,
        'fig_6': 6
    }

    def __init__(self):
        """Initialize the FigureMapper."""
        self.logger = logging.getLogger(__name__)
        # Track subfigure letter assignments for each figure
        self._subfigure_counters: Dict[str, int] = {}

    @classmethod
    def slide_to_figure(cls, slide_id: str) -> str:
        """
        Convert slide ID to figure ID.

        Args:
            slide_id: Slide identifier (e.g., 'slide_6')

        Returns:
            Figure identifier (e.g., 'fig_1')

        Raises:
            ValueError: If slide_id is not in the mapping

        Examples:
            >>> FigureMapper.slide_to_figure('slide_6')
            'fig_1'
            >>> FigureMapper.slide_to_figure('slide_8')
            'fig_3'
        """
        if slide_id not in cls.SLIDE_TO_FIGURE_MAP:
            raise ValueError(
                f"Unknown slide ID: {slide_id}. "
                f"Valid slide IDs: {list(cls.SLIDE_TO_FIGURE_MAP.keys())}"
            )
        return cls.SLIDE_TO_FIGURE_MAP[slide_id]

    @classmethod
    def figure_to_slide(cls, figure_id: str) -> str:
        """
        Convert figure ID to slide ID (for backward compatibility).

        Args:
            figure_id: Figure identifier (e.g., 'fig_1')

        Returns:
            Slide identifier (e.g., 'slide_6')

        Raises:
            ValueError: If figure_id is not in the mapping

        Examples:
            >>> FigureMapper.figure_to_slide('fig_1')
            'slide_6'
            >>> FigureMapper.figure_to_slide('fig_3')
            'slide_8'
        """
        if figure_id not in cls.FIGURE_TO_SLIDE_MAP:
            raise ValueError(
                f"Unknown figure ID: {figure_id}. "
                f"Valid figure IDs: {list(cls.FIGURE_TO_SLIDE_MAP.keys())}"
            )
        return cls.FIGURE_TO_SLIDE_MAP[figure_id]

    @classmethod
    def get_figure_number(cls, figure_id: str) -> int:
        """
        Extract numeric figure number from figure ID.

        Args:
            figure_id: Figure identifier (e.g., 'fig_1')

        Returns:
            Figure number as integer

        Examples:
            >>> FigureMapper.get_figure_number('fig_1')
            1
            >>> FigureMapper.get_figure_number('fig_3')
            3
        """
        if figure_id not in cls.FIGURE_NUMBERS:
            raise ValueError(f"Unknown figure ID: {figure_id}")
        return cls.FIGURE_NUMBERS[figure_id]

    @classmethod
    def is_valid_slide_id(cls, slide_id: str) -> bool:
        """
        Check if a slide ID is valid.

        Args:
            slide_id: Slide identifier to validate

        Returns:
            True if valid, False otherwise
        """
        return slide_id in cls.SLIDE_TO_FIGURE_MAP

    @classmethod
    def is_valid_figure_id(cls, figure_id: str) -> bool:
        """
        Check if a figure ID is valid.

        Args:
            figure_id: Figure identifier to validate

        Returns:
            True if valid, False otherwise
        """
        return figure_id in cls.FIGURE_NUMBERS

    def get_next_subfigure_letter(self, figure_id: str) -> str:
        """
        Get the next subfigure letter for a given figure.

        This method maintains a counter for each figure and returns sequential
        letters (a, b, c, etc.) for subfigures within that figure.

        Args:
            figure_id: Figure identifier (e.g., 'fig_1')

        Returns:
            Subfigure letter (e.g., 'a', 'b', 'c')

        Examples:
            >>> mapper = FigureMapper()
            >>> mapper.get_next_subfigure_letter('fig_1')
            'a'
            >>> mapper.get_next_subfigure_letter('fig_1')
            'b'
            >>> mapper.get_next_subfigure_letter('fig_2')
            'a'
        """
        if figure_id not in self._subfigure_counters:
            self._subfigure_counters[figure_id] = 0

        letter = chr(ord('a') + self._subfigure_counters[figure_id])
        self._subfigure_counters[figure_id] += 1

        return letter

    def reset_subfigure_counter(self, figure_id: str) -> None:
        """
        Reset the subfigure counter for a specific figure.

        Args:
            figure_id: Figure identifier
        """
        self._subfigure_counters[figure_id] = 0

    def reset_all_counters(self) -> None:
        """Reset all subfigure counters."""
        self._subfigure_counters.clear()

    def generate_subfigure_filename(
        self,
        figure_id: str,
        plot_type: str,
        file_format: str,
        subfigure_letter: Optional[str] = None
    ) -> str:
        """
        Generate a complete filename for a subfigure.

        Args:
            figure_id: Figure identifier (e.g., 'fig_1')
            plot_type: Type of plot (e.g., 'time_series')
            file_format: File extension (e.g., 'png', 'pdf')
            subfigure_letter: Optional explicit subfigure letter. If None,
                            uses the next letter from the counter.

        Returns:
            Complete filename (e.g., 'fig_2a_time_series.png')

        Examples:
            >>> mapper = FigureMapper()
            >>> mapper.generate_subfigure_filename('fig_2', 'time_series', 'png')
            'fig_2a_time_series.png'
            >>> mapper.generate_subfigure_filename('fig_2', 'diurnal_cycle', 'png')
            'fig_2b_diurnal_cycle.png'
        """
        if not self.is_valid_figure_id(figure_id):
            raise ValueError(f"Invalid figure ID: {figure_id}")

        if subfigure_letter is None:
            subfigure_letter = self.get_next_subfigure_letter(figure_id)

        return f"{figure_id}{subfigure_letter}_{plot_type}.{file_format}"

    def generate_ordered_subfigure_names(
        self,
        figure_id: str,
        plot_types: List[str],
        file_format: str
    ) -> Dict[str, str]:
        """
        Generate ordered subfigure filenames for multiple plot types.

        This method ensures consistent letter ordering across all plot types
        within a figure by resetting the counter and assigning letters sequentially.

        Args:
            figure_id: Figure identifier
            plot_types: List of plot types in desired order
            file_format: File extension

        Returns:
            Dictionary mapping plot_type to filename

        Examples:
            >>> mapper = FigureMapper()
            >>> plot_types = ['time_series', 'diurnal_cycle', 'temperature_difference']
            >>> mapper.generate_ordered_subfigure_names('fig_2', plot_types, 'png')
            {'time_series': 'fig_2a_time_series.png',
             'diurnal_cycle': 'fig_2b_diurnal_cycle.png',
             'temperature_difference': 'fig_2c_temperature_difference.png'}
        """
        # Reset counter to ensure consistent ordering
        self.reset_subfigure_counter(figure_id)

        filenames = OrderedDict()
        for plot_type in plot_types:
            filenames[plot_type] = self.generate_subfigure_filename(
                figure_id, plot_type, file_format
            )

        return filenames

    @classmethod
    def get_all_mappings(cls) -> Dict[str, str]:
        """
        Get complete slide-to-figure mapping.

        Returns:
            Dictionary of all slide-to-figure mappings
        """
        return cls.SLIDE_TO_FIGURE_MAP.copy()

    @classmethod
    def get_figure_title_mapping(cls) -> Dict[str, str]:
        """
        Get descriptive titles for each figure.

        Returns:
            Dictionary mapping figure IDs to descriptive titles
        """
        return {
            'fig_1': 'Tree Density Scenarios',
            'fig_2': 'Temperature Dynamics',
            'fig_3': 'Spatial Cooling Patterns',
            'fig_4': 'Vertical Cooling Profile',
            'fig_5': 'Age-Density-Cooling Relationship',
            'fig_6': 'Terrain-Following Transect Analysis'
        }

    def __repr__(self) -> str:
        """String representation of FigureMapper."""
        return (
            f"FigureMapper("
            f"mappings={len(self.SLIDE_TO_FIGURE_MAP)}, "
            f"active_counters={len(self._subfigure_counters)})"
        )
