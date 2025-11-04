"""
Output Manager for PALMPlot
Handles directory creation and file saving with support for figure-based
naming and subfigure lettering.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List
import matplotlib.pyplot as plt

from .figure_mapper import FigureMapper


class OutputManager:
    """
    Manages output directory structure and file saving with figure-based naming.

    This class integrates with FigureMapper to provide automatic subfigure
    lettering and proper figure directory organization.

    Attributes:
        base_directory: Base path for all outputs
        run_directory: Timestamped directory for current run
        figure_mapper: FigureMapper instance for naming conversions
    """

    def __init__(self, base_directory: str, use_figures: bool = True):
        """
        Initialize output manager

        Args:
            base_directory: Base directory for all outputs
            use_figures: If True, use figure-based naming (fig_1, fig_2).
                        If False, use legacy slide-based naming (slide_6, slide_7).
                        Default is True.
        """
        self.base_directory = Path(base_directory)
        self.logger = logging.getLogger(__name__)
        self.use_figures = use_figures

        # Create base directory
        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Create run-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_directory = self.base_directory / f"run_{timestamp}"
        self.run_directory.mkdir(exist_ok=True)

        # Initialize FigureMapper
        self.figure_mapper = FigureMapper()

        self.logger.info(f"Output directory created: {self.run_directory}")
        self.logger.info(f"Using {'figure-based' if use_figures else 'slide-based'} naming")
        
    def create_slide_directory(self, slide_id: str) -> Path:
        """
        Create directory for specific slide or figure.

        This method supports both legacy slide-based naming and new figure-based
        naming. If use_figures is True, it converts slide IDs to figure IDs.

        Args:
            slide_id: Slide identifier (e.g., 'slide_6') or figure identifier (e.g., 'fig_1')

        Returns:
            Path to slide/figure directory

        Examples:
            With use_figures=True and slide_id='slide_6':
                Creates and returns: run_TIMESTAMP/fig_1/
            With use_figures=False and slide_id='slide_6':
                Creates and returns: run_TIMESTAMP/slide_6/
        """
        # Determine the directory name
        if self.use_figures:
            # Convert slide ID to figure ID if needed
            if slide_id.startswith('slide_'):
                dir_name = self.figure_mapper.slide_to_figure(slide_id)
            elif slide_id.startswith('fig_'):
                dir_name = slide_id
            else:
                # Assume it's a legacy slide number
                dir_name = slide_id
        else:
            dir_name = slide_id

        # Create directory
        dir_path = self.run_directory / dir_name
        dir_path.mkdir(exist_ok=True)
        return dir_path

    def create_figure_directory(self, figure_id: str) -> Path:
        """
        Create directory for specific figure.

        Args:
            figure_id: Figure identifier (e.g., 'fig_1')

        Returns:
            Path to figure directory
        """
        figure_dir = self.run_directory / figure_id
        figure_dir.mkdir(exist_ok=True)
        return figure_dir
        
    def save_figure(self, fig: plt.Figure, slide_id: str,
                   plot_type: str, format: str = 'png',
                   dpi: Optional[int] = None,
                   subfigure_letter: Optional[str] = None) -> Path:
        """
        Save matplotlib figure to file with optional subfigure lettering.

        This method supports both legacy slide-based naming and new figure-based
        naming with automatic subfigure letter assignment.

        Args:
            fig: Matplotlib figure
            slide_id: Slide or figure identifier
            plot_type: Type of plot
            format: Output format (png, pdf, svg)
            dpi: Resolution for raster formats
            subfigure_letter: Optional explicit subfigure letter. If None and
                            use_figures is True, automatically assigns next letter.

        Returns:
            Path to saved file

        Examples:
            With use_figures=True, slide_id='slide_7', plot_type='time_series':
                Saves as: fig_2/fig_2a_time_series.png
            With use_figures=False, slide_id='slide_7', plot_type='time_series':
                Saves as: slide_7/slide_7_time_series.png
        """
        # Get directory (handles slide-to-figure conversion if needed)
        output_dir = self.create_slide_directory(slide_id)

        # Determine the actual ID for filename generation
        if self.use_figures:
            if slide_id.startswith('slide_'):
                figure_id = self.figure_mapper.slide_to_figure(slide_id)
            elif slide_id.startswith('fig_'):
                figure_id = slide_id
            else:
                figure_id = slide_id

            # Generate filename with subfigure letter
            filename = self.figure_mapper.generate_subfigure_filename(
                figure_id, plot_type, format, subfigure_letter
            )
        else:
            # Legacy naming: slide_X_plot_type.format
            filename = f"{slide_id}_{plot_type}.{format}"

        filepath = output_dir / filename

        # Configure save parameters
        save_kwargs = {
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'facecolor': 'white'
        }

        if dpi:
            save_kwargs['dpi'] = dpi

        if format in ['pdf', 'svg']:
            # Vector formats
            save_kwargs['format'] = format
            if format == 'pdf':
                save_kwargs['backend'] = 'pdf'
            elif format == 'svg':
                save_kwargs['backend'] = 'svg'

        try:
            fig.savefig(filepath, **save_kwargs)
            self.logger.info(f"Saved figure: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error saving figure {filepath}: {str(e)}")
            raise
            
    def reset_subfigure_counter(self, figure_id: str) -> None:
        """
        Reset the subfigure counter for a specific figure.

        This should be called before generating plots for a new figure to
        ensure subfigure letters start from 'a'.

        Args:
            figure_id: Figure identifier (e.g., 'fig_1')
        """
        self.figure_mapper.reset_subfigure_counter(figure_id)

    def get_figure_id_from_slide(self, slide_id: str) -> str:
        """
        Get figure ID from slide ID.

        Args:
            slide_id: Slide identifier (e.g., 'slide_6')

        Returns:
            Figure identifier (e.g., 'fig_1')
        """
        if slide_id.startswith('slide_'):
            return self.figure_mapper.slide_to_figure(slide_id)
        elif slide_id.startswith('fig_'):
            return slide_id
        else:
            raise ValueError(f"Invalid slide/figure ID: {slide_id}")

    def copy_config(self, config_path: str):
        """
        Copy configuration file to output directory

        Args:
            config_path: Path to configuration file
        """
        config_dest = self.run_directory / "config.yaml"
        shutil.copy2(config_path, config_dest)
        self.logger.info(f"Configuration copied to: {config_dest}")
        
    def create_summary_file(self, summary_content: str):
        """
        Create summary file with run information
        
        Args:
            summary_content: Content for summary file
        """
        summary_path = self.run_directory / "summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"PALMPlot Run Summary\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.run_directory}\n")
            f.write(f"\n{summary_content}")
            
        self.logger.info(f"Summary file created: {summary_path}")
        
    def get_latest_run_directory(self) -> Path:
        """
        Get the most recent run directory
        
        Returns:
            Path to latest run directory
        """
        run_dirs = [d for d in self.base_directory.iterdir() 
                   if d.is_dir() and d.name.startswith('run_')]
        
        if not run_dirs:
            return self.run_directory
            
        # Sort by modification time
        latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
        return latest
        
    def create_composite_directory(self) -> Path:
        """
        Create directory for composite plots combining multiple slides
        
        Returns:
            Path to composite directory
        """
        composite_dir = self.run_directory / "composite"
        composite_dir.mkdir(exist_ok=True)
        return composite_dir
        
    def clean_old_runs(self, keep_last: int = 5):
        """
        Clean old run directories, keeping only the most recent ones
        
        Args:
            keep_last: Number of recent runs to keep
        """
        run_dirs = [d for d in self.base_directory.iterdir() 
                   if d.is_dir() and d.name.startswith('run_')]
        
        if len(run_dirs) <= keep_last:
            return
            
        # Sort by modification time
        run_dirs.sort(key=lambda d: d.stat().st_mtime)
        
        # Remove older directories
        for old_dir in run_dirs[:-keep_last]:
            try:
                shutil.rmtree(old_dir)
                self.logger.info(f"Removed old run directory: {old_dir}")
            except Exception as e:
                self.logger.warning(f"Could not remove directory {old_dir}: {str(e)}")