"""
Base Plotter Module
Provides common functionality for all visualization modules
Fixed version with proper font handling and data extraction methods
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod
import warnings


class BasePlotter(ABC):
    """Base class providing common functionality for all plotters"""
    
    def __init__(self, config: Dict, output_manager):
        """
        Initialize base plotter

        Args:
            config: Configuration dictionary
            output_manager: Output manager instance
        """
        self.config = config
        self.output_manager = output_manager
        self.logger = logging.getLogger(__name__)

        # Determine if config uses figures or slides
        self._plots_key = 'figures' if 'figures' in config.get('plots', {}) else 'slides'

        # Set up matplotlib parameters
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """Configure matplotlib settings"""
        # Set default figure parameters
        plt.rcParams['figure.dpi'] = self.config['output']['dpi']
        plt.rcParams['savefig.dpi'] = self.config['output']['dpi']
        plt.rcParams['figure.figsize'] = self.config['plots']['global_settings']['figure_size']
        
        # Font handling with fallback
        try:
            # Try to use configured font
            font_family = self.config['plots']['global_settings']['font_family']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            if font_family in available_fonts:
                plt.rcParams['font.family'] = font_family
            else:
                # Use fallback fonts
                fallback_fonts = ['DejaVu Sans', 'Helvetica', 'sans-serif']
                for font in fallback_fonts:
                    if font in available_fonts or font == 'sans-serif':
                        plt.rcParams['font.family'] = font
                        self.logger.warning(f"Font '{font_family}' not found. Using '{font}' instead.")
                        break
                        
        except Exception as e:
            self.logger.warning(f"Error setting font: {str(e)}. Using matplotlib defaults.")
            
        # Set font sizes
        plt.rcParams['font.size'] = self.config['plots']['global_settings']['font_size']
        plt.rcParams['axes.titlesize'] = self.config['plots']['global_settings']['title_font_size']
        plt.rcParams['axes.labelsize'] = self.config['plots']['global_settings']['label_font_size']
        plt.rcParams['xtick.labelsize'] = self.config['plots']['global_settings']['font_size']
        plt.rcParams['ytick.labelsize'] = self.config['plots']['global_settings']['font_size']
        plt.rcParams['legend.fontsize'] = self.config['plots']['global_settings']['legend_font_size']
        
        # Grid settings
        plt.rcParams['axes.grid'] = self.config['plots']['global_settings']['grid']
        plt.rcParams['grid.alpha'] = self.config['plots']['global_settings']['grid_alpha']
        
        # Suppress font warnings
        warnings.filterwarnings('ignore', message='findfont: Font family')
        
    @abstractmethod
    def generate_plot(self, plot_type: str, data: Dict) -> plt.Figure:
        """
        Generate specific plot type - must be implemented by subclasses
        
        Args:
            plot_type: Type of plot to generate
            data: Loaded simulation data
            
        Returns:
            Matplotlib figure
        """
        pass
        
    @abstractmethod
    def available_plots(self) -> List[str]:
        """Return list of available plot types - must be implemented by subclasses"""
        pass
        
    def save_plot(self, fig: plt.Figure, plot_name: str, slide_id: str):
        """
        Save plot using output manager
        
        Args:
            fig: Matplotlib figure
            plot_name: Name of the plot
            slide_id: Slide identifier
        """
        self.output_manager.save_plot(fig, plot_name, slide_id)
        
    def _get_color_palette(self, palette_type: str, n_colors: int) -> List[str]:
        """
        Get color palette from configuration
        
        Args:
            palette_type: Type of palette (e.g., 'age', 'spacing')
            n_colors: Number of colors needed
            
        Returns:
            List of color codes
        """
        color_schemes = self.config['plots']['global_settings']['color_schemes']
        
        if palette_type in color_schemes:
            colors = color_schemes[palette_type]
            if isinstance(colors, list):
                return colors[:n_colors]
            else:
                # It's a colormap name
                cmap = plt.get_cmap(colors)
                return [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        else:
            # Default color palette
            cmap = plt.get_cmap('tab10')
            return [cmap(i % 10) for i in range(n_colors)]
            
    def _get_colormap(self, cmap_type: str):
        """
        Get colormap from configuration
        
        Args:
            cmap_type: Type of colormap (e.g., 'temperature', 'cooling')
            
        Returns:
            Matplotlib colormap
        """
        color_schemes = self.config['plots']['global_settings']['color_schemes']
        
        if cmap_type in color_schemes:
            return plt.get_cmap(color_schemes[cmap_type])
        else:
            # Default colormaps
            defaults = {
                'temperature': 'RdBu_r',
                'cooling': 'RdBu_r',
                'density': 'Greens'
            }
            return plt.get_cmap(defaults.get(cmap_type, 'viridis'))
            
    def _apply_common_formatting(self, ax, title: str = None, 
                               xlabel: str = None, ylabel: str = None):
        """
        Apply common formatting to axes
        
        Args:
            ax: Matplotlib axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if title:
            ax.set_title(title, fontsize=self.config['plots']['global_settings']['title_font_size'], 
                        weight='bold', pad=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config['plots']['global_settings']['label_font_size'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config['plots']['global_settings']['label_font_size'])
            
        # Apply grid settings
        if self.config['plots']['global_settings']['grid']:
            ax.grid(True, alpha=self.config['plots']['global_settings']['grid_alpha'], 
                   linestyle='--')
            
        # Apply tight layout if configured
        if self.config['plots']['global_settings']['publication_quality']['tight_layout']:
            plt.tight_layout()
            
    def _format_time_axis(self, ax, time_array):
        """
        Format time axis for better readability
        
        Args:
            ax: Matplotlib axes
            time_array: Array of datetime values
        """
        import matplotlib.dates as mdates
        
        # Determine appropriate time format based on time range
        time_range = time_array[-1] - time_array[0]
        
        if hasattr(time_range, 'days'):
            if time_range.days > 7:
                # Weekly format
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            elif time_range.days > 1:
                # Daily format
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            else:
                # Hourly format
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        else:
            # Default format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
            
        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
    def _extract_temperature_at_height(self, dataset, height: float):
        """
        Extract temperature at specified height
        
        Args:
            dataset: xarray dataset
            height: Height in meters
            
        Returns:
            Temperature data at specified height
        """
        # For PALM data, we typically use specific indices rather than height interpolation
        # Child domain (N02): first data level at zu_3d=21
        # Parent domain: first data level at zu_3d=25
        
        if 'N02' in dataset.encoding.get('source', ''):
            z_idx = 21  # First data level for child domain
        else:
            z_idx = 25  # First data level for parent domain
            
        # Extract temperature
        return dataset['ta'].isel(zu_3d=z_idx).mean(dim=['x', 'y'])
        
    def _add_statistics_box(self, ax, stats: Dict, location: str = 'upper right'):
        """
        Add statistics box to plot
        
        Args:
            ax: Matplotlib axes
            stats: Dictionary of statistics to display
            location: Location of the box
        """
        # Format statistics text
        stats_text = '\n'.join([f"{key}: {value:.2f}" for key, value in stats.items()])
        
        # Determine position based on location string
        loc_map = {
            'upper right': (0.98, 0.98),
            'upper left': (0.02, 0.98),
            'lower right': (0.98, 0.02),
            'lower left': (0.02, 0.02)
        }
        
        x, y = loc_map.get(location, (0.98, 0.98))
        ha = 'right' if 'right' in location else 'left'
        va = 'top' if 'upper' in location else 'bottom'
        
        # Add text box
        ax.text(x, y, stats_text, transform=ax.transAxes,
               verticalalignment=va, horizontalalignment=ha,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='gray', alpha=0.9),
               fontsize=self.config['plots']['global_settings']['font_size'] - 2)
               
    def close_figure(self, fig: plt.Figure):
        """
        Properly close a matplotlib figure to free memory
        
        Args:
            fig: Matplotlib figure to close
        """
        plt.close(fig)
        
    def _get_plot_settings(self, item_id: str) -> Dict:
        """
        Get settings for a specific plot item (figure or slide).

        This method provides backward compatibility by handling both
        'slides' and 'figures' config structures.

        Args:
            item_id: Figure or slide identifier (e.g., 'fig_3' or 'slide_8')

        Returns:
            Settings dictionary for the specified item

        Raises:
            KeyError: If item_id is not found in configuration

        Examples:
            # For slide_8 or fig_3 (same settings)
            settings = self._get_plot_settings('fig_3')
            settings = self._get_plot_settings('slide_8')
        """
        from ..utils.figure_mapper import FigureMapper

        plots_section = self.config['plots'][self._plots_key]

        # Try direct access first
        if item_id in plots_section:
            return plots_section[item_id]['settings']

        # If not found and using figures, try converting from slide ID
        if self._plots_key == 'figures' and item_id.startswith('slide_'):
            figure_id = FigureMapper.slide_to_figure(item_id)
            if figure_id in plots_section:
                return plots_section[figure_id]['settings']

        # If not found and using slides, try converting from figure ID
        if self._plots_key == 'slides' and item_id.startswith('fig_'):
            slide_id = FigureMapper.figure_to_slide(item_id)
            if slide_id in plots_section:
                return plots_section[slide_id]['settings']

        # If still not found, raise error
        raise KeyError(
            f"Plot settings for '{item_id}' not found in config. "
            f"Available items: {list(plots_section.keys())}"
        )

    def _validate_data(self, data: Dict) -> bool:
        """
        Validate that required data is present

        Args:
            data: Data dictionary to validate

        Returns:
            True if data is valid, False otherwise
        """
        if not data:
            self.logger.error("No data provided")
            return False

        # Check for required keys
        required_keys = ['simulations']
        for key in required_keys:
            if key not in data:
                self.logger.error(f"Missing required data key: {key}")
                return False

        # Check that we have at least one simulation
        if not data['simulations']:
            self.logger.error("No simulation data found")
            return False

        return True