"""
Spatial Cooling Plotter for Slide 8
Visualizes horizontal temperature distribution at 2m height
Fixed version with proper settings handling and dimension alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
import logging
import xarray as xr
import pandas as pd

from .base_plotter import BasePlotter


class SpatialCoolingPlotter(BasePlotter):
    """Creates visualizations for spatial cooling patterns"""
    
    def __init__(self, config: Dict, output_manager):
        """
        Initialize spatial cooling plotter
        
        Args:
            config: Configuration dictionary
            output_manager: Output manager instance
        """
        super().__init__(config, output_manager)
        self.logger = logging.getLogger(__name__)
        
    def generate_plot(self, plot_type: str, data: Dict) -> plt.Figure:
        """
        Generate specific plot type
        
        Args:
            plot_type: Type of plot to generate
            data: Loaded simulation data
            
        Returns:
            Matplotlib figure
        """
        if plot_type == "daytime_cooling":
            return self._plot_daytime_cooling(data)
        elif plot_type == "nighttime_cooling":
            return self._plot_nighttime_cooling(data)
        elif plot_type == "cooling_extent":
            return self._plot_cooling_extent(data)
        elif plot_type == "temperature_maps":
            return self._plot_temperature_maps(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    def _plot_daytime_cooling(self, data: Dict) -> plt.Figure:
        """Create spatial cooling pattern plot for daytime"""
        # Get settings - works with both figures and slides config
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work
        daytime_hour = settings['daytime_hour']
        height = settings['analysis_height']

        return self._create_spatial_comparison(data, daytime_hour, height,
                                              "Daytime Spatial Cooling Patterns (14:00)")
        
    def _plot_nighttime_cooling(self, data: Dict) -> plt.Figure:
        """Create spatial cooling pattern plot for nighttime"""
        # Get settings - works with both figures and slides config
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work
        nighttime_hour = settings['nighttime_hour']
        height = settings['analysis_height']

        return self._create_spatial_comparison(data, nighttime_hour, height,
                                              "Nighttime Spatial Cooling Patterns (02:00)")
        
    def _extract_spatial_temperature(self, dataset: xr.Dataset, hour: int, 
                                   height: float) -> np.ndarray:
        """
        Extract 2D temperature field at specific hour and height
        
        Args:
            dataset: xarray dataset containing temperature data
            hour: Hour of day (0-23)
            height: Height in meters
            
        Returns:
            2D numpy array of temperature values
        """
        try:
            # Find the appropriate height index
            # For child domain (N02), first data level is at zu_3d index 21
            # For parent domain, first data level is at zu_3d index 25
            if 'N02' in dataset.encoding.get('source', ''):
                z_idx = 21  # First data level for child domain
            else:
                z_idx = 25  # First data level for parent domain
                
            # Get temperature data
            temp_data = dataset['ta']
            
            # Select data for the specific hour
            # Convert time to pandas datetime for hour extraction
            time_pd = pd.to_datetime(dataset['time'].values)
            hour_mask = time_pd.hour == hour
            
            if not any(hour_mask):
                self.logger.warning(f"No data found for hour {hour}")
                # Return mean over all times as fallback
                return temp_data.isel(zu_3d=z_idx).mean(dim='time').values
                
            # Get temperature at specified hour and height
            temp_hourly = temp_data.isel(time=hour_mask, zu_3d=z_idx)
            
            # Average over all instances of this hour
            temp_field = temp_hourly.mean(dim='time').values
            
            return temp_field
            
        except Exception as e:
            self.logger.error(f"Error extracting spatial temperature: {str(e)}")
            raise
            
    def _calculate_cooling_effect(self, temp_field: np.ndarray, 
                                base_field: np.ndarray) -> np.ndarray:
        """
        Calculate cooling effect ensuring proper dimension alignment
        
        Args:
            temp_field: Temperature field from simulation
            base_field: Temperature field from base case
            
        Returns:
            Cooling effect field (base - simulation)
        """
        # Ensure both fields have the same shape
        if temp_field.shape != base_field.shape:
            self.logger.warning(f"Shape mismatch: {temp_field.shape} vs {base_field.shape}")
            
            # Try to align by taking the minimum dimensions
            min_x = min(temp_field.shape[0], base_field.shape[0])
            min_y = min(temp_field.shape[1], base_field.shape[1])
            
            temp_field = temp_field[:min_x, :min_y]
            base_field = base_field[:min_x, :min_y]
            
        # Calculate cooling (positive values mean cooling)
        cooling = base_field - temp_field
        
        return cooling
        
    def _create_spatial_comparison(self, data: Dict, hour: int, height: float,
                                  title: str) -> plt.Figure:
        """Create spatial comparison plot for all scenarios"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']

        # Get settings - works with both figures and slides config
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(ages), len(spacings) + 1, 
                                figsize=(20, 16), squeeze=False)
        
        # Get base case temperature if available
        base_temp_field = None
        if data.get('base_case') and 'av_3d_n02' in data['base_case']:
            try:
                base_data = data['base_case']['av_3d_n02']
                base_temp_field = self._extract_spatial_temperature(
                    base_data, hour, height
                )
            except Exception as e:
                self.logger.warning(f"Could not extract base case temperature: {str(e)}")
                
        # Color settings for temperature
        temp_vmin, temp_vmax = 20, 35  # Temperature range
        cooling_vmin, cooling_vmax = -6, 6  # Cooling range
        
        # Temperature colormap
        temp_cmap = self._get_colormap('temperature')
        
        # Cooling colormap with diverging colors
        cooling_cmap = self._get_colormap('cooling')
        cooling_norm = TwoSlopeNorm(vmin=cooling_vmin, vcenter=0, vmax=cooling_vmax)
        
        for i, age in enumerate(ages):
            # Plot base case in first column
            if base_temp_field is not None:
                im = axes[i, 0].imshow(base_temp_field, cmap=temp_cmap,
                                      vmin=temp_vmin, vmax=temp_vmax,
                                      origin='lower', aspect='equal')
                axes[i, 0].set_title("Base Case" if i == 0 else "")
                axes[i, 0].set_ylabel(f"{age} years", fontsize=12, weight='bold')
            else:
                axes[i, 0].text(0.5, 0.5, "No base case", 
                               transform=axes[i, 0].transAxes,
                               ha='center', va='center')
                
            # Plot each spacing scenario
            for j, spacing in enumerate(spacings):
                ax = axes[i, j + 1]
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            # Extract temperature field
                            temp_field = self._extract_spatial_temperature(
                                sim_data['av_3d_n02'], hour, height
                            )
                            
                            # Calculate cooling effect if base case available
                            if base_temp_field is not None:
                                cooling_field = self._calculate_cooling_effect(
                                    temp_field, base_temp_field
                                )
                                
                                # Apply smoothing if configured
                                if self.config['analysis']['spatial']['grid_interpolation']:
                                    sigma = self.config['analysis']['spatial']['smoothing_sigma']
                                    cooling_field = gaussian_filter(cooling_field, sigma=sigma)
                                    
                                # Plot cooling field
                                im = ax.imshow(cooling_field, cmap=cooling_cmap,
                                             norm=cooling_norm, origin='lower',
                                             aspect='equal')

                                # Add tree locations if requested
                                if settings['show_tree_locations']:
                                    self._overlay_tree_locations(ax, data, spacing)
                                    
                            else:
                                # Just plot temperature if no base case
                                im = ax.imshow(temp_field, cmap=temp_cmap,
                                             vmin=temp_vmin, vmax=temp_vmax,
                                             origin='lower', aspect='equal')
                                             
                        except Exception as e:
                            self.logger.warning(f"Error plotting {case_key}: {str(e)}")
                            ax.text(0.5, 0.5, "Error", transform=ax.transAxes,
                                   ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                           ha='center', va='center')
                    
                # Add title for first row
                if i == 0:
                    ax.set_title(f"{spacing}m spacing", fontsize=12, weight='bold')
                    
                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
                
        # Add main title
        fig.suptitle(title, fontsize=16, weight='bold')
        
        # Add colorbars
        # Temperature colorbar for base case
        cbar_temp_ax = fig.add_axes([0.08, 0.02, 0.35, 0.02])
        sm_temp = plt.cm.ScalarMappable(cmap=temp_cmap,
                                       norm=plt.Normalize(vmin=temp_vmin, vmax=temp_vmax))
        sm_temp.set_array([])
        cbar_temp = fig.colorbar(sm_temp, cax=cbar_temp_ax, orientation='horizontal')
        cbar_temp.set_label('Temperature (°C)', fontsize=12)
        
        # Cooling colorbar
        cbar_cool_ax = fig.add_axes([0.55, 0.02, 0.35, 0.02])
        sm_cool = plt.cm.ScalarMappable(cmap=cooling_cmap, norm=cooling_norm)
        sm_cool.set_array([])
        cbar_cool = fig.colorbar(sm_cool, cax=cbar_cool_ax, orientation='horizontal')
        cbar_cool.set_label('Cooling Effect (°C)', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        return fig
        
    def _plot_cooling_extent(self, data: Dict) -> plt.Figure:
        """Plot the spatial extent of cooling beyond tree canopy"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Placeholder for cooling extent analysis
        # This would analyze how far the cooling effect extends beyond the tree canopy
        
        ax.text(0.5, 0.5, "Cooling Extent Analysis\n(To be implemented)", 
               transform=ax.transAxes, ha='center', va='center', fontsize=14)
        
        return fig
        
    def _overlay_tree_locations(self, ax, data: Dict, spacing: int):
        """Overlay tree locations on the plot"""
        if 'tree_locations' not in data:
            return
            
        tree_key = f"{spacing}m_child"
        if tree_key in data['tree_locations']:
            locations = data['tree_locations'][tree_key]
            
            # Plot tree locations as small circles
            for loc in locations:
                circle = patches.Circle((loc[0], loc[1]), radius=2, 
                                      fill=False, edgecolor='black', 
                                      linewidth=1, alpha=0.5)
                ax.add_patch(circle)
                
    def available_plots(self) -> List[str]:
        """Return list of available plot types"""
        return ["daytime_cooling", "nighttime_cooling", "cooling_extent", "temperature_maps"]
    
    """
    Fixed temperature maps plotting methods for spatial_cooling.py
    Replace the existing _plot_temperature_maps, _plot_single_temperature_map, 
    and _extract_time_averaged_temperature methods with these versions
    """
    
    def _plot_temperature_maps(self, data: Dict) -> plt.Figure:
        """
        Create time-averaged temperature maps with separate figures for parent and child domains
        Organized in matrix layout with spacings as columns and ages as rows
        """
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work

        # Get analysis height and domain settings
        analysis_height_m = settings.get('analysis_height', 2.0)
        plot_parent = settings.get('plot_parent_domain', True)
        plot_child = settings.get('plot_child_domain', True)
        
        # Temperature range for consistent colormap
        temp_vmin = settings.get('temp_vmin', 24)
        temp_vmax = settings.get('temp_vmax', 30)
        temp_cmap = self._get_colormap('temperature')
        
        # We'll return a list of figures if both domains are requested
        figures = []
        
        # Create parent domain figure if requested
        if plot_parent:
            fig_parent = self._create_domain_matrix_plot(
                data, spacings, ages, 'parent', 'av_3d',
                temp_vmin, temp_vmax, temp_cmap, analysis_height_m
            )
            figures.append(fig_parent)
        
        # Create child domain figure if requested
        if plot_child:
            fig_child = self._create_domain_matrix_plot(
                data, spacings, ages, 'child', 'av_3d_n02',
                temp_vmin, temp_vmax, temp_cmap, analysis_height_m
            )
            figures.append(fig_child)
        
        # Return the appropriate figure(s)
        if len(figures) == 1:
            return figures[0]
        elif len(figures) == 2:
            # If both domains are plotted, save them separately and return the child domain figure
            # The main script will need to be modified to handle multiple figures
            # For now, we'll save the parent figure here and return the child figure
            self._save_additional_figure(figures[0], 'temperature_maps_parent')
            return figures[1]
        else:
            # No domains requested
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No domains selected for plotting", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
    
    def _create_domain_matrix_plot(self, data: Dict, spacings: List, ages: List,
                                   domain_name: str, data_key: str,
                                   temp_vmin: float, temp_vmax: float, temp_cmap,
                                   height_m: float) -> plt.Figure:
        """
        Create a matrix plot for a single domain with spacings as columns and ages as rows
        
        Args:
            data: Simulation data
            spacings: List of spacings
            ages: List of ages
            domain_name: 'parent' or 'child'
            data_key: 'av_3d' or 'av_3d_n02'
            temp_vmin, temp_vmax: Temperature range
            temp_cmap: Colormap
            height_m: Analysis height in meters
            
        Returns:
            Matplotlib figure
        """
        # Create figure with matrix layout
        n_rows = len(ages) + 1  # +1 for base case row
        n_cols = len(spacings)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        
        # Ensure axes is always 2D array
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Determine domain extent
        if domain_name == 'child':
            extent = [0, 200, 0, 200]
            is_child = True
        else:
            extent = [0, 400, 0, 400]
            is_child = False
        
        # Plot base case in first row (all columns)
        if data.get('base_case') and data_key in data['base_case']:
            base_temp = self._extract_time_averaged_temperature(
                data['base_case'][data_key], height_m, is_child
            )
            
            for col_idx in range(n_cols):
                ax = axes[0, col_idx]
                if base_temp is not None:
                    im = ax.imshow(base_temp, origin='lower', cmap=temp_cmap,
                                  vmin=temp_vmin, vmax=temp_vmax, extent=extent)
                    ax.set_title(f'Base Case\n{spacings[col_idx]}m column', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'Base Case\nNo data', 
                           transform=ax.transAxes, ha='center', va='center')
                
                ax.set_aspect('equal')
                if col_idx == 0:
                    ax.set_ylabel('Base Case', fontsize=12, weight='bold')
        else:
            # No base case data - clear first row
            for col_idx in range(n_cols):
                ax = axes[0, col_idx]
                ax.text(0.5, 0.5, 'Base Case\nNo data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_aspect('equal')
        
        # Plot simulation cases in matrix layout
        for row_idx, age in enumerate(ages):
            for col_idx, spacing in enumerate(spacings):
                ax = axes[row_idx + 1, col_idx]  # +1 to skip base case row
                case_key = f"{spacing}m_{age}yrs"
                
                # Add row labels (ages) on the left
                if col_idx == 0:
                    ax.set_ylabel(f'{age} years', fontsize=12, weight='bold')
                
                # Add column labels (spacings) on top row of simulations
                if row_idx == 0:
                    ax.set_title(f'{spacing}m', fontsize=12, weight='bold')
                
                if case_key in data['simulations'] and data_key in data['simulations'][case_key]:
                    # Extract temperature for this specific domain only
                    temp_field = self._extract_time_averaged_temperature(
                        data['simulations'][case_key][data_key], height_m, is_child
                    )
                    
                    if temp_field is not None:
                        im = ax.imshow(temp_field, origin='lower', cmap=temp_cmap,
                                      vmin=temp_vmin, vmax=temp_vmax, extent=extent)
                        
                        # Add grid
                        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
                    else:
                        ax.text(0.5, 0.5, 'No data', 
                               transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'No data', 
                           transform=ax.transAxes, ha='center', va='center')
                
                # Set aspect ratio
                ax.set_aspect('equal')
                
                # Add axis labels for bottom and left edges
                if row_idx == len(ages) - 1:  # Bottom row
                    ax.set_xlabel('X (m)', fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if col_idx == 0:  # Left column
                    ax.set_ylabel('Y (m)', fontsize=9)
                else:
                    ax.set_yticklabels([])
        
        # Add main title
        domain_text = "Parent Domain" if domain_name == 'parent' else "Child Domain"
        resolution = "10m resolution" if domain_name == 'parent' else "2m resolution"
        fig.suptitle(f'Time-Averaged Temperature at {height_m}m Height - {domain_text} ({resolution})', 
                    fontsize=16, weight='bold')
        
        # Add single colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=temp_cmap,
                                  norm=plt.Normalize(vmin=temp_vmin, vmax=temp_vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Temperature (°C)', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.91, 0.96])
        
        return fig
    
    def _save_additional_figure(self, fig: plt.Figure, name_suffix: str):
        """
        Save an additional figure using the output manager
        
        Args:
            fig: Figure to save
            name_suffix: Suffix for the filename
        """
        try:
            # This is a workaround since the main script expects single figures
            # In production, the main script should be modified to handle multiple figures
            formats = self.config['output']['formats']
            for fmt in formats:
                filename = f"slide_8_{name_suffix}.{fmt}"
                self.output_manager.save_plot(fig, name_suffix, 'slide_8')
        except Exception as e:
            self.logger.warning(f"Could not save additional figure: {str(e)}")
    
    def _plot_single_temperature_map(self, ax, case_data: Dict, title: str,
                                   vmin: float, vmax: float, cmap, height_m: float,
                                   plot_parent: bool, plot_child: bool):
        """
        Plot temperature map for a single case with correct orientation
        
        Args:
            ax: Matplotlib axes
            case_data: Data dictionary for the case
            title: Title for the subplot
            vmin, vmax: Temperature range
            cmap: Colormap
            height_m: Analysis height in meters
            plot_parent: Whether to plot parent domain
            plot_child: Whether to plot child domain
        """
        # Check which data is available
        has_parent = 'av_3d' in case_data
        has_child = 'av_3d_n02' in case_data
        
        if not has_parent and not has_child:
            ax.text(0.5, 0.5, f"No data available\n{title}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_aspect('equal')
            return
        
        # Extract temperature fields based on configuration
        parent_temp = None
        child_temp = None
        
        # Get parent domain temperature if requested and available
        if plot_parent and has_parent:
            parent_temp = self._extract_time_averaged_temperature(
                case_data['av_3d'], height_m, is_child=False
            )
        
        # Get child domain temperature if requested and available
        if plot_child and has_child:
            child_temp = self._extract_time_averaged_temperature(
                case_data['av_3d_n02'], height_m, is_child=True
            )
        
        # Plot based on what's available and requested
        if plot_parent and parent_temp is not None:
            # Plot parent domain
            # Correct orientation: no transpose needed for proper x-y orientation
            im = ax.imshow(parent_temp, origin='lower', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, 400, 0, 400])
            
            # Overlay child domain if requested and available
            if plot_child and child_temp is not None:
                # Child domain is centered in parent domain
                # Parent: 400x400m at 10m resolution (40x40 cells)
                # Child: 200x200m at 2m resolution (100x100 cells)
                # Child covers central 200x200m area (from 100m to 300m)
                child_extent = [100, 300, 100, 300]
                
                # Overlay child domain
                ax.imshow(child_temp, origin='lower', cmap=cmap,
                         vmin=vmin, vmax=vmax, extent=child_extent,
                         alpha=0.95, interpolation='bilinear')
                
                # Add boundary rectangle for child domain
                from matplotlib.patches import Rectangle
                rect = Rectangle((100, 100), 200, 200, linewidth=2, 
                               edgecolor='black', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
                # Add text label for child domain
                ax.text(105, 290, 'Child Domain (2m res)', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        elif plot_child and child_temp is not None:
            # Only child domain requested or available
            im = ax.imshow(child_temp, origin='lower', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, 200, 0, 200])
        else:
            # No data to plot based on settings
            ax.text(0.5, 0.5, f"No data for selected domains\n{title}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_aspect('equal')
            return
        
        # Set labels and title
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    def _extract_time_averaged_temperature(self, dataset: xr.Dataset, 
                                         height_m: float, is_child: bool) -> np.ndarray:
        """
        Extract time-averaged temperature at specified height with improved height selection
        
        Args:
            dataset: xarray dataset containing temperature data
            height_m: Height above ground in meters
            is_child: Whether this is child domain (affects z-index)
            
        Returns:
            2D array of time-averaged temperature
        """
        try:
            # Get z coordinate information
            z_coords = dataset['zu_3d'].values if 'zu_3d' in dataset.dims else None
            
            if z_coords is None:
                self.logger.error("No zu_3d coordinate found in dataset")
                return np.full((200, 200) if is_child else (400, 400), np.nan)
            
            # Get the first data index for this domain type
            first_data_idx = 21 if is_child else 25
            
            # Check if there's temperature data at lower indices
            # Some simulations might have data below the standard first_data_idx
            temp_var = dataset['ta']
            
            # Find the actual first index with valid temperature data
            actual_first_idx = first_data_idx
            for idx in range(first_data_idx):
                try:
                    test_data = temp_var.isel(zu_3d=idx, x=0, y=0, time=0).values
                    if not np.isnan(test_data) and test_data != 0:
                        actual_first_idx = idx
                        break
                except:
                    continue
                
            # Log information about available heights
            self.logger.info(f"{'Child' if is_child else 'Parent'} domain height analysis:")
            self.logger.info(f"  Standard first data index: {first_data_idx}")
            self.logger.info(f"  Actual first data index: {actual_first_idx}")
            self.logger.info(f"  Height at first data: {z_coords[actual_first_idx]:.2f}m")
            self.logger.info(f"  Requested height: {height_m}m")
            
            # If requested height is below the first available data level,
            # use the first available level and warn the user
            if height_m < z_coords[actual_first_idx]:
                self.logger.warning(
                    f"Requested height {height_m}m is below first available data "
                    f"at {z_coords[actual_first_idx]:.2f}m. Using lowest available level."
                )
                z_idx = actual_first_idx
            else:
                # Find the index closest to the requested height
                # Search through all valid levels starting from actual_first_idx
                valid_indices = range(actual_first_idx, len(z_coords))
                valid_heights = z_coords[actual_first_idx:]
                
                # Find closest height
                height_diff = np.abs(valid_heights - height_m)
                relative_idx = np.argmin(height_diff)
                z_idx = actual_first_idx + relative_idx
            
            actual_height = z_coords[z_idx]
            self.logger.info(f"  Using z_idx={z_idx} (actual height={actual_height:.2f}m)")
            
            # For very low heights, also check if surface temperature data is available
            if height_m <= 2.0 and actual_height > height_m + 2.0:
                self.logger.info(
                    "Note: For near-surface temperatures, consider using "
                    "surface temperature variables (t_surf) if available."
                )
            
            # Extract temperature at the selected height
            temp_data = dataset['ta'].isel(zu_3d=z_idx)
            
            # Average over time dimension
            temp_avg = temp_data.mean(dim='time')
            
            # Return the values without transposing to maintain correct orientation
            return temp_avg.values
            
        except Exception as e:
            self.logger.error(f"Error extracting time-averaged temperature: {str(e)}")
            self.logger.error(f"Dataset dimensions: {list(dataset.dims.keys())}")
            self.logger.error(f"Available variables: {list(dataset.data_vars.keys())}")
            # Return NaN array of appropriate size
            if is_child:
                return np.full((200, 200), np.nan)
            else:
                return np.full((400, 400), np.nan)