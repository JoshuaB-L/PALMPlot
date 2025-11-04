"""
Temperature Dynamics Plotter for Slide 7
Visualizes age and density effects on air temperature
Fixed version with proper time dimension and data extraction handling
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import xarray as xr

from .base_plotter import BasePlotter


class TemperatureDynamicsPlotter(BasePlotter):
    """Creates visualizations for temperature dynamics"""
    
    def __init__(self, config: Dict, output_manager):
        """
        Initialize temperature dynamics plotter
        
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
        if plot_type == "time_series":
            return self._plot_time_series(data)
        elif plot_type == "diurnal_cycle":
            return self._plot_diurnal_cycle(data)
        elif plot_type == "temperature_difference":
            return self._plot_temperature_difference(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    def _extract_temperature_at_point(self, dataset: xr.Dataset, x_idx: int = 100, 
                                     y_idx: int = 100, z_idx: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract temperature at a specific grid point
        
        Args:
            dataset: xarray dataset containing 'ta' variable
            x_idx: x index
            y_idx: y index  
            z_idx: zu_3d index
            
        Returns:
            Tuple of (time_array, temperature_array)
        """
        try:
            # Extract temperature at the specified point
            temp_data = dataset['ta'].isel(x=x_idx, y=y_idx, zu_3d=z_idx).squeeze()
            
            # Get time values (already converted to datetime by data loader)
            time_values = dataset['time'].values
            
            return time_values, temp_data.values
            
        except Exception as e:
            self.logger.error(f"Error extracting temperature: {str(e)}")
            raise
            
    def _plot_time_series(self, data: Dict) -> plt.Figure:
        """Create time series plot of temperature for all scenarios"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        
        # Use specific grid cell as in the working example
        x_idx, y_idx, z_idx = 100, 100, 14
        
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Colors and styles matching the working example
        spacing_colors = {
            10: '#1f77b4',  # Blue
            15: '#ff7f0e',  # Orange  
            20: '#2ca02c',  # Green
            25: '#d62728'   # Red
        }
        
        age_markers = {
            20: 'o',      # Circle
            40: 's',      # Square
            60: '^',      # Triangle up
            80: 'D'       # Diamond
        }
        
        age_linestyles = {
            20: '-',      # Solid
            40: '--',     # Dashed
            60: '-.',     # Dash-dot
            80: ':'       # Dotted
        }
        
        age_linewidths = {
            20: 2.5,
            40: 2.0,
            60: 2.0,
            80: 1.5
        }
        
        age_markersizes = {
            20: 6,
            40: 7,
            60: 6,
            80: 5
        }
        
        # Plot base case if available
        if data.get('base_case') and 'av_3d_n02' in data['base_case']:
            try:
                time_base, temp_base = self._extract_temperature_at_point(
                    data['base_case']['av_3d_n02'], x_idx, y_idx, z_idx
                )
                
                ax.plot(time_base, temp_base, 'k-o', label='PALM base simulation', 
                       linewidth=3, markersize=8, alpha=0.8)
            except Exception as e:
                self.logger.warning(f"Could not plot base case: {str(e)}")
        
        # Plot each scenario
        for spacing in spacings:
            for age in ages:
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            time_sim, temp_sim = self._extract_temperature_at_point(
                                sim_data['av_3d_n02'], x_idx, y_idx, z_idx
                            )
                            
                            ax.plot(time_sim, temp_sim,
                                   color=spacing_colors[spacing],
                                   marker=age_markers[age],
                                   linestyle=age_linestyles[age],
                                   label=f'{spacing}m {age}yrs',
                                   markersize=age_markersizes[age],
                                   linewidth=age_linewidths[age],
                                   alpha=0.8)
                                   
                        except Exception as e:
                            self.logger.warning(f"Error plotting {case_key}: {str(e)}")
        
        # Add weather data if configured
        settings = self._get_plot_settings('fig_2')  # or 'slide_7', both work
        if settings.get('include_weather_data', False):
            self._add_weather_data(ax)
        
        # Format plot
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Air Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Air Temperature Comparison - Forest Spacing and Age Variations', 
                    fontsize=14, fontweight='bold')
        
        # Create organized legend
        self._create_organized_legend(ax, spacings, ages)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45, fontsize=10)
        
        ax.yaxis.set_tick_params(labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
        
    def _plot_diurnal_cycle(self, data: Dict) -> plt.Figure:
        """Create diurnal cycle comparison plot"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        x_idx, y_idx, z_idx = 100, 100, 14
        
        # Create subplots for each spacing
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        age_colors = self._get_color_palette('age', len(ages))
        
        for idx, spacing in enumerate(spacings):
            ax = axes[idx]
            
            # Get base case for reference
            if data.get('base_case') and 'av_3d_n02' in data['base_case']:
                try:
                    time_base, temp_base = self._extract_temperature_at_point(
                        data['base_case']['av_3d_n02'], x_idx, y_idx, z_idx
                    )
                    
                    # Calculate hourly average
                    df_base = pd.DataFrame({'time': time_base, 'temp': temp_base})
                    df_base['hour'] = pd.to_datetime(df_base['time']).dt.hour
                    hourly_base = df_base.groupby('hour')['temp'].mean()
                    
                    ax.plot(hourly_base.index, hourly_base.values, 'k--', 
                           linewidth=2, label='Base', alpha=0.6)
                except Exception as e:
                    self.logger.warning(f"Could not plot base diurnal cycle: {str(e)}")
                    
            # Plot each age for this spacing
            for j, age in enumerate(ages):
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            time_sim, temp_sim = self._extract_temperature_at_point(
                                sim_data['av_3d_n02'], x_idx, y_idx, z_idx
                            )
                            
                            # Calculate hourly average
                            df_sim = pd.DataFrame({'time': time_sim, 'temp': temp_sim})
                            df_sim['hour'] = pd.to_datetime(df_sim['time']).dt.hour
                            hourly_sim = df_sim.groupby('hour')['temp'].mean()
                            
                            ax.plot(hourly_sim.index, hourly_sim.values,
                                   color=age_colors[j], linewidth=2.5,
                                   label=f'{age} years', marker='o', markersize=4)
                                   
                        except Exception as e:
                            self.logger.warning(f"Error plotting diurnal cycle for {case_key}: {str(e)}")
                            
            # Format subplot
            ax.set_title(f'{spacing}m Spacing', fontsize=14, weight='bold')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Temperature (°C)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim(0, 23)
            
            # Add shading for night hours
            ax.axvspan(0, 6, alpha=0.1, color='gray')
            ax.axvspan(20, 24, alpha=0.1, color='gray')
            
        # Main title
        fig.suptitle("Diurnal Temperature Cycles by Tree Configuration", 
                    fontsize=16, weight='bold')
                    
        plt.tight_layout()
        return fig
        
    def _plot_temperature_difference(self, data: Dict) -> plt.Figure:
        """Create temperature difference plot showing cooling effect"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        x_idx, y_idx, z_idx = 100, 100, 14
        
        # Check if base case is available
        if not data.get('base_case') or 'av_3d_n02' not in data['base_case']:
            self.logger.error("Base case required for temperature difference plot")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Base case data not available", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
            
        # Get base case temperature
        time_base, temp_base = self._extract_temperature_at_point(
            data['base_case']['av_3d_n02'], x_idx, y_idx, z_idx
        )
        
        # Create figure with subplots for different ages
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        spacing_colors = self._get_color_palette('spacing', len(spacings))
        
        for idx, age in enumerate(ages):
            ax = axes[idx]
            
            for i, spacing in enumerate(spacings):
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            time_sim, temp_sim = self._extract_temperature_at_point(
                                sim_data['av_3d_n02'], x_idx, y_idx, z_idx
                            )
                            
                            # Calculate cooling effect (ensure same time alignment)
                            min_length = min(len(temp_base), len(temp_sim))
                            cooling = temp_base[:min_length] - temp_sim[:min_length]
                            time_plot = time_base[:min_length]
                            
                            # Plot cooling effect
                            ax.plot(time_plot, cooling, 
                                   color=spacing_colors[i],
                                   linewidth=2.5, label=f'{spacing}m spacing',
                                   marker='o', markersize=3, markevery=6)
                                   
                        except Exception as e:
                            self.logger.warning(f"Error calculating cooling for {case_key}: {str(e)}")
                            
            # Format subplot
            ax.set_title(f'{age} Year Old Trees', fontsize=14, weight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cooling Effect (°C)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Format time axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            
            # Add statistics
            if 'cooling' in locals() and len(cooling) > 0:
                stats_text = f"Max cooling: {np.max(cooling):.2f}°C\nMean cooling: {np.mean(cooling):.2f}°C"
                if len(cooling) >= 48:  # At least 2 days of data
                    daytime_indices = []
                    for t in time_plot:
                        hour = pd.to_datetime(t).hour
                        if 6 <= hour <= 18:
                            daytime_indices.append(True)
                        else:
                            daytime_indices.append(False)
                    if any(daytime_indices):
                        daytime_cooling = cooling[daytime_indices]
                        stats_text += f"\nDaytime cooling: {np.mean(daytime_cooling):.2f}°C"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8), fontsize=9)
            
        # Main title
        fig.suptitle("Cooling Effect: Relative to Base Case (No Trees)", 
                    fontsize=16, weight='bold')
                    
        plt.tight_layout()
        return fig
        
    def _create_organized_legend(self, ax, spacings: List[int], ages: List[int]):
        """Create organized legend with clear groupings"""
        legend_elements = []
        
        # Add base case if it exists
        base_line = plt.Line2D([0], [0], color='black', marker='o', 
                              linestyle='-', linewidth=3, markersize=8, 
                              label='Base simulation')
        legend_elements.append(base_line)
        
        # Add spacing legend with distinct colors
        legend_elements.append(plt.Line2D([0], [0], color='white', alpha=0, label='Spacing:'))
        spacing_colors = {
            10: '#1f77b4',
            15: '#ff7f0e',
            20: '#2ca02c',
            25: '#d62728'
        }
        for spacing in spacings:
            legend_elements.append(plt.Line2D([0], [0], color=spacing_colors[spacing], 
                                            linestyle='-', linewidth=3, 
                                            label=f'  {spacing}m'))
        
        # Add age legend with distinct markers/styles
        legend_elements.append(plt.Line2D([0], [0], color='white', alpha=0, label='Age:'))
        age_markers = {20: 'o', 40: 's', 60: '^', 80: 'D'}
        age_linestyles = {20: '-', 40: '--', 60: '-.', 80: ':'}
        for age in ages:
            legend_elements.append(plt.Line2D([0], [0], color='gray', 
                                            marker=age_markers[age], 
                                            linestyle=age_linestyles[age], 
                                            linewidth=2, markersize=6,
                                            label=f'  {age}yrs'))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=10, frameon=True, fancybox=True, shadow=True)
        
    def _add_weather_data(self, ax):
        """Add weather data to plot if available"""
        # This is a placeholder for weather data integration
        # Implementation would depend on weather data availability
        pass
        
    def available_plots(self) -> List[str]:
        """Return list of available plot types"""
        return ["time_series", "diurnal_cycle", "temperature_difference"]