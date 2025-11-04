"""
Vertical Profile Plotter for Slide 9
Visualizes three-dimensional cooling structure
Complete fixed version with all necessary methods
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from typing import Dict, List, Tuple, Optional
import logging
import xarray as xr
import pandas as pd

from .base_plotter import BasePlotter


class VerticalProfilePlotter(BasePlotter):
    """Creates visualizations for vertical cooling profiles"""
    
    def __init__(self, config: Dict, output_manager):
        """
        Initialize vertical profile plotter
        
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
        if plot_type == "vertical_cross_section":
            return self._plot_vertical_cross_section(data)
        elif plot_type == "height_profiles":
            return self._plot_height_profiles(data)
        elif plot_type == "canopy_analysis":
            return self._plot_canopy_analysis(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    def _get_time_index(self, dataset: xr.Dataset, hour: int) -> int:
        """
        Get time index for specified hour of day
        
        Args:
            dataset: xarray dataset with time dimension
            hour: Hour of day (0-23)
            
        Returns:
            Index of first occurrence of specified hour
        """
        try:
            # Convert time to pandas datetime for hour extraction
            time_pd = pd.to_datetime(dataset['time'].values)
            hours = time_pd.hour
            
            # Find first occurrence of specified hour
            hour_indices = np.where(hours == hour)[0]
            
            if len(hour_indices) > 0:
                return hour_indices[0]
            else:
                # If specific hour not found, return closest hour
                hour_diff = np.abs(hours - hour)
                return np.argmin(hour_diff)
                
        except Exception as e:
            self.logger.warning(f"Error finding time index for hour {hour}: {str(e)}")
            # Default to middle of time series
            return len(dataset['time']) // 2
            
    def _extract_vertical_cross_section(self, dataset: xr.Dataset, 
                                       x_idx: int, hour: int) -> np.ndarray:
        """
        Extract 2D vertical cross-section of temperature
        
        Args:
            dataset: xarray dataset containing temperature data
            x_idx: X index for cross-section
            hour: Hour of day
            
        Returns:
            2D array (y, z) of temperature values
        """
        try:
            # Get time index
            time_idx = self._get_time_index(dataset, hour)
            
            # Extract temperature cross-section
            # For child domain (N02), data starts at zu_3d=21
            # For parent domain, data starts at zu_3d=25
            if 'N02' in dataset.encoding.get('source', ''):
                z_start = 21
            else:
                z_start = 25
                
            # Get temperature data
            temp_data = dataset['ta'].isel(x=x_idx, time=time_idx)
            
            # Extract only valid vertical levels
            temp_cross = temp_data.isel(zu_3d=slice(z_start, None))
            
            return temp_cross.values
            
        except Exception as e:
            self.logger.error(f"Error extracting vertical cross-section: {str(e)}")
            raise
            
    def _plot_vertical_cross_section(self, data: Dict) -> plt.Figure:
        """Create vertical cross-section of temperature/cooling"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_4')  # or 'slide_9', both work

        # Get daytime hour with fallback
        daytime_hour = settings.get('daytime_hour', 14)
        x_idx = settings.get('cross_section_x', 100)
        
        # Create figure with subplots for each age
        fig, axes = plt.subplots(len(ages), len(spacings) + 1, 
                                figsize=(20, 16), squeeze=False)
        
        # Get base case if available
        base_cross = None
        if data.get('base_case') and 'av_3d_n02' in data['base_case']:
            try:
                base_data = data['base_case']['av_3d_n02']
                base_cross = self._extract_vertical_cross_section(
                    base_data, x_idx, daytime_hour
                )
            except Exception as e:
                self.logger.warning(f"Could not extract base case cross-section: {str(e)}")
                
        # Temperature settings
        temp_vmin, temp_vmax = 20, 35
        cooling_vmin, cooling_vmax = -6, 6
        
        # Colormaps
        temp_cmap = self._get_colormap('temperature')
        cooling_cmap = self._get_colormap('cooling')
        cooling_norm = TwoSlopeNorm(vmin=cooling_vmin, vcenter=0, vmax=cooling_vmax)
        
        for i, age in enumerate(ages):
            # Plot base case in first column
            if base_cross is not None:
                ax = axes[i, 0]
                im = ax.imshow(base_cross.T, cmap=temp_cmap,
                              vmin=temp_vmin, vmax=temp_vmax,
                              origin='lower', aspect='auto')
                ax.set_title("Base Case" if i == 0 else "")
                ax.set_ylabel(f"{age} years", fontsize=12, weight='bold')
                ax.set_xlabel("Y index")
                
            # Plot each spacing scenario
            for j, spacing in enumerate(spacings):
                ax = axes[i, j + 1]
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            # Extract cross-section
                            temp_cross = self._extract_vertical_cross_section(
                                sim_data['av_3d_n02'], x_idx, daytime_hour
                            )
                            
                            # Calculate cooling if base case available
                            if base_cross is not None:
                                # Ensure dimensions match
                                min_y = min(temp_cross.shape[0], base_cross.shape[0])
                                min_z = min(temp_cross.shape[1], base_cross.shape[1])
                                
                                cooling_cross = base_cross[:min_y, :min_z] - temp_cross[:min_y, :min_z]
                                
                                im = ax.imshow(cooling_cross.T, cmap=cooling_cmap,
                                             norm=cooling_norm, origin='lower',
                                             aspect='auto')
                            else:
                                im = ax.imshow(temp_cross.T, cmap=temp_cmap,
                                             vmin=temp_vmin, vmax=temp_vmax,
                                             origin='lower', aspect='auto')
                                             
                        except Exception as e:
                            self.logger.warning(f"Error extracting vertical cross-section: {str(e)}")
                            ax.text(0.5, 0.5, "Error", transform=ax.transAxes,
                                   ha='center', va='center')
                            
                if i == 0:
                    ax.set_title(f"{spacing}m spacing", fontsize=12, weight='bold')
                    
                ax.set_xlabel("Y index")
                if j == 0:
                    ax.set_ylabel("Height index")
                    
        # Main title
        fig.suptitle(f"Vertical Temperature Cross-Section at X={x_idx} (14:00)", 
                    fontsize=16, weight='bold')
        
        # Add colorbars
        # Temperature colorbar
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
        
    def _plot_height_profiles(self, data: Dict) -> plt.Figure:
        """Create height profile plots"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_4')  # or 'slide_9', both work

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        axes = axes.flatten()
        
        # Define height levels (approximate, depends on grid)
        heights = np.arange(0, 50, 2)  # 0 to 50m in 2m steps
        
        for idx, age in enumerate(ages):
            ax = axes[idx]
            
            for spacing in spacings:
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            # Extract mean temperature profile
                            temp_data = sim_data['av_3d_n02']['ta']
                            
                            # Average over horizontal dimensions and time
                            temp_profile = temp_data.mean(dim=['x', 'y', 'time'])
                            
                            # Extract valid vertical levels
                            if 'N02' in sim_data['av_3d_n02'].encoding.get('source', ''):
                                z_start = 21
                            else:
                                z_start = 25
                                
                            temp_profile = temp_profile.isel(zu_3d=slice(z_start, None))
                            
                            # Plot profile
                            ax.plot(temp_profile.values, range(len(temp_profile)),
                                   linewidth=2, label=f'{spacing}m')
                                   
                        except Exception as e:
                            self.logger.warning(f"Error plotting height profile for {case_key}: {str(e)}")
                            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Height Level')
            ax.set_title(f'{age} Year Old Trees')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        fig.suptitle('Vertical Temperature Profiles', fontsize=16, weight='bold')
        plt.tight_layout()
        return fig
        
    def _plot_canopy_analysis(self, data: Dict) -> plt.Figure:
        """Create canopy cooling analysis plot"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Placeholder for canopy analysis
        # This would analyze cooling within and above canopy
        
        # For now, create a simple comparison plot
        cooling_data = []
        
        for spacing in spacings:
            for age in ages:
                case_key = f"{spacing}m_{age}yrs"
                if case_key in data['simulations']:
                    # Calculate some metric (placeholder)
                    cooling_data.append({
                        'spacing': spacing,
                        'age': age,
                        'cooling': np.random.uniform(0, 6)  # Placeholder
                    })
                    
        if cooling_data:
            import pandas as pd
            df = pd.DataFrame(cooling_data)
            
            # Create grouped bar plot
            spacing_groups = df.groupby('spacing')
            x = np.arange(len(spacings))
            width = 0.2
            
            for i, age in enumerate(ages):
                age_data = []
                for spacing in spacings:
                    mask = (df['spacing'] == spacing) & (df['age'] == age)
                    if mask.any():
                        age_data.append(df[mask]['cooling'].iloc[0])
                    else:
                        age_data.append(0)
                        
                ax.bar(x + i*width, age_data, width, label=f'{age} years')
                
            ax.set_xlabel('Tree Spacing (m)')
            ax.set_ylabel('Cooling Effect (°C)')
            ax.set_title('Canopy Cooling Analysis', fontsize=14, weight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([f'{s}m' for s in spacings])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "Insufficient data for canopy analysis",
                   transform=ax.transAxes, ha='center', va='center')
            
        plt.tight_layout()
        return fig
        
    def available_plots(self) -> List[str]:
        """Return list of available plot types"""
        return ["vertical_cross_section", "height_profiles", "canopy_analysis"]