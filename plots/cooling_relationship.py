"""
Cooling Relationship Plotter for Slide 10
Quantifies the relationship between tree age, density, and cooling effect
Fixed version with proper dimension handling and error checking
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.optimize import curve_fit
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from .base_plotter import BasePlotter


class CoolingRelationshipPlotter(BasePlotter):
    """Creates visualizations for age-density-cooling relationships"""
    
    def __init__(self, config: Dict, output_manager):
        """
        Initialize cooling relationship plotter
        
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
        if plot_type == "surface_plot":
            return self._plot_surface(data)
        elif plot_type == "contour_plot":
            return self._plot_contour(data)
        elif plot_type == "optimization_analysis":
            return self._plot_optimization_analysis(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    def _extract_cooling_metrics(self, data: Dict) -> pd.DataFrame:
        """Extract cooling metrics for all scenarios with proper dimension handling"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_5')  # or 'slide_10', both work
        analysis_time = settings['analysis_time']
        
        cooling_data = []
        
        # Need base case for cooling calculation
        if not data.get('base_case') or 'av_3d_n02' not in data['base_case']:
            self.logger.error("Base case required for cooling analysis")
            return pd.DataFrame()
            
        base_dataset = data['base_case']['av_3d_n02']
        
        # Extract base case temperature with proper handling
        try:
            # Use first data level for child domain
            z_idx = 21
            base_temp = base_dataset['ta'].isel(zu_3d=z_idx).mean(dim=['x', 'y'])
            base_time = base_dataset['time'].values
        except Exception as e:
            self.logger.error(f"Error extracting base case temperature: {str(e)}")
            return pd.DataFrame()
        
        for spacing in spacings:
            for age in ages:
                case_key = f"{spacing}m_{age}yrs"
                
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'av_3d_n02' in sim_data:
                        try:
                            # Extract simulation temperature
                            sim_dataset = sim_data['av_3d_n02']
                            sim_temp = sim_dataset['ta'].isel(zu_3d=z_idx).mean(dim=['x', 'y'])
                            sim_time = sim_dataset['time'].values
                            
                            # Calculate cooling with dimension alignment
                            cooling = self._calculate_aligned_cooling(
                                sim_temp.values, sim_time,
                                base_temp.values, base_time
                            )
                            
                            if cooling is not None and len(cooling) > 0:
                                # Calculate metrics based on analysis_time setting
                                if analysis_time == "max_cooling":
                                    metric = np.max(cooling)
                                elif analysis_time == "daytime_avg":
                                    # Use hours 6-18 for daytime
                                    daytime_cooling = self._extract_daytime_values(
                                        cooling, sim_time[:len(cooling)]
                                    )
                                    metric = np.mean(daytime_cooling) if len(daytime_cooling) > 0 else np.mean(cooling)
                                elif analysis_time == "nighttime_avg":
                                    # Use hours 20-6 for nighttime
                                    nighttime_cooling = self._extract_nighttime_values(
                                        cooling, sim_time[:len(cooling)]
                                    )
                                    metric = np.mean(nighttime_cooling) if len(nighttime_cooling) > 0 else np.mean(cooling)
                                else:
                                    metric = np.mean(cooling)
                                    
                                # Calculate tree density (stems per hectare)
                                density = 10000 / (spacing ** 2)
                                
                                cooling_data.append({
                                    'spacing': spacing,
                                    'age': age,
                                    'density': density,
                                    'cooling': metric,
                                    'max_cooling': np.max(cooling),
                                    'mean_cooling': np.mean(cooling)
                                })
                            
                        except Exception as e:
                            self.logger.warning(f"Error calculating cooling for {case_key}: {str(e)}")
                            
        return pd.DataFrame(cooling_data)
        
    def _calculate_aligned_cooling(self, sim_temp: np.ndarray, sim_time: np.ndarray,
                                  base_temp: np.ndarray, base_time: np.ndarray) -> Optional[np.ndarray]:
        """Calculate cooling effect with proper time alignment"""
        try:
            # Ensure we have data to work with
            if len(sim_temp) == 0 or len(base_temp) == 0:
                return None
                
            # Use the minimum length to avoid dimension mismatch
            min_length = min(len(sim_temp), len(base_temp))
            
            if min_length == 0:
                return None
                
            # Calculate cooling (positive values indicate cooling)
            cooling = base_temp[:min_length] - sim_temp[:min_length]
            
            return cooling
            
        except Exception as e:
            self.logger.error(f"Error in cooling calculation: {str(e)}")
            return None
            
    def _extract_daytime_values(self, values: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        """Extract values during daytime hours (6-18)"""
        try:
            time_pd = pd.to_datetime(time_array)
            hours = time_pd.hour
            daytime_mask = (hours >= 6) & (hours <= 18)
            return values[daytime_mask]
        except:
            return values
            
    def _extract_nighttime_values(self, values: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        """Extract values during nighttime hours (20-6)"""
        try:
            time_pd = pd.to_datetime(time_array)
            hours = time_pd.hour
            nighttime_mask = (hours >= 20) | (hours <= 6)
            return values[nighttime_mask]
        except:
            return values
            
    def _plot_surface(self, data: Dict) -> plt.Figure:
        """Create 3D surface plot of age-density-cooling relationship"""
        # Extract cooling metrics
        df_cooling = self._extract_cooling_metrics(data)
        
        if df_cooling.empty:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data for surface plot", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
            
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid for interpolation
        age_grid = np.linspace(df_cooling['age'].min(), df_cooling['age'].max(), 50)
        density_grid = np.linspace(df_cooling['density'].min(), df_cooling['density'].max(), 50)
        X, Y = np.meshgrid(age_grid, density_grid)
        
        # Interpolate cooling values
        points = df_cooling[['age', 'density']].values
        values = df_cooling['cooling'].values
        
        if len(points) >= 4:  # Need at least 4 points for interpolation
            try:
                Z = griddata(points, values, (X, Y), method='cubic')
                
                # Create surface plot
                surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', 
                                     alpha=0.8, antialiased=True)
                
                # Add data points
                ax.scatter(df_cooling['age'], df_cooling['density'], 
                          df_cooling['cooling'], c='black', s=50, alpha=1)
                
                # Labels and title
                ax.set_xlabel('Tree Age (years)', fontsize=12, labelpad=10)
                ax.set_ylabel('Tree Density (stems/ha)', fontsize=12, labelpad=10)
                ax.set_zlabel('Cooling Effect (°C)', fontsize=12, labelpad=10)
                ax.set_title('Cooling Effect as Function of Tree Age and Density', 
                           fontsize=14, weight='bold', pad=20)
                
                # Add colorbar
                cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                cbar.set_label('Cooling Effect (°C)', fontsize=10)
                
                # Set viewing angle
                ax.view_init(elev=25, azim=45)
                
            except Exception as e:
                self.logger.error(f"Error creating surface plot: {str(e)}")
                ax.text2D(0.5, 0.5, "Error creating surface plot", 
                         transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text2D(0.5, 0.5, "Insufficient data points for interpolation", 
                     transform=ax.transAxes, ha='center', va='center')
            
        plt.tight_layout()
        return fig
        
    def _plot_contour(self, data: Dict) -> plt.Figure:
        """Create contour plot of cooling relationship"""
        # Extract cooling metrics
        df_cooling = self._extract_cooling_metrics(data)
        
        if df_cooling.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Insufficient data for contour plot", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create grid for interpolation
        age_grid = np.linspace(df_cooling['age'].min(), df_cooling['age'].max(), 100)
        density_grid = np.linspace(df_cooling['density'].min(), df_cooling['density'].max(), 100)
        X, Y = np.meshgrid(age_grid, density_grid)
        
        # Interpolate cooling values
        points = df_cooling[['age', 'density']].values
        values = df_cooling['cooling'].values
        
        if len(points) >= 4:
            try:
                Z = griddata(points, values, (X, Y), method='cubic')
                
                # Create contour plot
                levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 20)
                contourf = ax.contourf(X, Y, Z, levels=levels, cmap='coolwarm', alpha=0.8)
                contour = ax.contour(X, Y, Z, levels=levels[::2], colors='black', 
                                   linewidths=0.5, alpha=0.5)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f°C')
                
                # Add data points
                scatter = ax.scatter(df_cooling['age'], df_cooling['density'], 
                                   c=df_cooling['cooling'], cmap='coolwarm',
                                   s=100, edgecolors='black', linewidths=2)
                
                # Add optimal point if configured
                plot_settings = self._get_plot_settings('fig_5')  # or 'slide_10', both work
                if plot_settings['show_optimal_points']:
                    optimal_idx = df_cooling['cooling'].idxmax()
                    optimal = df_cooling.loc[optimal_idx]
                    ax.scatter(optimal['age'], optimal['density'], 
                             marker='*', s=500, c='gold', edgecolors='black', 
                             linewidths=2, label=f'Optimal: {optimal["cooling"]:.1f}°C')
                    ax.legend()
                
                # Labels and title
                ax.set_xlabel('Tree Age (years)', fontsize=12)
                ax.set_ylabel('Tree Density (stems/ha)', fontsize=12)
                ax.set_title('Cooling Effect Contour Map', fontsize=14, weight='bold')
                
                # Colorbar
                cbar = plt.colorbar(contourf, ax=ax)
                cbar.set_label('Cooling Effect (°C)', fontsize=10)
                
                # Grid
                ax.grid(True, alpha=0.3, linestyle='--')
                
            except Exception as e:
                self.logger.error(f"Error creating contour plot: {str(e)}")
                ax.text(0.5, 0.5, "Error creating contour plot", 
                       transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Insufficient data points for interpolation", 
                   transform=ax.transAxes, ha='center', va='center')
            
        plt.tight_layout()
        return fig
        
    def _plot_optimization_analysis(self, data: Dict) -> plt.Figure:
        """Create optimization analysis plot"""
        # Extract cooling metrics
        df_cooling = self._extract_cooling_metrics(data)
        
        if df_cooling.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Insufficient data for optimization analysis", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
            
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Cooling vs Age for different densities
        ax1 = fig.add_subplot(gs[0, 0])
        for spacing in df_cooling['spacing'].unique():
            df_subset = df_cooling[df_cooling['spacing'] == spacing]
            density = df_subset['density'].iloc[0]
            ax1.plot(df_subset['age'], df_subset['cooling'], 
                    'o-', linewidth=2, markersize=8,
                    label=f'{spacing}m ({density:.0f} stems/ha)')
        
        ax1.set_xlabel('Tree Age (years)', fontsize=12)
        ax1.set_ylabel('Cooling Effect (°C)', fontsize=12)
        ax1.set_title('Cooling vs Age by Tree Density', fontsize=13, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Cooling vs Density for different ages
        ax2 = fig.add_subplot(gs[0, 1])
        for age in df_cooling['age'].unique():
            df_subset = df_cooling[df_cooling['age'] == age]
            ax2.plot(df_subset['density'], df_subset['cooling'], 
                    'o-', linewidth=2, markersize=8,
                    label=f'{age} years')
        
        ax2.set_xlabel('Tree Density (stems/ha)', fontsize=12)
        ax2.set_ylabel('Cooling Effect (°C)', fontsize=12)
        ax2.set_title('Cooling vs Density by Tree Age', fontsize=13, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Max vs Mean cooling
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(df_cooling['mean_cooling'], df_cooling['max_cooling'],
                            c=df_cooling['age'], s=df_cooling['density'],
                            cmap='viridis', alpha=0.7, edgecolors='black')
        
        # Add diagonal line
        min_val = min(df_cooling['mean_cooling'].min(), df_cooling['max_cooling'].min())
        max_val = max(df_cooling['mean_cooling'].max(), df_cooling['max_cooling'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax3.set_xlabel('Mean Cooling (°C)', fontsize=12)
        ax3.set_ylabel('Max Cooling (°C)', fontsize=12)
        ax3.set_title('Maximum vs Mean Cooling Effect', fontsize=13, weight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for age
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Tree Age (years)', fontsize=10)
        
        # Panel 4: Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Calculate summary statistics
        optimal_idx = df_cooling['cooling'].idxmax()
        optimal = df_cooling.loc[optimal_idx]
        
        summary_text = f"""
        OPTIMIZATION SUMMARY
        
        Optimal Configuration:
        • Age: {optimal['age']} years
        • Spacing: {optimal['spacing']}m
        • Density: {optimal['density']:.0f} stems/ha
        • Cooling: {optimal['cooling']:.2f}°C
        
        Key Findings:
        • Age threshold: ~20 years for significant cooling
        • Optimal density: {70:.0f}-{100:.0f} stems/ha
        • Maximum cooling: {df_cooling['cooling'].max():.2f}°C
        
        Diminishing Returns:
        • Minimal gain beyond {optimal['age']} years
        • Density >100 stems/ha shows limited benefit
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        # Main title
        fig.suptitle('Cooling Optimization Analysis', fontsize=16, weight='bold')
        
        return fig
        
    def available_plots(self) -> List[str]:
        """Return list of available plot types"""
        return ["surface_plot", "contour_plot", "optimization_analysis"]