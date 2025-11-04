"""
Tree Density Plotter for Slide 6
Visualizes tree density scenarios with different ages and spacings
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

from .base_plotter import BasePlotter


class TreeDensityPlotter(BasePlotter):
    """Creates visualizations for tree density scenarios"""
    
    def __init__(self, config: Dict, output_manager):
        """
        Initialize tree density plotter
        
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
        if plot_type == "density_matrix":
            return self._plot_density_matrix(data)
        elif plot_type == "trees_per_hectare":
            return self._plot_trees_per_hectare(data)
        elif plot_type == "lad_visualization":
            return self._plot_lad_visualization(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    def _plot_density_matrix(self, data: Dict) -> plt.Figure:
        """Create density matrix visualization showing tree arrangements"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        
        # Create figure with subplots for each combination
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(len(ages), len(spacings), figure=fig, 
                     hspace=0.3, wspace=0.2)
        
        # Get tree metadata if available
        tree_metadata = data.get('tree_metadata')
        
        for i, age in enumerate(ages):
            for j, spacing in enumerate(spacings):
                ax = fig.add_subplot(gs[i, j])
                
                # Plot tree arrangement
                self._plot_tree_arrangement(ax, spacing, age, data)
                
                # Add labels
                if i == 0:
                    ax.set_title(f"{spacing}m spacing", fontsize=12, weight='bold')
                if j == 0:
                    ax.set_ylabel(f"{age} years", fontsize=12, weight='bold')
                    
                # Calculate and display trees per hectare
                trees_per_ha = int(10000 / (spacing * spacing))
                ax.text(0.95, 0.05, f"{trees_per_ha} trees/ha", 
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
        # Add main title
        fig.suptitle("Tree Density Scenarios: Age and Spacing Matrix", 
                    fontsize=16, weight='bold')
        
        # Add tree size legend
        self._add_tree_size_legend(fig)
        
        return fig
        
    def _plot_tree_arrangement(self, ax: plt.Axes, spacing: int, age: int, data: Dict):
        """Plot individual tree arrangement for given spacing and age"""
        # Get tree locations
        tree_locations = data.get('tree_locations', {})
        location_key = f"{spacing}m_child"
        
        if location_key in tree_locations:
            locations = tree_locations[location_key]
            
            # Get tree metadata for this age
            tree_metadata = data.get('tree_metadata')
            if tree_metadata is not None:
                age_data = tree_metadata[tree_metadata['tree_age'] == age]
                if len(age_data) > 0:
                    # Use average crown radius for this age
                    crown_radius = age_data['crown_radius'].mean()
                else:
                    # Estimate crown radius based on age
                    crown_radius = self._estimate_crown_radius(age)
            else:
                crown_radius = self._estimate_crown_radius(age)
                
            # Scale for visualization
            plot_scale = 100 / spacing  # Adjust scale based on spacing
            
            # Plot trees as circles
            for loc in locations[:25]:  # Limit to 25 trees for clarity
                circle = plt.Circle((loc[0] * plot_scale, loc[1] * plot_scale), 
                                  crown_radius * plot_scale,
                                  facecolor='green', 
                                  edgecolor='darkgreen',
                                  alpha=0.6, 
                                  linewidth=1.5)
                ax.add_patch(circle)
                
            # Set plot limits and aspect
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 200)
            ax.set_aspect('equal')
            ax.axis('off')
            
        else:
            ax.text(0.5, 0.5, "No location data", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            
    def _estimate_crown_radius(self, age: int) -> float:
        """Estimate crown radius based on tree age"""
        # Based on Larsen & Kristoffersen (2002) data for Tilia
        # Simplified relationship
        if age <= 20:
            return 2.0 + (age / 20) * 2.0
        elif age <= 40:
            return 4.0 + ((age - 20) / 20) * 2.0
        elif age <= 60:
            return 6.0 + ((age - 40) / 20) * 1.5
        else:
            return 7.5 + ((age - 60) / 20) * 1.0
            
    def _plot_trees_per_hectare(self, data: Dict) -> plt.Figure:
        """Create bar chart showing trees per hectare for each scenario"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate trees per hectare
        trees_per_ha = [10000 / (s * s) for s in spacings]
        
        # Create grouped bar chart
        x = np.arange(len(spacings))
        width = 0.8 / len(ages)
        
        colors = self._get_color_palette('age', len(ages))
        
        for i, age in enumerate(ages):
            offset = (i - len(ages)/2 + 0.5) * width
            bars = ax.bar(x + offset, trees_per_ha, width, 
                          label=f'{age} years', color=colors[i])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
                       
        # Customize plot
        ax.set_xlabel('Tree Spacing (m)', fontsize=14, weight='bold')
        ax.set_ylabel('Trees per Hectare', fontsize=14, weight='bold')
        ax.set_title('Tree Density by Spacing Configuration', fontsize=16, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}m' for s in spacings])
        ax.legend(title='Tree Age', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add reference line for optimal density
        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                  label='Optimal density (100 trees/ha)')
        
        plt.tight_layout()
        return fig
        
    def _plot_lad_visualization(self, data: Dict) -> plt.Figure:
        """Create LAD visualization for different scenarios"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(ages), len(spacings), 
                                figsize=(16, 12), squeeze=False)
        
        # Get LAD colormap
        cmap = self._get_colormap('density')
        
        for i, age in enumerate(ages):
            for j, spacing in enumerate(spacings):
                ax = axes[i, j]
                
                # Load static driver data for this case
                case_key = f"{spacing}m_{age}yrs"
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    
                    if 'static_n02' in sim_data:
                        static_data = sim_data['static_n02']
                        
                        # Extract LAD data at canopy height
                        if 'lad' in static_data:
                            lad = static_data['lad']
                            
                            # Sum LAD over height to get column-integrated LAD
                            lad_integrated = lad.sum(dim='zlad')
                            
                            # Plot LAD distribution
                            im = ax.imshow(lad_integrated, cmap=cmap, 
                                         origin='lower', aspect='equal')
                            
                            # Add tree locations if available
                            location_key = f"{spacing}m_child"
                            if location_key in data['tree_locations']:
                                locations = data['tree_locations'][location_key]
                                ax.scatter(locations[:, 0], locations[:, 1], 
                                         c='red', s=10, marker='+', alpha=0.5)
                        else:
                            ax.text(0.5, 0.5, "No LAD data", 
                                   transform=ax.transAxes, ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, "No static data", 
                               transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "No simulation data", 
                           transform=ax.transAxes, ha='center', va='center')
                    
                # Add labels
                if i == 0:
                    ax.set_title(f"{spacing}m", fontsize=12)
                if j == 0:
                    ax.set_ylabel(f"{age} yrs", fontsize=12)
                    
                ax.set_xticks([])
                ax.set_yticks([])
                
        # Add main title
        fig.suptitle("Leaf Area Density Distribution", fontsize=16, weight='bold')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Column-integrated LAD (m²/m²)', fontsize=12)
        
        plt.tight_layout()
        return fig
        
    def _add_tree_size_legend(self, fig: plt.Figure):
        """Add legend showing tree sizes at different ages"""
        # Create legend axes
        legend_ax = fig.add_axes([0.85, 0.02, 0.12, 0.15])
        legend_ax.axis('off')
        
        # Plot example trees
        ages_example = [20, 40, 60, 80]
        positions = [(0.2, 0.8), (0.4, 0.8), (0.6, 0.8), (0.8, 0.8)]
        
        for age, pos in zip(ages_example, positions):
            radius = self._estimate_crown_radius(age) / 10  # Scale down
            circle = plt.Circle(pos, radius, facecolor='green', 
                              edgecolor='darkgreen', alpha=0.6)
            legend_ax.add_patch(circle)
            legend_ax.text(pos[0], pos[1] - radius - 0.1, f"{age}y", 
                          ha='center', va='top', fontsize=10)
            
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.text(0.5, 0.1, "Tree Age", ha='center', va='bottom', 
                      fontsize=12, weight='bold')
        
    def available_plots(self) -> List[str]:
        """Return list of available plot types"""
        return ["density_matrix", "trees_per_hectare", "lad_visualization"]