#!/usr/bin/env python3
"""
PALMPlot - PALM Simulation Data Visualization Package
Main module for orchestrating plot generation from PALM simulation data

Author: Joshua Brook-Lawson
Institution: University of Bath
"""

# Set matplotlib backend BEFORE any other imports
# Use 'Agg' (non-interactive) to prevent tkinter crashes in headless/non-GUI environments
import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Handle imports for both package and direct execution
# Handle imports for both package and direct execution
try:
    # Try relative imports first (when run as package)
    from .core.data_loader import PALMDataLoader
    from .core.config_handler import ConfigHandler
    from .plots import (
        TreeDensityPlotter,
        TemperatureDynamicsPlotter,
        SpatialCoolingPlotter,
        VerticalProfilePlotter,
        CoolingRelationshipPlotter,
        TerrainTransectPlotter
    )
    from .utils.logger import setup_logger
    from .utils.output_manager import OutputManager
    from .utils.figure_mapper import FigureMapper
except ImportError:
    # Fall back to absolute imports with package name
    from palmplot_thf.core.data_loader import PALMDataLoader
    from palmplot_thf.core.config_handler import ConfigHandler
    from palmplot_thf.plots import (
        TreeDensityPlotter,
        TemperatureDynamicsPlotter,
        SpatialCoolingPlotter,
        VerticalProfilePlotter,
        CoolingRelationshipPlotter,
        TerrainTransectPlotter
    )
    from palmplot_thf.utils.logger import setup_logger
    from palmplot_thf.utils.output_manager import OutputManager
    from palmplot_thf.utils.figure_mapper import FigureMapper


class PALMPlot:
    """Main class for PALMPlot visualization package"""
    
    def __init__(self, config_path: str):
        """
        Initialize PALMPlot with configuration file
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_handler = ConfigHandler(config_path)
        self.config = self.config_handler.load_config()
        
        # Setup logging
        self.logger = setup_logger(
            self.config['logging']['level'],
            self.config['logging']['log_file']
        )
        self.logger.info("Initializing PALMPlot")
        
        # Initialize components
        self.data_loader = PALMDataLoader(self.config)

        # Determine if we should use figure-based naming
        # Check if config uses 'figures' key or 'slides' key
        use_figures = 'figures' in self.config.get('plots', {})
        if not use_figures:
            # Check for legacy slide naming
            use_figures = not ('slides' in self.config.get('plots', {}))

        self.output_manager = OutputManager(
            self.config['output']['base_directory'],
            use_figures=use_figures
        )
        self.use_figures = use_figures

        # Initialize figure mapper
        self.figure_mapper = FigureMapper()

        # Initialize plotters
        self._init_plotters()
        
    def _init_plotters(self):
        """
        Initialize all plotter instances.

        Creates plotters using either slide-based or figure-based IDs depending
        on the configuration format.
        """
        # Determine which naming scheme to use based on config
        plots_config = self.config.get('plots', {})

        if 'figures' in plots_config:
            # New figure-based naming
            self.plotters = {
                'fig_1': TreeDensityPlotter(self.config, self.output_manager),
                'fig_2': TemperatureDynamicsPlotter(self.config, self.output_manager),
                'fig_3': SpatialCoolingPlotter(self.config, self.output_manager),
                'fig_4': VerticalProfilePlotter(self.config, self.output_manager),
                'fig_5': CoolingRelationshipPlotter(self.config, self.output_manager),
                'fig_6': TerrainTransectPlotter(self.config, self.output_manager)
            }
        else:
            # Legacy slide-based naming (backward compatibility)
            self.plotters = {
                'slide_6': TreeDensityPlotter(self.config, self.output_manager),
                'slide_7': TemperatureDynamicsPlotter(self.config, self.output_manager),
                'slide_8': SpatialCoolingPlotter(self.config, self.output_manager),
                'slide_9': VerticalProfilePlotter(self.config, self.output_manager),
                'slide_10': CoolingRelationshipPlotter(self.config, self.output_manager)
            }
        
    def run(self):
        """Execute plotting workflow based on configuration"""
        self.logger.info("Starting PALMPlot execution")

        # Load data
        self.logger.info("Loading PALM simulation data")
        data = self.data_loader.load_all_data()

        # Get the appropriate configuration section (figures or slides)
        plots_config = self.config['plots']
        if 'figures' in plots_config:
            items = plots_config['figures']
            item_type = "figure"
        else:
            items = plots_config['slides']
            item_type = "slide"

        # Process each item if enabled
        for item_id, item_config in items.items():
            if item_config['enabled']:
                self.logger.info(f"Processing {item_id}")
                self._process_item(item_id, item_config, data)

        self.logger.info("PALMPlot execution completed successfully")
        
    def _process_item(self, item_id: str, item_config: Dict, data: Dict):
        """
        Process individual figure/slide plots with proper subfigure lettering.

        Args:
            item_id: Figure or slide identifier (e.g., 'fig_1' or 'slide_6')
            item_config: Configuration for this item
            data: Loaded simulation data
        """
        try:
            plotter = self.plotters[item_id]

            # Reset subfigure counter for this figure
            if self.use_figures:
                figure_id = self.output_manager.get_figure_id_from_slide(item_id)
                self.output_manager.reset_subfigure_counter(figure_id)

            # Create output directory for this item
            item_dir = self.output_manager.create_slide_directory(item_id)

            # Generate plots - support both new dynamic and legacy formats
            # Check if using new dynamic format (no plot_types in config)
            if 'plot_types' in item_config:
                # Legacy format: iterate through plot_types dict
                plot_types_to_generate = [pt for pt, enabled in item_config['plot_types'].items() if enabled]
            else:
                # New dynamic format: use plotter's available_plots() method
                plot_types_to_generate = plotter.available_plots()
                self.logger.info(f"  Using dynamic plot generation: {len(plot_types_to_generate)} plots")

            for plot_type in plot_types_to_generate:
                self.logger.info(f"  Generating {plot_type} for {item_id}")

                # Generate plot
                fig = plotter.generate_plot(plot_type, data)

                # Get subfigure letter once per plot_type (outside format loop)
                # This ensures all formats of the same plot get the same letter
                if self.use_figures:
                    figure_id = self.output_manager.get_figure_id_from_slide(item_id)
                    subfigure_letter = self.output_manager.figure_mapper.get_next_subfigure_letter(figure_id)
                else:
                    subfigure_letter = None

                # Save in requested formats (using same letter for all formats)
                for fmt in self.config['output']['formats']:
                    output_path = self.output_manager.save_figure(
                        fig, item_id, plot_type, fmt, subfigure_letter=subfigure_letter
                    )
                    self.logger.info(f"    Saved: {output_path}")

                # Close figure to free memory
                plotter.close_figure(fig)

        except Exception as e:
            self.logger.error(f"Error processing {item_id}: {str(e)}")
            if self.config['general']['stop_on_error']:
                raise

    def _process_slide(self, slide_id: str, slide_config: Dict, data: Dict):
        """
        Legacy method for backward compatibility.
        Calls _process_item internally.

        Args:
            slide_id: Slide identifier (e.g., 'slide_6')
            slide_config: Configuration for this slide
            data: Loaded simulation data
        """
        self._process_item(slide_id, slide_config, data)
                

def main():
    """Main entry point for PALMPlot"""
    parser = argparse.ArgumentParser(
        description='PALMPlot - PALM Simulation Data Visualization'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without plotting'
    )
    parser.add_argument(
        '--list-plots',
        action='store_true',
        help='List available plots and exit'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    try:
        # Initialize PALMPlot
        palmplot = PALMPlot(args.config)
        
        if args.validate_only:
            print("Configuration validated successfully")
            sys.exit(0)
            
        if args.list_plots:
            print("\nAvailable plots:")
            for slide_id, plotter in palmplot.plotters.items():
                print(f"\n{slide_id}:")
                for plot_type in plotter.available_plots():
                    print(f"  - {plot_type}")
            sys.exit(0)
            
        # Run plotting
        palmplot.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
        

if __name__ == "__main__":
    main()