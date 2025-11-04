#!/usr/bin/env python3
"""
Quick test script for temperature maps plotting functionality
Run this to test the new time-averaged temperature maps plot
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add PALMPlot to path
sys.path.append('/home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf')

from core.data_loader import PALMDataLoader
from plots.spatial_cooling import SpatialCoolingPlotter
from utils.output_manager import OutputManager
import yaml

def test_temperature_maps():
    """Test the temperature maps plotting functionality"""
    
    print("Loading configuration...")
    # Load configuration
    config_path = Path('/home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/palmplot_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Initializing data loader...")
    # Initialize data loader
    data_loader = PALMDataLoader(config)
    
    # Load only necessary data for testing
    # You can modify this to load only specific cases for faster testing
    print("Loading data (this may take a moment)...")
    
    # Load just base case and one simulation for quick testing
    data = {
        'simulations': {},
        'tree_locations': {},
        'tree_metadata': None,
        'base_case': None
    }
    
    # Load base case
    print("Loading base case...")
    data['base_case'] = data_loader._load_base_case()
    
    # Load just one or two simulation cases for testing
    test_cases = [
        (10, 20),  # 10m spacing, 20 years
        (10, 80),  # 10m spacing, 80 years
        (25, 20),  # 25m spacing, 20 years
        (25, 80),  # 25m spacing, 80 years
    ]
    
    for spacing, age in test_cases:
        case_key = f"{spacing}m_{age}yrs"
        print(f"Loading simulation: {case_key}")
        case_data = data_loader._load_simulation_case(spacing, age)
        if case_data:
            data['simulations'][case_key] = case_data
    
    print("\nData loading complete. Creating plot...")
    
    # Initialize output manager with just the base directory
    output_base_dir = config['output']['base_directory']
    output_manager = OutputManager(output_base_dir)
    
    # Initialize plotter
    plotter = SpatialCoolingPlotter(config, output_manager)
    
    # Generate temperature maps plot
    try:
        fig = plotter._plot_temperature_maps(data)
        
        # Save the plot
        output_path = Path('/home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/results/test_temperature_maps.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        
        # Also save as PDF for better quality
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"PDF version saved to: {pdf_path}")
        
        # Display the plot (optional - comment out if running on server)
        # plt.show()
        
        plt.close(fig)
        
    except Exception as e:
        print(f"\nError creating plot: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_temperature_maps()