"""
PALMPlot Data Loader Module
Handles loading and preprocessing of PALM simulation data from NetCDF files
Fixed version with proper time dimension handling
"""

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta


class PALMDataLoader:
    """Handles loading and preprocessing of PALM simulation data"""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        
        # Set up paths
        self.sim_base_path = Path(config['data']['simulation_base_path'])
        self.tree_locations_path = Path(config['data']['tree_locations_path'])
        self.lad_wad_path = Path(config['data']['lad_wad_data_path'])
        
        # Extract parameters
        self.spacings = config['data']['spacings']
        self.ages = config['data']['ages']
        
        # Reference date for PALM simulations
        self.reference_date = np.datetime64('2018-08-07 00:00:00')
        
    def load_all_data(self) -> Dict:
        """
        Load all required data for plotting
        
        Returns:
            Dictionary containing all loaded data
        """
        self.logger.info("Loading all PALM simulation data")
        
        data = {
            'simulations': {},
            'tree_locations': {},
            'tree_metadata': None,
            'base_case': None
        }
        
        # Load base case if available
        data['base_case'] = self._load_base_case()
        
        # Load simulation data
        if self.config['general'].get('parallel_processing', False):
            data['simulations'] = self._load_simulations_parallel()
        else:
            data['simulations'] = self._load_simulations_sequential()
            
        # Load tree location data
        data['tree_locations'] = self._load_tree_locations()
        
        # Load tree metadata
        data['tree_metadata'] = self._load_tree_metadata()
        
        return data
        
    def _load_base_case(self) -> Optional[Dict]:
        """Load base case simulation data"""
        base_path = self.sim_base_path / "thf_base_2018080700"
        
        if not base_path.exists():
            self.logger.warning("Base case directory not found")
            return None
            
        try:
            base_data = {}
            
            # Load parent domain
            av_3d_path = base_path / "OUTPUT/merged_files/thf_base_2018080700_av_3d_merged.nc"
            if av_3d_path.exists():
                base_data['av_3d'] = self._load_and_process_netcdf(av_3d_path)
                
            # Load child domain
            av_3d_n02_path = base_path / "OUTPUT/merged_files/thf_base_2018080700_av_3d_N02_merged.nc"
            if av_3d_n02_path.exists():
                base_data['av_3d_n02'] = self._load_and_process_netcdf(av_3d_n02_path)
                
            # Load xy data
            av_xy_path = base_path / "OUTPUT/merged_files/thf_base_2018080700_av_xy_merged.nc"
            if av_xy_path.exists():
                base_data['av_xy'] = self._load_and_process_netcdf(av_xy_path)
                
            av_xy_n02_path = base_path / "OUTPUT/merged_files/thf_base_2018080700_av_xy_N02_merged.nc"
            if av_xy_n02_path.exists():
                base_data['av_xy_n02'] = self._load_and_process_netcdf(av_xy_n02_path)
                
            return base_data if base_data else None
            
        except Exception as e:
            self.logger.error(f"Error loading base case: {str(e)}")
            return None
            
    def _load_simulations_sequential(self) -> Dict:
        """Load simulation data sequentially"""
        simulations = {}
        
        for spacing in self.spacings:
            for age in self.ages:
                case_key = f"{spacing}m_{age}yrs"
                self.logger.info(f"Loading simulation: {case_key}")
                
                case_data = self._load_simulation_case(spacing, age)
                if case_data:
                    simulations[case_key] = case_data
                    
        return simulations
        
    def _load_simulations_parallel(self) -> Dict:
        """Load simulation data in parallel"""
        simulations = {}
        
        n_workers = self.config['general'].get('n_workers', 4)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_case = {}
            
            for spacing in self.spacings:
                for age in self.ages:
                    case_key = f"{spacing}m_{age}yrs"
                    future = executor.submit(self._load_simulation_case, spacing, age)
                    future_to_case[future] = case_key
                    
            # Collect results
            for future in as_completed(future_to_case):
                case_key = future_to_case[future]
                try:
                    case_data = future.result()
                    if case_data:
                        simulations[case_key] = case_data
                        self.logger.info(f"Successfully loaded: {case_key}")
                except Exception as e:
                    self.logger.error(f"Error loading {case_key}: {str(e)}")
                    
        return simulations
        
    def _load_simulation_case(self, spacing: int, age: int) -> Optional[Dict]:
        """
        Load data for a single simulation case
        
        Args:
            spacing: Tree spacing in meters
            age: Tree age in years
            
        Returns:
            Dictionary containing simulation data or None if loading fails
        """
        case_name = f"thf_forest_lad_spacing_{spacing}m_age_{age}yrs"
        case_path = self.sim_base_path / case_name
        
        if not case_path.exists():
            self.logger.warning(f"Case directory not found: {case_name}")
            return None
            
        case_data = {
            'spacing': spacing,
            'age': age,
            'name': case_name
        }
        
        try:
            # Load 3D averaged data (parent domain)
            av_3d_path = case_path / f"OUTPUT/merged_files/{case_name}_av_3d_merged.nc"
            if av_3d_path.exists():
                case_data['av_3d'] = self._load_and_process_netcdf(av_3d_path)
            
            # Load 3D averaged data (child domain)
            av_3d_n02_path = case_path / f"OUTPUT/merged_files/{case_name}_av_3d_N02_merged.nc"
            if av_3d_n02_path.exists():
                case_data['av_3d_n02'] = self._load_and_process_netcdf(av_3d_n02_path)
                
            # Load 2D averaged data (parent domain)
            av_xy_path = case_path / f"OUTPUT/merged_files/{case_name}_av_xy_merged.nc"
            if av_xy_path.exists():
                case_data['av_xy'] = self._load_and_process_netcdf(av_xy_path)
                
            # Load 2D averaged data (child domain)
            av_xy_n02_path = case_path / f"OUTPUT/merged_files/{case_name}_av_xy_N02_merged.nc"
            if av_xy_n02_path.exists():
                case_data['av_xy_n02'] = self._load_and_process_netcdf(av_xy_n02_path)
                
            # Load static driver (parent domain)
            static_path = case_path / f"INPUT/{case_name}_static"
            if static_path.exists():
                case_data['static'] = self._load_and_process_netcdf(static_path, is_static=True)
                
            # Load static driver (child domain)
            static_n02_path = case_path / f"INPUT/{case_name}_static_N02"
            if static_n02_path.exists():
                case_data['static_n02'] = self._load_and_process_netcdf(static_n02_path, is_static=True)
                
            return case_data
            
        except Exception as e:
            self.logger.error(f"Error loading case {case_name}: {str(e)}")
            return None
            
    def _load_and_process_netcdf(self, file_path: Path, is_static: bool = False) -> xr.Dataset:
        """
        Load NetCDF file and process time dimension if present
        
        Args:
            file_path: Path to NetCDF file
            is_static: Whether this is a static file (no time dimension)
            
        Returns:
            xarray Dataset with processed time
        """
        # Determine appropriate chunks based on file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        chunk_size = self.config['performance'].get('chunk_size', 100)
        
        if file_size_mb > chunk_size:
            # Use chunking for large files
            chunks = {
                'time': 'auto',  # Let xarray determine optimal chunk size
                'x': -1,  # Keep x dimension unchunked
                'y': -1   # Keep y dimension unchunked
            }
            ds = xr.open_dataset(file_path, chunks=chunks, decode_timedelta=False)
        else:
            # Load small files directly
            ds = xr.open_dataset(file_path, decode_timedelta=False)
            
        # Process time dimension if present and not static
        if 'time' in ds.dims and not is_static:
            # Convert PALM time (seconds since reference) to datetime
            time_seconds = ds['time'].values
            
            # Convert to datetime objects
            time_datetime = pd.to_datetime(self.reference_date) + pd.to_timedelta(time_seconds, unit='s')
            
            # Replace time coordinate
            ds = ds.assign_coords(time=time_datetime)
            
        return ds
            
    def _load_tree_locations(self) -> Dict:
        """Load tree location CSV files"""
        tree_locations = {}
        
        for spacing in self.spacings:
            # Load parent domain locations
            parent_file = self.tree_locations_path / f"thf_forest_{spacing}m_parent.csv"
            if parent_file.exists():
                df = pd.read_csv(parent_file)
                tree_locations[f"{spacing}m_parent"] = df[['i', 'j']].values
                
            # Load child domain locations
            child_file = self.tree_locations_path / f"thf_forest_{spacing}m_child.csv"
            if child_file.exists():
                df = pd.read_csv(child_file)
                tree_locations[f"{spacing}m_child"] = df[['i', 'j']].values
                
        return tree_locations
        
    def _load_tree_metadata(self) -> Optional[pd.DataFrame]:
        """Load tree metadata from LAD/WAD CSV file"""
        if not self.lad_wad_path.exists():
            self.logger.warning("LAD/WAD data file not found")
            return None
            
        try:
            # Read CSV with proper data types
            df = pd.read_csv(self.lad_wad_path)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['i', 'j', 'k', 'lad', 'wad', 'tree_age', 
                             'tree_height', 'trunk_height', 'trunk_diameter',
                             'crown_base', 'crown_radius', 'la', 'lai', 'max_lad']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading tree metadata: {str(e)}")
            return None
            
    def get_simulation_info(self) -> Dict:
        """Get information about available simulations"""
        info = {
            'spacings': self.spacings,
            'ages': self.ages,
            'total_simulations': len(self.spacings) * len(self.ages),
            'base_path': str(self.sim_base_path),
            'has_base_case': (self.sim_base_path / "thf_base_2018080700").exists()
        }
        
        return info