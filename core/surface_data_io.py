"""
Surface Data I/O Module

Handles reading and writing time-averaged surface data (av_xy variables) to NetCDF files.
Surface data files store multiple 2D variables that have been time-averaged from PALM av_xy outputs.

This module complements terrain_mask_io.py (for 3D atmospheric variables) by handling
2D surface variables (UTCI, radiation, surface temperature, etc.).

Author: PALMPlot Development Team
Date: 2025-11-08
"""

import os
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class SurfaceDataWriter:
    """Writes time-averaged surface data to NetCDF files following PALM conventions"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize surface data writer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def write_surface_data(self,
                          output_path: Path,
                          surface_data: Dict[str, np.ndarray],
                          coordinates: Dict[str, np.ndarray],
                          metadata: Dict,
                          compression: Optional[Dict] = None) -> None:
        """
        Write time-averaged surface data to NetCDF file.

        Args:
            output_path: Path to output NetCDF file
            surface_data: Dictionary of variable_name -> 2D array [y, x]
            coordinates: Dictionary of coordinate arrays (x, y)
            metadata: Dictionary of metadata (domain info, time averaging settings, etc.)
            compression: Optional compression settings

        Raises:
            ValueError: If input data is invalid
            IOError: If file cannot be written
        """
        self.logger.info(f"Writing surface data to: {output_path}")

        # Validate inputs
        self._validate_inputs(surface_data, coordinates)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get grid dimensions
        first_var = next(iter(surface_data.values()))
        ny, nx = first_var.shape

        try:
            # Create xarray Dataset
            ds = self._create_dataset(
                surface_data=surface_data,
                coordinates=coordinates,
                metadata=metadata,
                ny=ny,
                nx=nx
            )

            # Apply compression if requested
            encoding = {}
            if compression and compression.get('enabled', True):
                comp_level = compression.get('level', 4)
                encoding = self._create_compression_encoding(ds, comp_level)

            # Write to NetCDF
            ds.to_netcdf(
                output_path,
                format='NETCDF4',
                encoding=encoding,
                unlimited_dims=['time']
            )

            # Close dataset
            ds.close()

            self.logger.info(f"Successfully wrote surface data: {output_path.name}")
            self.logger.info(
                f"  Grid: {nx}×{ny}, Variables: {len(surface_data)}"
            )

        except Exception as e:
            self.logger.error(f"Failed to write surface data: {e}")
            raise IOError(f"Failed to write surface data to {output_path}: {e}")

    def _validate_inputs(self,
                        surface_data: Dict[str, np.ndarray],
                        coordinates: Dict[str, np.ndarray]) -> None:
        """Validate input data before writing."""

        if not surface_data:
            raise ValueError("surface_data cannot be empty")

        required_coords = ['x', 'y']
        for coord in required_coords:
            if coord not in coordinates:
                raise ValueError(f"Required coordinate '{coord}' not found")

        # Get expected shape from first variable
        first_var = next(iter(surface_data.values()))
        if first_var.ndim != 2:
            raise ValueError(
                f"Surface data must be 2D, got {first_var.ndim}D for first variable"
            )

        expected_shape = first_var.shape
        ny, nx = expected_shape

        # Validate all variables have same shape
        for var_name, var_data in surface_data.items():
            if var_data.ndim != 2:
                raise ValueError(
                    f"Variable '{var_name}' must be 2D, got {var_data.ndim}D"
                )

            if var_data.shape != expected_shape:
                raise ValueError(
                    f"Variable '{var_name}' shape {var_data.shape} "
                    f"does not match expected {expected_shape}"
                )

        # Validate coordinate lengths
        if len(coordinates['x']) != nx:
            raise ValueError(
                f"x coordinate length {len(coordinates['x'])} "
                f"does not match grid size {nx}"
            )

        if len(coordinates['y']) != ny:
            raise ValueError(
                f"y coordinate length {len(coordinates['y'])} "
                f"does not match grid size {ny}"
            )

    def _create_dataset(self,
                        surface_data: Dict[str, np.ndarray],
                        coordinates: Dict[str, np.ndarray],
                        metadata: Dict,
                        ny: int,
                        nx: int) -> xr.Dataset:
        """Create xarray Dataset with proper structure and metadata."""

        # Create coordinate arrays
        time = np.array([0.0])  # Single time point (after averaging)
        x = coordinates['x']
        y = coordinates['y']

        # Create coordinate dictionaries
        coords = {
            'time': ('time', time, {
                '_FillValue': np.nan,
                'long_name': 'time',
                'standard_name': 'time',
                'axis': 'T',
                'units': 'seconds',
            }),
            'x': ('x', x, {
                '_FillValue': np.nan,
                'units': 'meters',
                'axis': 'X',
            }),
            'y': ('y', y, {
                '_FillValue': np.nan,
                'units': 'meters',
                'axis': 'Y',
            }),
        }

        # Create data variables dictionary
        data_vars = {}

        # Add surface data variables
        for var_name, var_data in surface_data.items():
            # Add time dimension: [time, y, x]
            var_data_with_time = var_data[np.newaxis, ...]

            # Get variable-specific metadata
            var_units = self._get_variable_units(var_name, metadata)
            var_long_name = self._get_variable_long_name(var_name, metadata)

            data_vars[var_name] = (
                ['time', 'y', 'x'],
                var_data_with_time.astype(np.float32),
                {
                    '_FillValue': -999999.0,
                    'units': var_units,
                    'long_name': var_long_name,
                }
            )

        # Create Dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Add global attributes
        ds.attrs = self._create_global_attributes(metadata, len(surface_data))

        return ds

    def _create_global_attributes(self, metadata: Dict, n_vars: int) -> Dict:
        """Create global attributes following PALM conventions."""

        attrs = {
            'title': f"Time-averaged surface data for {metadata.get('case_name', 'unknown')}",
            'Conventions': 'CF-1.7',
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S %z'),
            'data_content': 'surface_data_time_averaged',
            'version': 2,  # Updated to version 2 for enhanced naming support

            # Time averaging information
            'time_averaging_method': metadata.get('time_averaging_method', 'mean'),
            'time_steps_used': metadata.get('time_steps_used', 0),
            'time_steps_total': metadata.get('time_steps_total', 0),
            'time_steps_corrupted': metadata.get('time_steps_corrupted', 0),

            # Enhanced naming metadata (NEW in version 2)
            'time_selection_mode': metadata.get('time_selection_mode', 'all_times_average'),

            # Domain information
            'domain_type': metadata.get('domain_type', 'unknown'),
            'grid_size_x': metadata.get('nx', 0),
            'grid_size_y': metadata.get('ny', 0),
            'resolution': metadata.get('resolution', 0.0),

            # Variable information
            'number_of_variables': n_vars,

            # Software information
            'source': 'PALMPlot surface_data_io module',
            'author': metadata.get('author', 'PALMPlot'),
            'institution': metadata.get('institution', ''),
        }

        # Add time range if specified (for time_window mode)
        if 'time_range_start' in metadata:
            attrs['time_range_start'] = metadata['time_range_start']
        if 'time_range_end' in metadata:
            attrs['time_range_end'] = metadata['time_range_end']

        # Add time index if specified (for single_time mode)
        if 'time_index' in metadata:
            attrs['time_index'] = metadata['time_index']

        # Add optional attributes
        if 'origin_x' in metadata:
            attrs['origin_x'] = metadata['origin_x']
        if 'origin_y' in metadata:
            attrs['origin_y'] = metadata['origin_y']
        if 'origin_z' in metadata:
            attrs['origin_z'] = metadata['origin_z']
        if 'rotation_angle' in metadata:
            attrs['rotation_angle'] = metadata['rotation_angle']

        return attrs

    def _get_variable_units(self, var_name: str, metadata: Dict) -> str:
        """
        Get units for variable.

        First checks metadata for variable-specific units,
        then falls back to standard units map.
        """
        # Check if variable units provided in metadata
        var_metadata = metadata.get('variable_metadata', {})
        if var_name in var_metadata and 'units' in var_metadata[var_name]:
            return var_metadata[var_name]['units']

        # Standard units map for common PALM variables
        units_map = {
            'bio_utci_xy': 'degC',
            'bio_pet_xy': 'degC',
            'rad_net_xy': 'W/m2',
            'rad_lw_in_xy': 'W/m2',
            'rad_lw_out_xy': 'W/m2',
            'rad_sw_in_xy': 'W/m2',
            'rad_sw_out_xy': 'W/m2',
            'ghf_xy': 'W/m2',
            'shf_xy': 'W/m2',
            'qsws_xy': 'kg/(m2*s)',
            'us_xy': 'm/s',
            't_surf_xy': 'K',
        }

        return units_map.get(var_name, '1')

    def _get_variable_long_name(self, var_name: str, metadata: Dict) -> str:
        """
        Get long name for variable.

        First checks metadata for variable-specific long name,
        then falls back to standard long name map.
        """
        # Check if variable long name provided in metadata
        var_metadata = metadata.get('variable_metadata', {})
        if var_name in var_metadata and 'long_name' in var_metadata[var_name]:
            return var_metadata[var_name]['long_name']

        # Standard long name map for common PALM variables
        long_name_map = {
            'bio_utci_xy': 'Universal Thermal Climate Index',
            'bio_pet_xy': 'Physiological Equivalent Temperature',
            'rad_net_xy': 'Net Radiation',
            'rad_lw_in_xy': 'Longwave Radiation Incoming',
            'rad_lw_out_xy': 'Longwave Radiation Outgoing',
            'rad_sw_in_xy': 'Shortwave Radiation Incoming',
            'rad_sw_out_xy': 'Shortwave Radiation Outgoing',
            'ghf_xy': 'Ground Heat Flux',
            'shf_xy': 'Sensible Heat Flux',
            'qsws_xy': 'Surface Moisture Flux',
            'us_xy': 'Friction Velocity',
            't_surf_xy': 'Surface Temperature',
        }

        return long_name_map.get(var_name, var_name)

    def _create_compression_encoding(self, ds: xr.Dataset, level: int) -> Dict:
        """Create encoding dictionary for NetCDF compression."""

        encoding = {}

        # Compress all data variables
        for var_name in ds.data_vars:
            if ds[var_name].ndim >= 2:  # Only compress multi-dimensional vars
                encoding[var_name] = {
                    'zlib': True,
                    'complevel': level,
                    'shuffle': True,
                }

        return encoding


class SurfaceDataReader:
    """Reads time-averaged surface data from NetCDF files"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize surface data reader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def read_surface_data(self,
                         input_path: Path,
                         variables: Optional[List[str]] = None,
                         validate: bool = True,
                         validation_settings: Optional[Dict] = None) -> Dict:
        """
        Read time-averaged surface data from NetCDF file.

        Args:
            input_path: Path to input NetCDF file
            variables: Optional list of variables to load (None = all)
            validate: Whether to validate surface data file
            validation_settings: Optional validation settings

        Returns:
            Dictionary containing:
                - 'surface_data': Dict of variable_name -> array [y, x]
                - 'coordinates': Dict of coordinate arrays
                - 'metadata': Dict of metadata

        Raises:
            FileNotFoundError: If surface data file does not exist
            ValueError: If validation fails
        """
        self.logger.info(f"Reading surface data from: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Surface data file not found: {input_path}")

        try:
            # Open dataset
            ds = xr.open_dataset(input_path, decode_timedelta=False)

            # Validate if requested
            if validate:
                self._validate_surface_data_file(ds, validation_settings or {})

            # Extract surface data
            result = {
                'surface_data': {},
                'coordinates': {},
                'metadata': {},
            }

            # Load coordinates
            result['coordinates']['x'] = ds['x'].values
            result['coordinates']['y'] = ds['y'].values
            if 'time' in ds.coords:
                result['coordinates']['time'] = ds['time'].values

            # Load surface data variables
            # Determine which variables to load
            if variables is None:
                # Load all data variables
                vars_to_load = list(ds.data_vars)
            else:
                vars_to_load = variables

            for var_name in vars_to_load:
                if var_name in ds.data_vars:
                    # Remove time dimension if present (we only store single time)
                    var_data = ds[var_name].values
                    if var_data.ndim == 3:  # [time, y, x]
                        var_data = var_data[0, ...]  # Get first (only) time step
                    result['surface_data'][var_name] = var_data
                else:
                    self.logger.warning(f"Variable '{var_name}' not found in surface data file")

            # Extract global metadata
            result['metadata']['attrs'] = dict(ds.attrs)

            # Close dataset
            ds.close()

            self.logger.info(
                f"Successfully read surface data: {len(result['surface_data'])} variables"
            )

            return result

        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to read surface data: {e}")
            raise IOError(f"Failed to read surface data from {input_path}: {e}")

    def _validate_surface_data_file(self, ds: xr.Dataset, settings: Dict) -> None:
        """
        Validate surface data file structure and contents.

        Args:
            ds: xarray Dataset to validate
            settings: Validation settings from config

        Raises:
            ValueError: If validation fails and on_mismatch='error'
        """
        errors = []
        warnings = []

        # Check required dimensions
        required_dims = ['y', 'x']
        for dim in required_dims:
            if dim not in ds.dims:
                errors.append(f"Required dimension '{dim}' not found")

        # Check required coordinates
        required_coords = ['y', 'x']
        for coord in required_coords:
            if coord not in ds.coords:
                errors.append(f"Required coordinate '{coord}' not found")

        # Check data_content attribute
        if ds.attrs.get('data_content') != 'surface_data_time_averaged':
            warnings.append(
                f"Unexpected data_content: '{ds.attrs.get('data_content')}' "
                f"(expected 'surface_data_time_averaged')"
            )

        # Check if file has any data variables
        if len(ds.data_vars) == 0:
            errors.append("No data variables found in surface data file")

        # Check age if max_age_days specified
        if settings.get('max_age_days'):
            creation_time_str = ds.attrs.get('creation_time')
            if creation_time_str:
                try:
                    # Parse creation time (remove timezone if present)
                    creation_time = datetime.strptime(
                        creation_time_str.split('+')[0].strip(),
                        '%Y-%m-%d %H:%M:%S'
                    )
                    age_days = (datetime.now() - creation_time).days
                    if age_days > settings['max_age_days']:
                        warnings.append(
                            f"Surface data file is {age_days} days old "
                            f"(max: {settings['max_age_days']})"
                        )
                except Exception as e:
                    warnings.append(f"Could not parse creation_time: {e}")

        # Handle errors and warnings
        if errors:
            msg = "Surface data validation failed:\n  " + "\n  ".join(errors)
            on_mismatch = settings.get('on_mismatch', 'recompute')
            if on_mismatch == 'error':
                raise ValueError(msg)
            else:
                self.logger.error(msg)

        if warnings:
            msg = "Surface data validation warnings:\n  " + "\n  ".join(warnings)
            self.logger.warning(msg)

    def check_surface_data_compatibility(self,
                                        surface_metadata: Dict,
                                        expected_grid_size: Tuple[int, int],
                                        expected_domain: str,
                                        validation_settings: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """
        Check if loaded surface data is compatible with current extraction settings.

        Args:
            surface_metadata: Metadata from loaded surface data
            expected_grid_size: Expected (ny, nx)
            expected_domain: Expected domain type ('parent' or 'child')
            validation_settings: Optional validation settings from config

        Returns:
            (is_compatible, list_of_issues)
        """
        issues = []
        settings = validation_settings or {}

        attrs = surface_metadata.get('attrs', {})

        # Check domain type if requested
        if settings.get('check_domain_type', True):
            cached_domain = attrs.get('domain_type', 'unknown')
            if cached_domain != expected_domain:
                issues.append(
                    f"Domain mismatch: cache is '{cached_domain}', expected '{expected_domain}'"
                )

        # Check grid size if requested
        if settings.get('check_grid_size', True):
            cached_nx = attrs.get('grid_size_x', 0)
            cached_ny = attrs.get('grid_size_y', 0)
            expected_ny, expected_nx = expected_grid_size

            if cached_nx != expected_nx or cached_ny != expected_ny:
                issues.append(
                    f"Grid size mismatch: cache is {cached_nx}×{cached_ny}, "
                    f"expected {expected_nx}×{expected_ny}"
                )

        is_compatible = len(issues) == 0

        if not is_compatible:
            self.logger.warning(
                f"Surface data compatibility check found {len(issues)} issue(s):\n  " +
                "\n  ".join(issues)
            )

        return is_compatible, issues


# Helper Functions

def generate_surface_data_filename(case_name: str, domain_type: str) -> str:
    """
    Generate standardized filename for surface data cache.

    Args:
        case_name: Simulation case name
        domain_type: 'parent' or 'child'

    Returns:
        Filename string (e.g., "case_name_parent_surface_data.nc")
    """
    return f"{case_name}_{domain_type}_surface_data.nc"


def find_existing_surface_data_file(cache_dir: Path,
                                   case_name: str,
                                   domain_type: str) -> Optional[Path]:
    """
    Find existing surface data file in cache directory.

    Args:
        cache_dir: Directory to search
        case_name: Simulation case name
        domain_type: 'parent' or 'child'

    Returns:
        Path to surface data file if found, None otherwise
    """
    if not cache_dir.exists():
        return None

    # Try exact filename match first
    expected_filename = generate_surface_data_filename(case_name, domain_type)
    expected_path = cache_dir / expected_filename

    if expected_path.exists():
        return expected_path

    # Try pattern matching (in case filename format changed)
    pattern = f"{case_name}_{domain_type}_surface_data*.nc"
    matches = list(cache_dir.glob(pattern))

    if matches:
        # Return most recent if multiple matches
        return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def validate_surface_data_file(file_path: Path,
                               logger: Optional[logging.Logger] = None) -> bool:
    """
    Quick validation of surface data file without full loading.

    Args:
        file_path: Path to surface data file
        logger: Optional logger

    Returns:
        True if file appears valid, False otherwise
    """
    logger = logger or logging.getLogger(__name__)

    try:
        ds = xr.open_dataset(file_path, decode_timedelta=False)

        # Check basic structure
        has_coords = 'x' in ds.coords and 'y' in ds.coords
        has_data = len(ds.data_vars) > 0
        correct_content = ds.attrs.get('data_content') == 'surface_data_time_averaged'

        ds.close()

        return has_coords and has_data and correct_content

    except Exception as e:
        logger.warning(f"Validation failed for {file_path}: {e}")
        return False
