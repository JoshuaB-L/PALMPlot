"""
Terrain-Following Mask I/O Module

Handles reading and writing terrain-following masks to NetCDF files following PALM conventions.
Masks are saved in PALM av_masked format with ku_above_surf dimension for terrain-relative levels.

Author: PALMPlot Development Team
Date: 2025-11-05
"""

import os
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class TerrainMaskWriter:
    """Writes terrain-following masks to NetCDF files following PALM conventions"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize terrain mask writer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def write_mask(self,
                   output_path: Path,
                   mask_data: Dict[str, np.ndarray],
                   source_levels: np.ndarray,
                   coordinates: Dict[str, np.ndarray],
                   metadata: Dict,
                   compression: Optional[Dict] = None) -> None:
        """
        Write terrain-following mask to NetCDF file.

        Args:
            output_path: Path to output NetCDF file
            mask_data: Dictionary of variable_name -> 3D array [ku_above_surf, y, x]
            source_levels: 2D array [y, x] of source zu_3d indices
            coordinates: Dictionary of coordinate arrays (x, y, ku_above_surf)
            metadata: Dictionary of metadata (domain info, settings, etc.)
            compression: Optional compression settings

        Raises:
            ValueError: If input data is invalid
            IOError: If file cannot be written
        """
        self.logger.info(f"Writing terrain mask to: {output_path}")

        # Validate inputs
        self._validate_inputs(mask_data, source_levels, coordinates)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare dimensions
        ny, nx = source_levels.shape
        n_levels = coordinates['ku_above_surf'].size

        try:
            # Create xarray Dataset
            ds = self._create_dataset(
                mask_data=mask_data,
                source_levels=source_levels,
                coordinates=coordinates,
                metadata=metadata,
                ny=ny,
                nx=nx,
                n_levels=n_levels
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

            self.logger.info(f"Successfully wrote terrain mask: {output_path.name}")
            self.logger.info(
                f"  Grid: {nx}×{ny}, Levels: {n_levels}, Variables: {len(mask_data)}"
            )

        except Exception as e:
            self.logger.error(f"Failed to write terrain mask: {e}")
            raise IOError(f"Failed to write terrain mask to {output_path}: {e}")

    def _validate_inputs(self,
                        mask_data: Dict[str, np.ndarray],
                        source_levels: np.ndarray,
                        coordinates: Dict[str, np.ndarray]) -> None:
        """Validate input data before writing."""

        if not mask_data:
            raise ValueError("mask_data cannot be empty")

        if source_levels.ndim != 2:
            raise ValueError(f"source_levels must be 2D, got {source_levels.ndim}D")

        required_coords = ['x', 'y', 'ku_above_surf']
        for coord in required_coords:
            if coord not in coordinates:
                raise ValueError(f"Required coordinate '{coord}' not found")

        ny, nx = source_levels.shape

        # Validate mask data shapes
        for var_name, var_data in mask_data.items():
            if var_data.ndim not in [3, 4]:
                raise ValueError(
                    f"Variable '{var_name}' must be 3D or 4D, got {var_data.ndim}D"
                )

            # Check spatial dimensions match
            if var_data.ndim == 3:
                if var_data.shape[1:] != (ny, nx):
                    raise ValueError(
                        f"Variable '{var_name}' spatial dimensions {var_data.shape[1:]} "
                        f"do not match source_levels {(ny, nx)}"
                    )
            else:  # 4D: [time, ku_above_surf, y, x]
                if var_data.shape[2:] != (ny, nx):
                    raise ValueError(
                        f"Variable '{var_name}' spatial dimensions {var_data.shape[2:]} "
                        f"do not match source_levels {(ny, nx)}"
                    )

    def _create_dataset(self,
                        mask_data: Dict[str, np.ndarray],
                        source_levels: np.ndarray,
                        coordinates: Dict[str, np.ndarray],
                        metadata: Dict,
                        ny: int,
                        nx: int,
                        n_levels: int) -> xr.Dataset:
        """Create xarray Dataset with proper structure and metadata."""

        # Create coordinate arrays
        time = np.array([0.0])  # Single time point
        ku_above_surf = coordinates['ku_above_surf']
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
            'ku_above_surf': ('ku_above_surf', ku_above_surf, {
                '_FillValue': np.nan,
                'units': '1',
                'long_name': 'grid point above terrain',
                'axis': 'Z',
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

        # Add mask variables
        for var_name, var_data in mask_data.items():
            # Ensure shape is [time, ku_above_surf, y, x]
            if var_data.ndim == 3:
                var_data = var_data[np.newaxis, ...]  # Add time dimension

            data_vars[var_name] = (
                ['time', 'ku_above_surf', 'y', 'x'],
                var_data.astype(np.float32),
                {
                    '_FillValue': -999999.0,
                    'units': self._get_variable_units(var_name),
                    'long_name': var_name,
                }
            )

        # Add source level array (metadata variable)
        data_vars['source_level'] = (
            ['y', 'x'],
            source_levels.astype(np.int32),
            {
                '_FillValue': -1,
                'units': '1',
                'long_name': 'Source zu_3d level index for base mask',
                'description': 'Index of vertical level used in terrain-following extraction',
            }
        )

        # Add terrain height if available
        if 'terrain_height' in metadata:
            data_vars['terrain_height'] = (
                ['y', 'x'],
                metadata['terrain_height'].astype(np.float32),
                {
                    '_FillValue': np.nan,
                    'units': 'meters',
                    'long_name': 'Terrain surface height',
                }
            )

        # Add buildings mask if available
        if 'buildings_mask' in metadata:
            data_vars['buildings_mask'] = (
                ['y', 'x'],
                metadata['buildings_mask'].astype(np.int32),
                {
                    '_FillValue': -1,
                    'units': '1',
                    'long_name': 'Building locations mask',
                    'flag_values': '0, 1',
                    'flag_meanings': 'open building',
                }
            )

        # Create Dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Add global attributes
        ds.attrs = self._create_global_attributes(metadata)

        return ds

    def _create_global_attributes(self, metadata: Dict) -> Dict:
        """Create global attributes following PALM conventions."""

        attrs = {
            'title': f"Terrain-following mask for {metadata.get('case_name', 'unknown')}",
            'Conventions': 'CF-1.7',
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S %z'),
            'data_content': 'terrain_following_mask',
            'version': 1,

            # Terrain-following specific
            'terrain_following_method': 'lowest_level_upward',
            'buildings_mask_applied': str(metadata.get('buildings_mask_applied', False)).lower(),
            'start_z_index': metadata.get('start_z_index', 0),
            'max_z_index': metadata.get('max_z_index', 0),
            'vertical_levels_stored': metadata.get('n_levels', 0),
            'zu_3d_coordinate_name': 'zu_3d',

            # Domain information
            'domain_type': metadata.get('domain_type', 'unknown'),
            'grid_size_x': metadata.get('nx', 0),
            'grid_size_y': metadata.get('ny', 0),
            'resolution': metadata.get('resolution', 0.0),

            # Software information
            'source': 'PALMPlot terrain_mask_io module',
            'author': metadata.get('author', 'PALMPlot'),
            'institution': metadata.get('institution', ''),
        }

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

    def _get_variable_units(self, var_name: str) -> str:
        """Get standard units for variable."""
        units_map = {
            'ta': 'degree_C',
            'theta': 'K',
            'q': 'kg/kg',
            'u': 'm/s',
            'v': 'm/s',
            'w': 'm/s',
            'wspeed': 'm/s',
            'wdir': 'degree',
            'ti': '1/s',
        }
        return units_map.get(var_name, '1')

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


class TerrainMaskReader:
    """Reads terrain-following masks from NetCDF files"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize terrain mask reader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def read_mask(self,
                  input_path: Path,
                  variables: Optional[List[str]] = None,
                  validate: bool = True,
                  validation_settings: Optional[Dict] = None) -> Dict:
        """
        Read terrain-following mask from NetCDF file.

        Args:
            input_path: Path to input NetCDF file
            variables: Optional list of variables to load (None = all)
            validate: Whether to validate mask file
            validation_settings: Optional validation settings

        Returns:
            Dictionary containing:
                - 'mask_data': Dict of variable_name -> array [ku_above_surf, y, x]
                - 'source_levels': 2D array of source zu_3d indices [y, x]
                - 'coordinates': Dict of coordinate arrays
                - 'metadata': Dict of metadata

        Raises:
            FileNotFoundError: If mask file does not exist
            ValueError: If validation fails
        """
        self.logger.info(f"Reading terrain mask from: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Terrain mask file not found: {input_path}")

        try:
            # Open dataset
            ds = xr.open_dataset(input_path, decode_timedelta=False)

            # Validate if requested
            if validate:
                self._validate_mask_file(ds, validation_settings or {})

            # Extract mask data
            result = {
                'mask_data': {},
                'source_levels': None,
                'coordinates': {},
                'metadata': {},
            }

            # Load coordinates
            result['coordinates']['x'] = ds['x'].values
            result['coordinates']['y'] = ds['y'].values
            result['coordinates']['ku_above_surf'] = ds['ku_above_surf'].values
            if 'time' in ds.coords:
                result['coordinates']['time'] = ds['time'].values

            # Load source levels (required metadata)
            if 'source_level' in ds.data_vars:
                result['source_levels'] = ds['source_level'].values
            else:
                self.logger.warning("'source_level' variable not found in mask file")

            # Load optional metadata variables
            if 'terrain_height' in ds.data_vars:
                result['metadata']['terrain_height'] = ds['terrain_height'].values
            if 'buildings_mask' in ds.data_vars:
                result['metadata']['buildings_mask'] = ds['buildings_mask'].values

            # Load mask data variables
            # Determine which variables to load
            if variables is None:
                # Load all non-coordinate, non-metadata variables
                vars_to_load = [v for v in ds.data_vars
                               if v not in ['source_level', 'terrain_height', 'buildings_mask']]
            else:
                vars_to_load = variables

            for var_name in vars_to_load:
                if var_name in ds.data_vars:
                    # Remove time dimension if present (we only store single time)
                    var_data = ds[var_name].values
                    if var_data.ndim == 4:  # [time, ku_above_surf, y, x]
                        var_data = var_data[0, ...]  # Get first (only) time step
                    result['mask_data'][var_name] = var_data
                else:
                    self.logger.warning(f"Variable '{var_name}' not found in mask file")

            # Extract global metadata
            result['metadata']['attrs'] = dict(ds.attrs)

            # Close dataset
            ds.close()

            self.logger.info(
                f"Successfully read terrain mask: {len(result['mask_data'])} variables, "
                f"{result['coordinates']['ku_above_surf'].size} levels"
            )

            return result

        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to read terrain mask: {e}")
            raise IOError(f"Failed to read terrain mask from {input_path}: {e}")

    def _validate_mask_file(self, ds: xr.Dataset, settings: Dict) -> None:
        """
        Validate terrain mask file structure and contents.

        Args:
            ds: xarray Dataset to validate
            settings: Validation settings from config

        Raises:
            ValueError: If validation fails and on_mismatch='error'
        """
        errors = []
        warnings = []

        # Check required dimensions
        required_dims = ['ku_above_surf', 'y', 'x']
        for dim in required_dims:
            if dim not in ds.dims:
                errors.append(f"Required dimension '{dim}' not found")

        # Check required coordinates
        required_coords = ['ku_above_surf', 'y', 'x']
        for coord in required_coords:
            if coord not in ds.coords:
                errors.append(f"Required coordinate '{coord}' not found")

        # Check for source_level variable
        if 'source_level' not in ds.data_vars:
            warnings.append("'source_level' variable not found (may be older format)")

        # Check data_content attribute
        if ds.attrs.get('data_content') != 'terrain_following_mask':
            warnings.append(
                f"Unexpected data_content: '{ds.attrs.get('data_content')}' "
                f"(expected 'terrain_following_mask')"
            )

        # Check mask age if max_age_days specified
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
                            f"Mask file is {age_days} days old "
                            f"(max: {settings['max_age_days']})"
                        )
                except Exception as e:
                    warnings.append(f"Could not parse creation_time: {e}")

        # Handle errors and warnings
        if errors:
            msg = "Terrain mask validation failed:\n  " + "\n  ".join(errors)
            on_mismatch = settings.get('on_mismatch', 'recompute')
            if on_mismatch == 'error':
                raise ValueError(msg)
            else:
                self.logger.error(msg)

        if warnings:
            msg = "Terrain mask validation warnings:\n  " + "\n  ".join(warnings)
            self.logger.warning(msg)

    def check_mask_compatibility(self,
                                 mask_metadata: Dict,
                                 expected_grid_size: Tuple[int, int],
                                 expected_domain: str,
                                 validation_settings: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """
        Check if loaded mask is compatible with current extraction settings.

        Args:
            mask_metadata: Metadata from loaded mask
            expected_grid_size: Expected (ny, nx)
            expected_domain: Expected domain type ('parent' or 'child')
            validation_settings: Optional validation settings from config

        Returns:
            (is_compatible, list_of_issues)
        """
        issues = []
        settings = validation_settings or {}

        attrs = mask_metadata.get('attrs', {})

        # Check domain type if requested
        if settings.get('check_domain_type', True):
            mask_domain = attrs.get('domain_type', 'unknown')
            if mask_domain != expected_domain:
                issues.append(
                    f"Domain mismatch: mask is '{mask_domain}', expected '{expected_domain}'"
                )

        # Check grid size if requested
        if settings.get('check_grid_size', True):
            mask_nx = attrs.get('grid_size_x', 0)
            mask_ny = attrs.get('grid_size_y', 0)
            expected_ny, expected_nx = expected_grid_size

            if mask_nx != expected_nx or mask_ny != expected_ny:
                issues.append(
                    f"Grid size mismatch: mask is {mask_nx}×{mask_ny}, "
                    f"expected {expected_nx}×{expected_ny}"
                )

        # Check z-coordinate if requested
        if settings.get('check_z_coordinate', True):
            zu_3d_name = attrs.get('zu_3d_coordinate_name', 'zu_3d')
            if zu_3d_name != 'zu_3d':
                issues.append(
                    f"Unexpected z-coordinate name: '{zu_3d_name}' (expected 'zu_3d')"
                )

        is_compatible = len(issues) == 0

        if not is_compatible:
            self.logger.warning(
                f"Mask compatibility check found {len(issues)} issue(s):\n  " +
                "\n  ".join(issues)
            )

        return is_compatible, issues
