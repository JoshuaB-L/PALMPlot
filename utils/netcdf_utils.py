"""
NetCDF Utility Functions for PALMPlot

Helper functions for working with NetCDF files, particularly for
terrain-following mask operations.

Author: PALMPlot Development Team
Date: 2025-11-05
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def copy_netcdf_metadata(source_ds: xr.Dataset, target_ds: xr.Dataset) -> xr.Dataset:
    """
    Copy relevant metadata from source to target dataset.

    Args:
        source_ds: Source dataset (original PALM output)
        target_ds: Target dataset (terrain mask)

    Returns:
        Updated target dataset with copied metadata
    """
    # Copy origin coordinates if present
    origin_attrs = [
        'origin_x', 'origin_y', 'origin_z',
        'origin_lat', 'origin_lon', 'rotation_angle'
    ]

    for attr in origin_attrs:
        if attr in source_ds.attrs:
            target_ds.attrs[attr] = source_ds.attrs[attr]

    return target_ds


def generate_mask_filename(case_name: str,
                          domain_type: str,
                          offsets: List[int],
                          suffix: str = '.nc') -> str:
    """
    Generate standardized filename for terrain mask.

    Args:
        case_name: Simulation case name
        domain_type: 'parent' or 'child'
        offsets: List of offset levels stored
        suffix: File suffix (default '.nc')

    Returns:
        Filename string

    Examples:
        >>> generate_mask_filename('thf_base', 'parent', [0, 1, 5])
        'thf_base_terrain_mask_parent_TF0-1-5.nc'

        >>> generate_mask_filename('thf_forest_10m_20yrs', 'child', [0])
        'thf_forest_10m_20yrs_terrain_mask_child_TF0.nc'

        >>> generate_mask_filename('thf_forest_10m_20yrs', 'parent', [0, 1, 2, 3, 4, 5])
        'thf_forest_10m_20yrs_terrain_mask_parent_TF0-5.nc'
    """
    # Domain suffix (empty for parent, _child for child)
    if domain_type == 'parent':
        domain_suffix = ''
    else:
        domain_suffix = f'_{domain_type}'

    # Generate offset string
    if len(offsets) == 0:
        offset_str = 'TF0'
    elif len(offsets) == 1:
        offset_str = f'TF{offsets[0]}'
    else:
        # Check if continuous range
        offsets_sorted = sorted(offsets)
        min_offset = min(offsets)
        max_offset = max(offsets)

        # Is it a continuous range?
        if offsets_sorted == list(range(min_offset, max_offset + 1)):
            # Continuous: use TF0-5 format
            offset_str = f'TF{min_offset}-{max_offset}'
        else:
            # Discontinuous: use TF0-1-5 format
            offset_str = 'TF' + '-'.join(map(str, offsets_sorted))

    return f"{case_name}_terrain_mask{domain_suffix}_{offset_str}{suffix}"


def find_existing_mask_file(cache_dir: Path,
                            case_name: str,
                            domain_type: str) -> Optional[Path]:
    """
    Find existing terrain mask file for case/domain.

    Searches for files matching the pattern and returns the most recent if multiple found.

    Args:
        cache_dir: Cache directory to search
        case_name: Simulation case name
        domain_type: 'parent' or 'child'

    Returns:
        Path to mask file if found, else None

    Example:
        >>> cache_dir = Path('./cache/terrain_masks')
        >>> mask_path = find_existing_mask_file(cache_dir, 'thf_base', 'parent')
        >>> if mask_path:
        ...     print(f"Found: {mask_path.name}")
        ... else:
        ...     print("No mask found")
    """
    if not cache_dir.exists():
        return None

    # Search for files matching pattern
    if domain_type == 'parent':
        domain_suffix = ''
    else:
        domain_suffix = f'_{domain_type}'

    pattern = f"{case_name}_terrain_mask{domain_suffix}_TF*.nc"

    matches = list(cache_dir.glob(pattern))

    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        # Multiple matches, return most recent
        return max(matches, key=lambda p: p.stat().st_mtime)


def parse_offset_specification(offset_spec, max_levels: int = 20) -> List[int]:
    """
    Parse offset specification into list of integers.

    Handles various input formats:
    - List of integers: [0, 1, 5, 10]
    - String "all": All levels from 0 to max_levels-1
    - String "range(...)": Python range notation

    Args:
        offset_spec: Offset specification (list, "all", or "range(...)")
        max_levels: Maximum number of levels for "all" mode

    Returns:
        List of offset integers

    Raises:
        ValueError: If offset_spec format is invalid

    Examples:
        >>> parse_offset_specification([0, 1, 5])
        [0, 1, 5]

        >>> parse_offset_specification("all", max_levels=5)
        [0, 1, 2, 3, 4]

        >>> parse_offset_specification("range(0, 6)")
        [0, 1, 2, 3, 4, 5]

        >>> parse_offset_specification("range(0, 11, 2)")
        [0, 2, 4, 6, 8, 10]
    """
    if isinstance(offset_spec, list):
        # Already a list
        return sorted(offset_spec)

    elif isinstance(offset_spec, str):
        if offset_spec == 'all':
            # All levels from 0 to max_levels-1
            return list(range(max_levels))

        elif offset_spec.startswith('range('):
            # Python range notation
            try:
                # Evaluate range expression safely
                offsets = list(eval(offset_spec))
                return sorted(offsets)
            except Exception as e:
                raise ValueError(
                    f"Invalid range specification '{offset_spec}': {e}"
                )
        else:
            raise ValueError(
                f"Invalid string offset specification '{offset_spec}'. "
                f"Must be 'all' or 'range(...)'"
            )
    else:
        raise ValueError(
            f"Invalid offset specification type {type(offset_spec)}. "
            f"Must be list or string"
        )


def get_mask_metadata_summary(mask_path: Path) -> Optional[Dict]:
    """
    Get summary metadata from mask file without loading full data.

    Useful for quickly checking mask properties before loading.

    Args:
        mask_path: Path to mask NetCDF file

    Returns:
        Dictionary with summary metadata, or None if file cannot be read

    Example:
        >>> mask_path = Path('./cache/thf_base_terrain_mask_parent_TF0-5.nc')
        >>> summary = get_mask_metadata_summary(mask_path)
        >>> if summary:
        ...     print(f"Grid: {summary['grid_size']}")
        ...     print(f"Levels: {summary['n_levels']}")
        ...     print(f"Variables: {summary['variables']}")
    """
    if not mask_path.exists():
        return None

    try:
        # Open dataset without loading data (decode_timedelta=False for speed)
        ds = xr.open_dataset(mask_path, decode_timedelta=False)

        summary = {
            'file_path': str(mask_path),
            'file_size_mb': mask_path.stat().st_size / (1024 * 1024),
            'grid_size': (ds.dims['x'], ds.dims['y']),
            'n_levels': ds.dims['ku_above_surf'],
            'variables': [v for v in ds.data_vars
                         if v not in ['source_level', 'terrain_height', 'buildings_mask']],
            'has_source_level': 'source_level' in ds.data_vars,
            'has_terrain_height': 'terrain_height' in ds.data_vars,
            'has_buildings_mask': 'buildings_mask' in ds.data_vars,
            'domain_type': ds.attrs.get('domain_type', 'unknown'),
            'creation_time': ds.attrs.get('creation_time', 'unknown'),
            'data_content': ds.attrs.get('data_content', 'unknown'),
        }

        ds.close()

        return summary

    except Exception:
        return None


def validate_mask_offsets(mask_path: Path, required_offsets: List[int]) -> Tuple[bool, List[int]]:
    """
    Check if mask file contains required offset levels.

    Args:
        mask_path: Path to mask NetCDF file
        required_offsets: List of required offset levels

    Returns:
        (has_all_offsets, missing_offsets)

    Example:
        >>> mask_path = Path('./cache/thf_base_terrain_mask_parent_TF0-5.nc')
        >>> required = [0, 1, 2, 10]
        >>> has_all, missing = validate_mask_offsets(mask_path, required)
        >>> if not has_all:
        ...     print(f"Missing offsets: {missing}")
        Missing offsets: [10]
    """
    if not mask_path.exists():
        return False, required_offsets

    try:
        ds = xr.open_dataset(mask_path, decode_timedelta=False)

        # Get number of available levels
        # Level 0 is below terrain (all fill values)
        # Level 1 = offset 0, Level 2 = offset 1, etc.
        n_levels = ds.dims['ku_above_surf']

        ds.close()

        # Check which offsets are available
        # offset N is stored in ku_above_surf level N+1
        available_offsets = list(range(n_levels - 1))  # -1 because level 0 is below terrain

        missing = [offset for offset in required_offsets
                  if offset not in available_offsets]

        has_all = len(missing) == 0

        return has_all, missing

    except Exception:
        return False, required_offsets
