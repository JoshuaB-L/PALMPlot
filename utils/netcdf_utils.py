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


def generate_mask_filename_enhanced(
    case_name: str,
    domain_type: str,
    offsets: List[int],
    time_mode: str = 'all_times_average',
    time_params: Optional[Dict] = None,
    building_mask: bool = True,
    suffix: str = '.nc'
) -> str:
    """
    Generate enhanced filename for terrain mask with time mode and mask mode.

    This is the new naming convention that supports:
    - Time selection modes (all_times, time_window, single_time)
    - Building mask toggle (masked/unmasked)

    Args:
        case_name: Simulation case name
        domain_type: 'parent' or 'child'
        offsets: List of offset levels stored
        time_mode: Time selection mode:
            - 'all_times_average': Average over all timesteps (default)
            - 'time_window': Average over specific time range
            - 'single_time': Single timestep snapshot
        time_params: Optional dict with time parameters:
            - For 'time_window': {'start': int, 'end': int}
            - For 'single_time': {'hour': int}
        building_mask: True = buildings excluded (NaN), False = filled through
        suffix: File suffix (default '.nc')

    Returns:
        Filename string

    Examples:
        >>> generate_mask_filename_enhanced('thf_base', 'parent', [0,1,2])
        'thf_base_terrain_mask_parent_all_times_average_masked_TF0-2.nc'

        >>> generate_mask_filename_enhanced(
        ...     'thf_base', 'child', [0,1,5,10],
        ...     time_mode='time_window',
        ...     time_params={'start': 30, 'end': 42},
        ...     building_mask=False
        ... )
        'thf_base_terrain_mask_child_time_window_30_42_unmasked_TF0-1-5-10.nc'

        >>> generate_mask_filename_enhanced(
        ...     'thf_forest_10m_20yrs', 'parent', [1],
        ...     time_mode='single_time',
        ...     time_params={'hour': 14}
        ... )
        'thf_forest_10m_20yrs_terrain_mask_parent_single_time_14_masked_TF1.nc'
    """
    # Domain suffix - always include domain type in enhanced naming
    domain_suffix = f'_{domain_type}'

    # Time mode string
    if time_mode == 'all_times_average':
        time_str = 'all_times_average'
    elif time_mode == 'time_window':
        if time_params and 'start' in time_params and 'end' in time_params:
            time_str = f"time_window_{time_params['start']}_{time_params['end']}"
        else:
            raise ValueError("time_window mode requires time_params with 'start' and 'end'")
    elif time_mode == 'single_time':
        if time_params and 'hour' in time_params:
            time_str = f"single_time_{time_params['hour']}"
        else:
            raise ValueError("single_time mode requires time_params with 'hour'")
    else:
        raise ValueError(f"Unknown time_mode: {time_mode}")

    # Building mask string
    mask_str = 'masked' if building_mask else 'unmasked'

    # Generate offset string (reuse logic from generate_mask_filename)
    if len(offsets) == 0:
        offset_str = 'TF0'
    elif len(offsets) == 1:
        offset_str = f'TF{offsets[0]}'
    else:
        offsets_sorted = sorted(offsets)
        min_offset = min(offsets)
        max_offset = max(offsets)

        if offsets_sorted == list(range(min_offset, max_offset + 1)):
            offset_str = f'TF{min_offset}-{max_offset}'
        else:
            offset_str = 'TF' + '-'.join(map(str, offsets_sorted))

    return f"{case_name}_terrain_mask{domain_suffix}_{time_str}_{mask_str}_{offset_str}{suffix}"


def parse_mask_filename_enhanced(filename: str) -> Optional[Dict]:
    """
    Parse enhanced mask filename to extract metadata.

    Args:
        filename: Mask filename (with or without .nc extension)

    Returns:
        Dictionary with parsed components, or None if parsing fails
        {
            'case_name': str,
            'domain_type': str,
            'time_mode': str,
            'time_params': dict,
            'building_mask': bool,
            'offsets': list[int]
        }

    Examples:
        >>> parse_mask_filename_enhanced(
        ...     'thf_base_terrain_mask_parent_all_times_average_masked_TF0-5.nc'
        ... )
        {'case_name': 'thf_base', 'domain_type': 'parent',
         'time_mode': 'all_times_average', 'time_params': {},
         'building_mask': True, 'offsets': [0,1,2,3,4,5]}

        >>> parse_mask_filename_enhanced(
        ...     'thf_forest_10m_20yrs_terrain_mask_child_time_window_30_42_unmasked_TF1-2-5.nc'
        ... )
        {'case_name': 'thf_forest_10m_20yrs', 'domain_type': 'child',
         'time_mode': 'time_window', 'time_params': {'start': 30, 'end': 42},
         'building_mask': False, 'offsets': [1,2,5]}
    """
    # Remove .nc extension if present
    if filename.endswith('.nc'):
        filename = filename[:-3]

    # Split filename into parts
    parts = filename.split('_')

    # Need at least: case_terrain_mask_domain_timemode_maskmode_TF
    if len(parts) < 6:
        return None

    # Check for 'terrain_mask' marker
    try:
        mask_idx = parts.index('terrain')
        if parts[mask_idx + 1] != 'mask':
            return None
    except (ValueError, IndexError):
        return None

    # Everything before 'terrain_mask' is case name
    case_name = '_'.join(parts[:mask_idx])

    # After 'terrain_mask': domain, time components, mask mode, TF
    remaining = parts[mask_idx + 2:]  # Skip 'terrain' and 'mask'

    if len(remaining) < 4:
        return None

    # Determine domain type
    # Could be: parent_all_times_average_masked_TF...
    # or: child_all_times_average_masked_TF...
    domain_idx = 0
    if remaining[domain_idx] in ['parent', 'child']:
        domain_type = remaining[domain_idx]
        time_start_idx = 1
    else:
        # Default domain is parent (no suffix)
        domain_type = 'parent'
        time_start_idx = 0

    # Find TF marker (should be last component)
    tf_idx = None
    for i, part in enumerate(remaining):
        if part.startswith('TF'):
            tf_idx = i
            break

    if tf_idx is None:
        return None

    # Parse time mode and mask mode (between domain and TF)
    time_mask_parts = remaining[time_start_idx:tf_idx]

    # Mask mode is always last before TF
    if len(time_mask_parts) < 2:
        return None

    mask_str = time_mask_parts[-1]
    time_parts = time_mask_parts[:-1]

    # Parse building mask
    if mask_str == 'masked':
        building_mask = True
    elif mask_str == 'unmasked':
        building_mask = False
    else:
        return None  # Invalid mask mode

    # Parse time mode
    time_params = {}
    if len(time_parts) == 2 and time_parts[0] == 'all' and time_parts[1] == 'times':
        # Legacy format: "all_times" → should be "all_times_average"
        time_mode = 'all_times_average'
    elif len(time_parts) == 3 and time_parts[0] == 'all' and time_parts[1] == 'times' and time_parts[2] == 'average':
        time_mode = 'all_times_average'
    elif len(time_parts) == 4 and time_parts[0] == 'time' and time_parts[1] == 'window':
        time_mode = 'time_window'
        try:
            time_params = {'start': int(time_parts[2]), 'end': int(time_parts[3])}
        except ValueError:
            return None
    elif len(time_parts) == 3 and time_parts[0] == 'single' and time_parts[1] == 'time':
        time_mode = 'single_time'
        try:
            time_params = {'hour': int(time_parts[2])}
        except ValueError:
            return None
    else:
        return None  # Unknown time mode

    # Parse offsets from TF string
    tf_str = remaining[tf_idx]
    if not tf_str.startswith('TF'):
        return None

    offset_str = tf_str[2:]  # Remove 'TF' prefix

    try:
        if '-' in offset_str:
            # Could be range (TF0-5) or list (TF0-1-5-10)
            offset_parts = offset_str.split('-')
            if len(offset_parts) == 2:
                # Range format TF0-5
                start, end = int(offset_parts[0]), int(offset_parts[1])
                offsets = list(range(start, end + 1))
            else:
                # List format TF0-1-5-10
                offsets = [int(x) for x in offset_parts]
        else:
            # Single offset TF0
            offsets = [int(offset_str)]
    except ValueError:
        return None

    return {
        'case_name': case_name,
        'domain_type': domain_type,
        'time_mode': time_mode,
        'time_params': time_params,
        'building_mask': building_mask,
        'offsets': offsets
    }


def find_existing_mask_file(cache_dir: Path,
                            case_name: str,
                            domain_type: str) -> Optional[Path]:
    """
    Find existing terrain mask file for case/domain (legacy function).

    Searches for files matching the pattern and returns the most recent if multiple found.

    Note: This is the legacy function that doesn't consider time_mode or building_mask.
    For new code, use find_existing_mask_file_enhanced() instead.

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


def find_existing_mask_file_enhanced(
    cache_dir: Path,
    case_name: str,
    domain_type: str,
    time_mode: str = 'all_times_average',
    time_params: Optional[Dict] = None,
    building_mask: bool = True,
    fallback_to_legacy: bool = True
) -> Optional[Path]:
    """
    Find existing terrain mask file with enhanced naming convention.

    Searches for cache files matching specific time_mode and building_mask settings.
    Can optionally fall back to legacy naming if enhanced cache not found.

    Args:
        cache_dir: Cache directory to search
        case_name: Simulation case name
        domain_type: 'parent' or 'child'
        time_mode: Time selection mode ('all_times_average', 'time_window', 'single_time')
        time_params: Time parameters (for time_window or single_time modes)
        building_mask: Building mask setting (True = masked, False = unmasked)
        fallback_to_legacy: If True, try legacy naming if enhanced not found

    Returns:
        Path to mask file if found, else None

    Examples:
        >>> cache_dir = Path('./cache/terrain_masks')
        >>>
        >>> # Find all_times_average masked cache
        >>> mask_path = find_existing_mask_file_enhanced(
        ...     cache_dir, 'thf_base', 'parent',
        ...     time_mode='all_times_average',
        ...     building_mask=True
        ... )
        >>>
        >>> # Find time_window unmasked cache
        >>> mask_path = find_existing_mask_file_enhanced(
        ...     cache_dir, 'thf_base', 'child',
        ...     time_mode='time_window',
        ...     time_params={'start': 30, 'end': 42},
        ...     building_mask=False
        ... )
    """
    if not cache_dir.exists():
        return None

    # Domain suffix
    if domain_type == 'parent':
        domain_suffix = ''
    else:
        domain_suffix = f'_{domain_type}'

    # Build time string
    if time_mode == 'all_times_average':
        time_str = 'all_times_average'
    elif time_mode == 'time_window':
        if time_params and 'start' in time_params and 'end' in time_params:
            time_str = f"time_window_{time_params['start']}_{time_params['end']}"
        else:
            return None  # Invalid time_params
    elif time_mode == 'single_time':
        if time_params and 'hour' in time_params:
            time_str = f"single_time_{time_params['hour']}"
        else:
            return None  # Invalid time_params
    else:
        return None  # Unknown time_mode

    # Build mask string
    mask_str = 'masked' if building_mask else 'unmasked'

    # Search pattern for enhanced naming
    # Example: thf_base_terrain_mask_parent_all_times_average_masked_TF*.nc
    pattern = f"{case_name}_terrain_mask{domain_suffix}_{time_str}_{mask_str}_TF*.nc"

    matches = list(cache_dir.glob(pattern))

    if len(matches) > 0:
        # Found enhanced cache file(s)
        if len(matches) == 1:
            return matches[0]
        else:
            # Multiple matches (different TF ranges), return most recent
            return max(matches, key=lambda p: p.stat().st_mtime)

    # No enhanced cache found
    if fallback_to_legacy:
        # Try legacy naming (no time_mode or mask_mode in filename)
        return find_existing_mask_file(cache_dir, case_name, domain_type)
    else:
        return None


def generate_surface_data_filename(
    case_name: str,
    domain_type: str,
    time_mode: str = 'all_times_average',
    time_params: Optional[Dict] = None,
    suffix: str = '.nc'
) -> str:
    """
    Generate filename for surface data cache file.

    Surface data files store time-averaged 2D surface variables (av_xy).

    Args:
        case_name: Simulation case name
        domain_type: 'parent' or 'child'
        time_mode: Time selection mode
        time_params: Time parameters (for time_window or single_time modes)
        suffix: File suffix (default '.nc')

    Returns:
        Filename string

    Examples:
        >>> generate_surface_data_filename('thf_base', 'parent')
        'thf_base_surface_data_parent_all_times_average.nc'

        >>> generate_surface_data_filename(
        ...     'thf_base', 'child',
        ...     time_mode='time_window',
        ...     time_params={'start': 30, 'end': 42}
        ... )
        'thf_base_surface_data_child_time_window_30_42.nc'
    """
    # Domain suffix
    if domain_type == 'parent':
        domain_suffix = ''
    else:
        domain_suffix = f'_{domain_type}'

    # Time mode string (same logic as terrain mask)
    if time_mode == 'all_times_average':
        time_str = 'all_times_average'
    elif time_mode == 'time_window':
        if time_params and 'start' in time_params and 'end' in time_params:
            time_str = f"time_window_{time_params['start']}_{time_params['end']}"
        else:
            raise ValueError("time_window mode requires time_params with 'start' and 'end'")
    elif time_mode == 'single_time':
        if time_params and 'hour' in time_params:
            time_str = f"single_time_{time_params['hour']}"
        else:
            raise ValueError("single_time mode requires time_params with 'hour'")
    else:
        raise ValueError(f"Unknown time_mode: {time_mode}")

    return f"{case_name}_surface_data{domain_suffix}_{time_str}{suffix}"


def find_existing_surface_data_file(
    cache_dir: Path,
    case_name: str,
    domain_type: str,
    time_mode: str = 'all_times_average',
    time_params: Optional[Dict] = None
) -> Optional[Path]:
    """
    Find existing surface data file with specific time mode.

    Args:
        cache_dir: Cache directory to search
        case_name: Simulation case name
        domain_type: 'parent' or 'child'
        time_mode: Time selection mode
        time_params: Time parameters

    Returns:
        Path to surface data file if found, else None
    """
    if not cache_dir.exists():
        return None

    # Build expected filename
    try:
        filename = generate_surface_data_filename(
            case_name, domain_type, time_mode, time_params
        )
    except ValueError:
        return None

    filepath = cache_dir / filename

    return filepath if filepath.exists() else None


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
