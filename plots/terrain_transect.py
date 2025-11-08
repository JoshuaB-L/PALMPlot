"""
Terrain Transect Plotter for fig_6
Visualizes terrain-following transects of temperature and water vapor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple, Optional, Union
import logging
import xarray as xr
from pathlib import Path

from .base_plotter import BasePlotter

# NEW: Import mask caching modules
try:
    from ..core.terrain_mask_io import TerrainMaskWriter, TerrainMaskReader
    from ..core.surface_data_io import (
        SurfaceDataWriter,
        SurfaceDataReader,
        generate_surface_data_filename,
        find_existing_surface_data_file
    )
    from ..utils.netcdf_utils import (
        generate_mask_filename,
        find_existing_mask_file,
        parse_offset_specification,
        copy_netcdf_metadata
    )
    MASK_CACHE_AVAILABLE = True
    SURFACE_DATA_CACHE_AVAILABLE = True
except ImportError as e:
    # Graceful degradation if modules not available
    MASK_CACHE_AVAILABLE = False
    SURFACE_DATA_CACHE_AVAILABLE = False
    import warnings
    warnings.warn(f"Mask caching modules not available: {e}")

# Import VariableMetadata for multi-variable support
try:
    from ..core.variable_metadata import VariableMetadata
    VARIABLE_METADATA_AVAILABLE = True
except ImportError as e:
    VARIABLE_METADATA_AVAILABLE = False
    import warnings
    warnings.warn(f"VariableMetadata not available: {e}")


class TerrainTransectPlotter(BasePlotter):
    """Creates visualizations for terrain-following transect analysis"""

    def __init__(self, config: Dict, output_manager):
        """
        Initialize terrain transect plotter

        Args:
            config: Configuration dictionary
            output_manager: Output manager instance
        """
        super().__init__(config, output_manager)
        self.logger = logging.getLogger(__name__)

        # Initialize variable metadata system
        if VARIABLE_METADATA_AVAILABLE:
            self.var_metadata = VariableMetadata(config, self.logger)
            self.logger.info("VariableMetadata system initialized")
        else:
            self.var_metadata = None
            self.logger.warning("VariableMetadata not available, using legacy mode")

        # Cache for terrain masks to avoid recomputation
        self._terrain_mask_cache = {}

    def available_plots(self) -> List[str]:
        """
        Return list of available plot types.

        Dynamically generates plot types from configuration based on:
        - variables: List of variables to plot
        - plot_matrix: domains and comparisons to generate
        - variable_overrides: Per-variable domain/comparison restrictions

        Returns:
            List of plot type strings in format: "{variable}_{domain}_{comparison}"
            e.g., ["temperature_parent_age", "utci_child_age", ...]
        """
        # Get full figure configuration (not just settings sub-block)
        plots_section = self.config['plots'][self._plots_key]
        fig_config = plots_section.get('fig_6', {})

        # Check if using new format (variables + plot_matrix at fig_6 level)
        if 'variables' in fig_config and 'plot_matrix' in fig_config:
            return self._generate_dynamic_plots(fig_config)
        else:
            # Fallback to legacy hard-coded list
            self.logger.warning("Using legacy plot type list. Consider migrating to new format.")
            return [
                "ta_parent_age",
                "ta_parent_spacing",
                "ta_child_age",
                "ta_child_spacing",
                "qv_parent_age",
                "qv_parent_spacing",
                "qv_child_age",
                "qv_child_spacing"
            ]

    def _generate_dynamic_plots(self, fig_config: Dict) -> List[str]:
        """
        Generate plot types dynamically from configuration.

        Args:
            fig_config: fig_6 configuration dictionary (full, not just settings)

        Returns:
            List of plot type strings
        """
        variables = fig_config.get('variables', [])
        plot_matrix = fig_config.get('plot_matrix', {})
        domains = plot_matrix.get('domains', ['parent'])
        comparisons = plot_matrix.get('comparisons', ['age'])
        overrides = fig_config.get('variable_overrides', {})

        plots = []
        for var in variables:
            # Apply per-variable overrides if they exist
            var_domains = overrides.get(var, {}).get('domains', domains)
            var_comps = overrides.get(var, {}).get('comparisons', comparisons)

            # Generate all combinations for this variable
            for domain in var_domains:
                for comp in var_comps:
                    plot_type = f"{var}_{domain}_{comp}"
                    plots.append(plot_type)

        self.logger.info(f"Generated {len(plots)} dynamic plot types: {plots}")
        return plots

    def generate_plot(self, plot_type: str, data: Dict) -> plt.Figure:
        """
        Generate specific plot type.

        Supports both dynamic format ("temperature_parent_age") and legacy format ("ta_parent_age").

        Args:
            plot_type: Type of plot to generate
            data: Loaded simulation data

        Returns:
            Matplotlib figure
        """
        settings = self._get_plot_settings('fig_6')

        # Try to parse as dynamic plot type: "{variable}_{domain}_{comparison}"
        parsed = self._parse_plot_type(plot_type)

        if parsed:
            variable, domain, comparison = parsed
            self.logger.info(f"Generating dynamic plot: variable={variable}, domain={domain}, comparison={comparison}")
            return self._create_comparison_plot(data, variable, domain, comparison, settings)
        else:
            # Fallback to legacy hard-coded methods
            self.logger.warning(f"Falling back to legacy plot method for: {plot_type}")
            plot_method_map = {
                "ta_parent_age": self._plot_ta_parent_age,
                "ta_parent_spacing": self._plot_ta_parent_spacing,
                "ta_child_age": self._plot_ta_child_age,
                "ta_child_spacing": self._plot_ta_child_spacing,
                "qv_parent_age": self._plot_qv_parent_age,
                "qv_parent_spacing": self._plot_qv_parent_spacing,
                "qv_child_age": self._plot_qv_child_age,
                "qv_child_spacing": self._plot_qv_child_spacing
            }

            if plot_type not in plot_method_map:
                raise ValueError(f"Unknown plot type: {plot_type}")

            return plot_method_map[plot_type](data)

    def _parse_plot_type(self, plot_type: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse dynamic plot type string into components.

        Args:
            plot_type: Plot type string (e.g., "temperature_parent_age", "radiation_net_child_spacing")

        Returns:
            Tuple of (variable, domain, comparison) or None if parsing fails
        """
        parts = plot_type.split('_')

        # Need at least 3 parts
        if len(parts) < 3:
            return None

        # Last two parts are domain and comparison
        comparison = parts[-1]
        domain = parts[-2]

        # Everything before that is the variable name (handles multi-word variables)
        variable = '_'.join(parts[:-2])

        # Validate domain and comparison
        valid_domains = ['parent', 'child']
        valid_comparisons = ['age', 'spacing']

        if domain not in valid_domains or comparison not in valid_comparisons:
            return None

        # Validate variable exists in metadata (if available)
        if self.var_metadata:
            try:
                self.var_metadata.get_variable_config(variable)
            except KeyError:
                self.logger.warning(f"Variable '{variable}' not found in metadata")
                return None

        return (variable, domain, comparison)

    def _find_variable_in_dataset(self, dataset, variable: str) -> Tuple[object, str]:
        """
        Find variable in dataset with wildcard support.

        Phase 5: Variable Discovery with Wildcard Support

        Args:
            dataset: xarray Dataset
            variable: Variable name from config (e.g., 'temperature', 'utci')

        Returns:
            Tuple of (xarray DataArray, actual PALM variable name)

        Raises:
            KeyError: If variable not found
        """
        if self.var_metadata:
            # Use metadata system for variable discovery (supports wildcards)
            return self.var_metadata.find_variable_in_dataset(dataset, variable)
        else:
            # Fallback to legacy hard-coded search for backward compatibility
            # Try ta, qv, theta
            for var_name in ['ta', 'qv', 'theta']:
                if var_name in dataset:
                    self.logger.debug(f"Found variable (legacy): {var_name}")
                    return dataset[var_name], var_name

            # Not found
            available_vars = list(dataset.data_vars.keys())
            raise KeyError(
                f"Variable '{variable}' not found using legacy search. "
                f"Available variables: {available_vars[:20]}..."
            )

    # ========================================================================
    # Plot Type Methods
    # ========================================================================

    def _plot_ta_parent_age(self, data: Dict) -> plt.Figure:
        """Parent domain temperature transect, varying ages"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'ta', 'parent', 'age', settings
        )

    def _plot_ta_parent_spacing(self, data: Dict) -> plt.Figure:
        """Parent domain temperature transect, varying spacings"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'ta', 'parent', 'spacing', settings
        )

    def _plot_ta_child_age(self, data: Dict) -> plt.Figure:
        """Child domain temperature transect, varying ages"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'ta', 'child', 'age', settings
        )

    def _plot_ta_child_spacing(self, data: Dict) -> plt.Figure:
        """Child domain temperature transect, varying spacings"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'ta', 'child', 'spacing', settings
        )

    def _plot_qv_parent_age(self, data: Dict) -> plt.Figure:
        """Parent domain water vapor transect, varying ages"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'qv', 'parent', 'age', settings
        )

    def _plot_qv_parent_spacing(self, data: Dict) -> plt.Figure:
        """Parent domain water vapor transect, varying spacings"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'qv', 'parent', 'spacing', settings
        )

    def _plot_qv_child_age(self, data: Dict) -> plt.Figure:
        """Child domain water vapor transect, varying ages"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'qv', 'child', 'age', settings
        )

    def _plot_qv_child_spacing(self, data: Dict) -> plt.Figure:
        """Child domain water vapor transect, varying spacings"""
        settings = self._get_plot_settings('fig_6')
        return self._create_comparison_plot(
            data, 'qv', 'child', 'spacing', settings
        )

    # ========================================================================
    # Core Processing Methods
    # ========================================================================

    def _extract_terrain_following_slice(self, dataset: xr.Dataset,
                                        static_dataset: xr.Dataset,
                                        domain_type: str,
                                        variable: str,
                                        z_offset: int,
                                        settings: Dict = None) -> Tuple[np.ndarray, str, bool]:
        """
        Extract TIME-AVERAGED data at constant height above terrain
        Uses xarray for time averaging (matching spatial_cooling.py approach)

        Phase 5 & 6: Now accepts variable parameter for dynamic variable discovery

        Args:
            dataset: xarray Dataset containing 3D averaged data (av_3d or av_xy)
            static_dataset: xarray Dataset containing topography
            domain_type: 'parent' or 'child'
            variable: Variable name from config (e.g., 'temperature', 'utci')
            z_offset: Integer offset from first grid point (0-indexed)
            settings: Plot settings dict (optional, used for time selection parameters)

        Returns:
            Tuple of (2D numpy array [y, x] TIME-AVERAGED, variable_name, needs_kelvin_conversion)
        """
        # Default settings if none provided
        if settings is None:
            settings = {}
        # Get first data level based on domain
        # NOTE: Buildings are very tall (up to 40m), so low-level data is sparse
        # We extract at the lowest height with reasonable data coverage that still shows
        # tree cooling effects while avoiding excessive NaN from buildings
        if domain_type == 'child':
            standard_first_level = 11  # Child domain N02: zu_3d[11] = 21.00m (68% coverage, above terrain+buildings)
        else:
            standard_first_level = 2   # Parent domain: zu_3d[2] = 15.00m (35% coverage, shows open areas)

        # Use consistent near-surface level for all scenarios
        # This balances data availability with capturing urban forest cooling effects
        first_level = standard_first_level
        self.logger.debug(f"Using near-surface z_idx={first_level} for {domain_type} domain")
        self.logger.info(f"Using near-surface data extraction: z_idx={first_level}")

        # Target z-index
        target_z = first_level + z_offset

        # Log detailed information about z-level selection
        self.logger.info(f"Z-level selection for {domain_type} domain:")
        self.logger.info(f"  First valid data level: z_idx={first_level}")
        self.logger.info(f"  Z-offset from config: {z_offset}")
        self.logger.info(f"  Target z-index: {target_z}")

        # For now, extract at constant z-level
        # TODO: Implement true terrain-following extraction using topography
        try:
            # Phase 5: Use dynamic variable discovery with wildcard support
            var_data, var_name_found = self._find_variable_in_dataset(dataset, variable)
            self.logger.info(f"Found variable '{variable}' as PALM variable '{var_name_found}' in dataset")

            # Validate target z-index is within bounds
            if 'zu_3d' in dataset.dims or 'zw_3d' in dataset.dims:
                z_dim = 'zu_3d' if 'zu_3d' in dataset.dims else 'zw_3d'
                z_coords = dataset[z_dim].values
                max_z_idx = len(z_coords) - 1

                if target_z > max_z_idx:
                    self.logger.error(f"Target z-index {target_z} exceeds maximum {max_z_idx}")
                    raise ValueError(f"Z-offset {z_offset} too large for domain {domain_type}")

                # Log actual height in meters
                actual_height_m = z_coords[target_z]
                self.logger.info(f"  Actual height: {actual_height_m:.2f}m (at zu_3d[{target_z}])")

                # Extract at target z-level (still has time dimension)
                slice_data_with_time = var_data.isel({z_dim: target_z})
                self.logger.debug(f"Extracted slice with time, shape: {slice_data_with_time.shape}")
            else:
                # Fall back to direct indexing
                slice_data_with_time = var_data[:, target_z, :, :]
                self.logger.debug(f"Extracted slice with time (direct), shape: {slice_data_with_time.shape}")

            # TIME AVERAGING: Use xarray.mean() on time dimension (matching spatial_cooling.py)
            # This is the KEY difference - do time averaging using xarray BEFORE converting to numpy

            # Get time selection method and parameters from settings
            time_selection_method = settings.get('time_selection_method', 'mean')
            total_time_steps = slice_data_with_time.shape[0]

            # Log time selection configuration (INFO level for visibility)
            msg = f"=== TIME SELECTION CONFIGURATION ==="
            print(msg)
            self.logger.info(msg)

            msg = f"  Domain: {domain_type}, Variable: {var_name_found}"
            print(msg)
            self.logger.info(msg)

            msg = f"  Total available time steps: {total_time_steps}"
            print(msg)
            self.logger.info(msg)

            msg = f"  Method: '{time_selection_method}'"
            print(msg)
            self.logger.info(msg)

            # Step 1: Apply time selection method
            skip_averaging = False  # Flag for single_timestep method

            if time_selection_method == 'single_timestep':
                # Extract a SINGLE time step (no averaging)
                time_index = settings.get('time_index', 0)

                # Validate time index
                if time_index < 0:
                    self.logger.warning(f"time_index={time_index} is negative, using 0")
                    time_index = 0
                if time_index >= total_time_steps:
                    self.logger.warning(f"time_index={time_index} exceeds available time steps ({total_time_steps}), using {total_time_steps - 1}")
                    time_index = total_time_steps - 1

                # Select single time step
                slice_data_with_time = slice_data_with_time.isel(time=[time_index])
                skip_averaging = True  # Don't average, just extract
                self.logger.info(f"  Selected single time step: {time_index}")
                self.logger.info(f"  No time averaging will be performed")

            elif time_selection_method == 'mean_timeframe':
                # User-specified time range for averaging
                time_start = settings.get('time_start', 0)
                time_end = settings.get('time_end', total_time_steps - 1)

                # Validate time range
                if time_start < 0:
                    self.logger.warning(f"time_start={time_start} is negative, using 0")
                    time_start = 0
                if time_end >= total_time_steps:
                    self.logger.warning(f"time_end={time_end} exceeds available time steps ({total_time_steps}), using {total_time_steps - 1}")
                    time_end = total_time_steps - 1
                if time_start > time_end:
                    self.logger.error(f"time_start ({time_start}) > time_end ({time_end}), using all time steps")
                    time_start = 0
                    time_end = total_time_steps - 1

                # Select time range
                time_indices = list(range(time_start, time_end + 1))
                slice_data_with_time = slice_data_with_time.isel(time=time_indices)

                msg = f"  Time range: steps {time_start} to {time_end} ({len(time_indices)} steps)"
                print(msg)
                self.logger.info(msg)

                msg = f"  Will average over selected time steps"
                print(msg)
                self.logger.info(msg)

            elif time_selection_method == 'mean':
                # Use all available time steps (default)
                self.logger.info(f"  Using all {total_time_steps} time steps")
                self.logger.info(f"  Will average over all time steps")

            else:
                # Unknown method, log warning and use default
                self.logger.warning(f"Unknown time_selection_method '{time_selection_method}', using 'mean' (all time steps)")
                self.logger.info(f"  Fallback: Using all {total_time_steps} time steps")

            # Step 2: CRITICAL FIX - Check for corrupted time steps with abnormally low values
            # (Skip for single_timestep since we're extracting one specific step)
            if not skip_averaging:
                # Sample a few non-building locations to detect corrupted time steps
                test_locations = [(50, 50), (75, 75), (100, 100)]
                corrupted_time_indices = set()

                self.logger.info(f"=== CORRUPTED STEP DETECTION ===")
                for y, x in test_locations:
                    test_series = slice_data_with_time.isel(y=y, x=x).values
                    # Find time steps where temperature is suspiciously low (<5°C) or zero
                    # Normal temperatures should be in range 15-40°C for this simulation
                    suspicious_mask = (test_series < 5.0) & (~np.isnan(test_series))
                    suspicious_indices = np.where(suspicious_mask)[0]
                    corrupted_time_indices.update(suspicious_indices.tolist())

                if len(corrupted_time_indices) > 0:
                    corrupted_list = sorted(list(corrupted_time_indices))
                    self.logger.warning(f"  Found {len(corrupted_list)} corrupted time step(s) with suspiciously low temperatures: {corrupted_list}")
                    self.logger.warning(f"  Excluding these time steps from averaging")

                    # Create mask of valid (non-corrupted) time steps
                    all_time_indices_in_range = list(range(slice_data_with_time.shape[0]))
                    valid_time_indices = [i for i in all_time_indices_in_range if i not in corrupted_time_indices]

                    if len(valid_time_indices) == 0:
                        self.logger.error("  All time steps in selected range are corrupted!")
                        raise ValueError("No valid time steps available for averaging")

                    # Select only valid time steps before averaging
                    slice_data_with_time = slice_data_with_time.isel(time=valid_time_indices)
                    self.logger.info(f"  Using {len(valid_time_indices)} valid time step(s) out of {len(all_time_indices_in_range)}")
                else:
                    self.logger.info(f"  No corrupted time steps detected")

            # Step 3: Perform time averaging (or extract single step)
            if skip_averaging:
                # Single timestep - just squeeze the time dimension
                slice_time_avg = slice_data_with_time.isel(time=0)
                self.logger.info(f"=== EXTRACTION COMPLETE ===")
                self.logger.info(f"  Single time step extracted (no averaging)")
                self.logger.info(f"  Output shape: {slice_time_avg.shape}")
            else:
                # Average over time dimension
                slice_time_avg = slice_data_with_time.mean(dim='time')
                self.logger.info(f"=== TIME AVERAGING COMPLETE ===")
                self.logger.info(f"  Averaged over {slice_data_with_time.shape[0]} time step(s)")
                self.logger.info(f"  Output shape: {slice_time_avg.shape}")

            # Now convert to numpy array [y, x]
            slice_array = slice_time_avg.values

            # CRITICAL FIX: Replace zeros with NaN (building locations)
            # The base case dataset has a bug where xarray.mean(dim='time') converts NaN to 0
            # at building locations, while forested scenarios preserve NaN correctly.
            # To ensure consistent behavior, replace any zeros with NaN.
            zero_count_before = np.sum(slice_array == 0)
            if zero_count_before > 0:
                self.logger.warning(f"Found {zero_count_before} zero values (likely building locations converted from NaN), replacing with NaN")
                slice_array[slice_array == 0] = np.nan

            # Check for NaN or invalid data
            if np.all(np.isnan(slice_array)):
                self.logger.error("Extracted slice contains only NaN values!")
                raise ValueError("Extracted data is all NaN")

            # Log data statistics
            valid_data = slice_array[~np.isnan(slice_array)]
            self.logger.info(f"Time-averaged data for {var_name_found}: "
                           f"shape={slice_array.shape}, "
                           f"min={np.min(valid_data):.2f}, "
                           f"max={np.max(valid_data):.2f}, "
                           f"mean={np.mean(valid_data):.2f}")

            # Determine if data needs Kelvin to Celsius conversion
            # PALM typically outputs ta in Kelvin (values ~297K for 24°C)
            # If mean value > 100, assume it's in Kelvin
            needs_conversion = np.mean(valid_data) > 100.0
            self.logger.info(f"Temperature unit detection: needs_kelvin_conversion={needs_conversion}")

            return slice_array, var_name_found, needs_conversion

        except Exception as e:
            self.logger.error(f"Error extracting terrain-following slice: {str(e)}")
            raise

    def _extract_terrain_following_transect_direct(self, dataset: xr.Dataset,
                                                   static_dataset: xr.Dataset,
                                                   domain_type: str,
                                                   variable: str,
                                                   z_offset: int,
                                                   transect_axis: str,
                                                   transect_location: int,
                                                   transect_width: int,
                                                   settings: Dict = None) -> Tuple[np.ndarray, np.ndarray, str, bool]:
        """
        Extract TIME-AVERAGED 1D transect directly from 4D data [time, z, y, x]
        MEMORY EFFICIENT: ~400× less memory than extracting full 2D slice first

        Phase 5: Now accepts variable parameter for dynamic variable discovery

        This method extracts the transect line directly from the 4D dataset,
        then performs time averaging on the 1D transect only, avoiding the need
        to create and store a full 2D spatial slice.

        Args:
            dataset: xarray Dataset containing 3D averaged data (av_3d or av_xy)
            static_dataset: xarray Dataset containing topography
            domain_type: 'parent' or 'child'
            variable: Variable name from config (e.g., 'temperature', 'utci')
            z_offset: Integer offset from first grid point (0-indexed)
            transect_axis: 'x' or 'y' - direction of transect
            transect_location: Grid index for transect location
            transect_width: Number of grid cells to average on each side
            settings: Plot settings dict (optional, used for time selection parameters)

        Returns:
            Tuple of (1D transect values, coordinates, variable_name, needs_kelvin_conversion)
        """
        # Default settings if none provided
        if settings is None:
            settings = {}

        # Log extraction method
        msg = f"\n=== DIRECT TRANSECT EXTRACTION ==="
        print(msg)
        self.logger.info(msg)

        msg = f"  Domain: {domain_type}"
        print(msg)
        self.logger.info(msg)

        msg = f"  Transect: axis={transect_axis}, location={transect_location}, width=±{transect_width}"
        print(msg)
        self.logger.info(msg)

        # Get first data level based on domain
        # NOTE: Buildings are very tall (up to 40m), so low-level data is sparse
        # We extract at the lowest height with reasonable data coverage
        if domain_type == 'child':
            standard_first_level = 11  # Child domain N02: zu_3d[11] = 21.00m (68% coverage)
        else:
            standard_first_level = 2   # Parent domain: zu_3d[2] = 15.00m (35% coverage)

        first_level = standard_first_level
        target_z = first_level + z_offset

        # Log z-level selection
        msg = f"  Z-level: first={first_level} (near-surface), offset={z_offset}, target={target_z}"
        print(msg)
        self.logger.info(msg)

        try:
            # Phase 5: Use dynamic variable discovery with wildcard support
            var_data, var_name_found = self._find_variable_in_dataset(dataset, variable)
            self.logger.info(f"Found variable '{variable}' as PALM variable '{var_name_found}' in dataset")

            # Validate z-index
            if 'zu_3d' in dataset.dims or 'zw_3d' in dataset.dims:
                z_dim = 'zu_3d' if 'zu_3d' in dataset.dims else 'zw_3d'
                z_coords = dataset[z_dim].values
                max_z_idx = len(z_coords) - 1

                if target_z > max_z_idx:
                    self.logger.error(f"Target z-index {target_z} exceeds maximum {max_z_idx}")
                    raise ValueError(f"Z-offset {z_offset} too large for domain {domain_type}")

                actual_height_m = z_coords[target_z]
                msg = f"  Height: {actual_height_m:.2f}m (zu_3d[{target_z}])"
                print(msg)
                self.logger.info(msg)
            else:
                z_dim = None  # Will use direct indexing

            # Get domain dimensions
            if transect_axis == 'x':
                ny, nx = var_data.shape[-2:]
                transect_length = nx
                y_min = max(0, transect_location - transect_width)
                y_max = min(ny, transect_location + transect_width + 1)

                msg = f"  Averaging over y=[{y_min}:{y_max}], extracting along x (length={transect_length})"
                print(msg)
                self.logger.info(msg)
            elif transect_axis == 'y':
                ny, nx = var_data.shape[-2:]
                transect_length = ny
                x_min = max(0, transect_location - transect_width)
                x_max = min(nx, transect_location + transect_width + 1)

                msg = f"  Averaging over x=[{x_min}:{x_max}], extracting along y (length={transect_length})"
                print(msg)
                self.logger.info(msg)
            else:
                raise ValueError(f"Invalid transect_axis: {transect_axis}. Must be 'x' or 'y'")

            # STEP 1: Extract 1D transect from spatial dimensions (still has time dimension)
            # This is the KEY efficiency gain: extract 1D instead of 2D
            if z_dim is not None:
                # Use xarray selection
                if transect_axis == 'x':
                    # Extract: var_data[time, target_z, y_min:y_max, :]
                    # Then average over y dimension to get [time, x]
                    transect_with_time = var_data.isel({z_dim: target_z, 'y': slice(y_min, y_max)}).mean(dim='y')
                else:  # transect_axis == 'y'
                    # Extract: var_data[time, target_z, :, x_min:x_max]
                    # Then average over x dimension to get [time, y]
                    transect_with_time = var_data.isel({z_dim: target_z, 'x': slice(x_min, x_max)}).mean(dim='x')
            else:
                # Direct indexing fallback
                if transect_axis == 'x':
                    transect_with_time = np.mean(var_data[:, target_z, y_min:y_max, :], axis=1)
                else:
                    transect_with_time = np.mean(var_data[:, target_z, :, x_min:x_max], axis=2)
                # Convert to xarray DataArray for consistent handling
                transect_with_time = xr.DataArray(transect_with_time, dims=['time', transect_axis])

            msg = f"  Transect extracted from spatial dims: shape={transect_with_time.shape}"
            print(msg)
            self.logger.info(msg)

            # STEP 2: TIME SELECTION AND AVERAGING
            # Get time selection method and parameters from settings
            time_selection_method = settings.get('time_selection_method', 'mean')
            total_time_steps = transect_with_time.shape[0]

            # Log time selection configuration
            msg = f"\n=== TIME SELECTION CONFIGURATION ==="
            print(msg)
            self.logger.info(msg)

            msg = f"  Domain: {domain_type}, Variable: {var_name_found}"
            print(msg)
            self.logger.info(msg)

            msg = f"  Total available time steps: {total_time_steps}"
            print(msg)
            self.logger.info(msg)

            msg = f"  Method: '{time_selection_method}'"
            print(msg)
            self.logger.info(msg)

            # Apply time selection method
            skip_averaging = False

            if time_selection_method == 'single_timestep':
                # Extract a SINGLE time step (no averaging)
                time_index = settings.get('time_index', 0)

                # Validate time index
                if time_index < 0:
                    self.logger.warning(f"time_index={time_index} is negative, using 0")
                    time_index = 0
                if time_index >= total_time_steps:
                    self.logger.warning(f"time_index={time_index} exceeds available steps ({total_time_steps}), using {total_time_steps - 1}")
                    time_index = total_time_steps - 1

                # Select single time step
                transect_with_time = transect_with_time.isel(time=[time_index])
                skip_averaging = True

                msg = f"  Selected single time step: {time_index}"
                print(msg)
                self.logger.info(msg)

                msg = f"  No time averaging will be performed"
                print(msg)
                self.logger.info(msg)

            elif time_selection_method == 'mean_timeframe':
                # User-specified time range for averaging
                time_start = settings.get('time_start', 0)
                time_end = settings.get('time_end', total_time_steps - 1)

                # Validate time range
                if time_start < 0:
                    self.logger.warning(f"time_start={time_start} is negative, using 0")
                    time_start = 0
                if time_end >= total_time_steps:
                    self.logger.warning(f"time_end={time_end} exceeds available steps ({total_time_steps}), using {total_time_steps - 1}")
                    time_end = total_time_steps - 1
                if time_start > time_end:
                    self.logger.error(f"time_start ({time_start}) > time_end ({time_end}), using all time steps")
                    time_start = 0
                    time_end = total_time_steps - 1

                # Select time range
                time_indices = list(range(time_start, time_end + 1))
                transect_with_time = transect_with_time.isel(time=time_indices)

                msg = f"  Time range: steps {time_start} to {time_end} ({len(time_indices)} steps)"
                print(msg)
                self.logger.info(msg)

                msg = f"  Will average over selected time steps"
                print(msg)
                self.logger.info(msg)

            elif time_selection_method == 'mean':
                # Use all available time steps (default)
                msg = f"  Using all {total_time_steps} time steps"
                print(msg)
                self.logger.info(msg)

                msg = f"  Will average over all time steps"
                print(msg)
                self.logger.info(msg)

            else:
                # Unknown method, log warning and use default
                self.logger.warning(f"Unknown time_selection_method '{time_selection_method}', using 'mean'")
                msg = f"  Fallback: Using all {total_time_steps} time steps"
                print(msg)
                self.logger.info(msg)

            # STEP 3: CORRUPTED STEP DETECTION (only for averaging methods)
            if not skip_averaging:
                # Sample a few locations to detect corrupted time steps
                test_locations = [int(transect_length * 0.25), int(transect_length * 0.5), int(transect_length * 0.75)]
                corrupted_time_indices = set()

                msg = f"\n=== CORRUPTED STEP DETECTION ==="
                print(msg)
                self.logger.info(msg)

                for idx in test_locations:
                    if idx < transect_length:
                        test_series = transect_with_time.isel({transect_axis: idx}).values
                        # Find time steps with suspiciously low temperatures (<5°C)
                        suspicious_mask = (test_series < 5.0) & (~np.isnan(test_series))
                        suspicious_indices = np.where(suspicious_mask)[0]
                        corrupted_time_indices.update(suspicious_indices.tolist())

                if len(corrupted_time_indices) > 0:
                    corrupted_list = sorted(list(corrupted_time_indices))

                    msg = f"  Found {len(corrupted_list)} corrupted time step(s) with suspiciously low temperatures: {corrupted_list}"
                    print(msg)
                    self.logger.warning(msg)

                    msg = f"  Excluding these time steps from averaging"
                    print(msg)
                    self.logger.warning(msg)

                    # Create mask of valid time steps
                    all_time_indices = list(range(transect_with_time.shape[0]))
                    valid_time_indices = [i for i in all_time_indices if i not in corrupted_time_indices]

                    if len(valid_time_indices) == 0:
                        self.logger.error("  All time steps in selected range are corrupted!")
                        raise ValueError("No valid time steps available for averaging")

                    # Select only valid time steps
                    transect_with_time = transect_with_time.isel(time=valid_time_indices)

                    msg = f"  Using {len(valid_time_indices)} valid time step(s) out of {len(all_time_indices)}"
                    print(msg)
                    self.logger.info(msg)
                else:
                    msg = f"  No corrupted time steps detected"
                    print(msg)
                    self.logger.info(msg)

            # STEP 4: Perform time averaging (or extract single step)
            msg = f"\n=== TIME PROCESSING ==="
            print(msg)
            self.logger.info(msg)

            if skip_averaging:
                # Single timestep - just squeeze the time dimension
                transect_final = transect_with_time.isel(time=0)

                msg = f"  Single time step extracted (no averaging)"
                print(msg)
                self.logger.info(msg)
            else:
                # Average over time dimension
                transect_final = transect_with_time.mean(dim='time')

                msg = f"  Averaged over {transect_with_time.shape[0]} time step(s)"
                print(msg)
                self.logger.info(msg)

            # Convert to numpy array
            transect_array = transect_final.values

            msg = f"  Output shape: {transect_array.shape}"
            print(msg)
            self.logger.info(msg)

            # Replace zeros with NaN (building locations)
            zero_count = np.sum(transect_array == 0)
            if zero_count > 0:
                self.logger.warning(f"  Found {zero_count} zero values (likely building locations), replacing with NaN")
                transect_array[transect_array == 0] = np.nan

            # Validate data
            if np.all(np.isnan(transect_array)):
                self.logger.error("Extracted transect contains only NaN values!")
                raise ValueError("Extracted transect is all NaN")

            # Log statistics
            valid_data = transect_array[~np.isnan(transect_array)]
            msg = f"  Data stats: min={np.min(valid_data):.2f}, max={np.max(valid_data):.2f}, mean={np.mean(valid_data):.2f}"
            print(msg)
            self.logger.info(msg)

            # Determine if data needs Kelvin to Celsius conversion
            needs_conversion = np.mean(valid_data) > 100.0

            msg = f"  Temperature unit detection: needs_kelvin_conversion={needs_conversion}"
            print(msg)
            self.logger.info(msg)

            # Generate coordinates
            coordinates = np.arange(transect_length)

            msg = f"\n=== DIRECT TRANSECT EXTRACTION COMPLETE ==="
            print(msg)
            self.logger.info(msg)

            return transect_array, coordinates, var_name_found, needs_conversion

        except Exception as e:
            self.logger.error(f"Error extracting direct transect: {str(e)}")
            raise

    # ========================================================================
    # Fill Value Detection Utilities (for terrain-following extraction)
    # ========================================================================

    def _detect_fill_values(self, data_array: xr.DataArray) -> Tuple[bool, float]:
        """
        Detect fill value from xarray DataArray attributes or use common defaults.

        PALM-LES data may use various conventions for marking missing/invalid data:
        - NaN (numpy/xarray convention)
        - _FillValue attribute (NetCDF convention)
        - missing_value attribute (alternative NetCDF convention)
        - Common sentinel values: -9999, -127

        Args:
            data_array: xarray DataArray to analyze

        Returns:
            Tuple of (has_fill_value, fill_value)
            - has_fill_value: Always True (defaults to NaN if no explicit fill value)
            - fill_value: The detected or default fill value
        """
        # Check for _FillValue attribute (NetCDF standard)
        if '_FillValue' in data_array.attrs:
            fill_val = float(data_array.attrs['_FillValue'])
            self.logger.debug(f"Detected _FillValue attribute: {fill_val}")
            return True, fill_val

        # Check for missing_value attribute (alternative NetCDF convention)
        if 'missing_value' in data_array.attrs:
            fill_val = float(data_array.attrs['missing_value'])
            self.logger.debug(f"Detected missing_value attribute: {fill_val}")
            return True, fill_val

        # Check for common sentinel values in the data
        # Note: Only check a sample to avoid expensive operations on large arrays
        sample_data = data_array.values.ravel()[:10000]  # Sample first 10k values
        sample_data_clean = sample_data[~np.isnan(sample_data)]  # Remove NaN for analysis

        if len(sample_data_clean) > 0:
            # Check for -9999 (common fill value)
            if np.any(np.isclose(sample_data_clean, -9999)):
                self.logger.debug("Detected -9999 fill value in data")
                return True, -9999.0

            # Check for -127 (sometimes used for integer data)
            if np.any(np.isclose(sample_data_clean, -127)):
                self.logger.debug("Detected -127 fill value in data")
                return True, -127.0

        # Default to NaN (xarray/numpy convention)
        self.logger.debug("No explicit fill value detected, defaulting to NaN")
        return True, np.nan

    def _is_fill_value(self, value: float, fill_value: float) -> bool:
        """
        Check if a value is a fill value (handles NaN comparison correctly).

        Args:
            value: Value to check
            fill_value: Reference fill value

        Returns:
            True if value matches fill_value (including NaN==NaN case)
        """
        # Handle NaN comparison (NaN != NaN in normal comparison)
        if np.isnan(fill_value):
            return np.isnan(value)
        else:
            # Use isclose for floating point comparison to handle precision issues
            # Also check exact equality for integer-like fill values
            return np.isclose(value, fill_value, rtol=1e-9, atol=1e-9) or value == fill_value

    # ========================================================================
    # Terrain Mask Caching Helper Methods
    # ========================================================================

    def _should_use_mask_cache(self, settings: Dict) -> Tuple[bool, str]:
        """
        Determine if mask caching should be used.

        Args:
            settings: Plot settings dictionary

        Returns:
            (use_cache, mode) where mode is 'save', 'load', 'auto', or 'disabled'
        """
        # DEBUG: Log MASK_CACHE_AVAILABLE status
        self.logger.debug(f"MASK_CACHE_AVAILABLE = {MASK_CACHE_AVAILABLE}")

        if not MASK_CACHE_AVAILABLE:
            self.logger.info("Mask caching disabled: modules not available")
            return False, 'disabled'

        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('mask_cache', {})

        # DEBUG: Log settings structure
        self.logger.debug(f"terrain_following settings keys: {list(tf_settings.keys())}")
        self.logger.debug(f"mask_cache settings: {cache_settings}")

        if not cache_settings.get('enabled', False):
            self.logger.info(f"Mask caching disabled: enabled={cache_settings.get('enabled', False)}")
            return False, 'disabled'

        mode = cache_settings.get('mode', 'auto')
        self.logger.info(f"Mask caching ENABLED: mode={mode}")
        return True, mode

    def _get_mask_cache_path(self,
                            case_name: str,
                            domain_type: str,
                            settings: Dict) -> Path:
        """
        Get path for mask cache file.

        Args:
            case_name: Simulation case name
            domain_type: 'parent' or 'child'
            settings: Plot settings dictionary

        Returns:
            Path to mask file
        """
        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('mask_cache', {})

        cache_dir = Path(cache_settings.get('cache_directory', './cache/terrain_masks'))
        cache_dir = cache_dir.expanduser().resolve()

        # Get offsets to include in filename
        levels = cache_settings.get('levels', {})
        offsets = levels.get('offsets', [0])

        # Parse offsets if string
        offsets_parsed = parse_offset_specification(
            offsets,
            max_levels=levels.get('max_levels', 20)
        )

        filename = generate_mask_filename(case_name, domain_type, offsets_parsed)

        return cache_dir / filename

    def _save_terrain_mask(self,
                          case_name: str,
                          domain_type: str,
                          mask_data_dict: Dict[str, np.ndarray],
                          source_levels: np.ndarray,
                          coordinates: Dict,
                          settings: Dict,
                          static_dataset: Optional[xr.Dataset] = None) -> None:
        """
        Save terrain-following mask to NetCDF file.

        Args:
            case_name: Simulation case name
            domain_type: 'parent' or 'child'
            mask_data_dict: Dictionary of variable_name -> 3D mask array [ku_above_surf, y, x]
            source_levels: Source zu_3d indices array [y, x]
            coordinates: Coordinate arrays (x, y, ku_above_surf)
            settings: Plot settings
            static_dataset: Optional static dataset for additional metadata
        """
        if not MASK_CACHE_AVAILABLE:
            self.logger.warning("Mask caching modules not available, cannot save mask")
            return

        output_path = self._get_mask_cache_path(case_name, domain_type, settings)

        self.logger.info(f"Saving terrain mask to: {output_path}")

        # Prepare metadata
        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('mask_cache', {})
        domain_settings = tf_settings.get(domain_type, {})

        metadata = {
            'case_name': case_name,
            'domain_type': domain_type,
            'buildings_mask_applied': tf_settings.get('buildings_mask', True),
            'start_z_index': domain_settings.get('start_z_index',
                                                tf_settings.get('start_z_index', 0)),
            'max_z_index': domain_settings.get('max_z_index',
                                              tf_settings.get('max_z_index', 20)),
            'n_levels': coordinates['ku_above_surf'].size,
            'nx': coordinates['x'].size,
            'ny': coordinates['y'].size,
            'resolution': settings.get('resolution', 0.0),
            'author': 'PALMPlot',
        }

        # Add origin info from static dataset if available
        if static_dataset is not None:
            if 'origin_x' in static_dataset.attrs:
                metadata['origin_x'] = static_dataset.attrs['origin_x']
            if 'origin_y' in static_dataset.attrs:
                metadata['origin_y'] = static_dataset.attrs['origin_y']
            if 'origin_z' in static_dataset.attrs:
                metadata['origin_z'] = static_dataset.attrs['origin_z']
            if 'rotation_angle' in static_dataset.attrs:
                metadata['rotation_angle'] = static_dataset.attrs['rotation_angle']

        # Get compression settings
        compression = cache_settings.get('compression', {'enabled': True, 'level': 4})

        # Write mask
        try:
            writer = TerrainMaskWriter(self.logger)
            writer.write_mask(
                output_path=output_path,
                mask_data=mask_data_dict,
                source_levels=source_levels,
                coordinates=coordinates,
                metadata=metadata,
                compression=compression
            )
        except Exception as e:
            self.logger.error(f"Failed to save terrain mask: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            # Don't raise - caching failure shouldn't stop execution

    def _load_terrain_mask(self,
                          case_name: str,
                          domain_type: str,
                          required_variables: List[str],
                          settings: Dict,
                          expected_grid_size: Tuple[int, int]) -> Optional[Dict]:
        """
        Load terrain-following mask from NetCDF file.

        Args:
            case_name: Simulation case name
            domain_type: 'parent' or 'child'
            required_variables: List of variables needed
            settings: Plot settings
            expected_grid_size: Expected (ny, nx) for validation

        Returns:
            Dictionary with mask data, or None if loading failed
        """
        if not MASK_CACHE_AVAILABLE:
            self.logger.warning("Mask caching modules not available, cannot load mask")
            return None

        # Find mask file
        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('mask_cache', {})
        cache_dir = Path(cache_settings.get('cache_directory', './cache/terrain_masks'))

        mask_path = find_existing_mask_file(cache_dir, case_name, domain_type)

        if mask_path is None:
            self.logger.info(f"No cached mask found for {case_name} ({domain_type})")
            return None

        self.logger.info(f"Found cached mask: {mask_path.name}")

        # Load mask
        reader = TerrainMaskReader(self.logger)

        try:
            validation_settings = cache_settings.get('validation', {})
            result = reader.read_mask(
                input_path=mask_path,
                variables=required_variables,
                validate=True,
                validation_settings=validation_settings
            )

            # Check compatibility
            is_compatible, issues = reader.check_mask_compatibility(
                mask_metadata=result['metadata'],
                expected_grid_size=expected_grid_size,
                expected_domain=domain_type,
                validation_settings=validation_settings
            )

            if not is_compatible:
                on_mismatch = validation_settings.get('on_mismatch', 'recompute')
                if on_mismatch == 'error':
                    raise ValueError(f"Mask compatibility check failed: {issues}")
                elif on_mismatch == 'warn':
                    self.logger.warning(
                        f"Mask compatibility issues detected but continuing: {issues}"
                    )
                else:  # recompute
                    self.logger.warning(
                        f"Mask compatibility check failed, will recompute: {issues}"
                    )
                    return None

            self.logger.info(
                f"Successfully loaded mask with {len(result['mask_data'])} variables, "
                f"{result['coordinates']['ku_above_surf'].size} levels"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to load mask: {e}")

            on_mismatch = cache_settings.get('validation', {}).get('on_mismatch', 'recompute')
            if on_mismatch == 'error':
                raise
            else:
                self.logger.warning("Falling back to mask computation")
                return None

    # ========================================================================
    # Surface Data Caching Methods (for av_xy variables)
    # ========================================================================

    def _get_surface_data_cache_path(self,
                                    case_name: str,
                                    domain_type: str,
                                    settings: Dict) -> Path:
        """
        Get path for surface data cache file.

        Args:
            case_name: Simulation case name
            domain_type: 'parent' or 'child'
            settings: Plot settings dictionary

        Returns:
            Path to surface data cache file
        """
        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('surface_data_cache', {})

        cache_dir = Path(cache_settings.get('cache_directory', './cache/surface_data'))
        cache_dir = cache_dir.expanduser().resolve()

        filename = generate_surface_data_filename(case_name, domain_type)

        return cache_dir / filename

    def _save_surface_data(self,
                          case_name: str,
                          domain_type: str,
                          surface_data_dict: Dict[str, np.ndarray],
                          coordinates: Dict,
                          metadata: Dict,
                          settings: Dict) -> None:
        """
        Save time-averaged surface data to NetCDF file.

        Args:
            case_name: Simulation case name
            domain_type: 'parent' or 'child'
            surface_data_dict: Dictionary of variable_name -> 2D array [y, x]
            coordinates: Coordinate arrays (x, y)
            metadata: Metadata dictionary
            settings: Plot settings
        """
        if not SURFACE_DATA_CACHE_AVAILABLE:
            self.logger.warning("Surface data caching modules not available, cannot save")
            return

        output_path = self._get_surface_data_cache_path(case_name, domain_type, settings)

        self.logger.info(f"Saving surface data to: {output_path}")

        # Prepare metadata
        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('surface_data_cache', {})

        # Add domain and grid info to metadata
        metadata['case_name'] = case_name
        metadata['domain_type'] = domain_type
        metadata['nx'] = coordinates['x'].size
        metadata['ny'] = coordinates['y'].size
        metadata['resolution'] = settings.get('resolution', 0.0)
        metadata['author'] = 'PALMPlot'

        # Get compression settings
        compression = cache_settings.get('compression', {'enabled': True, 'level': 4})

        # Write surface data
        try:
            writer = SurfaceDataWriter(self.logger)
            writer.write_surface_data(
                output_path=output_path,
                surface_data=surface_data_dict,
                coordinates=coordinates,
                metadata=metadata,
                compression=compression
            )
        except Exception as e:
            self.logger.error(f"Failed to save surface data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            # Don't raise - caching failure shouldn't stop execution

    def _load_surface_data(self,
                          case_name: str,
                          domain_type: str,
                          required_variables: Optional[List[str]],
                          settings: Dict,
                          expected_grid_size: Tuple[int, int]) -> Optional[Dict]:
        """
        Load time-averaged surface data from NetCDF file.

        Args:
            case_name: Simulation case name
            domain_type: 'parent' or 'child'
            required_variables: Optional list of variables needed (None = all)
            settings: Plot settings
            expected_grid_size: Expected (ny, nx) for validation

        Returns:
            Dictionary with surface data, or None if loading failed
        """
        if not SURFACE_DATA_CACHE_AVAILABLE:
            self.logger.warning("Surface data caching modules not available, cannot load")
            return None

        # Find surface data file
        tf_settings = settings.get('terrain_following', {})
        cache_settings = tf_settings.get('surface_data_cache', {})
        cache_dir = Path(cache_settings.get('cache_directory', './cache/surface_data'))

        surface_data_path = find_existing_surface_data_file(cache_dir, case_name, domain_type)

        if surface_data_path is None:
            self.logger.info(f"No cached surface data found for {case_name} ({domain_type})")
            return None

        self.logger.info(f"Found cached surface data: {surface_data_path.name}")

        # Load surface data
        reader = SurfaceDataReader(self.logger)

        try:
            validation_settings = cache_settings.get('validation', {})
            result = reader.read_surface_data(
                input_path=surface_data_path,
                variables=required_variables,
                validate=True,
                validation_settings=validation_settings
            )

            # Check compatibility
            is_compatible, issues = reader.check_surface_data_compatibility(
                surface_metadata=result['metadata'],
                expected_grid_size=expected_grid_size,
                expected_domain=domain_type,
                validation_settings=validation_settings
            )

            if not is_compatible:
                on_mismatch = validation_settings.get('on_mismatch', 'recompute')
                if on_mismatch == 'error':
                    raise ValueError(f"Surface data compatibility check failed: {issues}")
                elif on_mismatch == 'warn':
                    self.logger.warning(
                        f"Surface data compatibility issues detected but continuing: {issues}"
                    )
                else:  # recompute
                    self.logger.warning(
                        f"Surface data compatibility check failed, will recompute: {issues}"
                    )
                    return None

            self.logger.info(
                f"Successfully loaded surface data with {len(result['surface_data'])} variables"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to load surface data: {e}")

            on_mismatch = cache_settings.get('validation', {}).get('on_mismatch', 'recompute')
            if on_mismatch == 'error':
                raise
            else:
                self.logger.warning("Falling back to surface data extraction")
                return None

    # ========================================================================
    # Terrain-Following Extraction Method
    # ========================================================================

    def _extract_terrain_following(self,
                                     dataset: xr.Dataset,
                                     static_dataset: xr.Dataset,
                                     domain_type: str,
                                     variable: str,
                                     buildings_mask: bool,
                                     output_mode: str,
                                     transect_axis: str = None,
                                     transect_location: int = None,
                                     transect_width: int = 0,
                                     settings: Dict = None) -> Union[Tuple[np.ndarray, str, bool],
                                                                     Tuple[np.ndarray, np.ndarray, str, bool]]:
        """
        Extract terrain-following data by iterating from lowest vertical level upwards.

        Phase 5: Now accepts variable parameter for dynamic variable discovery

        This method implements a "bottom-up" filling algorithm:
        1. Start from zu_3d[0] (lowest vertical level)
        2. For each grid cell, use the first valid (non-fill) value encountered
        3. Iterate upwards through vertical levels, filling only empty cells
        4. Optional building masking: exclude buildings (leave as NaN) or fill through them

        The algorithm provides scientifically accurate terrain-following extraction by
        always using the lowest available valid data above terrain/buildings.

        Args:
            dataset: xarray Dataset containing 3D averaged data (av_3d or av_xy)
            static_dataset: xarray Dataset containing topography and buildings
            domain_type: 'parent' or 'child'
            variable: Variable name from config (e.g., 'temperature', 'utci')
            buildings_mask: If True, exclude building locations (leave as NaN)
                           If False, fill through buildings from level above
            output_mode: '2d' for full spatial field, '1d' for transect only
            transect_axis: 'x' or 'y' (required for 1d mode)
            transect_location: Grid index for transect (required for 1d mode)
            transect_width: Number of grid cells to average (for 1d mode)
            settings: Plot settings dict (optional, used for time selection parameters)

        Returns:
            If output_mode == '2d':
                Tuple of (filled_2d_array [y, x], variable_name, needs_kelvin_conversion)
            If output_mode == '1d':
                Tuple of (transect_1d_array, coordinates, variable_name, needs_kelvin_conversion)

        Raises:
            ValueError: If invalid output_mode or required parameters missing
            KeyError: If variable not found in dataset
        """
        # Validate parameters
        if output_mode not in ['2d', '1d']:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be '2d' or '1d'")

        if output_mode == '1d' and (transect_axis is None or transect_location is None):
            raise ValueError("transect_axis and transect_location required for 1d output mode")

        # Default settings if none provided
        if settings is None:
            settings = {}

        # ===== NEW: CHECK FOR CACHED MASK =====
        use_cache, cache_mode = self._should_use_mask_cache(settings)
        cached_result = None

        if use_cache and cache_mode in ['load', 'auto']:
            # Try to load existing mask
            case_name = settings.get('case_name', 'unknown')
            required_vars = [settings.get('variable', 'ta')]

            # Get grid size for validation
            ny = dataset.dims['y']
            nx = dataset.dims['x']

            cached_result = self._load_terrain_mask(
                case_name=case_name,
                domain_type=domain_type,
                required_variables=required_vars,
                settings=settings,
                expected_grid_size=(ny, nx)
            )

            if cached_result is not None:
                # Successfully loaded mask from cache
                self.logger.info("✓ Using cached terrain mask - skipping computation")

                # Extract appropriate data from cached mask
                mask_data = cached_result['mask_data']
                var_name = required_vars[0]

                if var_name not in mask_data:
                    self.logger.warning(
                        f"Variable '{var_name}' not found in cached mask, will recompute"
                    )
                    cached_result = None
                else:
                    # Get the mask data
                    cached_mask = mask_data[var_name]  # Shape: [ku_above_surf, y, x]

                    # Handle transect_z_offset if specified
                    tf_settings = settings.get('terrain_following', {})
                    domain_settings = tf_settings.get(domain_type, {})
                    transect_z_offset = domain_settings.get(
                        'transect_z_offset',
                        tf_settings.get('transect_z_offset', None)
                    )

                    # Extract appropriate level
                    if transect_z_offset is not None and transect_z_offset != 0:
                        # Direct mapping: ku_above_surf level N = offset N
                        # (Level 0 = offset 0 at terrain, Level 1 = offset 1, etc.)
                        ku_level = transect_z_offset
                        if ku_level < cached_mask.shape[0]:
                            filled_2d = cached_mask[ku_level, :, :]
                            self.logger.info(
                                f"Using cached offset level {transect_z_offset} "
                                f"(ku_above_surf[{ku_level}])"
                            )
                        else:
                            self.logger.warning(
                                f"Requested offset {transect_z_offset} not in cache "
                                f"(max ku_level: {cached_mask.shape[0]-1}), will recompute"
                            )
                            cached_result = None
                    else:
                        # Use base mask (offset 0 = ku_level 0)
                        filled_2d = cached_mask[0, :, :]
                        self.logger.info("Using cached base mask (offset 0)")

                    # If we successfully got cached data, return it
                    if cached_result is not None:
                        # Determine if conversion needed
                        needs_kelvin_conversion = False
                        if var_name in ['theta']:
                            needs_kelvin_conversion = True

                        # Return based on output mode
                        if output_mode == '2d':
                            return filled_2d, var_name, needs_kelvin_conversion

                        elif output_mode == '1d':
                            # Extract transect from cached 2D field
                            if transect_axis == 'x':
                                y_min = max(0, transect_location - transect_width)
                                y_max = min(ny, transect_location + transect_width + 1)
                                transect = np.nanmean(filled_2d[y_min:y_max, :], axis=0)
                                coords = dataset['x'].values if 'x' in dataset.coords else np.arange(nx)
                            else:  # y-axis
                                x_min = max(0, transect_location - transect_width)
                                x_max = min(nx, transect_location + transect_width + 1)
                                transect = np.nanmean(filled_2d[:, x_min:x_max], axis=1)
                                coords = dataset['y'].values if 'y' in dataset.coords else np.arange(ny)

                            return transect, coords, var_name, needs_kelvin_conversion

            elif cache_mode == 'load':
                # Load mode but no file found - error
                raise FileNotFoundError(
                    f"Mask cache mode is 'load' but no cached mask found for "
                    f"{case_name} ({domain_type})"
                )

        # If we get here, either caching is disabled or cache load failed
        # Continue with normal terrain-following computation

        # Log extraction configuration
        msg = f"\n=== TERRAIN-FOLLOWING EXTRACTION ==="
        print(msg)
        self.logger.info(msg)

        msg = f"  Domain: {domain_type}"
        print(msg)
        self.logger.info(msg)

        msg = f"  Output mode: {output_mode}"
        print(msg)
        self.logger.info(msg)

        msg = f"  Buildings mask: {buildings_mask} ({'exclude buildings' if buildings_mask else 'fill through buildings'})"
        print(msg)
        self.logger.info(msg)

        if output_mode == '1d':
            msg = f"  Transect: axis={transect_axis}, location={transect_location}, width=±{transect_width}"
            print(msg)
            self.logger.info(msg)

        try:
            # Get terrain-following specific settings
            tf_settings = settings.get('terrain_following', {})

            # Support domain-specific settings with fallback to global defaults
            # Check for domain-specific overrides (parent/child)
            domain_settings = tf_settings.get(domain_type, {})

            # Get start_z_index: domain-specific > global > default (0)
            start_z_index = domain_settings.get('start_z_index',
                                               tf_settings.get('start_z_index', 0))

            # Get max_z_index: domain-specific > global > default (None)
            max_z_index = domain_settings.get('max_z_index',
                                             tf_settings.get('max_z_index', None))

            # Get transect_z_offset: domain-specific > global > default (None/disabled)
            transect_z_offset = domain_settings.get('transect_z_offset',
                                                    tf_settings.get('transect_z_offset', None))

            # Log which settings source was used
            if domain_type in tf_settings:
                self.logger.debug(f"Using domain-specific terrain_following settings for '{domain_type}'")
            else:
                self.logger.debug(f"Using global terrain_following settings for '{domain_type}'")

            msg = f"  Vertical iteration: start_z={start_z_index}"
            if max_z_index is not None:
                msg += f", max_z={max_z_index}"
            else:
                msg += ", max_z=all levels"
            print(msg)
            self.logger.info(msg)

            # Phase 5: Use dynamic variable discovery with wildcard support
            var_data, var_name_found = self._find_variable_in_dataset(dataset, variable)
            self.logger.info(f"Found variable '{variable}' as PALM variable '{var_name_found}' in dataset")

            # Phase 6: Dimensionality Detection - Check if 2D (surface) or 3D (atmospheric)
            is_2d_surface = False
            z_dim = None

            # Check for 3D atmospheric dimensions (zu_3d, zw_3d)
            if 'zu_3d' in var_data.dims:
                z_dim = 'zu_3d'
                is_2d_surface = False
            elif 'zw_3d' in var_data.dims:
                z_dim = 'zw_3d'
                is_2d_surface = False
            # Check for 2D surface dimensions (zu1_xy, zu_xy)
            elif 'zu1_xy' in var_data.dims:
                z_dim = 'zu1_xy'
                is_2d_surface = True
            elif 'zu_xy' in var_data.dims:
                z_dim = 'zu_xy'
                is_2d_surface = True
            else:
                # Check if already 2D (time, y, x)
                if len(var_data.dims) == 3 and 'time' in var_data.dims:
                    is_2d_surface = True
                    z_dim = None
                    msg = "  Variable is 2D (no vertical dimension)"
                    print(msg)
                    self.logger.info(msg)
                else:
                    available_dims = list(var_data.dims)
                    raise ValueError(
                        f"Could not find recognized vertical dimension in variable '{var_name_found}'. "
                        f"Available dimensions: {available_dims}. "
                        f"Expected: zu_3d, zw_3d, zu1_xy, or zu_xy"
                    )

            # Handle 2D surface variables differently (NO terrain-following)
            if is_2d_surface:
                msg = f"\n=== SURFACE VARIABLE DETECTED (2D) ==="
                print(msg)
                self.logger.info(msg)
                msg = f"  Variable '{var_name_found}' is a surface variable (file_type: av_xy)"
                print(msg)
                self.logger.info(msg)
                msg = f"  Skipping terrain-following extraction, using surface level directly"
                print(msg)
                self.logger.info(msg)

                # ===== CHECK SURFACE DATA CACHE =====
                # Try to load from cache if enabled
                tf_settings = settings.get('terrain_following', {})
                cache_settings = tf_settings.get('surface_data_cache', {})
                cache_enabled = cache_settings.get('enabled', False)
                cache_mode = cache_settings.get('mode', 'disabled')

                if cache_enabled and cache_mode in ['load', 'update']:
                    # Get case_name from settings (added by _extract_scenario_data)
                    case_name = settings.get('case_name', 'unknown')

                    # Get grid size for validation
                    ny, nx = var_data.shape[-2:]  # Last two dimensions are y, x
                    expected_grid_size = (ny, nx)

                    # Try to load from cache
                    cached_surface_data = self._load_surface_data(
                        case_name=case_name,
                        domain_type=domain_type,
                        required_variables=[var_name_found],
                        settings=settings,
                        expected_grid_size=expected_grid_size
                    )

                    # Check if the requested variable is in cache
                    if cached_surface_data and var_name_found in cached_surface_data['surface_data']:
                        msg = f"  ✓ Found '{var_name_found}' in surface data cache, using cached data"
                        print(msg)
                        self.logger.info(msg)

                        filled_2d = cached_surface_data['surface_data'][var_name_found]
                        needs_conversion = False  # Cached data already processed

                        msg = f"\n=== SURFACE DATA LOADED FROM CACHE ==="
                        print(msg)
                        self.logger.info(msg)
                        msg = f"  2D field shape: {filled_2d.shape}"
                        print(msg)
                        self.logger.info(msg)
                        msg = f"  Data range: min={np.nanmin(filled_2d):.2f}, max={np.nanmax(filled_2d):.2f}, mean={np.nanmean(filled_2d):.2f}"
                        print(msg)
                        self.logger.info(msg)

                        # Return based on output mode
                        if output_mode == '1d':
                            # Extract 1D transect from cached 2D surface
                            transect_values, coordinates = self._extract_transect_line(
                                filled_2d, transect_axis, transect_location, transect_width
                            )
                            return transect_values, coordinates, var_name_found, needs_conversion
                        else:  # '2d'
                            return filled_2d, var_name_found, needs_conversion
                    else:
                        msg = f"  Variable '{var_name_found}' not in cache, will extract"
                        print(msg)
                        self.logger.info(msg)

                # Extract surface data directly
                if z_dim and z_dim in var_data.dims:
                    # Has zu1_xy or zu_xy dimension - extract first level
                    slice_data_with_time = var_data.isel({z_dim: 0})
                    msg = f"  Extracted surface level ({z_dim}[0])"
                    print(msg)
                    self.logger.info(msg)
                else:
                    # Already 2D
                    slice_data_with_time = var_data
                    msg = f"  Variable is already 2D (time, y, x)"
                    print(msg)
                    self.logger.info(msg)

                # Apply time averaging
                msg = "\n=== TIME SELECTION CONFIGURATION ==="
                print(msg)
                self.logger.info(msg)

                time_selection_method = settings.get('time_selection_method', 'mean')
                total_time_steps = slice_data_with_time.shape[0]
                msg = f"  Total available time steps: {total_time_steps}"
                print(msg)
                self.logger.info(msg)
                msg = f"  Method: '{time_selection_method}'"
                print(msg)
                self.logger.info(msg)

                # Time averaging with corruption detection
                msg = f"  Time processing: Averaging with corrupted step detection..."
                print(msg)
                self.logger.info(msg)

                # Detect and exclude corrupted time steps
                n_time_before = slice_data_with_time.shape[0]
                valid_time_mask = []

                for t_idx in range(n_time_before):
                    slice_t = slice_data_with_time.isel(time=t_idx)
                    test_values = slice_t.values.ravel()
                    test_values_clean = test_values[~np.isnan(test_values)]

                    if len(test_values_clean) > 0:
                        # For temperature: suspicious if any value < 5.0°C
                        # For other variables: check for extreme outliers or NaN
                        if var_name_found in ['ta', 'theta', 'pt']:
                            suspicious = np.any(test_values_clean < 5.0)
                        else:
                            # For non-temperature variables, check for all NaN or extreme outliers
                            suspicious = False
                        valid_time_mask.append(not suspicious)
                    else:
                        valid_time_mask.append(True)

                valid_time_mask = np.array(valid_time_mask)
                n_corrupted = np.sum(~valid_time_mask)
                n_valid = np.sum(valid_time_mask)

                if n_corrupted > 0:
                    msg = f"  Found {n_corrupted} corrupted time steps, using {n_valid} valid steps"
                    print(msg)
                    self.logger.warning(msg)

                    valid_indices = np.where(valid_time_mask)[0]
                    slice_data_with_time = slice_data_with_time.isel(time=valid_indices)
                else:
                    msg = f"  All {n_time_before} time steps are valid"
                    print(msg)
                    self.logger.info(msg)

                # Perform time averaging using xarray
                filled_2d = slice_data_with_time.mean(dim='time').values

                msg = f"  Time averaging complete"
                print(msg)
                self.logger.info(msg)

                # Check if unit conversion needed (Kelvin to Celsius)
                # PALM typically outputs temperature in Kelvin (values ~297K for 24°C)
                # If mean value > 100, assume it's in Kelvin
                valid_data = filled_2d[~np.isnan(filled_2d)]
                if len(valid_data) > 0:
                    data_mean = np.mean(valid_data)
                    needs_conversion = data_mean > 100.0
                else:
                    needs_conversion = False

                msg = f"  Temperature unit detection: needs_kelvin_conversion={needs_conversion}"
                print(msg)
                self.logger.info(msg)

                msg = f"\n=== SURFACE EXTRACTION COMPLETE ==="
                print(msg)
                self.logger.info(msg)
                msg = f"  2D field shape: {filled_2d.shape}"
                print(msg)
                self.logger.info(msg)
                msg = f"  Data range: min={np.nanmin(filled_2d):.2f}, max={np.nanmax(filled_2d):.2f}, mean={np.nanmean(filled_2d):.2f}"
                print(msg)
                self.logger.info(msg)

                # ===== SAVE TO SURFACE DATA CACHE =====
                # Save to cache if enabled (if not already loaded from cache)
                if cache_enabled and cache_mode in ['save', 'update']:
                    # Get case_name from settings (added by _extract_scenario_data)
                    case_name = settings.get('case_name', 'unknown')

                    # Check if this variable should be cached
                    # Get list of variables to cache for this domain
                    domain_cache_settings = cache_settings.get(domain_type, {})
                    cache_variables = domain_cache_settings.get('variables', [])

                    # Check if variable should be cached
                    should_cache = False
                    if cache_variables == 'auto':
                        # Auto mode: cache all av_xy variables
                        should_cache = True
                    elif isinstance(cache_variables, list):
                        # Check if variable in list (using the internal variable name from config)
                        should_cache = variable in cache_variables

                    if should_cache:
                        msg = f"  Saving '{var_name_found}' to surface data cache..."
                        print(msg)
                        self.logger.info(msg)

                        # Get grid dimensions from data
                        ny, nx = filled_2d.shape

                        # Start with new variable
                        surface_data_dict = {var_name_found: filled_2d}

                        # Check if cache file already exists (for multi-variable merging)
                        cache_path = self._get_surface_data_cache_path(case_name, domain_type, settings)

                        if cache_path.exists():
                            msg = f"  Found existing cache file, will merge variables"
                            print(msg)
                            self.logger.info(msg)
                            try:
                                # Load existing variables
                                existing_cache = self._load_surface_data(
                                    case_name=case_name,
                                    domain_type=domain_type,
                                    required_variables=None,  # Load all
                                    settings=settings,
                                    expected_grid_size=(ny, nx)
                                )

                                if existing_cache and 'surface_data' in existing_cache:
                                    # Merge existing variables with new variable
                                    for existing_var, existing_data in existing_cache['surface_data'].items():
                                        if existing_var != var_name_found:
                                            surface_data_dict[existing_var] = existing_data
                                            msg = f"    Keeping existing variable '{existing_var}' in cache"
                                            print(msg)
                                            self.logger.info(msg)
                                        else:
                                            msg = f"    Updating variable '{var_name_found}' in cache"
                                            print(msg)
                                            self.logger.info(msg)

                                    msg = f"  Multi-variable cache: {len(surface_data_dict)} total variables"
                                    print(msg)
                                    self.logger.info(msg)
                            except Exception as e:
                                msg = f"  Could not merge with existing cache: {e}"
                                print(msg)
                                self.logger.warning(msg)

                        # Prepare coordinates
                        coords = {
                            'x': dataset['x'].values if 'x' in dataset.coords else np.arange(nx, dtype=float),
                            'y': dataset['y'].values if 'y' in dataset.coords else np.arange(ny, dtype=float),
                        }

                        # Prepare metadata
                        metadata = {
                            'time_averaging_method': time_selection_method,
                            'time_steps_used': n_valid,
                            'time_steps_total': n_time_before,
                            'time_steps_corrupted': n_corrupted,
                        }

                        # Add variable-specific metadata
                        if self.var_metadata:
                            try:
                                var_config = self.var_metadata.get_variable_config(variable)
                                metadata['variable_metadata'] = {
                                    var_name_found: {
                                        'units': var_config.get('units_out', ''),
                                        'long_name': var_config.get('label', var_name_found),
                                    }
                                }
                            except:
                                pass

                        # Save surface data
                        self._save_surface_data(
                            case_name=case_name,
                            domain_type=domain_type,
                            surface_data_dict=surface_data_dict,
                            coordinates=coords,
                            metadata=metadata,
                            settings=settings
                        )

                        var_list = list(surface_data_dict.keys())
                        msg = f"  ✓ Surface data saved to cache: {len(surface_data_dict)} variable(s) - {var_list}"
                        print(msg)
                        self.logger.info(msg)

                # Return based on output mode
                if output_mode == '1d':
                    # Extract 1D transect from 2D surface
                    transect_values, coordinates = self._extract_transect_line(
                        filled_2d, transect_axis, transect_location, transect_width
                    )
                    return transect_values, coordinates, var_name_found, needs_conversion
                else:  # '2d'
                    return filled_2d, var_name_found, needs_conversion

            # ===== 3D ATMOSPHERIC VARIABLE - Continue with terrain-following =====

            # Detect fill values
            has_fill_value, fill_value = self._detect_fill_values(var_data)
            msg = f"  Fill value: {fill_value} ({'NaN' if np.isnan(fill_value) else fill_value})"
            print(msg)
            self.logger.info(msg)

            # Get vertical coordinates
            z_coords = dataset[z_dim].values
            n_z_levels = len(z_coords)

            msg = f"  Vertical levels: {n_z_levels} (heights: {z_coords[0]:.2f}m to {z_coords[-1]:.2f}m)"
            print(msg)
            self.logger.info(msg)

            # Validate z-index range
            if start_z_index < 0 or start_z_index >= n_z_levels:
                self.logger.error(f"start_z_index {start_z_index} out of range [0, {n_z_levels})")
                raise ValueError(f"Invalid start_z_index: {start_z_index}")

            if max_z_index is not None:
                if max_z_index < start_z_index or max_z_index >= n_z_levels:
                    self.logger.warning(f"max_z_index {max_z_index} out of range, using all levels")
                    max_z_index = None

            # Determine effective max z-index
            effective_max_z = max_z_index if max_z_index is not None else (n_z_levels - 1)

            # STEP 1: TIME AVERAGING
            # Apply time selection/averaging using existing logic (same as other extraction methods)
            msg = f"\n=== TIME SELECTION CONFIGURATION ==="
            print(msg)
            self.logger.info(msg)

            time_selection_method = settings.get('time_selection_method', 'mean')
            total_time_steps = var_data.shape[0]

            msg = f"  Total available time steps: {total_time_steps}"
            print(msg)
            self.logger.info(msg)

            msg = f"  Method: '{time_selection_method}'"
            print(msg)
            self.logger.info(msg)

            # Apply time selection (same logic as _extract_terrain_following_slice)
            var_data_time_selected = var_data

            if time_selection_method == 'single_timestep':
                time_index = settings.get('time_index', 0)
                if time_index < 0 or time_index >= total_time_steps:
                    self.logger.warning(f"time_index {time_index} out of range, using 0")
                    time_index = 0
                var_data_time_selected = var_data.isel(time=[time_index])
                self.logger.info(f"  Selected single time step: {time_index}")

            elif time_selection_method == 'mean_timeframe':
                time_start = settings.get('time_start', 0)
                time_end = settings.get('time_end', total_time_steps - 1)

                # Validate and clip range
                time_start = max(0, min(time_start, total_time_steps - 1))
                time_end = max(time_start, min(time_end, total_time_steps - 1))

                time_indices = list(range(time_start, time_end + 1))
                var_data_time_selected = var_data.isel(time=time_indices)
                self.logger.info(f"  Selected time range: [{time_start}, {time_end}] ({len(time_indices)} steps)")

            else:  # 'mean' - use all time steps
                self.logger.info(f"  Using all {total_time_steps} time steps")

            # Perform time averaging (or extract single step)
            if time_selection_method == 'single_timestep':
                # Extract single time step (no averaging)
                var_data_time_avg = var_data_time_selected.isel(time=0)
                msg = f"  Time processing: Extracted single timestep (no averaging)"
            else:
                # Perform averaging with corrupted step detection
                msg = f"  Time processing: Averaging with corrupted step detection..."
                print(msg)
                self.logger.info(msg)

                # Detect and exclude corrupted time steps (T < 5°C indicates corrupted data)
                n_time_before = var_data_time_selected.shape[0]
                valid_time_mask = []

                for t_idx in range(n_time_before):
                    # Use dictionary unpacking to avoid syntax error with dynamic dimension name
                    selection = {'time': t_idx, z_dim: start_z_index}
                    slice_t = var_data_time_selected.isel(**selection)
                    test_values = slice_t.values.ravel()
                    test_values_clean = test_values[~np.isnan(test_values)]

                    if len(test_values_clean) > 0:
                        suspicious = np.any(test_values_clean < 5.0)
                        valid_time_mask.append(not suspicious)
                    else:
                        valid_time_mask.append(True)

                valid_time_mask = np.array(valid_time_mask)
                n_corrupted = np.sum(~valid_time_mask)
                n_valid = np.sum(valid_time_mask)

                if n_corrupted > 0:
                    msg = f"  Found {n_corrupted} corrupted time steps, using {n_valid} valid steps"
                    print(msg)
                    self.logger.warning(msg)

                    valid_indices = np.where(valid_time_mask)[0]
                    var_data_time_selected = var_data_time_selected.isel(time=valid_indices)
                else:
                    msg = f"  All {n_time_before} time steps are valid"
                    print(msg)
                    self.logger.info(msg)

                # Perform time averaging using xarray
                var_data_time_avg = var_data_time_selected.mean(dim='time')
                msg = f"  Time averaging complete"

            print(msg)
            self.logger.info(msg)

            # Now var_data_time_avg has shape [zu_3d, y, x] (time dimension removed)

            # STEP 2: LOAD BUILDING MASK (if needed)
            building_mask_2d = None
            if buildings_mask and static_dataset is not None and 'buildings_2d' in static_dataset:
                building_mask_2d = static_dataset['buildings_2d'].values > 0
                n_building_cells = np.sum(building_mask_2d)
                total_cells = building_mask_2d.size
                pct_buildings = 100 * n_building_cells / total_cells

                msg = f"  Building mask loaded: {n_building_cells}/{total_cells} cells ({pct_buildings:.1f}%) are buildings"
                print(msg)
                self.logger.info(msg)
            elif buildings_mask and (static_dataset is None or 'buildings_2d' not in static_dataset):
                self.logger.warning("buildings_mask=True but buildings_2d not available in static dataset")
                buildings_mask = False  # Disable mask if data not available

            # STEP 3: TERRAIN-FOLLOWING ITERATION
            msg = f"\n=== VERTICAL ITERATION (TERRAIN-FOLLOWING) ==="
            print(msg)
            self.logger.info(msg)

            # Get spatial dimensions
            ny, nx = var_data_time_avg.shape[-2:]

            msg = f"  Spatial dimensions: y={ny}, x={nx}"
            print(msg)
            self.logger.info(msg)

            # Initialize output array with NaN
            filled_array = np.full((ny, nx), np.nan, dtype=np.float64)

            # Track which z-level each cell was filled from (for debugging/visualization)
            source_level_array = np.full((ny, nx), -1, dtype=np.int32)

            # Iterate through vertical levels from bottom to top
            n_z_to_process = effective_max_z - start_z_index + 1
            n_filled_total = 0

            for z_idx in range(start_z_index, effective_max_z + 1):
                # Extract 2D slice at this level
                slice_2d = var_data_time_avg.isel({z_dim: z_idx}).values

                # Find cells that are:
                # 1. Currently unfilled (NaN in filled_array)
                # 2. Have valid data at this level (not fill_value)
                # 3. Not masked by buildings (if buildings_mask=True)

                currently_unfilled = np.isnan(filled_array)
                has_valid_data = ~self._is_fill_value(slice_2d, fill_value)

                # If buildings_mask=True, exclude building locations
                if buildings_mask and building_mask_2d is not None:
                    not_building = ~building_mask_2d
                else:
                    not_building = np.ones_like(currently_unfilled, dtype=bool)

                # Combine all conditions
                fillable_mask = currently_unfilled & has_valid_data & not_building

                n_filled_this_level = np.sum(fillable_mask)

                # Fill cells that meet all conditions
                filled_array[fillable_mask] = slice_2d[fillable_mask]
                source_level_array[fillable_mask] = z_idx

                n_filled_total += n_filled_this_level

                # Log progress every 5 levels or on first/last level
                if z_idx == start_z_index or z_idx == effective_max_z or (z_idx - start_z_index) % 5 == 0:
                    height_m = z_coords[z_idx]
                    n_unfilled_remaining = np.sum(np.isnan(filled_array))
                    pct_filled = 100 * n_filled_total / (ny * nx)

                    msg = f"  Level {z_idx:2d} ({height_m:6.2f}m): filled {n_filled_this_level:5d} cells | " \
                          f"Total filled: {n_filled_total:6d} ({pct_filled:5.1f}%) | Remaining: {n_unfilled_remaining:6d}"
                    print(msg)
                    self.logger.info(msg)

            # Final statistics
            n_filled_final = np.sum(~np.isnan(filled_array))
            n_unfilled_final = np.sum(np.isnan(filled_array))
            pct_filled_final = 100 * n_filled_final / (ny * nx)

            msg = f"\n=== TERRAIN-FOLLOWING ITERATION COMPLETE ==="
            print(msg)
            self.logger.info(msg)

            msg = f"  Processed {n_z_to_process} vertical levels (zu_3d[{start_z_index}] to zu_3d[{effective_max_z}])"
            print(msg)
            self.logger.info(msg)

            msg = f"  Final coverage: {n_filled_final}/{ny*nx} cells ({pct_filled_final:.1f}%) filled, {n_unfilled_final} remaining NaN"
            print(msg)
            self.logger.info(msg)

            if n_filled_final == 0:
                self.logger.error("No valid data found in any vertical level!")
                raise ValueError("Terrain-following extraction produced all NaN")

            # Log source level statistics
            valid_source_levels = source_level_array[source_level_array >= 0]
            if len(valid_source_levels) > 0:
                min_source = np.min(valid_source_levels)
                max_source = np.max(valid_source_levels)
                mean_source = np.mean(valid_source_levels)

                msg = f"  Source levels: min=zu_3d[{min_source}] ({z_coords[min_source]:.2f}m), " \
                      f"max=zu_3d[{max_source}] ({z_coords[max_source]:.2f}m), " \
                      f"mean=zu_3d[{int(mean_source)}] ({z_coords[int(mean_source)]:.2f}m)"
                print(msg)
                self.logger.info(msg)

            # STEP 3.5: APPLY VERTICAL OFFSET (if transect_z_offset is specified)
            if transect_z_offset is not None and transect_z_offset != 0:
                msg = f"\n=== APPLYING VERTICAL OFFSET ==="
                print(msg)
                self.logger.info(msg)

                msg = f"  Offset: {transect_z_offset} grid points {'upward' if transect_z_offset > 0 else 'downward'}"
                print(msg)
                self.logger.info(msg)

                # VECTORIZED IMPLEMENTATION (much faster than nested loops)

                # Get mask of cells with valid source levels
                valid_source_mask = source_level_array >= 0
                n_valid_sources = np.sum(valid_source_mask)

                # Calculate target z-levels for all cells (vectorized)
                target_z_array = source_level_array + transect_z_offset

                # Check which target levels are within bounds (vectorized)
                in_bounds_mask = (target_z_array >= 0) & (target_z_array < n_z_levels) & valid_source_mask
                n_offset_out_of_bounds = n_valid_sources - np.sum(in_bounds_mask)

                # Convert xarray to numpy for fast indexing (shape: [n_z_levels, ny, nx])
                full_data_array = var_data_time_avg.values

                # Create meshgrid for y, x indices
                y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

                # Initialize offset array with NaN
                offset_array = np.full((ny, nx), np.nan, dtype=np.float64)

                # For in-bounds cells, extract values using advanced indexing
                if np.any(in_bounds_mask):
                    # Get target z-levels for in-bounds cells (convert to int for indexing)
                    target_zs = target_z_array[in_bounds_mask].astype(int)
                    y_coords = y_indices[in_bounds_mask]
                    x_coords = x_indices[in_bounds_mask]

                    # Extract values: full_data_array[z, y, x] (vectorized!)
                    extracted_values = full_data_array[target_zs, y_coords, x_coords]

                    # Check for fill values (vectorized)
                    if np.isnan(fill_value):
                        valid_value_mask = ~np.isnan(extracted_values)
                    else:
                        valid_value_mask = ~(np.isclose(extracted_values, fill_value, rtol=1e-9, atol=1e-9) |
                                           (extracted_values == fill_value))

                    # Create temporary array for assignments
                    temp_values = np.full(np.sum(in_bounds_mask), np.nan)
                    temp_values[valid_value_mask] = extracted_values[valid_value_mask]

                    # Assign back to offset_array (vectorized)
                    offset_array[in_bounds_mask] = temp_values

                # Statistics
                n_offset_applied = np.sum(~np.isnan(offset_array))
                n_offset_fill_value = np.sum(in_bounds_mask) - n_offset_applied

                msg = f"  Offset results:"
                print(msg)
                self.logger.info(msg)

                msg = f"    - Successfully offset: {n_offset_applied}/{n_valid_sources} cells"
                print(msg)
                self.logger.info(msg)

                if n_offset_out_of_bounds > 0:
                    msg = f"    - Out of bounds: {n_offset_out_of_bounds} cells (target z-level outside dataset)"
                    print(msg)
                    self.logger.warning(msg)

                if n_offset_fill_value > 0:
                    msg = f"    - Fill values: {n_offset_fill_value} cells (target contained fill value)"
                    print(msg)
                    self.logger.warning(msg)

                # Log offset level statistics
                valid_offset_sources = source_level_array[~np.isnan(offset_array)]
                if len(valid_offset_sources) > 0:
                    offset_target_levels = valid_offset_sources + transect_z_offset
                    min_target = np.min(offset_target_levels)
                    max_target = np.max(offset_target_levels)
                    mean_target = np.mean(offset_target_levels)

                    msg = f"  Target levels after offset: min=zu_3d[{int(min_target)}] ({z_coords[int(min_target)]:.2f}m), " \
                          f"max=zu_3d[{int(max_target)}] ({z_coords[int(max_target)]:.2f}m), " \
                          f"mean=zu_3d[{int(mean_target)}] ({z_coords[int(mean_target)]:.2f}m)"
                    print(msg)
                    self.logger.info(msg)

                # Replace filled_array with offset_array for downstream use
                filled_array = offset_array
                n_filled_final = np.sum(~np.isnan(filled_array))

                msg = f"  Final coverage after offset: {n_filled_final}/{ny*nx} cells ({100*n_filled_final/(ny*nx):.1f}%)"
                print(msg)
                self.logger.info(msg)

                if n_filled_final == 0:
                    self.logger.error("No valid data after applying vertical offset!")
                    raise ValueError("Vertical offset produced all NaN - try reducing offset or adjusting max_z_index")

            # STEP 4: GENERATE OUTPUT BASED ON OUTPUT_MODE
            valid_data = filled_array[~np.isnan(filled_array)]
            if len(valid_data) > 0:
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                data_mean = np.mean(valid_data)

                msg = f"  Data range: min={data_min:.2f}, max={data_max:.2f}, mean={data_mean:.2f}"
                print(msg)
                self.logger.info(msg)

                # Detect if conversion from Kelvin needed
                needs_kelvin_conversion = data_mean > 100.0
                msg = f"  Temperature unit: needs_kelvin_conversion={needs_kelvin_conversion}"
                print(msg)
                self.logger.info(msg)
            else:
                needs_kelvin_conversion = False

            # ===== NEW: SAVE MASK IF CACHING ENABLED =====
            # DEBUG: Log cache status
            self.logger.info(f"Cache status: use_cache={use_cache}, cache_mode={cache_mode}, cached_result={cached_result is not None}")

            if use_cache and cache_mode in ['save', 'auto']:
                # Check if we should save (auto mode only saves if file didn't exist)
                should_save = (cache_mode == 'save') or \
                             (cache_mode == 'auto' and cached_result is None)


                if should_save:
                    try:
                        self.logger.info("Saving terrain mask to cache...")

                        # Get cache settings
                        tf_settings = settings.get('terrain_following', {})
                        cache_settings = tf_settings.get('mask_cache', {})
                        levels = cache_settings.get('levels', {})

                        # Parse offsets to save
                        offsets_to_save = parse_offset_specification(
                            levels.get('offsets', [0]),
                            max_levels=levels.get('max_levels', 20)
                        )

                        # Compute masks for all requested offsets
                        # We already have filled_array which is either:
                        # - Base terrain-following (if transect_z_offset is None or 0)
                        # - Offset terrain-following (if transect_z_offset was specified)

                        # Determine which offset level we currently have
                        current_offset = transect_z_offset if (transect_z_offset is not None and transect_z_offset != 0) else 0

                        # Determine maximum ku_level needed
                        # Direct mapping: ku_above_surf level N = offset N
                        max_ku_level = max(offsets_to_save) + 1

                        # Initialize 3D mask array [ku_above_surf, y, x]
                        mask_3d = np.full((max_ku_level, ny, nx), fill_value, dtype=np.float32)

                        # For each requested offset, compute the mask
                        for offset in offsets_to_save:
                            ku_level = offset  # Direct mapping: ku_above_surf level N = offset N

                            if ku_level >= max_ku_level:
                                continue

                            # If this is the offset we already computed, use it
                            if offset == current_offset:
                                mask_3d[ku_level, :, :] = filled_array
                                self.logger.info(f"  Using computed data for offset {offset} (ku_level {ku_level})")
                                continue

                            # Otherwise, compute this offset level
                            self.logger.info(f"  Computing offset {offset} (ku_level {ku_level})...")

                            if offset == 0:
                                # This is the base terrain-following mask
                                # We need to use source_level_array without offset
                                target_z_array = source_level_array.copy()
                            else:
                                # Apply offset to source levels
                                target_z_array = source_level_array + offset

                            # Check which cells are valid (in bounds and have valid source)
                            valid_mask = (source_level_array >= 0) & \
                                       (target_z_array >= 0) & \
                                       (target_z_array < n_z_levels)

                            if np.any(valid_mask):
                                # Use vectorized extraction (same as main computation)
                                full_data_array = var_data_time_avg.values
                                y_indices, x_indices = np.meshgrid(
                                    np.arange(ny), np.arange(nx), indexing='ij'
                                )

                                target_zs = target_z_array[valid_mask].astype(int)
                                y_coords = y_indices[valid_mask]
                                x_coords = x_indices[valid_mask]

                                extracted_values = full_data_array[target_zs, y_coords, x_coords]

                                # Check for fill values
                                if np.isnan(fill_value):
                                    valid_value_mask = ~np.isnan(extracted_values)
                                else:
                                    valid_value_mask = ~(
                                        np.isclose(extracted_values, fill_value, rtol=1e-9, atol=1e-9) |
                                        (extracted_values == fill_value)
                                    )

                                # Create temporary array for assignments
                                temp_values = np.full(np.sum(valid_mask), np.nan, dtype=np.float32)
                                temp_values[valid_value_mask] = extracted_values[valid_value_mask]

                                # Assign to 3D mask
                                mask_3d[ku_level, valid_mask] = temp_values

                                n_valid = np.sum(~np.isnan(mask_3d[ku_level, :, :]))
                                self.logger.info(f"    Computed {n_valid} valid cells for offset {offset}")

                        # Prepare mask data dictionary - START WITH NEW VARIABLE
                        mask_data_dict = {var_name_found: mask_3d}

                        # Check if we should merge with existing cache file
                        case_name = settings.get('case_name', 'unknown')
                        cache_path = self._get_mask_cache_path(case_name, domain_type, settings)

                        if cache_path.exists():
                            self.logger.info(f"Found existing cache file, will merge variables: {cache_path.name}")
                            try:
                                # Load existing variables from cache
                                existing_mask = self._load_terrain_mask(
                                    case_name=case_name,
                                    domain_type=domain_type,
                                    required_variables=None,  # Load all variables
                                    settings=settings,
                                    expected_grid_size=(ny, nx)
                                )

                                if existing_mask and 'mask_data' in existing_mask:
                                    # Merge existing variables with new variable
                                    for existing_var, existing_data in existing_mask['mask_data'].items():
                                        if existing_var != var_name_found:
                                            # Keep existing variables that aren't being updated
                                            mask_data_dict[existing_var] = existing_data
                                            self.logger.info(f"  Keeping existing variable '{existing_var}' in cache")
                                        else:
                                            self.logger.info(f"  Updating variable '{var_name_found}' in cache")

                                    self.logger.info(f"Multi-variable cache: {len(mask_data_dict)} total variables")
                            except Exception as e:
                                self.logger.warning(f"Could not merge with existing cache: {e}")
                                # Continue with just the new variable

                        # Prepare coordinates
                        coords = {
                            'x': dataset['x'].values if 'x' in dataset.coords else np.arange(nx, dtype=float),
                            'y': dataset['y'].values if 'y' in dataset.coords else np.arange(ny, dtype=float),
                            'ku_above_surf': np.arange(max_ku_level, dtype=float),
                        }

                        # Save mask (now with potentially multiple variables)
                        self._save_terrain_mask(
                            case_name=case_name,
                            domain_type=domain_type,
                            mask_data_dict=mask_data_dict,
                            source_levels=source_level_array,
                            coordinates=coords,
                            settings=settings,
                            static_dataset=static_dataset
                        )

                        var_count = len(mask_data_dict)
                        var_list = list(mask_data_dict.keys())
                        self.logger.info(f"✓ Terrain mask saved to cache: {var_count} variable(s) - {var_list}")

                    except Exception as e:
                        self.logger.error(f"Failed to save terrain mask (non-fatal): {e}")
                        # Don't raise - caching failure shouldn't stop execution
                        import traceback
                        self.logger.debug(traceback.format_exc())

            if output_mode == '2d':
                # Return full 2D field
                msg = f"\n=== OUTPUT: 2D field ({ny}x{nx}) ==="
                print(msg)
                self.logger.info(msg)

                return filled_array, var_name_found, needs_kelvin_conversion

            else:  # output_mode == '1d'
                # Extract transect from filled 2D field
                msg = f"\n=== OUTPUT: 1D transect extraction ==="
                print(msg)
                self.logger.info(msg)

                transect_values, coordinates = self._extract_transect_line(
                    filled_array, transect_axis, transect_location, transect_width
                )

                msg = f"  Transect length: {len(transect_values)}"
                print(msg)
                self.logger.info(msg)

                n_valid_transect = np.sum(~np.isnan(transect_values))
                msg = f"  Valid points: {n_valid_transect}/{len(transect_values)} ({100*n_valid_transect/len(transect_values):.1f}%)"
                print(msg)
                self.logger.info(msg)

                return transect_values, coordinates, var_name_found, needs_kelvin_conversion

        except Exception as e:
            self.logger.error(f"Error in terrain-following extraction: {str(e)}")
            raise

    def _time_average_data(self, data_3d: np.ndarray, method: str = 'mean') -> np.ndarray:
        """
        Extract representative 2D field from time series

        Args:
            data_3d: 3D array [time, y, x]
            method: 'max_variation' (select time with max spatial variation),
                   'mean' (average all times), or
                   'time_XX' (specific time index, e.g., 'time_38')

        Returns:
            2D array [y, x] with selected/averaged values
        """
        if method == 'max_variation':
            # Find time step with maximum spatial variation
            std_devs = np.array([np.nanstd(data_3d[t, :, :]) for t in range(data_3d.shape[0])])
            best_time = np.argmax(std_devs)
            self.logger.info(f"Selected time index {best_time} with spatial std={std_devs[best_time]:.3f}")
            return data_3d[best_time, :, :]
        elif method == 'mean':
            # Average over all time steps - use nanmean to preserve NaN (e.g., buildings)
            return np.nanmean(data_3d, axis=0) 
        elif method.startswith('time_'):
            # Extract specific time index
            time_idx = int(method.split('_')[1])
            if time_idx < 0 or time_idx >= data_3d.shape[0]:
                self.logger.warning(f"Time index {time_idx} out of range [0, {data_3d.shape[0]}), using max_variation")
                std_devs = np.array([np.nanstd(data_3d[t, :, :]) for t in range(data_3d.shape[0])])
                time_idx = np.argmax(std_devs)
            self.logger.info(f"Using specified time index {time_idx}")
            return data_3d[time_idx, :, :]
        else:
            self.logger.warning(f"Unknown method '{method}', using max_variation")
            std_devs = np.array([np.nanstd(data_3d[t, :, :]) for t in range(data_3d.shape[0])])
            best_time = np.argmax(std_devs)
            return data_3d[best_time, :, :]

    def _extract_transect_line(self, data_2d: np.ndarray,
                               axis: str,
                               location: int,
                               width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 1D transect line from 2D field with width averaging

        Args:
            data_2d: 2D array [y, x]
            axis: 'x' or 'y' - direction of transect
            location: Grid index for transect location
            width: Number of grid cells to average on each side

        Returns:
            Tuple of (transect_values, coordinates)
        """
        ny, nx = data_2d.shape

        if axis == 'x':
            # Transect runs along x-axis (constant y)
            y_min = max(0, location - width)
            y_max = min(ny, location + width + 1)

            # Average over y range - use nanmean to preserve NaN values (e.g., buildings)
            transect = np.nanmean(data_2d[y_min:y_max, :], axis=0)
            coordinates = np.arange(nx)

        elif axis == 'y':
            # Transect runs along y-axis (constant x)
            x_min = max(0, location - width)
            x_max = min(nx, location + width + 1)

            # Average over x range - use nanmean to preserve NaN values (e.g., buildings)
            transect = np.nanmean(data_2d[:, x_min:x_max], axis=1)
            coordinates = np.arange(ny)

        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x' or 'y'")

        return transect, coordinates

    def _get_building_lad_masks(self, static_dataset: xr.Dataset,
                                axis: str,
                                location: int,
                                width: int,
                                domain_type: str) -> Dict:
        """
        Extract building and LAD arrays along transect line

        Args:
            static_dataset: xarray Dataset from static file
            axis: 'x' or 'y' - transect direction
            location: Grid index for transect location
            width: Number of grid cells to average
            domain_type: 'parent' or 'child'

        Returns:
            Dict with:
            - 'buildings': 1D boolean array marking building locations
            - 'lad': 1D float array with LAD values (or None if not available)
            - 'coordinates': 1D array of x or y coordinates
        """
        result = {
            'buildings': None,
            'lad': None,
            'coordinates': None
        }

        try:
            # Extract building mask if available
            if 'buildings_2d' in static_dataset:
                buildings_2d = static_dataset['buildings_2d'].values

                # Extract along transect
                if axis == 'x':
                    ny, nx = buildings_2d.shape
                    y_min = max(0, location - width)
                    y_max = min(ny, location + width + 1)
                    buildings_slice = np.any(buildings_2d[y_min:y_max, :] > 0, axis=0)
                    result['coordinates'] = np.arange(nx)
                else:
                    ny, nx = buildings_2d.shape
                    x_min = max(0, location - width)
                    x_max = min(nx, location + width + 1)
                    buildings_slice = np.any(buildings_2d[:, x_min:x_max] > 0, axis=1)
                    result['coordinates'] = np.arange(ny)

                result['buildings'] = buildings_slice

            # Extract LAD if available
            if 'lad' in static_dataset:
                lad_data = static_dataset['lad']

                # Get first data level based on domain (same as temperature data)
                if domain_type == 'child':
                    first_level = 21  # Child domain N02 - first valid data level
                else:
                    first_level = 25  # Parent domain - first valid data level

                # Sum LAD over vertical column
                if len(lad_data.shape) == 3:  # [z, y, x]
                    lad_column_sum = np.sum(lad_data.values, axis=0)
                else:
                    lad_column_sum = lad_data.values

                # Extract along transect
                if axis == 'x':
                    y_min = max(0, location - width)
                    y_max = min(lad_column_sum.shape[0], location + width + 1)
                    lad_slice = np.mean(lad_column_sum[y_min:y_max, :], axis=0)
                else:
                    x_min = max(0, location - width)
                    x_max = min(lad_column_sum.shape[1], location + width + 1)
                    lad_slice = np.mean(lad_column_sum[:, x_min:x_max], axis=1)

                result['lad'] = lad_slice

        except Exception as e:
            self.logger.warning(f"Could not extract building/LAD masks: {str(e)}")

        return result

    def _create_comparison_plot(self, data: Dict,
                               variable: str,
                               domain: str,
                               comparison_type: str,
                               settings: Dict) -> plt.Figure:
        """
        Create complete transect comparison plot

        Args:
            data: Loaded simulation data
            variable: 'ta' or 'qv'
            domain: 'parent' or 'child'
            comparison_type: 'age' or 'spacing'
            settings: Plot settings from config

        Returns:
            Matplotlib figure
        """
        # Get scenarios to compare
        scenarios = self._get_scenarios_to_compare(comparison_type, settings)

        # Extract data for each scenario
        scenarios_data = []
        missing_scenarios = []
        for scenario in scenarios:
            scenario_data = self._extract_scenario_data(
                data, scenario, variable, domain, settings
            )
            if scenario_data is not None:
                scenarios_data.append(scenario_data)
            elif scenario['spacing'] is not None:  # Skip warning for base case
                missing_scenarios.append(scenario['label'])

        # Warn if scenarios are missing
        if missing_scenarios:
            self.logger.warning(
                f"Missing {len(missing_scenarios)} scenario(s): {', '.join(missing_scenarios)}. "
                f"Ensure data.spacings and data.ages in config include all values "
                f"used in fig_6 settings (constant_spacing, varying_ages, constant_age, varying_spacings)."
            )

        # Create the plot
        fig = self._create_transect_plot(
            scenarios_data, variable, domain, comparison_type, settings
        )

        return fig

    def _get_scenarios_to_compare(self, comparison_type: str,
                                  settings: Dict) -> List[Dict]:
        """
        Get list of scenarios to compare based on comparison type
        Supports multiple constant groups with distinct color shading

        Args:
            comparison_type: 'age' or 'spacing'
            settings: Plot settings from config

        Returns:
            List of scenario dictionaries
        """
        scenarios = []

        # Always include base case with dashed line to distinguish from forested scenarios
        scenarios.append({
            'spacing': None,
            'age': None,
            'label': 'No Trees',
            'color': '#8B0000',  # Dark red
            'linestyle': '--',   # Dashed line for base case
            'linewidth': 2.5
        })

        if comparison_type == 'age':
            # Varying ages, constant spacing(s)
            constant_spacing_raw = settings['age_comparison']['constant_spacing']
            varying_ages = settings['age_comparison']['varying_ages']

            # Handle both single value and list
            constant_spacings = constant_spacing_raw if isinstance(constant_spacing_raw, list) else [constant_spacing_raw]

            # Base colors for age groups (one color per spacing group)
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

            # Create scenarios for each constant spacing group
            for spacing_idx, spacing in enumerate(constant_spacings):
                base_color = base_colors[spacing_idx % len(base_colors)]

                # Generate shades of the base color for different ages
                for age_idx, age in enumerate(varying_ages):
                    # Create lighter/darker shades: darkest for youngest, lightest for oldest
                    shade_factor = 0.4 + (age_idx / max(len(varying_ages) - 1, 1)) * 0.6
                    color = self._adjust_color_brightness(base_color, shade_factor)

                    scenarios.append({
                        'spacing': spacing,
                        'age': age,
                        'label': f'{spacing}m {age}yrs',
                        'color': color,
                        'linestyle': '-',
                        'linewidth': 2.0
                    })

        else:  # spacing comparison
            # Varying spacings, constant age(s)
            constant_age_raw = settings['spacing_comparison']['constant_age']
            varying_spacings = settings['spacing_comparison']['varying_spacings']

            # Handle both single value and list
            constant_ages = constant_age_raw if isinstance(constant_age_raw, list) else [constant_age_raw]

            # Base colors for spacing groups (one color per age group)
            base_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']  # Blue, Green, Purple, Orange

            # Create scenarios for each constant age group
            for age_idx, age in enumerate(constant_ages):
                base_color = base_colors[age_idx % len(base_colors)]

                # Generate shades of the base color for different spacings
                for spacing_idx, spacing in enumerate(varying_spacings):
                    # Create lighter/darker shades: darkest for densest, lightest for sparsest
                    shade_factor = 0.4 + (spacing_idx / max(len(varying_spacings) - 1, 1)) * 0.6
                    color = self._adjust_color_brightness(base_color, shade_factor)

                    scenarios.append({
                        'spacing': spacing,
                        'age': age,
                        'label': f'{spacing}m {age}yrs',
                        'color': color,
                        'linestyle': '-',
                        'linewidth': 2.0
                    })

        return scenarios

    def _adjust_color_brightness(self, hex_color: str, factor: float) -> str:
        """
        Adjust brightness of a hex color

        Args:
            hex_color: Hex color string (e.g., '#1f77b4')
            factor: Brightness factor (0.0 = black, 1.0 = original, >1.0 = lighter)

        Returns:
            Adjusted hex color string
        """
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')

        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Adjust brightness
        if factor < 1.0:
            # Darken: move towards black
            r = int(r * factor)
            g = int(g * factor)
            b = int(b * factor)
        else:
            # Lighten: move towards white
            r = int(r + (255 - r) * (factor - 1.0))
            g = int(g + (255 - g) * (factor - 1.0))
            b = int(b + (255 - b) * (factor - 1.0))

        # Clamp values
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        return f'#{r:02x}{g:02x}{b:02x}'

    def _extract_scenario_data(self, data: Dict,
                               scenario: Dict,
                               variable: str,
                               domain: str,
                               settings: Dict) -> Optional[Dict]:
        """
        Extract transect data for a single scenario
        Supports three extraction methods based on settings['extraction_method']:
          - 'slice_2d' (default): Extract full 2D slice at fixed height, then extract transect
          - 'transect_direct': Extract 1D transect directly from 4D data (~400× more memory efficient)
          - 'terrain_following': Fill from lowest vertical level upwards (true terrain-following)

        Args:
            data: All loaded simulation data
            scenario: Scenario specification
            variable: 'ta' or 'qv'
            domain: 'parent' or 'child'
            settings: Plot settings

        Returns:
            Dict with transect data or None if scenario not available
        """
        try:
            # Get dataset key
            if scenario['spacing'] is None:
                # Base case
                if 'base_case' not in data or data['base_case'] is None:
                    self.logger.warning("Base case data not available")
                    return None
                case_data = data['base_case']
                self.logger.info(f"Loading base case data for '{scenario['label']}'")
            else:
                # Tree scenario
                case_key = f"{scenario['spacing']}m_{scenario['age']}yrs"
                if case_key not in data['simulations']:
                    self.logger.warning(f"Scenario {case_key} not available")
                    return None
                case_data = data['simulations'][case_key]

            # Get appropriate dataset based on domain AND variable file type
            # Phase 4: File type routing based on variable metadata
            if self.var_metadata:
                try:
                    file_type = self.var_metadata.get_file_type(variable)
                    self.logger.debug(f"Variable '{variable}' requires file type: {file_type}")
                except KeyError:
                    # Variable not in metadata, assume av_3d for backward compatibility
                    self.logger.warning(f"Variable '{variable}' not in metadata, assuming av_3d")
                    file_type = 'av_3d'
            else:
                # No metadata available, fall back to legacy behavior (assume 3D)
                file_type = 'av_3d'

            # Select dataset key based on file type and domain
            if domain == 'child':
                dataset_key = f"{file_type}_n02"
            else:
                dataset_key = file_type

            # Check if dataset exists
            if dataset_key not in case_data:
                self.logger.error(
                    f"Required dataset '{dataset_key}' not found for variable '{variable}', "
                    f"domain '{domain}', scenario '{scenario['label']}'. "
                    f"Available datasets: {list(case_data.keys())}"
                )
                return None

            dataset = case_data[dataset_key]
            self.logger.info(f"Using dataset '{dataset_key}' for variable '{variable}' in {domain} domain")

            # Get static dataset for building/LAD masks
            if domain == 'child':
                static_dataset = case_data.get('static_n02')
            else:
                static_dataset = case_data.get('static')

            # ===== NEW: Add case_name to settings for mask caching =====
            # Extract case name from scenario
            if scenario['spacing'] is None:
                # Base case
                case_name = 'thf_base_2018080700'  # Base case name
            else:
                # Tree scenario - construct full case name
                case_name = f"thf_forest_lad_spacing_{scenario['spacing']}m_age_{scenario['age']}yrs"

            # Add to settings for use by caching system
            settings = settings.copy()  # Don't modify original settings
            settings['case_name'] = case_name
            settings['variable'] = variable  # Also add variable name for cache

            # Get extraction method from settings (default to slice_2d for backward compatibility)
            extraction_method = settings.get('extraction_method', 'slice_2d')

            # Get common parameters with domain-specific override support
            # For terrain_following method, check for domain-specific settings first
            if extraction_method == 'terrain_following':
                tf_settings = settings.get('terrain_following', {})
                domain_settings = tf_settings.get(domain, {})

                # Use domain-specific if available, else fall back to global settings
                transect_axis = domain_settings.get('transect_axis', settings.get('transect_axis'))
                transect_location = domain_settings.get('transect_location', settings.get('transect_location'))
                transect_width = domain_settings.get('transect_width', settings.get('transect_width', 0))
                z_offset = domain_settings.get('terrain_mask_height_z', settings.get('terrain_mask_height_z', 0))

                if domain in tf_settings:
                    self.logger.debug(f"Using domain-specific transect settings for '{domain}': "
                                     f"axis={transect_axis}, location={transect_location}, width={transect_width}")
            else:
                # For other methods, use global settings with defaults
                z_offset = settings.get('terrain_mask_height_z', 0)
                transect_axis = settings.get('transect_axis', 'x')
                transect_location = settings.get('transect_location', 100)
                transect_width = settings.get('transect_width', 0)

            # Route to appropriate extraction method
            if extraction_method == 'transect_direct':
                # DIRECT METHOD: Extract 1D transect directly from 4D data
                # Memory efficient, but no 2D context for visualization
                msg = f"\n=== Using DIRECT transect extraction for {scenario['label']} ==="
                print(msg)
                self.logger.info(msg)

                transect_values, coordinates, var_name, needs_conversion = \
                    self._extract_terrain_following_transect_direct(
                        dataset, static_dataset, domain, variable, z_offset,
                        transect_axis, transect_location, transect_width, settings
                    )

                # Get building/LAD masks if static data available
                masks = None
                if static_dataset is not None:
                    masks = self._get_building_lad_masks(
                        static_dataset, transect_axis, transect_location,
                        transect_width, domain
                    )

                return {
                    'transect_values': transect_values,
                    'coordinates': coordinates,
                    'xy_slice': None,  # No 2D context available with direct method
                    'label': scenario['label'],
                    'color': scenario['color'],
                    'linestyle': scenario.get('linestyle', '-'),
                    'linewidth': scenario.get('linewidth', 2.0),
                    'masks': masks,
                    'needs_kelvin_conversion': needs_conversion
                }

            elif extraction_method == 'terrain_following':
                # TERRAIN-FOLLOWING METHOD: Fill from lowest vertical level upwards
                # Iterates through z-levels, using first valid data encountered
                msg = f"\n=== Using TERRAIN-FOLLOWING extraction for {scenario['label']} ==="
                print(msg)
                self.logger.info(msg)

                # Get terrain-following specific settings
                tf_settings = settings.get('terrain_following', {})
                output_mode = tf_settings.get('output_mode', '2d')
                buildings_mask = tf_settings.get('buildings_mask', True)

                msg = f"  Output mode: {output_mode}, Buildings mask: {buildings_mask}"
                print(msg)
                self.logger.info(msg)

                if output_mode == '1d':
                    # 1D transect output (more memory efficient)
                    transect_values, coordinates, var_name, needs_conversion = \
                        self._extract_terrain_following(
                            dataset, static_dataset, domain, variable, buildings_mask, '1d',
                            transect_axis, transect_location, transect_width, settings
                        )

                    # Get building/LAD masks if static data available
                    masks = None
                    if static_dataset is not None:
                        masks = self._get_building_lad_masks(
                            static_dataset, transect_axis, transect_location,
                            transect_width, domain
                        )

                    return {
                        'transect_values': transect_values,
                        'coordinates': coordinates,
                        'xy_slice': None,  # No 2D context with 1D mode
                        'label': scenario['label'],
                        'color': scenario['color'],
                        'linestyle': scenario.get('linestyle', '-'),
                        'linewidth': scenario.get('linewidth', 2.0),
                        'masks': masks,
                        'needs_kelvin_conversion': needs_conversion
                    }
                else:  # '2d' mode
                    # 2D full field output (provides spatial context)
                    slice_2d, var_name, needs_conversion = \
                        self._extract_terrain_following(
                            dataset, static_dataset, domain, variable, buildings_mask, '2d',
                            settings=settings
                        )

                    # Extract transect from 2D field (using existing function)
                    transect_values, coordinates = self._extract_transect_line(
                        slice_2d, transect_axis, transect_location, transect_width
                    )

                    # Get building/LAD masks if static data available
                    masks = None
                    if static_dataset is not None:
                        masks = self._get_building_lad_masks(
                            static_dataset, transect_axis, transect_location,
                            transect_width, domain
                        )

                    return {
                        'transect_values': transect_values,
                        'coordinates': coordinates,
                        'xy_slice': slice_2d,  # 2D context available
                        'label': scenario['label'],
                        'color': scenario['color'],
                        'linestyle': scenario.get('linestyle', '-'),
                        'linewidth': scenario.get('linewidth', 2.0),
                        'masks': masks,
                        'needs_kelvin_conversion': needs_conversion
                    }

            else:  # extraction_method == 'slice_2d' (default)
                # SLICE METHOD: Extract full 2D slice, then extract transect
                # Less memory efficient, but provides 2D context for visualization
                msg = f"\n=== Using 2D slice extraction for {scenario['label']} ==="
                print(msg)
                self.logger.info(msg)

                # Extract terrain-following slice (ALREADY TIME-AVERAGED using xarray)
                slice_2d, var_name, needs_conversion = self._extract_terrain_following_slice(
                    dataset, static_dataset, domain, variable, z_offset, settings
                )

                # NOTE: Time averaging is done INSIDE _extract_terrain_following_slice()
                # using xarray.mean(dim='time') with optional time range selection via time_selection_method

                self.logger.info(f"Time-averaged 2D field ready for transect extraction: "
                               f"shape={slice_2d.shape}")

                # Extract transect line from 2D slice
                transect_values, coordinates = self._extract_transect_line(
                    slice_2d, transect_axis, transect_location, transect_width
                )

                # Log transect data
                self.logger.info(f"Transect extracted: "
                               f"length={len(transect_values)}, "
                               f"min={np.nanmin(transect_values):.2f}, "
                               f"max={np.nanmax(transect_values):.2f}, "
                               f"mean={np.nanmean(transect_values):.2f}")

                # Get building/LAD masks if static data available
                masks = None
                if static_dataset is not None:
                    masks = self._get_building_lad_masks(
                        static_dataset, transect_axis, transect_location,
                        transect_width, domain
                    )

                return {
                    'transect_values': transect_values,
                    'coordinates': coordinates,
                    'xy_slice': slice_2d,  # 2D context available with slice method
                    'label': scenario['label'],
                    'color': scenario['color'],
                    'linestyle': scenario.get('linestyle', '-'),  # Preserve linestyle (default to solid)
                    'linewidth': scenario.get('linewidth', 2.0),  # Preserve linewidth (default to 2.0)
                    'masks': masks,
                    'needs_kelvin_conversion': needs_conversion
                }

        except Exception as e:
            self.logger.error(f"Error extracting data for {scenario['label']}: {str(e)}")
            return None

    def _create_transect_plot(self, scenarios_data: List[Dict],
                             variable: str,
                             domain: str,
                             comparison_type: str,
                             settings: Dict) -> plt.Figure:
        """
        Create complete transect plot
        Layout adapts based on data availability:
          - Two panels (transect + map) if 2D context data available
          - Single panel (transect only) if using direct extraction method

        Args:
            scenarios_data: List of scenario data dictionaries
            variable: 'ta' or 'qv'
            domain: 'parent' or 'child'
            comparison_type: 'age' or 'spacing'
            settings: Plot settings

        Returns:
            Matplotlib figure
        """
        # Check if we have 2D context data (xy_slice) for map visualization
        has_2d_context = len(scenarios_data) > 0 and scenarios_data[0].get('xy_slice') is not None

        # Create figure with adaptive layout
        if has_2d_context:
            # Two-panel layout: transect + map
            fig, (ax_transect, ax_map) = plt.subplots(
                2, 1, figsize=(12, 8),
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3}
            )
            msg = "Creating two-panel plot (transect + map)"
            print(msg)
            self.logger.info(msg)
        else:
            # Single-panel layout: transect only
            fig, ax_transect = plt.subplots(figsize=(12, 6))
            ax_map = None
            msg = "Creating single-panel plot (transect only - direct extraction)"
            print(msg)
            self.logger.info(msg)

        # Determine variable properties from metadata system
        if self.var_metadata:
            try:
                var_config = self.var_metadata.get_variable_config(variable)
                var_display_name = var_config.get('label', variable.replace('_', ' ').title())
                var_units = var_config.get('units_out', '')
                var_label = f"{var_display_name} ({var_units})" if var_units else var_display_name
                var_range_config = var_config.get('value_range', 'auto')
                cmap = var_config.get('colormap', 'viridis')
                self.logger.info(f"Using metadata for variable '{variable}': label='{var_label}', range={var_range_config}, cmap={cmap}")
            except KeyError:
                # Fallback if variable not in metadata
                self.logger.warning(f"Variable '{variable}' not found in metadata, using legacy defaults")
                if variable == 'ta':
                    var_label = 'Air Temperature (°C)'
                    var_range_config = settings.get('temperature_range', [24.0, 28.6])
                    cmap = settings.get('temperature_cmap', 'RdBu_r')
                else:
                    var_label = variable.replace('_', ' ').title()
                    var_range_config = 'auto'
                    cmap = 'viridis'
        else:
            # No metadata system available - use legacy hardcoded logic
            if variable == 'ta':
                var_label = 'Air Temperature (°C)'
                var_range_config = settings.get('temperature_range', [24.0, 28.6])
                cmap = settings.get('temperature_cmap', 'RdBu_r')
            else:  # qv
                var_label = 'Water Vapor Mixing Ratio (kg/kg-1)'
                var_range_config = settings.get('qv_range', [0.00081, 0.00095])
                cmap = settings.get('qv_cmap', 'YlGnBu')

        # Get domain resolution for x-axis conversion
        if domain == 'parent':
            resolution = 10.0  # meters
        else:
            resolution = 2.0   # meters

        # Determine if we need Kelvin to Celsius conversion from first scenario
        needs_conversion = False
        if len(scenarios_data) > 0 and 'needs_kelvin_conversion' in scenarios_data[0]:
            needs_conversion = scenarios_data[0]['needs_kelvin_conversion']

        self.logger.info(f"Plotting with Kelvin conversion: {needs_conversion}")

        # Collect all y-values for automatic range detection
        all_y_values = []

        # Plot transect lines
        for scenario in scenarios_data:
            transect_values = scenario['transect_values']
            coordinates = scenario['coordinates']

            # Debug logging of transect values
            num_zeros = np.sum(transect_values == 0)
            num_nans = np.sum(np.isnan(transect_values))
            self.logger.debug(f"{scenario['label']}: Raw transect has {len(transect_values)} points, "
                            f"{num_nans} NaNs, {num_zeros} zeros, "
                            f"min={np.nanmin(transect_values):.2f}, max={np.nanmax(transect_values):.2f}")

            # Convert to physical units
            x_coords = coordinates * resolution

            # Apply unit conversion if needed
            if variable == 'ta' and needs_conversion:
                y_values = transect_values - 273.15  # Kelvin to Celsius
                # Use nanmin/nanmax to ignore NaN values in logging
                valid_original = transect_values[(~np.isnan(transect_values)) & (transect_values != 0)]
                valid_converted = y_values[(~np.isnan(y_values)) & (y_values != -273.15)]
                if len(valid_original) > 0 and len(valid_converted) > 0:
                    self.logger.debug(f"{scenario['label']}: Converting K to C, "
                                    f"original range [{np.min(valid_original):.2f}, {np.max(valid_original):.2f}], "
                                    f"converted range [{np.min(valid_converted):.2f}, {np.max(valid_converted):.2f}]")
            else:
                y_values = transect_values
                valid_y = y_values[(~np.isnan(y_values)) & (y_values != 0)]
                if len(valid_y) > 0:
                    self.logger.debug(f"{scenario['label']}: No conversion, "
                                    f"value range [{np.min(valid_y):.2f}, {np.max(valid_y):.2f}]")

            # Collect ONLY VALID y-values for auto-scaling
            # Filter out both NaN AND zero values (zeros indicate missing/invalid data in buildings)
            valid_mask = ~np.isnan(y_values) & (y_values != 0)
            valid_values = y_values[valid_mask]
            all_y_values.extend(valid_values)

            # Debug logging of filtered values
            num_nans = np.sum(np.isnan(y_values))
            num_zeros = np.sum(y_values == 0)
            num_valid = len(valid_values)
            if num_valid > 0:
                self.logger.debug(f"{scenario['label']}: {num_valid} valid values after filtering "
                                f"(excluded {num_nans} NaNs, {num_zeros} zeros), "
                                f"range [{np.min(valid_values):.2f}, {np.max(valid_values):.2f}]")

            # Replace zeros with NaN for proper visualization
            # Zeros in base case data represent building locations and should create line breaks
            y_values_plot = y_values.copy()
            y_values_plot[y_values_plot == 0] = np.nan

            # Get line styling from scenario (with defaults)
            linestyle = scenario.get('linestyle', '-')
            linewidth = scenario.get('linewidth', 2.0)

            ax_transect.plot(
                x_coords, y_values_plot,  # Plot with NaN instead of zeros
                label=scenario['label'],
                color=scenario['color'],
                linestyle=linestyle,
                linewidth=linewidth
            )

        # Add building and LAD shading (from first scenario with masks)
        for scenario in scenarios_data:
            if scenario['masks'] is not None:
                masks = scenario['masks']

                # Building shading
                if masks['buildings'] is not None:
                    coordinates = scenario['coordinates']
                    x_coords = coordinates * resolution

                    for i, is_building in enumerate(masks['buildings']):
                        if is_building:
                            ax_transect.axvspan(
                                x_coords[i] - resolution/2,
                                x_coords[i] + resolution/2,
                                alpha=settings.get('building_alpha', 0.5),
                                color=settings.get('building_color', 'grey'),
                                zorder=0
                            )

                # LAD shading (green where LAD > 0)
                if masks['lad'] is not None:
                    coordinates = scenario['coordinates']
                    x_coords = coordinates * resolution

                    for i, lad_value in enumerate(masks['lad']):
                        if lad_value > 0:
                            ax_transect.axvspan(
                                x_coords[i] - resolution/2,
                                x_coords[i] + resolution/2,
                                alpha=settings.get('lad_alpha', 0.5),
                                color=settings.get('lad_color', 'green'),
                                zorder=0
                            )

                # Only use masks from first valid scenario
                break

        # Determine actual range to use (auto or fixed)
        if var_range_config == "auto":
            # Automatic scaling based on actual data
            self.logger.info(f"Auto-scaling with {len(all_y_values)} valid values collected")
            if len(all_y_values) > 0:
                data_min = np.min(all_y_values)
                data_max = np.max(all_y_values)
                self.logger.info(f"  Data range: min={data_min:.2f}, max={data_max:.2f}")
                # Add 5% padding for visual clarity
                padding = (data_max - data_min) * 0.05
                var_range = [data_min - padding, data_max + padding]
                self.logger.info(f"  Auto Y-axis range (with 5% padding): [{var_range[0]:.2f}, {var_range[1]:.2f}]")
            else:
                # Fallback to default if no data
                var_range = [24.0, 28.6] if variable == 'ta' else [0.00081, 0.00095]
                self.logger.warning(f"No data for auto-scaling, using default range: {var_range}")
        else:
            # Use fixed range from config (legacy mode)
            var_range = var_range_config
            self.logger.info(f"Using fixed Y-axis range from config: {var_range}")

        # Format top panel
        ax_transect.set_ylabel(var_label)
        ax_transect.set_ylim(var_range)
        ax_transect.legend(loc='best', framealpha=0.9)
        ax_transect.grid(True, alpha=0.3)

        # Create title using variable display name from metadata
        domain_str = domain.capitalize()
        # Use the display name extracted from metadata (or fallback)
        if self.var_metadata:
            try:
                var_config = self.var_metadata.get_variable_config(variable)
                var_str = var_config.get('label', variable.replace('_', ' ').title())
            except KeyError:
                var_str = variable.replace('_', ' ').title()
        else:
            var_str = 'Air Temperature' if variable == 'ta' else variable.replace('_', ' ').title()

        comp_str = 'Tree Ages' if comparison_type == 'age' else 'Tree Spacing'
        title = f'THF Forest Spacing {domain_str} - {comp_str} - {var_str} Average Transect'
        ax_transect.set_title(title, fontsize=14, fontweight='bold')

        # Plot XY plan view in bottom panel (only if 2D context available)
        if ax_map is not None and len(scenarios_data) > 0:
            xy_slice = scenarios_data[0]['xy_slice']

            if xy_slice is not None:
                # Convert to physical units
                if variable == 'ta' and needs_conversion:
                    xy_slice_plot = xy_slice - 273.15  # Kelvin to Celsius
                    self.logger.info(f"XY map: Converting K to C, "
                                   f"original range [{np.nanmin(xy_slice):.2f}, {np.nanmax(xy_slice):.2f}], "
                                   f"converted range [{np.nanmin(xy_slice_plot):.2f}, {np.nanmax(xy_slice_plot):.2f}]")
                else:
                    xy_slice_plot = xy_slice
                    self.logger.info(f"XY map: No conversion, "
                                   f"value range [{np.nanmin(xy_slice_plot):.2f}, {np.nanmax(xy_slice_plot):.2f}]")

                # Plot as colored map
                extent_x = xy_slice.shape[1] * resolution
                extent_y = xy_slice.shape[0] * resolution
                self.logger.info(f"XY map extent: x=[0, {extent_x}], y=[0, {extent_y}], shape={xy_slice.shape}")

                im = ax_map.imshow(
                    xy_slice_plot,
                    cmap=cmap,
                    vmin=var_range[0],
                    vmax=var_range[1],
                    origin='lower',
                    extent=[0, extent_x, 0, extent_y],
                    aspect='equal'
                )

                # Add transect line - handle domain-specific settings for terrain_following
                # Check if we're using terrain_following with domain-specific settings
                extraction_method = settings.get('extraction_method', 'slice_2d')
                if extraction_method == 'terrain_following':
                    tf_settings = settings.get('terrain_following', {})
                    domain_settings = tf_settings.get(domain, {})

                    # Use domain-specific if available, else fall back to global settings
                    transect_axis = domain_settings.get('transect_axis',
                                                       settings.get('transect_axis', 'x'))
                    transect_location = domain_settings.get('transect_location',
                                                           settings.get('transect_location', 100))
                else:
                    # For other methods, use global settings
                    transect_axis = settings.get('transect_axis', 'x')
                    transect_location = settings.get('transect_location', 100)

                if transect_axis == 'x':
                    # Horizontal line at constant y
                    y_pos = transect_location * resolution
                    ax_map.axhline(
                        y_pos,
                        color=settings.get('transect_line_color', 'magenta'),
                        linestyle=settings.get('transect_line_style', '--'),
                        linewidth=2,
                        label='Transect Line'
                    )
                else:
                    # Vertical line at constant x
                    x_pos = transect_location * resolution
                    ax_map.axvline(
                        x_pos,
                        color=settings.get('transect_line_color', 'magenta'),
                        linestyle=settings.get('transect_line_style', '--'),
                        linewidth=2,
                        label='Transect Line'
                    )

                # Format bottom panel
                ax_map.set_xlabel('X Axis (m)')
                ax_map.set_ylabel('Y Axis (m)')
                ax_map.set_aspect('equal')

        plt.tight_layout()

        return fig
