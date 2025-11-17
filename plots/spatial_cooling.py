"""
Spatial Cooling Plotter for Slide 8
Visualizes horizontal temperature distribution at 2m height
Fixed version with proper settings handling and dimension alignment
Enhanced to support multiple variables with terrain-following and PCM support
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
import logging
import xarray as xr
import pandas as pd

from .base_plotter import BasePlotter
from pathlib import Path

try:
    from ..core.variable_metadata import VariableMetadata
except ImportError:
    from core.variable_metadata import VariableMetadata

# Import terrain-following extraction from terrain_transect
try:
    from .terrain_transect import TerrainTransectPlotter
except ImportError:
    from plots.terrain_transect import TerrainTransectPlotter


class SpatialCoolingPlotter(BasePlotter):
    """Creates visualizations for spatial cooling patterns"""

    def __init__(self, config: Dict, output_manager):
        """
        Initialize spatial cooling plotter

        Args:
            config: Configuration dictionary
            output_manager: Output manager instance
        """
        super().__init__(config, output_manager)
        self.logger = logging.getLogger(__name__)

        # Initialize VariableMetadata for dynamic variable handling
        self.var_metadata = VariableMetadata(config, self.logger)

        # Create helper instance for terrain-following extraction
        # This allows us to borrow methods from terrain_transect.py
        self._terrain_helper = TerrainTransectPlotter(config, output_manager)

        # Get cache directories from GLOBAL terrain_following config (NEW)
        self.terrain_mask_cache_dir = None
        self.surface_data_cache_dir = None

        try:
            # Get global terrain_following settings
            tf_global = config.get('plots', {}).get('terrain_following', {})

            if tf_global:
                # Get terrain mask cache directory
                mask_cache = tf_global.get('mask_cache', {})
                if mask_cache.get('enabled', False):
                    cache_dir = mask_cache.get('cache_directory')
                    if cache_dir:
                        self.terrain_mask_cache_dir = Path(cache_dir).expanduser().resolve()
                        self.logger.info(f"Terrain mask cache directory: {self.terrain_mask_cache_dir}")

                # Get surface data cache directory
                surface_cache = tf_global.get('surface_data_cache', {})
                if surface_cache.get('enabled', False):
                    cache_dir = surface_cache.get('cache_directory')
                    if cache_dir:
                        self.surface_data_cache_dir = Path(cache_dir).expanduser().resolve()
                        self.logger.info(f"Surface data cache directory: {self.surface_data_cache_dir}")
            else:
                self.logger.warning("No global terrain_following config found, caching disabled")
        except Exception as e:
            self.logger.warning(f"Could not determine cache directories: {e}")

    # ========================================================================
    # Terrain-Following Extraction Methods (delegate to terrain_transect.py)
    # ========================================================================

    def _extract_terrain_following_2d(
        self,
        dataset: xr.Dataset,
        static_dataset: xr.Dataset,
        domain_type: str,
        variable: str,
        settings: Dict
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Extract 2D terrain-following slice by delegating to terrain_transect.py.

        This method borrows the proven terrain-following extraction logic
        from fig_6 (terrain_transect.py) and applies it to fig_3's 2D spatial plots.

        Args:
            dataset: xarray Dataset containing variable data
            static_dataset: xarray Dataset containing topography
            domain_type: 'parent' or 'child'
            variable: Variable name (config key)
            settings: Extraction settings dict (must include terrain_following config)

        Returns:
            Tuple of (2D array [y, x], variable_name, needs_kelvin_conversion)

        Raises:
            ValueError: If extraction fails
        """
        self.logger.info(f"Delegating 2D terrain-following extraction to terrain_helper")

        # Delegate to terrain_transect's _extract_terrain_following() method
        # with output_mode='2d' to get full spatial field
        try:
            data_2d, var_name, needs_conversion = self._terrain_helper._extract_terrain_following(
                dataset=dataset,
                static_dataset=static_dataset,
                domain_type=domain_type,
                variable=variable,
                buildings_mask=settings.get('terrain_following', {}).get('buildings_mask', True),
                output_mode='2d',  # Request 2D field (not transect)
                settings=settings
            )

            self.logger.info(f"Successfully extracted 2D terrain-following data for {variable}")
            self.logger.info(f"  Shape: {data_2d.shape}, Variable: {var_name}, Needs K→C: {needs_conversion}")

            return data_2d, var_name, needs_conversion

        except Exception as e:
            self.logger.error(f"Failed to extract terrain-following data: {e}")
            raise ValueError(f"Terrain-following extraction failed for {variable}: {e}")

    def _is_pcm_variable(self, variable: str, dataset: xr.Dataset) -> bool:
        """
        Check if variable is a PCM (Plant Canopy Model) variable.

        Delegates to terrain_helper for consistent PCM detection.

        Args:
            variable: Variable name (config key)
            dataset: xarray Dataset to check

        Returns:
            bool: True if variable uses zpc_3d coordinate (PCM variable)
        """
        return self._terrain_helper._is_pcm_variable(variable, dataset)

    # ========================================================================
    # Time Window and Cache Key Management (NEW)
    # ========================================================================

    def _determine_time_mode(self, hour) -> Tuple[str, Optional[Dict]]:
        """
        Determine standardized time_mode and parameters from hour specification.

        Args:
            hour: Time specification - can be:
                  - int: single hour of day (0-23) → 'single_time'
                  - list [start, end]: simulation hour range → 'time_window'
                  - None: all times → 'all_times_average'

        Returns:
            Tuple of (time_mode: str, time_params: Optional[Dict])

            Examples:
                hour=14 → ('single_time', {'hour': 14})
                hour=[30, 42] → ('time_window', {'start': 30, 'end': 42})
                hour=None → ('all_times_average', None)
        """
        if hour is None:
            return 'all_times_average', None

        elif isinstance(hour, list) and len(hour) == 2:
            # Time window: [start_hour, end_hour]
            return 'time_window', {'start': hour[0], 'end': hour[1]}

        elif isinstance(hour, int):
            # Single hour of day
            return 'single_time', {'hour': hour}

        else:
            # Default fallback
            self.logger.warning(f"Unexpected hour format: {hour}, defaulting to all_times_average")
            return 'all_times_average', None

    def _load_from_enhanced_cache(
        self,
        dataset: xr.Dataset,
        static_dataset: xr.Dataset,
        variable: str,
        domain_type: str,
        time_mode: str,
        time_params: Optional[Dict],
        settings: Dict
    ) -> Optional[np.ndarray]:
        """
        Load terrain-following data from enhanced cache with time_mode and building_mask.

        Args:
            dataset: xarray Dataset containing variable data
            static_dataset: xarray Dataset with topography
            variable: Variable name (config key)
            domain_type: 'parent' or 'child'
            time_mode: 'all_times_average', 'time_window', or 'single_time'
            time_params: Time parameters dict (start/end for time_window, hour for single_time)
            settings: Extraction settings dict

        Returns:
            2D numpy array if cache hit, None if cache miss
        """
        if self.terrain_mask_cache_dir is None:
            self.logger.debug("Terrain mask cache disabled")
            return None

        try:
            # Get case name
            case_name = self._get_case_name_from_dataset(dataset)
            if case_name is None:
                self.logger.debug("Could not determine case name for cache lookup")
                return None

            # Get building mask setting
            building_mask = settings.get('terrain_following', {}).get('buildings_mask', True)

            # Import enhanced cache utilities
            try:
                from ..utils.netcdf_utils import find_existing_mask_file_enhanced
            except ImportError:
                from utils.netcdf_utils import find_existing_mask_file_enhanced

            # Try to find enhanced cache file
            cache_file = find_existing_mask_file_enhanced(
                cache_dir=self.terrain_mask_cache_dir,
                case_name=case_name,
                domain_type=domain_type,
                time_mode=time_mode,
                time_params=time_params,
                building_mask=building_mask,
                fallback_to_legacy=True  # Enable fallback to old naming
            )

            if cache_file is None:
                self.logger.debug(
                    f"Cache miss: {case_name}, {domain_type}, "
                    f"time_mode={time_mode}, building_mask={building_mask}"
                )
                return None

            # Load and extract variable
            self.logger.info(f"Loading from enhanced cache: {cache_file.name}")

            # Get PALM variable name
            palm_name = self.var_metadata.get_palm_name(variable)

            cache_ds = xr.open_dataset(cache_file)

            if palm_name not in cache_ds.data_vars:
                self.logger.debug(f"Variable '{palm_name}' not in cache file {cache_file.name}")
                cache_ds.close()
                return None

            # Extract data at z_offset
            z_offset = settings.get('terrain_mask_height_z', 0)
            var_data = cache_ds[palm_name]

            if 'ku_above_surf' in var_data.dims:
                var_data = var_data.sel(ku_above_surf=z_offset)
            else:
                self.logger.warning(f"ku_above_surf dimension not found in cached {palm_name}")
                cache_ds.close()
                return None

            # Squeeze out time dimension if present
            if 'time' in var_data.dims:
                var_data = var_data.isel(time=0)

            data_array = var_data.values
            cache_ds.close()

            # Clean invalid data
            data_array = self._clean_invalid_data(data_array, variable)

            self.logger.info(f"✓ Successfully loaded {variable} from enhanced cache")
            return data_array

        except Exception as e:
            self.logger.debug(f"Error loading from enhanced cache: {e}")
            return None

    def _save_to_enhanced_cache(
        self,
        data_2d: np.ndarray,
        dataset: xr.Dataset,
        static_dataset: xr.Dataset,
        variable: str,
        domain_type: str,
        time_mode: str,
        time_params: Optional[Dict],
        settings: Dict
    ) -> bool:
        """
        Save terrain-following data to enhanced cache.

        NOTE: Cache saving is handled automatically by terrain_helper's
        _extract_terrain_following() method. When cache mode is 'auto' or 'save',
        the terrain_helper computes terrain masks for all requested offset levels
        and saves them to the cache directory specified in the config.

        This method is intentionally a no-op because:
        1. terrain_helper already saves the full 3D mask (multiple offset levels)
        2. We only extract a single 2D slice here (one offset level)
        3. Saving individual slices would create redundant cache files

        The cache saving happens inside _extract_terrain_following() around line 3183
        of terrain_transect.py, which is called by our _extract_terrain_following_2d()
        delegation method.

        Args:
            data_2d: 2D data array to save (unused)
            dataset: Original dataset (unused)
            static_dataset: Static dataset with topography (unused)
            variable: Variable name (unused)
            domain_type: 'parent' or 'child' (unused)
            time_mode: Time mode string (unused)
            time_params: Time parameters dict (unused)
            settings: Extraction settings (unused)

        Returns:
            False (cache saving handled by terrain_helper)
        """
        # Cache saving is automatically handled by terrain_helper during extraction
        # No additional action needed here
        self.logger.debug("Cache saving handled by terrain_helper during extraction")
        return False

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _load_static_dataset(self, dataset: xr.Dataset) -> Optional[xr.Dataset]:
        """
        Load the corresponding static dataset for a given dataset.

        Args:
            dataset: xarray Dataset to find corresponding static file for

        Returns:
            Static dataset or None if not found
        """
        try:
            # Extract static file path from dataset source
            source = dataset.encoding.get('source', '')
            if not source:
                self.logger.warning("Could not determine static file path - no source in dataset")
                return None

            # Convert path to find corresponding static file
            # Example: .../OUTPUT/merged_files/case_av_3d_merged.nc -> .../INPUT/case_static
            source_path = Path(source)
            case_dir = source_path.parent.parent.parent  # Go up from OUTPUT/merged_files/

            # Determine case name and domain suffix
            filename = source_path.stem  # e.g., "thf_base_2018080700_av_3d_N02_merged"
            is_child = 'N02' in filename or '_n02' in filename.lower()

            # Extract case name (remove _av_3d, _N02, _merged, etc.)
            case_name = filename.split('_av_')[0]  # Get part before _av_3d

            # Build static file path
            domain_suffix = '_N02' if is_child else ''
            static_file = case_dir / 'INPUT' / f"{case_name}_static{domain_suffix}"

            if not static_file.exists():
                self.logger.warning(f"Static file not found: {static_file}")
                return None

            # Load static dataset
            static_ds = xr.open_dataset(static_file)
            self.logger.info(f"Loaded static dataset from {static_file.name}")
            return static_ds

        except Exception as e:
            self.logger.warning(f"Error loading static dataset: {str(e)}")
            return None

    def _get_case_name_from_dataset(self, dataset: xr.Dataset) -> Optional[str]:
        """
        Extract case name from dataset source attribute

        Args:
            dataset: xarray dataset

        Returns:
            Case name or None if not found
        """
        try:
            source = dataset.encoding.get('source', '')
            if source:
                # Extract case name from path
                # Patterns to match:
                #   - Tree scenarios: thf_forest_lad_spacing_10m_age_20yrs
                #   - Base case: thf_base_2018080700
                parts = Path(source).parts
                for part in parts:
                    # Check for tree scenario pattern
                    if part.startswith('thf_forest_lad_spacing_'):
                        return part
                    # Check for base case pattern
                    if part.startswith('thf_base_'):
                        return part
        except Exception as e:
            self.logger.debug(f"Could not extract case name: {e}")
        return None

    def _build_cache_file_paths(self, case_name: str, is_child: bool, z_offset: int) -> List[Path]:
        """
        Build list of possible terrain mask cache file paths

        Args:
            case_name: Simulation case name (e.g., 'thf_forest_lad_spacing_10m_age_20yrs')
            is_child: Whether this is child domain
            z_offset: Z-offset for terrain-following extraction

        Returns:
            List of possible cache file paths (ordered by preference)
        """
        if self.terrain_mask_cache_dir is None:
            return []

        # Determine domain suffix
        domain_suffix = '_child' if is_child else ''

        # Build filename based on available offsets
        # The cache files are named with offset ranges like TF0-10 or TF1-2-5-10
        # TF1-2-5-10 files contain more variables (including ta, q, theta, etc.)
        # TF0-10 files only contain PCM variables
        # We prefer TF1-2-5-10 files for atmospheric variables

        # Order matters: prefer TF1-2-5-10 first (has atmospheric vars)
        possible_patterns = [
            f"{case_name}_terrain_mask{domain_suffix}_TF1-2-5-10.nc",  # Full variable set
            f"{case_name}_terrain_mask{domain_suffix}_TF0-10.nc",      # PCM variables only
        ]

        valid_files = []
        for pattern in possible_patterns:
            cache_file = self.terrain_mask_cache_dir / pattern
            if cache_file.exists():
                # Verify this file contains our z_offset
                try:
                    ds = xr.open_dataset(cache_file)
                    ku_values = ds['ku_above_surf'].values
                    ds.close()
                    if z_offset in ku_values:
                        self.logger.debug(f"Found cache file: {cache_file}")
                        valid_files.append(cache_file)
                except Exception as e:
                    self.logger.debug(f"Error checking cache file {cache_file}: {e}")
                    continue

        return valid_files

    def _load_topography(self, dataset: xr.Dataset) -> Optional[np.ndarray]:
        """
        Load topography data (terrain height) from static file

        Args:
            dataset: xarray dataset (used to find corresponding static file)

        Returns:
            2D numpy array of terrain heights, or None if not found
        """
        try:
            # Extract static file path from dataset source
            source = dataset.encoding.get('source', '')
            if not source:
                self.logger.warning("Could not determine static file path - no source in dataset")
                return None

            # Convert path to find corresponding static file
            # Example: .../OUTPUT/merged_files/case_av_3d_merged.nc -> .../INPUT/case_static
            source_path = Path(source)
            case_dir = source_path.parent.parent.parent  # Go up from OUTPUT/merged_files/

            # Determine case name and domain suffix
            filename = source_path.stem  # e.g., "thf_base_2018080700_av_3d_N02_merged"
            is_child = 'N02' in filename or '_n02' in filename.lower()

            # Extract case name (remove _av_3d, _N02, _merged, etc.)
            case_name = filename.split('_av_')[0]  # Get part before _av_3d

            # Build static file path
            domain_suffix = '_N02' if is_child else ''
            static_file = case_dir / 'INPUT' / f"{case_name}_static{domain_suffix}"

            if not static_file.exists():
                self.logger.warning(f"Static file not found: {static_file}")
                return None

            # Load topography data
            static_ds = xr.open_dataset(static_file)
            if 'zt' not in static_ds.data_vars:
                self.logger.warning(f"Topography variable 'zt' not found in {static_file}")
                static_ds.close()
                return None

            zt = static_ds['zt'].values
            static_ds.close()

            self.logger.info(f"Loaded topography from {static_file.name}, shape: {zt.shape}")
            return zt

        except Exception as e:
            self.logger.warning(f"Error loading topography: {str(e)}")
            return None

    def _compute_terrain_following_mask(self, dataset: xr.Dataset, z_dim: str = 'zu_3d',
                                       topography: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute terrain-following mask (ku_above_surf) from topography data

        This determines, for each horizontal grid point, the vertical index of the
        first grid level above the terrain surface.

        Args:
            dataset: xarray dataset containing vertical coordinates
            z_dim: Name of vertical coordinate ('zu_3d' or 'zpc_3d')
            topography: 2D array of terrain heights (if None, will load from static file)

        Returns:
            2D numpy array of indices (ku_above_surf), or None if computation fails
        """
        try:
            # Load topography if not provided
            if topography is None:
                topography = self._load_topography(dataset)
                if topography is None:
                    self.logger.warning("Cannot compute terrain mask without topography data")
                    return None

            # Get vertical coordinate
            if z_dim not in dataset.coords:
                self.logger.warning(f"Vertical coordinate '{z_dim}' not found in dataset")
                return None

            z_levels = dataset[z_dim].values
            ny, nx = topography.shape
            nz = len(z_levels)

            # Initialize ku_above_surf array
            ku_above_surf = np.zeros((ny, nx), dtype=np.int32)

            # Compute terrain-following indices
            # For each horizontal grid point, find first vertical level above terrain
            for iy in range(ny):
                for ix in range(nx):
                    terrain_height = topography[iy, ix]

                    # Find first level above terrain
                    for k in range(nz):
                        if z_levels[k] > terrain_height:
                            ku_above_surf[iy, ix] = k
                            break
                    else:
                        # If no level found above terrain, use last level
                        ku_above_surf[iy, ix] = nz - 1

            self.logger.info(f"Computed terrain-following mask: ku_above_surf range [{ku_above_surf.min()}, {ku_above_surf.max()}]")
            return ku_above_surf

        except Exception as e:
            self.logger.error(f"Error computing terrain-following mask: {str(e)}")
            return None

    def _clean_invalid_data(self, data_array: np.ndarray, variable: str) -> np.ndarray:
        """
        Clean invalid data values and replace with NaN

        Args:
            data_array: Data array to clean
            variable: Variable name to determine appropriate thresholds

        Returns:
            Cleaned data array with invalid values replaced by NaN
        """
        # Make a copy to avoid modifying the original
        cleaned = data_array.copy()

        # Replace fill values with NaN
        fill_value = -999999.0
        cleaned = np.where(np.abs(cleaned - fill_value) < 1e-6, np.nan, cleaned)

        # Variable-specific invalid value handling
        if variable in ['temperature', 'theta', 'potential_temperature']:
            # Temperature can be in Kelvin (200-400K) or Celsius (-100 to 100°C)
            # Auto-detect based on data range
            valid_data = cleaned[~np.isnan(cleaned)]
            if len(valid_data) > 0:
                data_median = np.median(valid_data)

                # If median > 100, assume Kelvin; otherwise assume Celsius
                if data_median > 100:
                    # Kelvin: reasonable range 200-400K
                    invalid_mask = (cleaned < 200) | (cleaned > 400)
                else:
                    # Celsius: reasonable range -50 to 70°C
                    invalid_mask = (cleaned < -50) | (cleaned > 70)

                # Also mark exact zeros as invalid (missing data indicator)
                invalid_mask = invalid_mask | (np.abs(cleaned) < 1e-10)
                cleaned = np.where(invalid_mask, np.nan, cleaned)

        elif variable in ['humidity_q', 'qv', 'water_vapor']:
            # Water vapor mixing ratio should be >= 0
            # Negative values or very large values are invalid
            invalid_mask = (cleaned < 0) | (cleaned > 0.1)  # > 100 g/kg is unrealistic
            cleaned = np.where(invalid_mask, np.nan, cleaned)

        elif variable in ['pressure', 'p']:
            # Atmospheric pressure should be positive and reasonable
            invalid_mask = (cleaned < 50000) | (cleaned > 120000)  # 500-1200 hPa range
            cleaned = np.where(invalid_mask, np.nan, cleaned)

        # Generic check: replace exactly zero values with NaN for most atmospheric variables
        # (except variables where zero is physically meaningful, like vertical velocity)
        elif variable not in ['w', 'vertical_velocity', 'pcm_transpirationrate', 'pcm_heatrate',
                              'pcm_latentrate', 'pcm_transpirationvolume']:
            # For most variables, exact zeros often indicate missing/invalid data
            zero_threshold = 1e-15
            cleaned = np.where(np.abs(cleaned) < zero_threshold, np.nan, cleaned)

        return cleaned

    def _load_from_terrain_mask_cache(self, dataset: xr.Dataset, variable: str,
                                      z_offset: int, is_child: bool) -> Optional[np.ndarray]:
        """
        Load variable data from terrain mask cache

        NOTE: Cache files are created by fig_6 and contain time-averaged terrain-following
        data. They have already been processed for PCM zero-to-NaN conversion if applicable.

        Args:
            dataset: Original dataset (used to get case name)
            variable: Variable name to load (e.g., 'temperature')
            z_offset: Z-offset for ku_above_surf dimension
            is_child: Whether this is child domain

        Returns:
            2D numpy array of data or None if not found in cache
        """
        try:
            # Get case name
            case_name = self._get_case_name_from_dataset(dataset)
            if case_name is None:
                print(f"[fig_3] Cache: Could not determine case name")
                self.logger.debug("Could not determine case name for cache lookup")
                return None

            print(f"[fig_3] Cache: case_name={case_name}")

            # Get PALM variable name
            palm_name = self.var_metadata.get_palm_name(variable)
            print(f"[fig_3] Cache: Looking for PALM variable '{palm_name}'")

            # Build list of possible cache file paths (ordered by preference)
            cache_files = self._build_cache_file_paths(case_name, is_child, z_offset)
            if not cache_files:
                print(f"[fig_3] Cache: No cache files found for offset={z_offset}")
                return None

            # Try each cache file until we find one with the required variable
            for cache_file in cache_files:
                print(f"[fig_3] Cache: Trying file: {cache_file.name}")

                try:
                    cache_ds = xr.open_dataset(cache_file)

                    # Check if variable exists in this cache file
                    if palm_name not in cache_ds.data_vars:
                        available_vars = list(cache_ds.data_vars.keys())[:5]
                        print(f"[fig_3] Cache: '{palm_name}' not in {cache_file.name} (has {available_vars}...)")
                        cache_ds.close()
                        continue  # Try next file

                    # Variable found! Extract data
                    print(f"[fig_3] Cache: ✓ Found '{palm_name}' in {cache_file.name}")

                    # Extract data at specified ku_above_surf level
                    var_data = cache_ds[palm_name]

                    # Select the z_offset level
                    if 'ku_above_surf' in var_data.dims:
                        var_data = var_data.sel(ku_above_surf=z_offset)
                    else:
                        self.logger.warning(f"ku_above_surf dimension not found in cached {palm_name}")
                        cache_ds.close()
                        continue  # Try next file

                    # Squeeze out time dimension (cache is time-averaged)
                    if 'time' in var_data.dims:
                        var_data = var_data.isel(time=0)

                    # Get numpy array
                    data_array = var_data.values

                    cache_ds.close()

                    # Clean invalid data (fill values, zeros, out-of-range values)
                    data_array = self._clean_invalid_data(data_array, variable)

                    # Report data range for debugging
                    valid_data = data_array[~np.isnan(data_array)]
                    if len(valid_data) > 0:
                        print(f"[fig_3] Cache: Data range: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}]")
                    else:
                        print(f"[fig_3] Cache: WARNING - All data is NaN after cleaning!")

                    print(f"[fig_3] Cache: Successfully loaded {palm_name} from cache")
                    self.logger.info(f"Loaded {variable} (PALM: {palm_name}) from cache at offset={z_offset}")
                    return data_array

                except Exception as e:
                    print(f"[fig_3] Cache: Error reading {cache_file.name}: {e}")
                    continue  # Try next file

            # If we get here, none of the cache files had the variable
            print(f"[fig_3] Cache: Variable '{palm_name}' not found in any cache file")
            return None

        except Exception as e:
            self.logger.warning(f"Error loading from cache: {str(e)}")
            return None

    def generate_plot(self, plot_type: str, data: Dict) -> plt.Figure:
        """
        Generate specific plot type
        
        Args:
            plot_type: Type of plot to generate
            data: Loaded simulation data
            
        Returns:
            Matplotlib figure
        """
        if plot_type == "daytime_cooling":
            return self._plot_daytime_cooling(data)
        elif plot_type == "nighttime_cooling":
            return self._plot_nighttime_cooling(data)
        elif plot_type == "cooling_extent":
            return self._plot_cooling_extent(data)
        elif plot_type == "temperature_maps":
            return self._plot_temperature_maps(data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
    def _plot_daytime_cooling(self, data: Dict) -> plt.Figure:
        """Create spatial cooling pattern plot for daytime"""
        # Get settings - works with both figures and slides config
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work

        # Get variable from config (default to temperature for backward compatibility)
        fig_config = self.config['plots'].get('figures', {}).get('fig_3',
                                              self.config['plots'].get('slides', {}).get('slide_8', {}))
        variable = fig_config.get('variable', 'temperature')

        daytime_hour = settings['daytime_hour']

        # Generate dynamic title
        time_label = f"{daytime_hour}" if isinstance(daytime_hour, int) else f"hours {daytime_hour[0]}-{daytime_hour[1]}"
        title = self._generate_figure_title(variable, f"Daytime ({time_label})")

        return self._create_spatial_comparison(data, variable, daytime_hour, settings, title)

    def _plot_nighttime_cooling(self, data: Dict) -> plt.Figure:
        """Create spatial cooling pattern plot for nighttime"""
        # Get settings - works with both figures and slides config
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work

        # Get variable from config
        fig_config = self.config['plots'].get('figures', {}).get('fig_3',
                                              self.config['plots'].get('slides', {}).get('slide_8', {}))
        variable = fig_config.get('variable', 'temperature')

        nighttime_hour = settings['nighttime_hour']

        # Generate dynamic title
        time_label = f"{nighttime_hour}" if isinstance(nighttime_hour, int) else f"hours {nighttime_hour[0]}-{nighttime_hour[1]}"
        title = self._generate_figure_title(variable, f"Nighttime ({time_label})")

        return self._create_spatial_comparison(data, variable, nighttime_hour, settings, title)
        
    def _extract_spatial_data(self, dataset: xr.Dataset, variable: str,
                             hour, height: float) -> np.ndarray:
        """
        Extract 2D variable field at specific hour or time window and height

        Args:
            dataset: xarray dataset containing variable data
            variable: Variable name to extract (e.g., 'ta', 'q', 'bio_utci*_xy')
            hour: Single hour (int, 0-23) or time range (list, [start, end] in simulation hours 0-48)
            height: Height in meters

        Returns:
            2D numpy array of variable values
        """
        try:
            # Find the appropriate height index
            # For child domain (N02), first data level is at zu_3d index 21
            # For parent domain, first data level is at zu_3d index 25
            if 'N02' in dataset.encoding.get('source', ''):
                z_idx = 21  # First data level for child domain
            else:
                z_idx = 25  # First data level for parent domain

            # Get variable data using VariableMetadata for dynamic lookup
            var_data, var_name = self.var_metadata.find_variable_in_dataset(dataset, variable)
            self.logger.debug(f"Found variable '{var_name}' for requested '{variable}'")

            # Handle time selection
            if isinstance(hour, list) and len(hour) == 2:
                # Time range: [start_hour, end_hour] in simulation hours
                start_hour, end_hour = hour
                self.logger.info(f"Extracting time window: simulation hours {start_hour}-{end_hour}")

                # Get time coordinates in simulation hours (assuming time in seconds)
                time_seconds = dataset['time'].values.astype('timedelta64[s]').astype(float)
                time_hours = time_seconds / 3600.0

                # Filter out NaN/inf values from time conversion issues (e.g., NaT values)
                valid_time_mask = np.isfinite(time_hours) & (time_hours >= 0)
                time_mask = valid_time_mask & (time_hours >= start_hour) & (time_hours <= end_hour)

                if not any(time_mask):
                    self.logger.warning(f"No data found for time range {start_hour}-{end_hour}")
                    return var_data.isel(zu_3d=z_idx).mean(dim='time').values

                # Extract and average over time window
                var_window = var_data.isel(time=time_mask, zu_3d=z_idx)
                var_field = var_window.mean(dim='time').values

            else:
                # Single hour: hour of day (0-23)
                time_pd = pd.to_datetime(dataset['time'].values)
                hour_mask = time_pd.hour == hour

                if not any(hour_mask):
                    self.logger.warning(f"No data found for hour {hour}")
                    result = var_data.isel(zu_3d=z_idx).mean(dim='time').values
                    return self._clean_invalid_data(result, variable)

                # Get variable data at specified hour and height
                var_hourly = var_data.isel(time=hour_mask, zu_3d=z_idx)
                var_field = var_hourly.mean(dim='time').values

            # Clean invalid data (convert zeros to NaN for proper masking)
            var_field = self._clean_invalid_data(var_field, variable)

            return var_field

        except Exception as e:
            self.logger.error(f"Error extracting spatial data for variable '{variable}': {str(e)}")
            raise

    def _extract_spatial_data_terrain_following(self, dataset: xr.Dataset, variable: str,
                                               hour, z_offset: int = 0) -> np.ndarray:
        """
        Extract 2D variable field at specific hour or time window using terrain-following coordinates.

        This method now uses the enhanced cache system with time_mode detection and
        delegates extraction to terrain_helper for consistency with fig_6.

        Args:
            dataset: xarray dataset containing variable data
            variable: Variable name to extract (e.g., 'ta', 'q', 'bio_utci*_xy')
            hour: Single hour (int, 0-23) or time range (list, [start, end] in simulation hours 0-48)
            z_offset: Z-index offset from terrain surface (0 = first grid point above surface)

        Returns:
            2D numpy array of variable values
        """
        try:
            # Determine domain type
            is_child = 'N02' in dataset.encoding.get('source', '')
            domain_type = 'child' if is_child else 'parent'

            # Determine time_mode and parameters from hour specification
            time_mode, time_params = self._determine_time_mode(hour)

            self.logger.info(
                f"Extracting {variable} in terrain-following mode: "
                f"domain={domain_type}, time_mode={time_mode}, z_offset={z_offset}"
            )

            # Get settings (need terrain_following config for building_mask)
            settings = self._get_plot_settings('fig_3')
            settings['terrain_mask_height_z'] = z_offset  # Inject z_offset into settings

            # CRITICAL FIX: Merge global terrain_following config into settings
            # The terrain_helper needs the full terrain_following config with cache settings
            global_tf_config = self.config['plots'].get('terrain_following', {})
            if 'terrain_following' not in settings:
                # Use global config as base
                settings['terrain_following'] = global_tf_config.copy()
            else:
                # Merge fig-specific overrides with global defaults
                merged_tf = global_tf_config.copy()
                merged_tf.update(settings['terrain_following'])
                settings['terrain_following'] = merged_tf

            self.logger.debug(f"Merged terrain_following config: cache enabled={settings['terrain_following'].get('mask_cache', {}).get('enabled', False)}")

            # STEP 1: Try loading from enhanced cache
            cached_data = self._load_from_enhanced_cache(
                dataset=dataset,
                static_dataset=self._load_static_dataset(dataset),
                variable=variable,
                domain_type=domain_type,
                time_mode=time_mode,
                time_params=time_params,
                settings=settings
            )

            if cached_data is not None:
                self.logger.info(f"✓ Using cached data for {variable} (time_mode={time_mode})")
                return cached_data

            # STEP 2: Cache miss - delegate extraction to terrain_helper
            self.logger.info(f"Cache miss - delegating extraction to terrain_helper")

            # We need to handle time selection BEFORE delegating
            # The terrain_helper expects settings with time selection info at ROOT level
            # CRITICAL FIX: Set parameters at root level, NOT inside terrain_following dict

            # Update settings with time selection method (AT ROOT LEVEL)
            if time_mode == 'all_times_average':
                settings['time_selection_method'] = 'mean'
            elif time_mode == 'time_window':
                settings['time_selection_method'] = 'mean_timeframe'
                settings['time_start'] = time_params['start']
                settings['time_end'] = time_params['end']
                self.logger.info(f"Set time window: hours {time_params['start']}-{time_params['end']}")
            elif time_mode == 'single_time':
                settings['time_selection_method'] = 'single_timestep'
                settings['time_index'] = time_params['hour']
                self.logger.info(f"Set single time: hour {time_params['hour']}")

            # Extract case name from dataset encoding for cache naming
            case_name = dataset.encoding.get('source', 'unknown')
            # Remove path and extension to get clean case name
            if '/' in case_name:
                case_name = case_name.split('/')[-1]
            if '.nc' in case_name:
                case_name = case_name.replace('.nc', '')
            settings['case_name'] = case_name

            # Load static dataset if not already loaded
            static_dataset = self._load_static_dataset(dataset)

            # Delegate to terrain_helper for extraction
            data_2d, var_name, needs_conversion = self._extract_terrain_following_2d(
                dataset=dataset,
                static_dataset=static_dataset,
                domain_type=domain_type,
                variable=variable,
                settings=settings
            )

            # Apply Kelvin to Celsius conversion if needed
            if needs_conversion:
                data_2d = data_2d - 273.15
                self.logger.info(f"Converted {var_name} from Kelvin to Celsius")

            # STEP 3: Save to enhanced cache for future use
            save_success = self._save_to_enhanced_cache(
                data_2d=data_2d,
                dataset=dataset,
                static_dataset=static_dataset,
                variable=variable,
                domain_type=domain_type,
                time_mode=time_mode,
                time_params=time_params,
                settings=settings
            )

            if save_success:
                self.logger.info(f"✓ Saved extraction to enhanced cache")
            else:
                self.logger.debug(f"Cache saving skipped or failed")

            return data_2d

        except Exception as e:
            self.logger.error(f"Error extracting terrain-following data for variable '{variable}': {str(e)}")
            raise

    def _extract_data_unified(self, dataset: xr.Dataset, variable: str,
                             hour: int, settings: Dict) -> np.ndarray:
        """
        Unified extraction method that routes to appropriate extraction method

        Args:
            dataset: xarray dataset containing variable data
            variable: Variable name to extract
            hour: Hour of day (0-23)
            settings: Figure-specific settings from configuration

        Returns:
            2D numpy array of variable values
        """
        # Get extraction method from settings (default to 'slice')
        extraction_method = settings.get('extraction_method', 'slice')

        if extraction_method == 'terrain_following':
            # Get z-offset for terrain-following extraction
            z_offset = settings.get('terrain_mask_height_z', 0)
            self.logger.info(f"Using terrain-following extraction with z_offset={z_offset}")
            return self._extract_spatial_data_terrain_following(dataset, variable, hour, z_offset)
        else:
            # Use absolute slice method
            height = settings.get('analysis_height', 2.0)
            self.logger.info(f"Using slice extraction at height={height}m")
            return self._extract_spatial_data(dataset, variable, hour, height)

    def _calculate_cooling_effect(self, temp_field: np.ndarray,
                                base_field: np.ndarray) -> np.ndarray:
        """
        Calculate cooling effect ensuring proper dimension alignment

        Args:
            temp_field: Temperature field from simulation
            base_field: Temperature field from base case

        Returns:
            Cooling effect field (base - simulation)
        """
        # Ensure both fields have the same shape
        if temp_field.shape != base_field.shape:
            self.logger.warning(f"Shape mismatch: {temp_field.shape} vs {base_field.shape}")

            # Try to align by taking the minimum dimensions
            min_x = min(temp_field.shape[0], base_field.shape[0])
            min_y = min(temp_field.shape[1], base_field.shape[1])

            temp_field = temp_field[:min_x, :min_y]
            base_field = base_field[:min_x, :min_y]

        # Calculate cooling (positive values mean cooling)
        cooling = base_field - temp_field

        return cooling

    def _nan_aware_gaussian_filter(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian smoothing that preserves NaN boundaries and doesn't expand masked regions

        Args:
            data: 2D array with potential NaN values
            sigma: Gaussian kernel sigma

        Returns:
            Smoothed array with NaN regions preserved
        """
        # Create mask of valid (non-NaN) data
        valid_mask = ~np.isnan(data)

        # Replace NaN with zeros for filtering
        data_filled = np.where(valid_mask, data, 0.0)

        # Apply gaussian filter to data and to the mask
        smoothed_data = gaussian_filter(data_filled, sigma=sigma)
        smoothed_mask = gaussian_filter(valid_mask.astype(float), sigma=sigma)

        # Avoid division by zero: where smoothed_mask is very small, keep as NaN
        result = np.where(smoothed_mask > 0.01, smoothed_data / smoothed_mask, np.nan)

        # Preserve original NaN regions: if original was NaN, keep it NaN
        result = np.where(valid_mask, result, np.nan)

        return result
        
    def _create_spatial_comparison(self, data: Dict, variable: str, hour,
                                  settings: Dict, title: str) -> plt.Figure:
        """Create spatial comparison plot for all scenarios"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']

        # Create figure with subplots
        fig, axes = plt.subplots(len(ages), len(spacings) + 1,
                                figsize=(20, 16), squeeze=False)

        # Get base case data if available
        base_field = None
        if data.get('base_case') and 'av_3d_n02' in data['base_case']:
            try:
                base_data = data['base_case']['av_3d_n02']
                base_field = self._extract_data_unified(base_data, variable, hour, settings)
            except Exception as e:
                self.logger.warning(f"Could not extract base case data: {str(e)}")

        # Collect all data for auto-scaling
        all_absolute_data = []
        all_difference_data = []

        if base_field is not None:
            all_absolute_data.append(base_field)

        # First pass: collect all data for scaling
        for i, age in enumerate(ages):
            for j, spacing in enumerate(spacings):
                case_key = f"{spacing}m_{age}yrs"
                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]
                    if 'av_3d_n02' in sim_data:
                        try:
                            field = self._extract_data_unified(
                                sim_data['av_3d_n02'], variable, hour, settings
                            )
                            all_absolute_data.append(field)

                            if base_field is not None:
                                diff_field = self._calculate_cooling_effect(field, base_field)
                                all_difference_data.append(diff_field)
                        except Exception as e:
                            self.logger.warning(f"Error extracting data for {case_key}: {str(e)}")

        # Get scaling parameters
        if all_absolute_data:
            combined_absolute = np.concatenate([d.ravel() for d in all_absolute_data])
            absolute_params = self._get_variable_scale_params(variable, combined_absolute, settings)
        else:
            # Fallback defaults
            absolute_params = {'vmin': 20, 'vmax': 35, 'cmap': self._get_colormap('temperature')}

        if all_difference_data:
            combined_difference = np.concatenate([d.ravel() for d in all_difference_data])
            difference_params = self._get_difference_scale_params(variable, combined_difference, settings)
        else:
            # Fallback defaults
            cooling_cmap = self._get_colormap('cooling')
            difference_params = {
                'vmin': -6, 'vmax': 6, 'cmap': cooling_cmap,
                'norm': TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
            }
        
        # Second pass: plot all data
        for i, age in enumerate(ages):
            # Plot base case in first column
            if base_field is not None:
                im = axes[i, 0].imshow(base_field, cmap=absolute_params['cmap'],
                                      vmin=absolute_params['vmin'], vmax=absolute_params['vmax'],
                                      origin='lower', aspect='equal', interpolation='none')
                axes[i, 0].set_title("Base Case" if i == 0 else "")
                axes[i, 0].set_ylabel(f"{age} years", fontsize=12, weight='bold')
            else:
                axes[i, 0].text(0.5, 0.5, "No base case",
                               transform=axes[i, 0].transAxes,
                               ha='center', va='center')

            # Plot each spacing scenario
            for j, spacing in enumerate(spacings):
                ax = axes[i, j + 1]
                case_key = f"{spacing}m_{age}yrs"

                if case_key in data['simulations']:
                    sim_data = data['simulations'][case_key]

                    if 'av_3d_n02' in sim_data:
                        try:
                            # Extract variable field using unified method
                            field = self._extract_data_unified(
                                sim_data['av_3d_n02'], variable, hour, settings
                            )

                            # Calculate difference if base case available
                            if base_field is not None:
                                diff_field = self._calculate_cooling_effect(field, base_field)

                                # Apply NaN-aware smoothing if configured
                                if self.config['analysis']['spatial']['grid_interpolation']:
                                    sigma = self.config['analysis']['spatial']['smoothing_sigma']
                                    # Use NaN-aware smoothing to prevent expansion of masked regions
                                    diff_field = self._nan_aware_gaussian_filter(diff_field, sigma)

                                # Plot difference field
                                im = ax.imshow(diff_field, cmap=difference_params['cmap'],
                                             norm=difference_params['norm'], origin='lower',
                                             aspect='equal', interpolation='none')

                                # Add tree locations if requested
                                if settings.get('show_tree_locations', False):
                                    self._overlay_tree_locations(ax, data, spacing)

                            else:
                                # Just plot absolute values if no base case
                                im = ax.imshow(field, cmap=absolute_params['cmap'],
                                             vmin=absolute_params['vmin'], vmax=absolute_params['vmax'],
                                             origin='lower', aspect='equal', interpolation='none')

                        except Exception as e:
                            self.logger.warning(f"Error plotting {case_key}: {str(e)}")
                            ax.text(0.5, 0.5, "Error", transform=ax.transAxes,
                                   ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                           ha='center', va='center')

                # Add title for first row
                if i == 0:
                    ax.set_title(f"{spacing}m spacing", fontsize=12, weight='bold')

                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])

        # Add main title
        fig.suptitle(title, fontsize=16, weight='bold')

        # Add colorbars with dynamic labels
        # Absolute value colorbar for base case
        cbar_abs_ax = fig.add_axes([0.08, 0.02, 0.35, 0.02])
        sm_abs = plt.cm.ScalarMappable(cmap=absolute_params['cmap'],
                                       norm=plt.Normalize(vmin=absolute_params['vmin'],
                                                         vmax=absolute_params['vmax']))
        sm_abs.set_array([])
        cbar_abs = fig.colorbar(sm_abs, cax=cbar_abs_ax, orientation='horizontal')
        cbar_abs.set_label(self._generate_colorbar_label(variable), fontsize=12)

        # Difference colorbar
        cbar_diff_ax = fig.add_axes([0.55, 0.02, 0.35, 0.02])
        sm_diff = plt.cm.ScalarMappable(cmap=difference_params['cmap'], norm=difference_params['norm'])
        sm_diff.set_array([])
        cbar_diff = fig.colorbar(sm_diff, cax=cbar_diff_ax, orientation='horizontal')
        cbar_diff.set_label(self._generate_difference_label(variable), fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        return fig
        
    def _plot_cooling_extent(self, data: Dict) -> plt.Figure:
        """Plot the spatial extent of cooling beyond tree canopy"""
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Placeholder for cooling extent analysis
        # This would analyze how far the cooling effect extends beyond the tree canopy
        
        ax.text(0.5, 0.5, "Cooling Extent Analysis\n(To be implemented)", 
               transform=ax.transAxes, ha='center', va='center', fontsize=14)
        
        return fig
        
    def _overlay_tree_locations(self, ax, data: Dict, spacing: int):
        """Overlay tree locations on the plot"""
        if 'tree_locations' not in data:
            return
            
        tree_key = f"{spacing}m_child"
        if tree_key in data['tree_locations']:
            locations = data['tree_locations'][tree_key]
            
            # Plot tree locations as small circles
            for loc in locations:
                circle = patches.Circle((loc[0], loc[1]), radius=2, 
                                      fill=False, edgecolor='black', 
                                      linewidth=1, alpha=0.5)
                ax.add_patch(circle)
                
    def _generate_figure_title(self, variable: str, time_info: str) -> str:
        """
        Generate dynamic figure title based on variable and time

        Args:
            variable: Variable name (e.g., 'ta', 'q')
            time_info: Time information string (e.g., 'Daytime', 'Time-Averaged')

        Returns:
            Formatted title string
        """
        var_metadata = self.var_metadata.get_plot_metadata(variable)
        var_label = var_metadata.get('label', variable)

        return f"{time_info} Spatial Patterns: {var_label}"

    def _generate_colorbar_label(self, variable: str) -> str:
        """
        Generate dynamic colorbar label with units

        Args:
            variable: Variable name (e.g., 'ta', 'q')

        Returns:
            Formatted colorbar label with units
        """
        var_metadata = self.var_metadata.get_plot_metadata(variable)
        var_label = var_metadata.get('label', variable)
        var_units = var_metadata.get('units', '')

        if var_units:
            return f"{var_label} ({var_units})"
        else:
            return var_label

    def _generate_difference_label(self, variable: str) -> str:
        """
        Generate label for difference plots (e.g., cooling effect)

        Args:
            variable: Variable name (e.g., 'ta', 'q')

        Returns:
            Formatted difference label
        """
        var_metadata = self.var_metadata.get_plot_metadata(variable)
        var_label = var_metadata.get('label', variable)
        var_units = var_metadata.get('units', '')

        # For temperature: "Cooling Effect (°C)"
        # For humidity: "Humidity Difference (kg/kg)"
        effect_name = "Effect" if "temperature" in var_label.lower() or "ta" in variable.lower() else "Difference"

        if var_units:
            return f"{var_label} {effect_name} ({var_units})"
        else:
            return f"{var_label} {effect_name}"

    def _get_variable_scale_params(self, variable: str, data_array: np.ndarray,
                                   settings: Dict) -> Dict:
        """
        Get variable-specific scaling parameters for colormaps

        Args:
            variable: Variable name
            data_array: Data array to analyze for auto-scaling
            settings: Figure-specific settings

        Returns:
            Dictionary with 'vmin', 'vmax', 'cmap' keys
        """
        # Get variable-specific settings (if they exist)
        var_settings = settings.get('variable_settings', {}).get(variable, {})

        # Determine if auto-scaling is enabled
        auto_scale = var_settings.get('auto_scale', True)

        # Get colormap (variable-specific or default)
        cmap_name = var_settings.get('cmap', 'RdBu_r')
        cmap = self._get_colormap(cmap_name)

        if auto_scale:
            # Auto-scale from data
            percentile_clip = var_settings.get('percentile_clip', None)

            if percentile_clip:
                # Clip outliers using percentiles
                vmin = np.nanpercentile(data_array, percentile_clip)
                vmax = np.nanpercentile(data_array, 100 - percentile_clip)
                self.logger.info(f"Auto-scaled {variable} with {percentile_clip}% clip: [{vmin:.4f}, {vmax:.4f}]")
            else:
                # Use full data range
                vmin = np.nanmin(data_array)
                vmax = np.nanmax(data_array)
                self.logger.info(f"Auto-scaled {variable} from data: [{vmin:.4f}, {vmax:.4f}]")

        else:
            # Use fixed range from config or defaults
            vmin = var_settings.get('vmin', np.nanmin(data_array))
            vmax = var_settings.get('vmax', np.nanmax(data_array))
            self.logger.info(f"Using fixed scale for {variable}: [{vmin:.4f}, {vmax:.4f}]")

        return {
            'vmin': vmin,
            'vmax': vmax,
            'cmap': cmap
        }

    def _get_difference_scale_params(self, variable: str, diff_array: np.ndarray,
                                    settings: Dict) -> Dict:
        """
        Get scaling parameters for difference plots (cooling effect)

        Args:
            variable: Variable name
            diff_array: Difference data array
            settings: Figure-specific settings

        Returns:
            Dictionary with 'vmin', 'vmax', 'cmap', 'norm' keys
        """
        # Get variable-specific settings for differences
        var_settings = settings.get('variable_settings', {}).get(variable, {})
        diff_settings = var_settings.get('difference', {})

        # Determine if auto-scaling is enabled for differences
        auto_scale = diff_settings.get('auto_scale', True)

        # Get colormap (use diverging colormap for differences)
        cmap_name = diff_settings.get('cmap', 'RdBu_r')
        cmap = self._get_colormap(cmap_name)

        if auto_scale:
            # Auto-scale symmetrically around zero
            percentile_clip = diff_settings.get('percentile_clip', None)

            if percentile_clip:
                # Use percentiles to determine range
                abs_max = np.nanpercentile(np.abs(diff_array), 100 - percentile_clip)
            else:
                # Use full data range
                abs_max = np.nanmax(np.abs(diff_array))

            vmin = -abs_max
            vmax = abs_max
            self.logger.info(f"Auto-scaled {variable} difference: [{vmin:.4f}, {vmax:.4f}]")

        else:
            # Use fixed range from config
            vmin = diff_settings.get('vmin', -6.0)
            vmax = diff_settings.get('vmax', 6.0)
            self.logger.info(f"Using fixed difference scale for {variable}: [{vmin:.4f}, {vmax:.4f}]")

        # Create diverging norm centered at zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        return {
            'vmin': vmin,
            'vmax': vmax,
            'cmap': cmap,
            'norm': norm
        }

    def available_plots(self) -> List[str]:
        """Return list of available plot types"""
        return ["daytime_cooling", "nighttime_cooling", "cooling_extent", "temperature_maps"]
    
    """
    Fixed temperature maps plotting methods for spatial_cooling.py
    Replace the existing _plot_temperature_maps, _plot_single_temperature_map, 
    and _extract_time_averaged_temperature methods with these versions
    """
    
    def _plot_temperature_maps(self, data: Dict) -> plt.Figure:
        """
        Create time-averaged variable maps with separate figures for parent and child domains
        Organized in matrix layout with spacings as columns and ages as rows
        """
        spacings = self.config['data']['spacings']
        ages = self.config['data']['ages']
        settings = self._get_plot_settings('fig_3')  # or 'slide_8', both work

        # Get variable from config
        fig_config = self.config['plots'].get('figures', {}).get('fig_3',
                                              self.config['plots'].get('slides', {}).get('slide_8', {}))
        variable = fig_config.get('variable', 'temperature')

        # Get domain settings
        plot_parent = settings.get('plot_parent_domain', True)
        plot_child = settings.get('plot_child_domain', True)
        
        # We'll return a list of figures if both domains are requested
        figures = []

        # Create parent domain figure if requested
        if plot_parent:
            fig_parent = self._create_domain_matrix_plot(
                data, spacings, ages, variable, 'parent', 'av_3d', settings
            )
            figures.append(fig_parent)

        # Create child domain figure if requested
        if plot_child:
            fig_child = self._create_domain_matrix_plot(
                data, spacings, ages, variable, 'child', 'av_3d_n02', settings
            )
            figures.append(fig_child)
        
        # Return the appropriate figure(s)
        if len(figures) == 1:
            return figures[0]
        elif len(figures) == 2:
            # If both domains are plotted, save them separately and return the child domain figure
            # The main script will need to be modified to handle multiple figures
            # For now, we'll save the parent figure here and return the child figure
            self._save_additional_figure(figures[0], 'temperature_maps_parent')
            return figures[1]
        else:
            # No domains requested
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No domains selected for plotting", 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
    
    def _create_domain_matrix_plot(self, data: Dict, spacings: List, ages: List,
                                   variable: str, domain_name: str, data_key: str,
                                   settings: Dict) -> plt.Figure:
        """
        Create a matrix plot for a single domain with spacings as columns and ages as rows

        Args:
            data: Simulation data
            spacings: List of spacings
            ages: List of ages
            variable: Variable name to plot
            domain_name: 'parent' or 'child'
            data_key: 'av_3d' or 'av_3d_n02'
            settings: Figure-specific settings

        Returns:
            Matplotlib figure
        """
        # Create figure with matrix layout
        n_rows = len(ages) + 1  # +1 for base case row
        n_cols = len(spacings)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        # Ensure axes is always 2D array
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Determine domain extent
        if domain_name == 'child':
            extent = [0, 200, 0, 200]
            is_child = True
        else:
            extent = [0, 400, 0, 400]
            is_child = False

        # Collect all data for auto-scaling
        all_data = []

        # Plot base case in first row (all columns)
        base_data_field = None
        if data.get('base_case') and data_key in data['base_case']:
            base_data_field = self._extract_time_averaged_unified(
                data['base_case'][data_key], variable, is_child, settings
            )
            if base_data_field is not None:
                all_data.append(base_data_field)

        # First pass: collect all data for scaling
        for age in ages:
            for spacing in spacings:
                case_key = f"{spacing}m_{age}yrs"
                if case_key in data['simulations'] and data_key in data['simulations'][case_key]:
                    try:
                        field = self._extract_time_averaged_unified(
                            data['simulations'][case_key][data_key], variable, is_child, settings
                        )
                        if field is not None:
                            all_data.append(field)
                    except Exception as e:
                        self.logger.warning(f"Error extracting data for {case_key}: {str(e)}")

        # Get scaling parameters
        if all_data:
            combined_data = np.concatenate([d.ravel() for d in all_data])
            scale_params = self._get_variable_scale_params(variable, combined_data, settings)
        else:
            # Fallback defaults
            scale_params = {'vmin': 20, 'vmax': 35, 'cmap': self._get_colormap('temperature')}

        # Second pass: plot all data
        # Plot base case
        if base_data_field is not None:
            for col_idx in range(n_cols):
                ax = axes[0, col_idx]
                im = ax.imshow(base_data_field, origin='lower', cmap=scale_params['cmap'],
                              vmin=scale_params['vmin'], vmax=scale_params['vmax'], extent=extent,
                              interpolation='none')
                ax.set_title(f'Base Case\n{spacings[col_idx]}m column', fontsize=10)
                ax.set_aspect('equal')
                if col_idx == 0:
                    ax.set_ylabel('Base Case', fontsize=12, weight='bold')
        else:
            # No base case data - clear first row
            for col_idx in range(n_cols):
                ax = axes[0, col_idx]
                ax.text(0.5, 0.5, 'Base Case\nNo data',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_aspect('equal')

        # Plot simulation cases in matrix layout
        for row_idx, age in enumerate(ages):
            for col_idx, spacing in enumerate(spacings):
                ax = axes[row_idx + 1, col_idx]  # +1 to skip base case row
                case_key = f"{spacing}m_{age}yrs"

                # Add row labels (ages) on the left
                if col_idx == 0:
                    ax.set_ylabel(f'{age} years', fontsize=12, weight='bold')

                # Add column labels (spacings) on top row of simulations
                if row_idx == 0:
                    ax.set_title(f'{spacing}m', fontsize=12, weight='bold')

                if case_key in data['simulations'] and data_key in data['simulations'][case_key]:
                    # Extract variable field for this specific domain only
                    field = self._extract_time_averaged_unified(
                        data['simulations'][case_key][data_key], variable, is_child, settings
                    )

                    if field is not None:
                        im = ax.imshow(field, origin='lower', cmap=scale_params['cmap'],
                                      vmin=scale_params['vmin'], vmax=scale_params['vmax'], extent=extent,
                                      interpolation='none')

                        # Add grid
                        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
                    else:
                        ax.text(0.5, 0.5, 'No data',
                               transform=ax.transAxes, ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'No data',
                           transform=ax.transAxes, ha='center', va='center')

                # Set aspect ratio
                ax.set_aspect('equal')

                # Add axis labels for bottom and left edges
                if row_idx == len(ages) - 1:  # Bottom row
                    ax.set_xlabel('X (m)', fontsize=9)
                else:
                    ax.set_xticklabels([])

                if col_idx == 0:  # Left column
                    ax.set_ylabel('Y (m)', fontsize=9)
                else:
                    ax.set_yticklabels([])

        # Add main title with dynamic variable info
        domain_text = "Parent Domain" if domain_name == 'parent' else "Child Domain"
        resolution = "10m resolution" if domain_name == 'parent' else "2m resolution"
        var_metadata = self.var_metadata.get_plot_metadata(variable)
        var_label = var_metadata.get('label', variable)
        fig.suptitle(f'Time-Averaged {var_label} - {domain_text} ({resolution})',
                    fontsize=16, weight='bold')

        # Add single colorbar on the right with dynamic label
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=scale_params['cmap'],
                                  norm=plt.Normalize(vmin=scale_params['vmin'], vmax=scale_params['vmax']))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(self._generate_colorbar_label(variable), fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.91, 0.96])
        
        return fig
    
    def _save_additional_figure(self, fig: plt.Figure, name_suffix: str):
        """
        Save an additional figure using the output manager

        Args:
            fig: Figure to save
            name_suffix: Suffix for the filename
        """
        try:
            # This is a workaround since the main script expects single figures
            # In production, the main script should be modified to handle multiple figures
            formats = self.config['output']['formats']
            output_dir = self.output_manager.get_figure_dir('fig_3')

            for fmt in formats:
                filename = f"fig_3_{name_suffix}.{fmt}"
                filepath = output_dir / filename
                fig.savefig(filepath, format=fmt, dpi=self.config['output']['dpi'],
                           bbox_inches='tight', transparent=self.config['output']['transparent_background'])
                self.logger.info(f"    Saved additional figure: {filepath}")
        except Exception as e:
            self.logger.warning(f"Could not save additional figure: {str(e)}")
    
    def _plot_single_temperature_map(self, ax, case_data: Dict, title: str,
                                   vmin: float, vmax: float, cmap, height_m: float,
                                   plot_parent: bool, plot_child: bool):
        """
        Plot temperature map for a single case with correct orientation
        
        Args:
            ax: Matplotlib axes
            case_data: Data dictionary for the case
            title: Title for the subplot
            vmin, vmax: Temperature range
            cmap: Colormap
            height_m: Analysis height in meters
            plot_parent: Whether to plot parent domain
            plot_child: Whether to plot child domain
        """
        # Check which data is available
        has_parent = 'av_3d' in case_data
        has_child = 'av_3d_n02' in case_data
        
        if not has_parent and not has_child:
            ax.text(0.5, 0.5, f"No data available\n{title}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_aspect('equal')
            return
        
        # Extract temperature fields based on configuration
        parent_temp = None
        child_temp = None
        
        # Get parent domain temperature if requested and available
        if plot_parent and has_parent:
            parent_temp = self._extract_time_averaged_temperature(
                case_data['av_3d'], height_m, is_child=False
            )
        
        # Get child domain temperature if requested and available
        if plot_child and has_child:
            child_temp = self._extract_time_averaged_temperature(
                case_data['av_3d_n02'], height_m, is_child=True
            )
        
        # Plot based on what's available and requested
        if plot_parent and parent_temp is not None:
            # Plot parent domain
            # Correct orientation: no transpose needed for proper x-y orientation
            im = ax.imshow(parent_temp, origin='lower', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, 400, 0, 400],
                          interpolation='none')
            
            # Overlay child domain if requested and available
            if plot_child and child_temp is not None:
                # Child domain is centered in parent domain
                # Parent: 400x400m at 10m resolution (40x40 cells)
                # Child: 200x200m at 2m resolution (100x100 cells)
                # Child covers central 200x200m area (from 100m to 300m)
                child_extent = [100, 300, 100, 300]
                
                # Overlay child domain
                ax.imshow(child_temp, origin='lower', cmap=cmap,
                         vmin=vmin, vmax=vmax, extent=child_extent,
                         alpha=0.95, interpolation='bilinear')
                
                # Add boundary rectangle for child domain
                from matplotlib.patches import Rectangle
                rect = Rectangle((100, 100), 200, 200, linewidth=2, 
                               edgecolor='black', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
                # Add text label for child domain
                ax.text(105, 290, 'Child Domain (2m res)', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        elif plot_child and child_temp is not None:
            # Only child domain requested or available
            im = ax.imshow(child_temp, origin='lower', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, 200, 0, 200],
                          interpolation='none')
        else:
            # No data to plot based on settings
            ax.text(0.5, 0.5, f"No data for selected domains\n{title}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_aspect('equal')
            return
        
        # Set labels and title
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    def _extract_time_averaged_data_terrain_following(self, dataset: xr.Dataset,
                                                      variable: str,
                                                      z_offset: int = 0) -> np.ndarray:
        """
        Extract time-averaged variable data using terrain-following coordinates

        This method delegates to _extract_spatial_data_terrain_following() with hour=None
        to trigger all-times-average extraction mode. This ensures consistent terrain-following
        logic across all plot types.

        Args:
            dataset: xarray dataset containing variable data
            variable: Variable name to extract
            z_offset: Z-index offset from terrain surface

        Returns:
            2D array of time-averaged variable values
        """
        try:
            self.logger.info(
                f"Extracting time-averaged {variable} using terrain-following "
                f"(all timesteps average, z_offset={z_offset})"
            )

            # Delegate to _extract_spatial_data_terrain_following with hour=None
            # This triggers 'all_times_average' mode in _determine_time_mode()
            # which properly uses the terrain_helper for on-the-fly computation
            result = self._extract_spatial_data_terrain_following(
                dataset=dataset,
                variable=variable,
                hour=None,  # None triggers all_times_average mode
                z_offset=z_offset
            )

            return result

        except Exception as e:
            self.logger.error(f"Error extracting terrain-following time-averaged data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Determine appropriate array size
            is_child = 'N02' in dataset.encoding.get('source', '')
            return np.full((200, 200) if is_child else (400, 400), np.nan)

    def _extract_time_averaged_data(self, dataset: xr.Dataset, variable: str,
                                   height_m: float, is_child: bool) -> np.ndarray:
        """
        Extract time-averaged variable data at specified height with improved height selection

        Args:
            dataset: xarray dataset containing variable data
            variable: Variable name to extract (e.g., 'ta', 'q', 'bio_utci*_xy')
            height_m: Height above ground in meters
            is_child: Whether this is child domain (affects z-index)

        Returns:
            2D array of time-averaged variable values
        """
        try:
            # Get z coordinate information
            z_coords = dataset['zu_3d'].values if 'zu_3d' in dataset.dims else None

            if z_coords is None:
                self.logger.error("No zu_3d coordinate found in dataset")
                return np.full((200, 200) if is_child else (400, 400), np.nan)

            # Get the first data index for this domain type
            first_data_idx = 21 if is_child else 25

            # Get variable data using VariableMetadata for dynamic lookup
            var_data, var_name = self.var_metadata.find_variable_in_dataset(dataset, variable)
            self.logger.debug(f"Found variable '{var_name}' for requested '{variable}'")

            # Find the actual first index with valid variable data
            actual_first_idx = first_data_idx
            for idx in range(first_data_idx):
                try:
                    test_data = var_data.isel(zu_3d=idx, x=0, y=0, time=0).values
                    if not np.isnan(test_data) and test_data != 0:
                        actual_first_idx = idx
                        break
                except:
                    continue

            # Log information about available heights
            self.logger.info(f"{'Child' if is_child else 'Parent'} domain height analysis:")
            self.logger.info(f"  Variable: {var_name}")
            self.logger.info(f"  Standard first data index: {first_data_idx}")
            self.logger.info(f"  Actual first data index: {actual_first_idx}")
            self.logger.info(f"  Height at first data: {z_coords[actual_first_idx]:.2f}m")
            self.logger.info(f"  Requested height: {height_m}m")

            # If requested height is below the first available data level,
            # use the first available level and warn the user
            if height_m < z_coords[actual_first_idx]:
                self.logger.warning(
                    f"Requested height {height_m}m is below first available data "
                    f"at {z_coords[actual_first_idx]:.2f}m. Using lowest available level."
                )
                z_idx = actual_first_idx
            else:
                # Find the index closest to the requested height
                # Search through all valid levels starting from actual_first_idx
                valid_indices = range(actual_first_idx, len(z_coords))
                valid_heights = z_coords[actual_first_idx:]

                # Find closest height
                height_diff = np.abs(valid_heights - height_m)
                relative_idx = np.argmin(height_diff)
                z_idx = actual_first_idx + relative_idx

            actual_height = z_coords[z_idx]
            self.logger.info(f"  Using z_idx={z_idx} (actual height={actual_height:.2f}m)")

            # Extract variable data at the selected height
            var_at_height = var_data.isel(zu_3d=z_idx)

            # Average over time dimension
            var_avg = var_at_height.mean(dim='time')

            # Clean invalid data (convert zeros to NaN for proper masking)
            result = var_avg.values
            result = self._clean_invalid_data(result, variable)

            # Return the values without transposing to maintain correct orientation
            return result

        except Exception as e:
            self.logger.error(f"Error extracting time-averaged data for variable '{variable}': {str(e)}")
            self.logger.error(f"Dataset dimensions: {list(dataset.dims.keys())}")
            self.logger.error(f"Available variables: {list(dataset.data_vars.keys())}")
            # Return NaN array of appropriate size
            if is_child:
                return np.full((200, 200), np.nan)
            else:
                return np.full((400, 400), np.nan)

    def _extract_time_averaged_unified(self, dataset: xr.Dataset, variable: str,
                                      is_child: bool, settings: Dict) -> np.ndarray:
        """
        Unified time-averaged extraction method that routes to appropriate method

        Args:
            dataset: xarray dataset containing variable data
            variable: Variable name to extract
            is_child: Whether this is child domain
            settings: Figure-specific settings from configuration

        Returns:
            2D array of time-averaged variable values
        """
        # Get extraction method from settings (default to 'slice')
        extraction_method = settings.get('extraction_method', 'slice')

        if extraction_method == 'terrain_following':
            # Get z-offset for terrain-following extraction
            z_offset = settings.get('terrain_mask_height_z', 0)
            self.logger.info(f"Using terrain-following time-averaged extraction with z_offset={z_offset}")
            return self._extract_time_averaged_data_terrain_following(dataset, variable, z_offset)
        else:
            # Use absolute height method
            height_m = settings.get('analysis_height', 2.0)
            self.logger.info(f"Using slice time-averaged extraction at height={height_m}m")
            return self._extract_time_averaged_data(dataset, variable, height_m, is_child)