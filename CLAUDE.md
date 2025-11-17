# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PALMPlot is a Python package for visualizing PALM-LES (Parallelized Large-Eddy Simulation Model) simulation data, specifically analyzing temperature effects of urban tree scenarios (Tilia trees in Berlin's THF). The package processes NetCDF files from PALM simulations and generates publication-quality plots comparing different tree ages (20, 40, 60, 80 years) and spacings (10, 15, 20, 25m).

## Running the Package

### Main execution
```bash
# Run with configuration file
python -m palmplot_thf palmplot_config.yaml

# Validate configuration only
python -m palmplot_thf palmplot_config.yaml --validate-only

# List available plots
python -m palmplot_thf palmplot_config.yaml --list-plots
```

### Testing
```bash
# Test specific plotting functionality
python test_temperature_maps.py

# Test fig_3 terrain-following implementation
python test_fig3_implementation.py

# Test with fig_3 test configuration
python -m palmplot_thf palmplot_config_fig3_test.yaml
```

## Architecture

### Core Components

**Data Loading (`core/data_loader.py`)**
- `PALMDataLoader`: Loads NetCDF files from PALM simulations
- Handles both parent domain (10m resolution, 400x400m) and child domain (2m resolution, 200x200m)
- Processes time dimensions by converting PALM time (seconds since reference date 2018-08-07) to datetime
- Supports chunked loading for large files (>100MB)
- Key domains:
  - Parent: zu_3d index 25 is first data level
  - Child (N02): zu_3d index 21 is first data level

**Configuration (`core/config_handler.py`)**
- `ConfigHandler`: Validates and manages YAML configuration using schema validation
- Validates paths, creates output directories, expands relative paths
- Settings organized by: general, data, output, plots (slides), analysis, logging, performance, validation

**Main Orchestration (`__main__.py`)**
- `PALMPlot`: Main class that initializes components and orchestrates plotting workflow
- Loads data once, then generates multiple plots based on figure configuration
- Each figure (fig_1 through fig_5) corresponds to a different analysis/visualization
  - fig_1 (formerly slide_6): Tree Density Scenarios
  - fig_2 (formerly slide_7): Temperature Dynamics
  - fig_3 (formerly slide_8): Spatial Cooling Patterns
  - fig_4 (formerly slide_9): Vertical Cooling Profile
  - fig_5 (formerly slide_10): Age-Density-Cooling Relationship
- Supports backward compatibility with legacy slide-based naming

### Plotting Architecture

**Base Plotter (`plots/base_plotter.py`)**
- Abstract base class for all plotters
- Handles matplotlib configuration, font fallback, color schemes
- Common utilities: color palettes, time axis formatting, statistics boxes
- All plotters must implement: `generate_plot()` and `available_plots()`

**Plotter Modules** (all inherit from `BasePlotter`)
- `TreeDensityPlotter` (fig_1): Visualizes tree density scenarios in matrix layout
- `TemperatureDynamicsPlotter` (fig_2): Time series and diurnal temperature cycles
- `SpatialCoolingPlotter` (fig_3): Horizontal spatial patterns with variable selection, terrain-following support, time window averaging, and auto-scaling
- `VerticalProfilePlotter` (fig_4): Vertical cross-sections and height profiles
- `CoolingRelationshipPlotter` (fig_5): Age-density-cooling surface plots and optimization

**Output Management (`utils/output_manager.py`)**
- `OutputManager`: Creates timestamped run directories (run_YYYYMMDD_HHMMSS)
- Organizes outputs by figure (fig_1/, fig_2/, etc.)
- Supports automatic subfigure lettering (a, b, c, etc.)
- Output files follow pattern: `fig_Xa_plot_type.format` (e.g., `fig_2a_time_series.png`)
- Supports multiple formats: PNG, PDF, SVG
- Backward compatible with legacy slide-based naming

**Figure Mapping (`utils/figure_mapper.py`)**
- `FigureMapper`: Manages conversion between slide and figure IDs
- Handles subfigure letter assignment (a, b, c, etc.)
- Provides centralized mapping: slide_6→fig_1, slide_7→fig_2, etc.

### Data Flow

1. Configuration loaded and validated via `ConfigHandler`
2. `PALMDataLoader` loads:
   - Base case simulation (thf_base_2018080700)
   - All spacing×age combinations (16 total scenarios)
   - Tree location CSVs
   - LAD/WAD metadata
3. Data structure passed to plotters:
   ```python
   data = {
       'simulations': {
           '10m_20yrs': {'av_3d': ds, 'av_3d_n02': ds, 'av_xy': ds, ...},
           ...
       },
       'base_case': {'av_3d': ds, 'av_3d_n02': ds, ...},
       'tree_locations': {'10m_parent': array, '10m_child': array, ...},
       'tree_metadata': df
   }
   ```
4. Each plotter extracts needed variables (e.g., 'ta' for temperature) at appropriate height indices
5. Figures saved in multiple formats to organized output directories

## Key Implementation Details

### Height/Vertical Index Handling
- PALM data uses zu_3d coordinate for vertical levels
- First data levels differ by domain:
  - Parent domain: zu_3d[25] is ~2m height
  - Child domain (N02): zu_3d[21] is ~2m height
- Always use `.isel(zu_3d=idx)` not interpolation for extracting specific heights
- Check actual heights with `dataset['zu_3d'].values` when debugging

### Time Dimension Processing
- PALM outputs time as seconds since 2018-08-07 00:00:00
- `PALMDataLoader._load_and_process_netcdf()` converts to pandas datetime
- Extract specific hours: `pd.to_datetime(dataset['time'].values).hour == hour`
- Time averaging: use `.mean(dim='time')` after selecting time slices

### Domain Nesting
- Child domain (200×200m) is nested in center of parent domain (400×400m)
- Child spans from 100m to 300m in both x and y
- When overlaying: use `extent=[100, 300, 100, 300]` for child on parent plot
- File naming: files with 'N02' suffix are child domain

### Orientation and Plotting
- Use `origin='lower'` in `imshow()` for correct spatial orientation
- Do NOT transpose temperature arrays - maintain as-is from NetCDF
- Extent format: `[x_min, x_max, y_min, y_max]`

### Configuration Structure
- Enable/disable figures: `plots.figures.fig_X.enabled`
- Enable/disable specific plot types: `plots.figures.fig_X.plot_types.plot_name`
- Figure-specific settings: `plots.figures.fig_X.settings`
- Access in plotter: Use `self._get_plot_settings('fig_3')` for cross-compatible access
- The `_get_plot_settings()` helper handles both 'figures' and 'slides' config structures
- Legacy slide-based configuration still supported for backward compatibility

### Terrain-Following Extraction and Enhanced Caching

**Overview**
Terrain-following extraction is a method for extracting 3D data at a constant height above terrain/buildings, rather than at absolute height. This is critical for accurate analysis in complex urban terrain. Both fig_3 and fig_6 support terrain-following with enhanced caching.

**Global Configuration**
Shared terrain-following settings are defined in `plots.terrain_following`:
```yaml
plots:
  terrain_following:
    default_method: 'terrain_following'  # or 'slice' for absolute height
    buildings_mask: true                 # Enable/disable building masking
    buildings_mask_mode: 'natural_mask'  # 'natural_mask' (default) or 'static_mask'
    time_selection_method: 'mean'        # 'mean', 'mean_timeframe', 'single_timestep'

    mask_cache:
      enabled: true
      mode: 'auto'  # 'auto', 'load', or 'save'
      cache_directory: "/path/to/cache/terrain_masks"
      levels:
        max_levels: 25
        offsets: [0, 1, 2, 5, 10]  # Heights above surface to cache
      validation:
        check_time_mode: true
        check_building_mask: true
        on_mismatch: 'recompute'  # 'warn', 'recompute', or 'error'

    surface_data_cache:
      enabled: true
      mode: 'auto'
      cache_directory: "/path/to/cache/surface_data"
```

**Enhanced Cache Naming Convention**
Cache files now include time mode and building mask state for proper differentiation:

**Terrain Mask Files:**
- Pattern: `{case_name}_terrain_mask_{domain}_{time_mode}_{mask_mode}_TF{offsets}.nc`
- Examples:
  - `thf_base_terrain_mask_parent_all_times_average_masked_TF0-2.nc`
  - `thf_forest_10m_20yrs_terrain_mask_child_time_window_30_42_unmasked_TF0-1-5-10.nc`
  - `thf_forest_15m_40yrs_terrain_mask_parent_single_time_14_masked_TF1.nc`

**Surface Data Files:**
- Pattern: `{case_name}_surface_data_{domain}_{time_mode}.nc`
- Examples:
  - `thf_base_surface_data_child_all_times_average.nc`
  - `thf_forest_10m_20yrs_surface_data_parent_time_window_21_30.nc`

**Time Modes:**
- `all_times_average`: Average over all simulation timesteps
- `time_window_{start}_{end}`: Average over specific simulation hour range
- `single_time_{hour}`: Single timestep snapshot at specific hour of day

**Building Mask Modes:**

The `buildings_mask_mode` parameter controls how building regions are identified and masked during terrain-following extraction:

1. **`natural_mask` (DEFAULT, RECOMMENDED)**
   - Relies on fill values in atmospheric data to naturally determine masked regions
   - Does NOT use the static `buildings_2d` field as a 2D mask
   - Respects actual building heights: atmospheric data above building roofs is included
   - Eliminates artificial "outline zone" artifacts around buildings
   - More physically accurate for vertical profiles
   - **How it works:** The `has_valid_data` check already excludes grid cells where atmospheric variables have fill values (buildings/terrain). The natural_mask mode simply removes the additional static 2D mask constraint, allowing the algorithm to fill cells above building roofs where valid atmospheric data exists.

2. **`static_mask` (LEGACY)**
   - Uses `buildings_2d` from static driver files to create a 2D boolean mask
   - Applies this fixed mask at ALL vertical levels
   - Excludes building locations even when extraction height is above building roofs
   - Can create "masking outline zones" in output due to static 2D constraint
   - Useful for explicit building visualization or legacy comparisons
   - **How it works:** Loads `buildings_2d > 0` to create boolean mask, inverts it (`~building_mask_2d`), and applies via boolean AND in the fillable condition at every vertical level.

**Comparison:**
```
Vertical profile at building location:

natural_mask:                  static_mask:
z=40m: ✓ Valid atmospheric    z=40m: ✗ Masked (2D mask applied)
z=30m: ✓ Valid atmospheric    z=30m: ✗ Masked (2D mask applied)
z=20m: Building roof           z=20m: Building roof
z=10m: ✗ Fill value (inside)  z=10m: ✗ Fill value + static mask
z=0m:  ✗ Fill value (terrain)  z=0m:  ✗ Fill value + static mask

Result: Natural mode includes atmospheric data above building roofs (physically accurate)
        Static mode excludes ALL building locations regardless of height (over-masking)
```

**Configuration Examples:**
```yaml
# Recommended: Natural mask (default)
terrain_following:
  buildings_mask: true
  buildings_mask_mode: 'natural_mask'

# Legacy: Static mask for explicit building visualization
terrain_following:
  buildings_mask: true
  buildings_mask_mode: 'static_mask'

# No masking: Fill through all regions
terrain_following:
  buildings_mask: false
  # buildings_mask_mode is ignored when buildings_mask: false
```

**Backward Compatibility:**
The system automatically falls back to legacy cache file naming if enhanced files are not found. Legacy format: `{case_name}_terrain_mask_{domain}_TF{offsets}.nc`

**Fig_3 Time Window Support**
fig_3 (SpatialCoolingPlotter) now supports flexible time specifications:

```yaml
fig_3:
  settings:
    extraction_method: "terrain_following"
    terrain_mask_height_z: 0  # Offset from terrain surface

    # Time window (simulation hours 33-42)
    daytime_hour: [33, 42]

    # Single hour of day (6 AM)
    nighttime_hour: 6

    # All times average (omit or set to null)
    # some_plot_hour: null
```

**Implementation Details:**
- `spatial_cooling.py` delegates terrain-following extraction to `terrain_transect.py` via `_terrain_helper` instance
- Time mode detection: `_determine_time_mode(hour)` converts hour specs to standardized (time_mode, time_params)
- Enhanced cache loading: `_load_from_enhanced_cache()` searches for matching cache files with time/mask parameters
- Cache files are shared between fig_3 and fig_6 when parameters match

**Variable Selection and Auto-Scaling**
fig_3 supports dynamic variable selection with automatic scaling:

```yaml
fig_3:
  variable: "temperature"  # or "humidity_q", "utci", etc.

  settings:
    variable_settings:
      temperature:
        auto_scale: true  # Automatically determine vmin/vmax from data
        percentile_clip: 2  # Optional: clip outliers at 2nd/98th percentile
        cmap: "RdBu_r"
        difference:
          auto_scale: true  # Auto-scale difference plots
          cmap: "RdBu_r"
```

Titles, colorbar labels, and scales are automatically updated based on selected variable.

**PCM Variable Support**
Both fig_3 and fig_6 support Plant Canopy Model (PCM) variables:
- PCM variables use `zpc_3d` vertical coordinate (canopy levels)
- Automatically detected via `_is_pcm_variable(variable, dataset)`
- Special handling for zero-to-NaN conversion (PCM uses 0.0 for "no canopy")
- Examples: `pcm_lad` (Leaf Area Density), `pcm_transpirationrate`

**Key Methods in spatial_cooling.py:**
- `_determine_time_mode(hour)`: Converts hour specification to (time_mode, time_params)
- `_extract_terrain_following_2d()`: Delegates to terrain_helper for extraction
- `_load_from_enhanced_cache()`: Loads cached data with time/mask filtering
- `_is_pcm_variable()`: Delegates PCM detection to terrain_helper
- `_load_static_dataset()`: Loads corresponding static file for terrain data

**Cache Validation:**
Cache files include metadata validated on load:
- Grid size (nx, ny)
- Domain type (parent/child)
- Z-coordinate name
- Time selection mode
- Building mask state
- Maximum age (optional warning if cache is old)

On mismatch, the system can:
- `warn`: Log warning but use cache anyway
- `recompute`: Discard cache and recompute
- `error`: Raise error and stop

## File Paths and Naming Conventions

### Input Data Locations
- Simulation base: `/mnt/f/phd/thf_forest_study/thf_forest_lad_cases_000/`
- Case naming: `thf_forest_lad_spacing_{spacing}m_age_{age}yrs/`
- NetCDF files: `OUTPUT/merged_files/{case_name}_av_3d{_N02}_merged.nc`
- Static files: `INPUT/{case_name}_static{_N02}`

### Output Organization
- Base: configured in `output.base_directory`
- Run structure: `base_directory/run_YYYYMMDD_HHMMSS/fig_X/`
- Filename pattern: `fig_Xa_plot_type.format` where 'a' is auto-incremented subfigure letter
- Examples:
  - `fig_2a_time_series.png` - First plot in figure 2
  - `fig_2b_diurnal_cycle.png` - Second plot in figure 2
  - `fig_2c_temperature_difference.png` - Third plot in figure 2
- Legacy slide-based naming: `slide_X_plot_type.format` (still supported)

## Common Development Tasks

### Adding a New Plot Type
1. Add method to appropriate plotter class (e.g., `_plot_new_analysis()`)
2. Update `generate_plot()` to route to new method
3. Add to `available_plots()` return list
4. Add plot type to config: `plots.figures.fig_X.plot_types.new_analysis: true`
5. The new plot will automatically receive the next available subfigure letter

### Accessing Figure Settings in Plotters
- Use `self._get_plot_settings('fig_X')` instead of direct config access
- This method handles both 'figures' and 'slides' config structures
- Example: `settings = self._get_plot_settings('fig_3')`
- Works with both 'fig_3' and 'slide_8' identifiers (automatically converted)

### Modifying Temperature Extraction
- Temperature variable is 'ta' (configurable in `data.variables.temperature`)
- Height selection logic in `BasePlotter._extract_temperature_at_height()` or plotter-specific methods
- Always check domain type (child vs parent) to use correct z-index

### Debugging Data Loading Issues
- Check log file: configured in `logging.log_file`
- Verify NetCDF structure: `dataset.dims`, `dataset.data_vars`, `dataset.coords`
- Test with single case: modify `test_temperature_maps.py` to load specific cases
- Disable parallel processing: set `general.parallel_processing: false`

### Using Terrain-Following in New Plotters
To add terrain-following support to a new plotter:

1. **Import TerrainTransectPlotter** in your plotter's `__init__`:
   ```python
   from .terrain_transect import TerrainTransectPlotter

   def __init__(self, config, output_manager):
       super().__init__(config, output_manager)
       self._terrain_helper = TerrainTransectPlotter(config, output_manager)
   ```

2. **Read global terrain_following config** for cache directories:
   ```python
   tf_global = config.get('plots', {}).get('terrain_following', {})
   mask_cache = tf_global.get('mask_cache', {})
   if mask_cache.get('enabled', False):
       self.terrain_mask_cache_dir = Path(mask_cache['cache_directory']).expanduser()
   ```

3. **Delegate extraction to terrain_helper**:
   ```python
   data_2d, var_name, needs_conversion = self._terrain_helper._extract_terrain_following(
       dataset=dataset,
       static_dataset=static_dataset,
       domain_type='child',  # or 'parent'
       variable=variable,
       buildings_mask=True,
       output_mode='2d',
       settings=settings
   )
   ```

4. **Use enhanced cache methods** for time-aware caching:
   - Determine time mode: `time_mode, time_params = self._determine_time_mode(hour)`
   - Load from cache: `self._load_from_enhanced_cache(..., time_mode, time_params, ...)`
   - Cache files automatically differentiate by time window and building mask

5. **Test your implementation**:
   - Create test config with terrain-following enabled
   - Run with cache mode='auto' to verify cache creation
   - Verify cache file names follow enhanced convention
   - Test with different time modes (single_time, time_window, all_times_average)

## Dependencies

Key Python packages used:
- xarray: NetCDF file handling
- numpy: Array operations
- matplotlib: Plotting
- pandas: Data manipulation and time handling
- scipy: Spatial smoothing (gaussian_filter)
- yaml: Configuration parsing
- schema: Configuration validation

## Notes

- The package uses dual import strategy (relative and absolute) to support both package and direct execution
- Font handling includes fallback to DejaVu Sans if configured font unavailable
- Temperature data is in Celsius
- Spatial smoothing optional via `analysis.spatial.grid_interpolation` and `smoothing_sigma`
- Memory management: figures are explicitly closed after saving to prevent memory leaks
