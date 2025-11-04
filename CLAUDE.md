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
- `SpatialCoolingPlotter` (fig_3): Horizontal temperature distribution, cooling patterns, time-averaged temperature maps
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
