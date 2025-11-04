# Implementation Plan: fig_6 - Terrain-Following Transect Analysis

## Overview
Create a new plotting module for terrain-following transect analysis showing air temperature (ta) and water vapor mixing ratio (qv) profiles across tree scenarios, with separate files for age-varying and spacing-varying comparisons.

## Phase 1: Create New Plotter Module

### File: `/home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/plots/terrain_transect.py`

**Class Structure:**
```python
class TerrainTransectPlotter(BasePlotter):
```

**Key Methods:**
1. `generate_plot(plot_type, data)` - Main router for plot types
2. `_plot_ta_parent_age()` - fig_6a: Parent domain temperature, varying ages
3. `_plot_ta_parent_spacing()` - fig_6a variant: Parent domain temperature, varying spacings
4. `_plot_ta_child_age()` - fig_6b: Child domain temperature, varying ages
5. `_plot_ta_child_spacing()` - fig_6b variant: Child domain temperature, varying spacings
6. `_plot_qv_parent_age()` - fig_6c: Parent domain water vapor, varying ages
7. `_plot_qv_parent_spacing()` - fig_6c variant: Parent domain water vapor, varying spacings
8. `_plot_qv_child_age()` - fig_6d: Child domain water vapor, varying ages
9. `_plot_qv_child_spacing()` - fig_6d variant: Child domain water vapor, varying spacings

**Core Processing Methods:**
1. `_extract_terrain_following_slice()` - Apply terrain mask and extract data at z-offset above topography
2. `_time_average_data()` - Average over all timesteps
3. `_extract_transect_line()` - Extract 1D line through x or y axis with width averaging
4. `_get_building_lad_masks()` - Extract building and LAD arrays along transect
5. `_create_transect_plot()` - Main plotting function creating dual-panel layout

**Plotting Components:**
- Top panel: Line plot with shaded regions (buildings=grey 50%, trees=green 50%)
- Bottom panel: XY plan view with transect line (magenta dashed)
- Color schemes: Temperature (RdBu_r), Water vapor (custom green scale)

## Phase 2: Update Configuration

### File: `palmplot_config_full.yaml`

**Add fig_6 section:**
```yaml
fig_6:
  enabled: true
  title: "Terrain-Following Transect Analysis"
  description: "Air temperature and water vapor transects above terrain"
  plot_types:
    ta_parent_age: true           # fig_6a_ta_parent_age
    ta_parent_spacing: true       # fig_6a_ta_parent_spacing
    ta_child_age: true            # fig_6b_ta_child_age
    ta_child_spacing: true        # fig_6b_ta_child_spacing
    qv_parent_age: true           # fig_6c_qv_parent_age
    qv_parent_spacing: true       # fig_6c_qv_parent_spacing
    qv_child_age: true            # fig_6d_qv_child_age
    qv_child_spacing: true        # fig_6d_qv_child_spacing
  settings:
    # Terrain-following mask
    terrain_mask_height_z: 0      # Z-index offset (0=first grid point above topography)

    # Transect configuration
    transect_axis: "x"            # "x" or "y" - which axis to slice through
    transect_location: 200        # Grid index location of transect line
    transect_width: 5             # Grid cells to average on each side

    # Scenario comparisons (user configurable)
    age_comparison:
      constant_spacing: 10        # meters
      varying_ages: [20, 40, 60, 80]  # years
    spacing_comparison:
      constant_age: 40            # years
      varying_spacings: [10, 15, 20, 25]  # meters

    # Color schemes
    temperature_cmap: "RdBu_r"
    temperature_range: [24.0, 28.6]
    qv_cmap: "YlGnBu"
    qv_range: [0.00081, 0.00095]

    # Shading
    building_alpha: 0.5
    building_color: "grey"
    lad_alpha: 0.5
    lad_color: "green"
    transect_line_color: "magenta"
    transect_line_style: "--"
```

## Phase 3: Update Main Integration Files

### File: `__main__.py`
- Add `TerrainTransectPlotter` import
- Add to plotters dictionary: `'fig_6': TerrainTransectPlotter(...)`

### File: `plots/__init__.py`
- Import: `from .terrain_transect import TerrainTransectPlotter`
- Add to `__all__` list
- Add to `PLOTTER_REGISTRY`: `"fig_6": TerrainTransectPlotter`

### File: `utils/figure_mapper.py`
- Update `SLIDE_TO_FIGURE_MAP` if needed for backward compatibility

## Phase 4: Implementation Details

### Terrain-Following Extraction Algorithm:
```python
def _extract_terrain_following_slice(self, dataset, domain_type, z_offset):
    """
    Extract data at a constant height above terrain

    Algorithm:
    1. Load topography from static file (zt variable)
    2. Load ta or qv from av_3d files
    3. For each (x,y) point:
       - Find first zu_3d index above topography
       - Add z_offset to get target index
       - Extract data[time, z_target, y, x]
    4. Return 3D array [time, y, x]

    Args:
        dataset: xarray Dataset containing 3D averaged data
        domain_type: 'parent' or 'child' (determines first data level)
        z_offset: Integer offset from first grid point (0-indexed)

    Returns:
        3D numpy array [time, y, x] with terrain-following slice
    """
```

### Time Averaging:
```python
def _time_average_data(self, data_3d):
    """
    Average data over time dimension

    Args:
        data_3d: 3D array [time, y, x]

    Returns:
        2D array [y, x] with time-averaged values
    """
```

### Transect Extraction:
```python
def _extract_transect_line(self, data_2d, axis, location, width):
    """
    Extract 1D transect line from 2D field with width averaging

    Algorithm:
    1. If axis='x':
       - Slice at y=location
       - Average over y_range = [location-width, location+width]
       - Return 1D array along x
    2. If axis='y':
       - Slice at x=location
       - Average over x_range = [location-width, location+width]
       - Return 1D array along y

    Args:
        data_2d: 2D array [y, x]
        axis: 'x' or 'y' - direction of transect
        location: Grid index for transect location
        width: Number of grid cells to average on each side

    Returns:
        1D array along specified axis
    """
```

### Building and LAD Mask Extraction:
```python
def _get_building_lad_masks(self, static_dataset, transect_params):
    """
    Extract building and LAD arrays along transect line

    Args:
        static_dataset: xarray Dataset from static file
        transect_params: Dict with axis, location, width

    Returns:
        Dict with:
        - 'buildings': 1D boolean array marking building locations
        - 'lad': 1D float array with LAD values
        - 'coordinates': 1D array of x or y coordinates
    """
```

### Main Plotting Function:
```python
def _create_transect_plot(self, scenarios_data, variable, domain, comparison_type):
    """
    Create complete transect plot with dual-panel layout

    Layout:
    ┌─────────────────────────────────────┐
    │  Top Panel: Line Plot               │
    │  - Multiple scenario lines          │
    │  - Building shading (grey, 50%)     │
    │  - LAD shading (green, 50%)         │
    │  - Legend with scenario labels      │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │  Bottom Panel: XY Plan View         │
    │  - Colored patches for scenarios    │
    │  - Magenta dashed transect line     │
    │  - Grid showing domain structure    │
    └─────────────────────────────────────┘

    Args:
        scenarios_data: List of dicts, each containing:
            - 'transect_values': 1D array of variable values
            - 'xy_slice': 2D array for plan view
            - 'label': Scenario name (e.g., "10m 20yrs")
            - 'color': Line color
        variable: 'ta' or 'qv'
        domain: 'parent' or 'child'
        comparison_type: 'age' or 'spacing'

    Returns:
        matplotlib Figure object
    """
```

## Phase 5: Data Requirements

### NetCDF Variables Needed:

**From av_3d or av_3d_N02:**
- `ta`: Air temperature (K)
- `qv`: Water vapor mixing ratio (kg/kg)
- Dimensions: `[time, zu_3d, y, x]`

**From static or static_N02:**
- `zt`: Topography height (m)
- `buildings_2d`: Building mask (binary)
- `lad`: Leaf area density (m²/m³) - optional
- Dimensions: `[y, x]` or `[z, y, x]` for LAD

**Coordinates:**
- `x`: X-axis coordinates (m)
- `y`: Y-axis coordinates (m)
- `zu_3d`: Vertical coordinates (m)
- `time`: Time coordinate (seconds or datetime)

### Domain-Specific Parameters:

| Domain | Resolution | Grid Size | First Data Level (zu_3d index) |
|--------|------------|-----------|-------------------------------|
| Parent | 10m        | 400×400   | 25                            |
| Child  | 2m         | 200×200   | 21                            |

## Phase 6: Scenario Color Mapping

### Color Schemes:

**For Age Comparisons:**
```python
AGE_COLORS = {
    'no_trees': '#8B0000',    # Dark red
    20: '#1f77b4',            # Blue
    40: '#2ca02c',            # Green
    60: '#d62728',            # Red
    80: '#ff7f0e'             # Orange
}
```

**For Spacing Comparisons:**
```python
SPACING_COLORS = {
    'no_trees': '#8B0000',    # Dark red
    10: '#1f77b4',            # Blue
    15: '#2ca02c',            # Green
    20: '#9467bd',            # Purple
    25: '#ff7f0e'             # Orange
}
```

## Phase 7: Testing Strategy

### Unit Tests:
1. Test terrain-following extraction with synthetic topography
2. Test transect extraction in both x and y directions
3. Test time averaging with known input
4. Test building/LAD mask extraction

### Integration Tests:
1. Run single plot type with minimal config
2. Verify output file naming convention
3. Check plot dimensions and layout

### Full Validation:
1. Execute all 8 plot types
2. Visual comparison with example images
3. Verify color scales match examples
4. Check legend labels and formatting

## Phase 8: Error Handling

### Graceful Fallbacks:
1. **Missing LAD data**: Skip LAD shading, log warning
2. **Missing building data**: Skip building shading, log warning
3. **Invalid z_offset**: Clamp to valid range, log warning
4. **Missing scenarios**: Skip unavailable combinations, log info
5. **Transect out of bounds**: Adjust to domain limits, log warning

### Validation Checks:
```python
def _validate_config(self, settings):
    """Validate fig_6 configuration parameters"""
    # Check terrain_mask_height_z is non-negative
    # Check transect_axis is 'x' or 'y'
    # Check transect_location is within domain bounds
    # Check scenario combinations exist in loaded data
```

## Phase 9: File Outputs

### Expected Directory Structure:
```
results/full_analysis/run_TIMESTAMP/
└── fig_6/
    ├── fig_6a_ta_parent_age.png
    ├── fig_6a_ta_parent_age.pdf
    ├── fig_6a_ta_parent_spacing.png
    ├── fig_6a_ta_parent_spacing.pdf
    ├── fig_6b_ta_child_age.png
    ├── fig_6b_ta_child_age.pdf
    ├── fig_6b_ta_child_spacing.png
    ├── fig_6b_ta_child_spacing.pdf
    ├── fig_6c_qv_parent_age.png
    ├── fig_6c_qv_parent_age.pdf
    ├── fig_6c_qv_parent_spacing.png
    ├── fig_6c_qv_parent_spacing.pdf
    ├── fig_6d_qv_child_age.png
    ├── fig_6d_qv_child_age.pdf
    ├── fig_6d_qv_child_spacing.png
    └── fig_6d_qv_child_spacing.pdf
```

**Total: 16 files (8 PNG + 8 PDF)**

## Phase 10: Performance Considerations

### Optimization Strategies:
1. **Cache terrain masks**: Compute once per domain, reuse for all scenarios
2. **Lazy loading**: Only load required variables (ta or qv)
3. **Vectorized operations**: Use numpy broadcasting for terrain-following extraction
4. **Memory management**: Process one scenario at a time if memory constrained
5. **Parallel plotting**: Can generate different plot types concurrently

### Expected Processing Time:
- Terrain extraction per scenario: ~2-5 seconds
- Time averaging: ~1 second
- Plotting per figure: ~3-5 seconds
- **Total for all 8 plots: ~2-4 minutes**

## Key Design Decisions

1. **Modular Structure**: Separate methods for each plot type enables easy testing and maintenance
2. **Reusable Core Functions**: Terrain extraction, time averaging, and transect extraction are generic and can be reused
3. **Flexible Configuration**: All parameters user-configurable via YAML for maximum flexibility
4. **Consistent Naming**: Follows established `fig_Xa_plot_type.format` convention
5. **Error Handling**: Graceful fallbacks if LAD/building data missing
6. **Performance**: Cache terrain masks and static data to avoid recomputation
7. **Separate Files**: Each comparison type (age vs spacing) generates separate output files for clarity
8. **User-Configurable Scenarios**: Allow users to specify which spacing/age combinations to compare

## Estimated Code Complexity

### Lines of Code:
- Core extraction functions: ~200 lines
- Plotting layout function: ~300 lines
- Individual plot type methods: ~400 lines (8 methods × 50 lines)
- Helper functions: ~100 lines
- Documentation and comments: ~200 lines
- **Total for terrain_transect.py: ~1,200 lines**

### Configuration:
- YAML additions: ~50 lines

### Integration:
- `__main__.py` updates: ~5 lines
- `plots/__init__.py` updates: ~10 lines
- `utils/figure_mapper.py` updates: ~5 lines
- **Total integration: ~20 lines**

### **Grand Total: ~1,270 lines of code**

## Implementation Order

1. **Setup** (30 min)
   - Create terrain_transect.py skeleton
   - Add basic class structure inheriting from BasePlotter
   - Add to __init__.py and __main__.py

2. **Core Functions** (3 hours)
   - Implement `_extract_terrain_following_slice()`
   - Implement `_time_average_data()`
   - Implement `_extract_transect_line()`
   - Test with sample data

3. **Mask Extraction** (1 hour)
   - Implement `_get_building_lad_masks()`
   - Handle missing data gracefully

4. **Plotting Layout** (3 hours)
   - Implement `_create_transect_plot()`
   - Create dual-panel layout
   - Add shading for buildings/LAD
   - Add transect line to plan view

5. **Plot Type Methods** (4 hours)
   - Implement all 8 `_plot_*` methods
   - Test each with real data
   - Verify color schemes match examples

6. **Configuration** (1 hour)
   - Add fig_6 section to YAML
   - Add validation logic
   - Document all parameters

7. **Integration** (1 hour)
   - Update all integration files
   - Test end-to-end workflow

8. **Testing & Refinement** (2 hours)
   - Run full test suite
   - Visual validation against examples
   - Fix any issues
   - Performance optimization

**Total Estimated Time: 15-16 hours**

## Dependencies

### Python Packages:
- numpy
- matplotlib
- xarray
- scipy (for potential interpolation)
- logging (built-in)

### Data Files Required:
- Parent domain: `*_av_3d_merged.nc`, `*_static`
- Child domain: `*_av_3d_N02_merged.nc`, `*_static_N02`
- All scenario combinations specified in config

### Existing Code Dependencies:
- `BasePlotter` class
- `OutputManager` for file saving
- `PALMDataLoader` for data access
- `FigureMapper` for naming consistency

## Documentation Updates

### Files to Update:
1. **CLAUDE.md**: Add fig_6 documentation
2. **README.md**: Update plot types list
3. **Code docstrings**: Full documentation for all methods
4. **Example config**: Add commented fig_6 section

### Documentation Sections:
- Overview of terrain-following transect analysis
- Configuration parameter descriptions
- Expected outputs and interpretation
- Troubleshooting common issues
