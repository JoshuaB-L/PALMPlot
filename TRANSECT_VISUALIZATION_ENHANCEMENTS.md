# Transect Visualization Enhancements

**Date**: 2025-11-08
**Features**: Three new enhancements to transect 2D slice visualization

---

## Overview

Enhanced the transect plotting functionality with three key improvements:

1. **X-axis extent matching** - 2D slice x-axis now matches the line plot above
2. **Transect width cropping** - Optional cropping of 2D slice to show only a band around the transect line
3. **Horizontal colorbar** - Full-width colorbar below x-axis showing variable values

---

## 1. X-Axis Extent Matching

### What Changed

Previously, the 2D slice plot used the full domain extent `[0, domain_size]` regardless of the line plot extent. Now, the 2D slice x-axis automatically matches the line plot extent.

### Benefits

- Visual alignment between line plot and 2D slice
- Clearer correspondence between features in both panels
- Consistent spatial reference across both visualizations

### Implementation

The code now:
1. Extracts the coordinates from the first scenario's transect data
2. Calculates `x_extent_min` and `x_extent_max` from these coordinates
3. Uses these values in the `extent` parameter of `imshow()`

---

## 2. Transect Visualization Width Cropping

### What Changed

Added a new configuration parameter `transect_visualization_width` that crops the 2D slice to show only a band of specified width centered on the transect line.

### Configuration

```yaml
plots:
  figures:
    fig_4:
      settings:
        terrain_following:
          child:  # or parent
            transect_visualization_width: 20  # ±20 grid cells around transect line
```

**Parameter Details**:
- **Type**: Integer (number of grid cells)
- **Default**: `null` (no cropping, show full domain)
- **Units**: Grid cells (not meters)
- **Behavior**: Crops perpendicular to transect axis
  - For `transect_axis: "x"`: Crops in y-direction
  - For `transect_axis: "y"`: Crops in x-direction

### Examples

**Example 1: Child domain with 2m resolution**
```yaml
child:
  transect_axis: "x"
  transect_location: 100  # Middle of 200×200 domain
  transect_visualization_width: 20
```
Result: Shows a 80m band (40 cells × 2m/cell) centered on the transect line

**Example 2: Parent domain with 10m resolution**
```yaml
parent:
  transect_axis: "x"
  transect_location: 200
  transect_visualization_width: 10
```
Result: Shows a 200m band (20 cells × 10m/cell) centered on the transect line

**Example 3: No cropping (default)**
```yaml
child:
  transect_axis: "x"
  transect_location: 100
  # transect_visualization_width: null  # or omit entirely
```
Result: Shows full domain width

### Benefits

- Focus on relevant area around the transect
- Reduce visual clutter from distant features
- Highlight details near the transect line
- Better visualization of localized features (buildings, trees, etc.)

### Implementation Details

The cropping logic:
1. Determines transect axis direction
2. Calculates crop indices: `[location - width, location + width + 1]`
3. Clips to valid domain bounds: `[0, domain_size]`
4. Crops the numpy array: `array[y_min:y_max, :]` or `array[:, x_min:x_max]`
5. Updates extent to match cropped region
6. Transect line remains centered in the cropped view

---

## 3. Horizontal Colorbar

### What Changed

Automatically adds a horizontal colorbar below the x-axis of the 2D slice plot that spans the full width.

### Features

**Horizontal Colorbar**:
- Orientation: Horizontal (below x-axis)
- Width: Spans full width of the plot
- Height: 5% of the 2D slice panel height
- Position: Below the "X Axis (m)" label with 0.6 units padding
- Label: Shows variable name and units (e.g., "Air Temperature (°C)")
- Colormap: Matches the 2D slice colormap
- Value range: Matches the variable range used in the plot

### Benefits

- Clear visual reference for variable values
- Easy to interpret colors in the 2D slice
- Professional appearance matching standard scientific plots
- Consistent with colorbar conventions
- Full-width display maximizes readability

### Examples

**Typical colorbar labels**:
- Temperature plots: "Air Temperature (°C)" with range [24, 28]
- UTCI plots: "UTCI (°C)" with range [20, 35]
- Radiation plots: "Net Radiation (W/m²)" with range [0, 200]

---

## Usage Examples

### Minimal Configuration (Full Domain)

```yaml
plots:
  figures:
    fig_4:
      enabled: true
      settings:
        domain: "child"
        variable: "temperature"
        extraction_method: "terrain_following"
        terrain_following:
          output_mode: "2d"
          child:
            transect_axis: "x"
            transect_location: 100
```

**Result**:
- ✅ X-axis matches line plot
- ✅ Full domain shown (no cropping)
- ✅ Horizontal colorbar added

### Cropped View Configuration

```yaml
plots:
  figures:
    fig_4:
      enabled: true
      settings:
        domain: "child"
        variable: "temperature"
        extraction_method: "terrain_following"
        terrain_following:
          output_mode: "2d"
          child:
            transect_axis: "x"
            transect_location: 100
            transect_visualization_width: 20  # ±20 cells (40m total width)
```

**Result**:
- ✅ X-axis matches line plot (full 200m)
- ✅ Y-axis cropped to 40m band around transect
- ✅ Horizontal colorbar shows variable values

---

## Technical Details

### Modified Files

**File**: `plots/terrain_transect.py`
**Method**: `_generate_plot_multivar()`
**Lines**: 3346-3547

### Key Changes

1. **Extract line plot coordinates** (lines 3381-3389):
   ```python
   coordinates = scenarios_data[0]['coordinates']
   x_coords_line = coordinates * resolution
   x_extent_min = x_coords_line.min()
   x_extent_max = x_coords_line.max()
   ```

2. **Crop 2D slice if requested** (lines 3391-3431):
   ```python
   if transect_visualization_width is not None:
       if transect_axis == 'x':
           y_min = max(0, transect_location - transect_visualization_width)
           y_max = min(ny, transect_location + transect_visualization_width + 1)
           xy_slice_plot = xy_slice_plot[y_min:y_max, :]
           extent_y_min = y_min * resolution
           extent_y_max = y_max * resolution
   ```

3. **Update extent to match** (lines 3451-3458):
   ```python
   im = ax_map.imshow(
       xy_slice_plot,
       extent=[extent_x_min, extent_x_max, extent_y_min, extent_y_max],
       aspect='auto'  # Allow non-square pixels
   )
   ```

4. **Add horizontal colorbar** (lines 3487-3497):
   ```python
   from mpl_toolkits.axes_grid1 import make_axes_locatable
   divider = make_axes_locatable(ax_map)
   cax = divider.append_axes("bottom", size="5%", pad=0.6)
   cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
   cbar.set_label(var_label, fontsize=10)
   ```

### Aspect Ratio Change

Changed from `aspect='equal'` to `aspect='auto'` to allow:
- X-axis to match line plot extent (potentially stretched)
- Y-axis to show cropped region (potentially compressed)
- Non-square pixels when cropping is applied

This is intentional to ensure x-axis alignment with the line plot.

---

## Backward Compatibility

All changes are **fully backward compatible**:

1. If `transect_visualization_width` is not specified → full domain shown (same as before)
2. Horizontal colorbar is added automatically (no config needed)
3. Existing configurations work without modification
4. X-axis extent matching is automatic (no config needed)

---

## Testing

### Test Configuration

See `palmplot_config_terrain_following_test.yaml` for working example:

```yaml
plots:
  figures:
    fig_4:
      enabled: true
      settings:
        terrain_following:
          child:
            transect_visualization_width: 20
```

### Expected Behavior

**With `transect_visualization_width: 20`**:
1. Line plot: Shows full transect (200 grid cells for child domain)
2. 2D slice: Shows ±20 cells (40 cells total, 80m width)
3. X-axis: Both plots aligned (0-400m for child)
4. Colorbar: Horizontal bar below x-axis showing variable values

**With `transect_visualization_width: null` or omitted**:
1. Line plot: Shows full transect (200 grid cells)
2. 2D slice: Shows full domain (200 grid cells, 400m)
3. X-axis: Both plots aligned (0-400m)
4. Colorbar: Horizontal bar below x-axis showing variable values

---

## Future Enhancements

Possible future additions:
1. User-configurable colorbar position (`top`/`bottom`)
2. User-configurable colorbar size (`size` parameter)
3. Option to disable colorbar (`show_colorbar: false`)
4. Vertical colorbar option (`orientation: vertical`)
5. Custom colorbar tick locations and labels

---

## Summary

**What's New**:
- ✅ X-axis extent matching between line plot and 2D slice
- ✅ Optional transect width cropping via `transect_visualization_width`
- ✅ Automatic horizontal colorbar below x-axis
- ✅ Fully backward compatible
- ✅ Professional scientific visualization standard

**Configuration**:
- Single new parameter: `transect_visualization_width` (optional)
- No changes required to existing configs

**Benefits**:
- Better visual alignment
- Focus on relevant features
- Professional appearance
- Clear spatial reference

---

**Status**: ✅ Implementation Complete
**Testing**: Ready for user validation
**Documentation**: Complete
