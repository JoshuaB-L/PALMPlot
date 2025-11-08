# Critical Fixes Applied - 2025-11-07

## Summary

Three critical issues have been fixed to complete the multi-variable support implementation:

1. ✅ **XY Variable Time Averaging Error** - Fixed missing method
2. ✅ **Multi-Variable Caching** - Fixed single-variable limitation
3. ✅ **Per-Domain Cache Control** - Added config structure

---

## Fix #1: XY Variable Extraction Error

### Problem
XY surface variables (UTCI, radiation) were failing with error:
```
Error in terrain-following extraction: 'TerrainTransectPlotter' object has no attribute '_needs_kelvin_conversion'
```

This caused all XY plots to be empty and show wrong titles ("Water Vapor Mixing Ratio" instead of "UTCI").

### Root Cause
Phase 6 implementation (dimensionality detection) had TWO bugs:
1. ❌ Called non-existent `_time_average_with_corruption_detection()` method
2. ❌ Called non-existent `_needs_kelvin_conversion()` method

Both methods don't exist - the codebase uses inline implementations instead.

### Fixes Applied

#### Fix 1a: Time Averaging (Line 1514-1561)
Replaced the non-existent `_time_average_with_corruption_detection()` with inline time averaging logic:

```python
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
            suspicious = False
        valid_time_mask.append(not suspicious)
    else:
        valid_time_mask.append(True)

# ... filter and average valid time steps
filled_2d = slice_data_with_time.mean(dim='time').values
```

#### Fix 1b: Unit Conversion Check (Lines 1563-1575)
Replaced the non-existent `_needs_kelvin_conversion()` method with inline detection:

```python
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
```

### Expected Result
- XY plots (UTCI, radiation) should now show data
- Terminal output should show:
  ```
  === SURFACE VARIABLE DETECTED (2D) ===
  Time processing: Averaging with corrupted step detection...
  All X time steps are valid
  Time averaging complete
  Temperature unit detection: needs_kelvin_conversion=False
  === SURFACE EXTRACTION COMPLETE ===
  Data range: min=XX.XX, max=XX.XX, mean=XX.XX
  ```
- **NO ERROR MESSAGES**

---

## Fix #2: Multi-Variable Terrain Mask Caching

### Problem
Only ONE variable was being written to each cache file, even though multiple variables were being extracted:
- Parent domain cache: only `rh`
- Child domain cache: only `ta`

Each time a new variable was extracted, it **overwrote** the previous cache file instead of **adding to** it.

### Root Cause
**File**: `plots/terrain_transect.py` line 2047

The code created a mask dictionary with only the current variable:
```python
mask_data_dict = {var_name_found: mask_3d}  # ONLY ONE VARIABLE!
```

Then immediately wrote it to cache, overwriting any existing file.

### Fix Applied
**File**: `plots/terrain_transect.py` lines 2046-2100

Implemented cache merging logic:

```python
# Start with new variable
mask_data_dict = {var_name_found: mask_3d}

# Check if cache file already exists
cache_path = self._get_mask_cache_path(case_name, domain_type, settings)

if cache_path.exists():
    self.logger.info(f"Found existing cache file, will merge variables")
    try:
        # Load existing variables from cache
        existing_mask = self._load_terrain_mask(
            case_name=case_name,
            domain_type=domain_type,
            required_variables=None,  # Load all
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

# Save mask (now with multiple variables)
self._save_terrain_mask(...)

var_list = list(mask_data_dict.keys())
self.logger.info(f"✓ Terrain mask saved to cache: {len(mask_data_dict)} variable(s) - {var_list}")
```

### How It Works
1. **First variable extracted** (e.g., `ta`):
   - No cache file exists
   - Writes cache with: `{ta: mask_data}`

2. **Second variable extracted** (e.g., `rh`):
   - Cache file exists
   - Loads existing variables: `{ta: ...}`
   - Adds new variable: `{ta: ..., rh: ...}`
   - Writes combined cache

3. **Third variable extracted** (e.g., `qv`):
   - Cache file exists
   - Loads existing: `{ta: ..., rh: ...}`
   - Adds new: `{ta: ..., rh: ..., qv: ...}`
   - Writes combined cache

### Expected Result
After running all plots, cache files should contain **all** extracted variables:
- Parent domain cache: `ta`, `rh`, and any other 3D variables
- Child domain cache: `ta`, `rh`, and any other 3D variables

Terminal output should show:
```
Found existing cache file, will merge variables: ...
  Keeping existing variable 'ta' in cache
  Updating variable 'rh' in cache
Multi-variable cache: 2 total variables
✓ Terrain mask saved to cache: 2 variable(s) - ['ta', 'rh']
```

---

## Fix #3: Per-Domain Cache Variable Control

### Problem
No way to specify which variables to cache for parent vs child domains independently from plotting variables.

### Fix Applied
**File**: `palmplot_config_terrain_following_test.yaml` lines 453-467

Updated config structure from single `variables` setting to per-domain control:

**OLD structure**:
```yaml
terrain_mask_cache:
  enabled: true
  mode: "save"
  variables: "auto"  # Applied to both domains
```

**NEW structure**:
```yaml
terrain_mask_cache:
  enabled: true
  mode: "save"

  # Per-domain variable control
  parent:
    # Variables to cache for parent domain (3D atmospheric variables only)
    # Options:
    #   - "auto": Cache all 3D variables enabled in plots for parent domain
    #   - List: ["temperature", "relative_humidity"] for specific variables
    # NOTE: Only variables with terrain_following: true are cached
    # NOTE: XY surface variables (file_type: av_xy) are NOT cached
    variables: "auto"

  child:
    # Variables to cache for child domain (3D atmospheric variables only)
    variables: "auto"
```

### Usage Examples

**Cache all plotted variables** (recommended):
```yaml
parent:
  variables: "auto"
child:
  variables: "auto"
```

**Cache specific variables for each domain**:
```yaml
parent:
  variables: ["temperature", "relative_humidity"]
child:
  variables: ["temperature"]  # Only temperature for child
```

**Mixed approach**:
```yaml
parent:
  variables: "auto"  # All plotted variables
child:
  variables: ["temperature", "relative_humidity"]  # Specific subset
```

### Important Notes
- Only **3D atmospheric variables** are cached (variables with `terrain_following: true`)
- **2D surface variables** (XY variables like UTCI, radiation) are NOT cached
- Surface variables are extracted directly from `zu1_xy[0]` - no mask needed
- Each domain has a separate cache file

---

## What to Test

### 1. Run the test
```bash
python -m palmplot_thf palmplot_config_terrain_following_test.yaml
```

### 2. Check terminal output for:

**XY Variable Extraction**:
```
=== SURFACE VARIABLE DETECTED (2D) ===
  Variable 'bio_utci*_xy' is a surface variable (file_type: av_xy)
  Skipping terrain-following extraction, using surface level directly

=== TIME SELECTION CONFIGURATION ===
  Total available time steps: 48
  Method: 'mean'
  Time processing: Averaging with corrupted step detection...
  All 48 time steps are valid

=== SURFACE EXTRACTION COMPLETE ===
  2D field shape: (200, 200)
  Data range: min=XX.XX, max=XX.XX, mean=XX.XX
```

**Multi-Variable Caching**:
```
Found existing cache file, will merge variables: ...
  Keeping existing variable 'ta' in cache
  Keeping existing variable 'rh' in cache
  Updating variable 'qv' in cache
Multi-variable cache: 3 total variables
✓ Terrain mask saved to cache: 3 variable(s) - ['ta', 'rh', 'qv']
```

### 3. Check generated plots:
- [ ] All 9 plots generated without errors
- [ ] XY plots (UTCI, radiation) show actual data (not empty)
- [ ] Temperature plots still work as before
- [ ] Relative humidity plots still work

### 4. Check cache files:
```bash
# Find cache files
find . -name "*terrain_mask*.nc"

# Inspect variables in cache (example)
ncdump -h path/to/cache_file.nc
```

Expected: Each cache file should contain **multiple data variables**, not just one.

Example output:
```
variables:
    float ta(time, ku_above_surf, y, x) ;
    float rh(time, ku_above_surf, y, x) ;
    float qv(time, ku_above_surf, y, x) ;
    int source_level(y, x) ;
```

---

## Remaining Implementation Phases

### Phase 7: Unit Conversion Framework (~2 hours)
Replace hard-coded Kelvin detection with config-driven conversions using `variable_metadata.convert_units()`.

### Phase 9: Plotting Metadata Integration (~3 hours)
Use variable metadata for plot styling (colormaps, ranges, units, formatting).

### Phase 10: Testing & Validation (~5 hours)
Comprehensive test suite, edge cases, documentation, performance validation.

---

## If Issues Occur

### XY plots still empty
- Check terminal for `SURFACE VARIABLE DETECTED` message
- Verify time averaging completes without errors
- Check data range in terminal output (should not be all NaN)

### Cache files still have single variable
- Check terminal for "Found existing cache file, will merge" message
- If merging fails, check warning message
- Verify cache file permissions (writable)

### Config errors
- Validate YAML syntax
- Ensure `parent:` and `child:` sections exist under `terrain_mask_cache`
- Check indentation (YAML is sensitive to whitespace)

### Send me the terminal output
If errors occur, provide the full terminal output showing:
- Variable detection messages
- Time averaging messages
- Cache merging messages
- Any error tracebacks

---

## Architecture Summary

### Cache File Structure

Each scenario/domain combination has ONE cache file containing ALL variables:

```
thf_forest_lad_spacing_10m_age_20yrs_parent_terrain_mask.nc
├── ta(time, ku_above_surf, y, x)      # Temperature
├── rh(time, ku_above_surf, y, x)      # Relative humidity
├── qv(time, ku_above_surf, y, x)      # Specific humidity
├── source_level(y, x)                  # Source zu_3d indices
├── terrain_height(y, x)                # Terrain surface height
└── buildings_mask(y, x)                # Building locations

thf_forest_lad_spacing_10m_age_20yrs_child_terrain_mask.nc
├── ta(time, ku_above_surf, y, x)
├── rh(time, ku_above_surf, y, x)
└── ...
```

### Why XY Variables Are NOT Cached

2D surface variables (UTCI, radiation) are NOT cached because:
1. They use different dimensions (`zu1_xy` instead of `ku_above_surf`)
2. They don't use terrain-following extraction
3. They're extracted directly from surface level (`zu1_xy[0]`)
4. No mask computation needed - just time averaging

Caching is only beneficial for expensive terrain-following computation, which XY variables don't use.

---

## Success Criteria

### Phase 6 Complete (Dimensionality Detection)
- [x] Code distinguishes 2D surface vs 3D atmospheric variables
- [x] XY variables bypass terrain-following
- [ ] XY variables show data (awaiting user test)

### Phase 8 Complete (Multi-Variable Caching)
- [x] Multiple variables written to single cache file
- [x] Cache merging logic implemented
- [x] Per-domain cache control in config
- [ ] All extracted variables present in cache (awaiting user test)

### Ready for Phase 7 & 9
- Unit conversion framework
- Plotting metadata integration

---

**Date**: 2025-11-07
**Status**: Phases 6 & 8 Complete - Ready for Testing
**Next**: User testing required before continuing with Phases 7, 9, 10
