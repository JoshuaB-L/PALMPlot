# XY Variable Plotting - Final Fix

**Date**: 2025-11-07
**Issue**: XY surface variables (UTCI, radiation) showing empty/incorrect data
**Root Cause Identified**: `AttributeError: '_needs_kelvin_conversion' method doesn't exist`

---

## Problem Analysis

### What Was Happening

1. ✅ **XY variables detected correctly**:
   ```
   === SURFACE VARIABLE DETECTED (2D) ===
   Variable 'bio_utci*_xy' is a surface variable (file_type: av_xy)
   Skipping terrain-following extraction, using surface level directly
   ```

2. ✅ **Surface extraction successful**:
   ```
   Extracted surface level (zu1_xy[0])
   ```

3. ✅ **Time averaging completed**:
   ```
   === TIME SELECTION CONFIGURATION ===
   All 49 time steps are valid
   Time averaging complete
   ```

4. ❌ **CRASH during unit conversion check**:
   ```
   Error in terrain-following extraction: 'TerrainTransectPlotter' object has no attribute '_needs_kelvin_conversion'
   ```

5. ❌ **No data returned to plotting**:
   - Extraction failed → no data added to scenarios_data
   - Plot created with empty data
   - Showed wrong title "Water Vapor Mixing Ratio" (default/previous variable)

### Root Cause

**File**: `plots/terrain_transect.py` line 1563 (before fix)

In my Phase 6 implementation (2D surface variable fix), I incorrectly called:
```python
needs_conversion = self._needs_kelvin_conversion(filled_2d, var_name_found)
```

This method **does not exist**! The codebase uses inline checks instead:
```python
needs_conversion = np.mean(valid_data) > 100.0
```

---

## Solution Applied

### Fix Implementation

**File**: `plots/terrain_transect.py` lines 1563-1575

Replaced the non-existent method call with inline unit conversion detection:

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

### How It Works

1. **Extracts valid (non-NaN) values** from the 2D surface array
2. **Computes mean value**
3. **Applies heuristic**:
   - If mean > 100 → data is in Kelvin (e.g., 297K for 24°C)
   - If mean < 100 → data is already in target units
4. **Sets `needs_conversion` flag** for plotting code

### Why This Works for XY Variables

For typical XY surface variables:
- **UTCI**: Values range 15-45°C → mean < 100 → `needs_conversion = False` ✓
- **Radiation**: Values range 0-800 W/m² → mean could be > 100
  - BUT radiation is NOT a temperature, so conversion won't be applied anyway
  - The plotting code only converts `if variable == 'ta' and needs_conversion`
- **Surface Temperature**: Values ~297K → mean > 100 → `needs_conversion = True` ✓

---

## Expected Results After Fix

### Terminal Output Should Show:

```
=== SURFACE VARIABLE DETECTED (2D) ===
  Variable 'bio_utci*_xy' is a surface variable (file_type: av_xy)
  Skipping terrain-following extraction, using surface level directly
  Extracted surface level (zu1_xy[0])

=== TIME SELECTION CONFIGURATION ===
  Total available time steps: 49
  Method: 'mean'
  Time processing: Averaging with corrupted step detection...
  All 49 time steps are valid
  Time averaging complete

  Temperature unit detection: needs_kelvin_conversion=False    <-- NEW LINE

=== SURFACE EXTRACTION COMPLETE ===
  2D field shape: (200, 200)
  Data range: min=XX.XX, max=XX.XX, mean=XX.XX
```

**NO MORE ERROR!** The extraction should complete successfully and return data.

### Plots Should Show:

1. **UTCI plots**: Real data with values in the 15-45°C range
2. **Radiation plots**: Real data with values in the 0-800 W/m² range
3. **Correct titles**: "UTCI" and "Net Radiation" (not "Water Vapor Mixing Ratio")

---

## Testing Instructions

### 1. Run the Fixed Code

```bash
cd /home/joshuabl/phd/thf_forest_study/code/python
python -m palmplot_thf palmplot_thf/palmplot_config_multivar_test.yaml
```

### 2. Check Terminal Output

Look for these indicators of success:

**For UTCI (bio_utci*_xy)**:
```
=== SURFACE VARIABLE DETECTED (2D) ===
  Variable 'bio_utci*_xy' is a surface variable
  Temperature unit detection: needs_kelvin_conversion=False
=== SURFACE EXTRACTION COMPLETE ===
  2D field shape: (200, 200)
  Data range: min=25.XX, max=35.XX, mean=30.XX  <-- Should show real values
```

**For Radiation (rad_net*_xy)**:
```
=== SURFACE VARIABLE DETECTED (2D) ===
  Variable 'rad_net*_xy' is a surface variable
  Temperature unit detection: needs_kelvin_conversion=False
=== SURFACE EXTRACTION COMPLETE ===
  2D field shape: (400, 400)
  Data range: min=0.XX, max=600.XX, mean=350.XX  <-- Should show real values
```

**NO ERROR MESSAGES** like:
```
Error in terrain-following extraction: 'TerrainTransectPlotter' object has no attribute '_needs_kelvin_conversion'
```

### 3. Check Generated Plots

**Files to check**:
- `fig_6e_utci_child_age.png`
- `fig_6h_radiation_net_parent_age.png`
- `fig_6i_radiation_net_child_age.png`

**What to verify**:
- [ ] Plots are NOT empty (not just blank axes)
- [ ] Data values are realistic (UTCI: 15-45°C, Radiation: 0-800 W/m²)
- [ ] Titles show correct variable names (not "Water Vapor Mixing Ratio")
- [ ] Color scales show data variation (not flat/uniform)

### 4. Check Cache Files (Optional)

Since XY variables are **NOT cached**, you won't see them in cache files. This is expected behavior.

**Cache files should contain**:
- Parent domain: `ta`, `rh`, `q` (3D atmospheric variables only)
- Child domain: `ta`, `rh` (3D atmospheric variables only)

---

## Technical Details

### Why XY Variables Are Not Cached

2D surface variables are **NOT cached** because:
1. **Different dimensions**: `zu1_xy` vs `ku_above_surf` (incompatible)
2. **No terrain-following**: Extracted directly from surface level
3. **Fast extraction**: No expensive iteration needed
4. **Small data**: Single time-averaged 2D array per variable

Caching is only beneficial for expensive terrain-following computation (3D atmospheric variables).

### Complete Extraction Flow for XY Variables

```
1. Variable requested: "utci"
   ↓
2. Metadata lookup: palm_name="bio_utci*_xy", file_type="av_xy"
   ↓
3. Dataset selection: av_xy_n02 (child domain)
   ↓
4. Variable discovery: Wildcard match → "bio_utci_xy" found
   ↓
5. Dimensionality detection: zu1_xy dimension → 2D surface variable
   ↓
6. Extract surface level: zu1_xy[0]
   ↓
7. Time averaging: Mean over all valid timesteps
   ↓
8. Unit conversion check: mean < 100 → needs_conversion=False
   ↓
9. Return: (filled_2d, var_name, needs_conversion)
   ↓
10. Plotting: Use data with correct variable name and units
```

---

## Summary of All Fixes

### Phase 6 (Original Implementation)
✅ Dimensionality detection (2D vs 3D)
✅ Surface variable bypass of terrain-following
✅ Direct surface extraction
❌ **BUG**: Called non-existent `_needs_kelvin_conversion()` method

### Phase 6 Fix (This Fix)
✅ Replaced method call with inline unit conversion detection
✅ Proper error handling for empty data
✅ Logging of conversion decision

---

## Files Modified

1. `plots/terrain_transect.py` - Lines 1563-1575: Fixed unit conversion check
2. `XY_VARIABLE_FIX_FINAL.md` - This documentation

---

## Next Steps

1. **Test the fix** - Run with `palmplot_config_multivar_test.yaml`
2. **Verify XY plots show data** - Check UTCI and radiation plots
3. **Verify cache files** - Confirm multi-variable caching works for 3D variables
4. **Continue with Phase 7-10** - Unit conversion framework, metadata integration, testing

---

## Success Criteria

✅ **No errors** during XY variable extraction
✅ **UTCI plots** show realistic data (15-45°C range)
✅ **Radiation plots** show realistic data (0-800 W/m² range)
✅ **Correct titles** on all plots
✅ **Cache files** contain multiple 3D variables (ta, rh, q)

---

**Status**: Fix applied, ready for testing
**Confidence**: High - root cause identified and properly fixed
**Impact**: Critical - enables XY variable plotting for UTCI, radiation, and all other surface variables
