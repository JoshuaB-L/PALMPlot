# Multi-Variable Test Run - Error Analysis

**Date**: 2025-11-08
**Config**: palmplot_config_multivar_test.yaml
**Status**: 4 Errors Identified, 2 Fixed, 1 Partially Diagnosed, 1 Cascading

---

## Executive Summary

User attempted production run with all variables and multiple cases but encountered:
1. ✅ **FIXED**: Config/data mismatch - plots requested non-existent scenarios
2. ✅ **FIXED**: matplotlib backend crash - tkinter not suitable for headless environment
3. ⚠️ **DIAGNOSED**: Cache variable merge issue - only 1 variable saved per file type
4. ✅ **CASCADING**: Empty slice warnings - caused by Error #1

---

## Error #1: Config/Data Mismatch (CRITICAL) ✅ FIXED

### Problem
Config defined available scenarios as:
```yaml
data:
  spacings: [10, 15]    # Line 27
  ages: [20, 40]        # Line 28
```

But plot settings requested:
```yaml
age_comparison:
  constant_spacing: [10, 15, 20, 25]    # Line 560 - includes 20, 25 that don't exist
  varying_ages: [20, 40, 60, 80]         # Line 561 - includes 60, 80 that don't exist
```

### Impact
- Plots tried to load 16 scenarios (4 spacings × 4 ages)
- Only 4 scenarios exist in data: 10m_20yrs, 10m_40yrs, 15m_20yrs, 15m_40yrs
- 12 scenarios failed to load → cascaded to Error #4

### Root Cause
Configuration inconsistency - plot settings don't match data availability

### Fix Applied
**File**: `palmplot_config_multivar_test.yaml` lines 560-561

```yaml
# BEFORE (BROKEN)
constant_spacing: [10, 15, 20, 25]
varying_ages: [20, 40, 60, 80]

# AFTER (FIXED)
constant_spacing: [10, 15]              # MUST MATCH data.spacings
varying_ages: [20, 40]                  # MUST MATCH data.ages
```

### Verification
Run config and verify no "scenario not found" errors in logs.

---

## Error #2: matplotlib Backend Crash (CRITICAL) ✅ FIXED

### Problem
Terminal output showed:
```
RuntimeError: main thread is not in main loop
Aborted (core dumped)
```

### Root Cause
- matplotlib defaulting to tkinter backend
- tkinter requires GUI/display and main thread event loop
- Fails in headless/non-GUI environments (WSL, SSH, background jobs)
- Causes core dump when trying to initialize

### Impact
- Complete script termination
- No plots generated
- Data processing incomplete

### Fix Applied
**File**: `__main__.py` lines 10-13

```python
# Set matplotlib backend BEFORE any other imports
# Use 'Agg' (non-interactive) to prevent tkinter crashes in headless/non-GUI environments
import matplotlib
matplotlib.use('Agg')
```

### Why This Works
- 'Agg' backend is non-interactive (no GUI needed)
- Safe for headless environments, SSH sessions, background jobs
- Fully supports all output formats (PNG, PDF, SVG)
- Must be set BEFORE any matplotlib submodules are imported

### Verification
Run script and verify no tkinter-related errors or core dumps.

---

## Error #3: Cache Variable Merge Issue ⚠️ DIAGNOSED

### Problem
Terminal output showed:
- Only `ta` (temperature) saved to av_3d cache files
- Only `utci` saved to av_xy cache files
- Should have saved:
  - **av_3d**: 9 variables (ta, q, qv, rh, theta, wspeed, wdir, p, ti)
  - **av_xy**: 8 variables (utci, pet, rad_net, rad_sw_in, rad_sw_out, rad_lw_in, rad_lw_out, tsurf, shf, ghf)

### Expected Behavior
Config correctly sets `variables: "auto"` for both cache types (lines 461-475 and 537-547), which should cache all enabled variables.

### Code Analysis

#### Cache File Naming (Correct)
```python
# surface_data_io.py line 569
filename = f"{case_name}_{domain_type}_surface_data.nc"
```
- Filename based ONLY on case_name and domain_type (NOT variable name)
- All variables for a case/domain should share ONE file
- This is the expected and correct behavior

#### Merge Logic (Appears Correct)
**Surface data** (terrain_transect.py lines 1851-1891):
```python
# Start with new variable
surface_data_dict = {var_name_found: filled_2d}

# Check if cache file exists
if cache_path.exists():
    try:
        # Load existing variables
        existing_cache = self._load_surface_data(...)

        if existing_cache and 'surface_data' in existing_cache:
            # Merge: keep existing vars, update current var
            for existing_var, existing_data in existing_cache['surface_data'].items():
                if existing_var != var_name_found:
                    surface_data_dict[existing_var] = existing_data  # Keep
                else:
                    # Update current variable

    except Exception as e:
        self.logger.warning(f"Could not merge with existing cache: {e}")
        # Continue with just the new variable  ← PROBLEM HERE!

# Save merged dict
self._save_surface_data(..., surface_data_dict, ...)
```

**Terrain mask** (terrain_transect.py lines 2404-2437): Same pattern

### Hypothesis: Merge Failing Silently

#### Possible Causes
1. **Exception during load**: `_load_surface_data()` or `_load_terrain_mask()` throwing exception
   - Caught by try/except (lines 1887-1890, 2434-2436)
   - Warning logged but execution continues with only new variable
   - New variable overwrites existing file

2. **File write mode**: NetCDF writer using mode='w' (overwrite)
   - Writer: `terrain_mask_io.py` line 85: `ds.to_netcdf(output_path, ...)`
   - No `mode` parameter specified → defaults to 'w' (overwrite)
   - **But should still work** because merged dict contains all variables

3. **Load validation failure**: Cached file fails validation checks
   - Lines 483-497 define validation settings
   - If validation fails and `on_mismatch: "recompute"`, might skip merge
   - Or raises exception that's caught

4. **Variable name mismatch**: var_name_found might not match stored names
   - PALM wildcard expansion (bio_utci*_xy → bio_utci_xy)
   - If stored as 'bio_utci_xy' but searching for 'bio_utci*_xy', won't match

### Diagnostic Steps Needed

**To confirm issue**, check the log file for warnings:
```bash
grep "Could not merge" /home/joshuabl/phd/thf/thf_forest_study/code/python/palmplot_thf/logs/*.log
```

Expected to find warnings like:
```
Could not merge with existing cache: [specific error message]
```

**To identify root cause**, add debug logging before merge:
```python
if cache_path.exists():
    self.logger.info(f"DEBUG: Cache file exists, attempting merge: {cache_path}")
    self.logger.info(f"DEBUG: New variable to add/update: {var_name_found}")
    try:
        existing_cache = self._load_surface_data(...)
        if existing_cache:
            self.logger.info(f"DEBUG: Loaded cache, keys: {existing_cache.keys()}")
            if 'surface_data' in existing_cache:
                existing_vars = list(existing_cache['surface_data'].keys())
                self.logger.info(f"DEBUG: Existing variables in cache: {existing_vars}")
```

### Recommended Fix

**Option 1: Better error handling and logging**
```python
except Exception as e:
    self.logger.error(f"CRITICAL: Merge failed, will lose existing variables!")
    self.logger.error(f"  Error: {e}")
    self.logger.error(f"  Existing cache: {cache_path}")
    self.logger.error(f"  New variable: {var_name_found}")
    import traceback
    self.logger.error(f"  Traceback: {traceback.format_exc()}")
    # Still continue, but with better visibility
```

**Option 2: Fail-fast on merge errors**
```python
except Exception as e:
    raise RuntimeError(
        f"Failed to merge with existing cache {cache_path}. "
        f"New variable '{var_name_found}' cannot be added without losing existing data. "
        f"Original error: {e}"
    )
```

**Option 3: Pre-load all variables before any extraction**
- Change architecture: Determine all variables to cache upfront
- Extract all variables for a case before writing cache
- Write once with complete variable set
- Avoids iterative merge entirely

### Current Status
⚠️ **Partially diagnosed** - merge logic appears correct but likely failing with suppressed exceptions. Need actual log file from failed run to confirm hypothesis.

---

## Error #4: Empty Slice RuntimeWarning ✅ CASCADING

### Problem
```
RuntimeWarning: Mean of empty slice
```

### Root Cause
Cascading error from Error #1. When plots try to extract data for non-existent scenarios (20m, 25m, 60yrs, 80yrs), the data arrays are empty, causing `.mean()` to fail on empty arrays.

### Fix
Resolved by fixing Error #1 (config mismatch). Once scenarios match available data, this warning will disappear.

### Verification
After fixing Error #1, verify no "Mean of empty slice" warnings in output.

---

## Summary of Fixes Applied

### ✅ Fixed (2/4)

1. **Config file** (`palmplot_config_multivar_test.yaml`)
   - Lines 560-561: Matched age_comparison to data availability
   - Comment added: "MUST MATCH data.spacings/ages above"

2. **matplotlib backend** (`__main__.py`)
   - Lines 10-13: Set backend to 'Agg' before imports
   - Prevents tkinter crashes in headless environments

### ⚠️ Diagnosed But Not Fixed (1/4)

3. **Cache merge issue**
   - Root cause identified: Likely exception during merge
   - Exception caught and suppressed → each variable overwrites previous
   - Need log file from actual failed run to confirm
   - Recommended: Add diagnostic logging or fail-fast behavior

### ✅ Auto-Resolved (1/4)

4. **Empty slice warnings**
   - Cascading effect of Error #1
   - Will disappear once config fix is applied

---

## Next Steps

### Immediate (Required Before Production Run)

1. **Test config fix**:
   ```bash
   python -m palmplot_thf palmplot_config_multivar_test.yaml
   ```
   - Verify only 4 scenarios are processed (10m_20yrs, 10m_40yrs, 15m_20yrs, 15m_40yrs)
   - Verify no "scenario not found" errors

2. **Verify backend fix**:
   - Check for no tkinter errors or core dumps
   - Verify plots are generated successfully

3. **Check cache variable counts**:
   - After run completes, inspect cache files:
   ```bash
   ls -lh /home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/cache/surface_data/
   ls -lh /home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/cache/terrain_masks/
   ```
   - Use `ncdump -h <file>` to check variable counts
   - **Expected**:
     - av_3d files: 9 variables (ta, q, qv, rh, theta, wspeed, wdir, p, ti)
     - av_xy files: 10 variables (utci, pet, rad_net, rad_sw_in, rad_sw_out, rad_lw_in, rad_lw_out, tsurf, shf, ghf)

4. **Review logs for merge warnings**:
   ```bash
   grep -i "merge" /home/joshuabl/phd/thf/thf_forest_study/code/python/palmplot_thf/logs/*.log
   grep -i "could not" /home/joshuabl/phd/thf/thf_forest_study/code/python/palmplot_thf/logs/*.log
   ```

### If Cache Issue Persists

1. **Add diagnostic logging** (recommended fix in code section above)
2. **Check NetCDF file contents** after first variable:
   ```bash
   # After first variable extracts
   ncdump -h <cache_file>.nc
   # Should show 1 variable

   # After second variable extracts
   ncdump -h <cache_file>.nc
   # Should show 2 variables (if merge works)
   ```

3. **Manual cache deletion** for clean test:
   ```bash
   rm -rf /home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/cache/surface_data/*
   rm -rf /home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/cache/terrain_masks/*
   ```

### Full Production Run (After Verification)

If you need all scenarios (not just test subset):
```yaml
data:
  spacings: [10, 15, 20, 25]    # Full dataset
  ages: [20, 40, 60, 80]        # Full dataset

# And update age_comparison to match:
age_comparison:
  constant_spacing: [10, 15, 20, 25]
  varying_ages: [20, 40, 60, 80]
```

**Warning**: Ensure data files exist for all 16 combinations before running!

---

## Files Modified

1. `palmplot_config_multivar_test.yaml` - Lines 560-561 (config mismatch fix)
2. `__main__.py` - Lines 10-13 (backend fix)

## Files Analyzed

1. `palmplot_config_multivar_test.yaml` - Configuration
2. `__main__.py` - Entry point
3. `plots/base_plotter.py` - matplotlib setup
4. `plots/terrain_transect.py` - Cache merge logic
5. `core/terrain_mask_io.py` - NetCDF writer
6. `core/surface_data_io.py` - Surface data writer

---

**Analysis completed**: 2025-11-08
**Fixes applied**: 2 critical, 1 cascading auto-resolved
**Remaining**: 1 partially diagnosed (cache merge) - needs log file verification
