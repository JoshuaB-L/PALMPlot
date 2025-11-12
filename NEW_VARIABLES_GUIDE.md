# Adding New Variables Guide

**Date**: 2025-11-08
**New Variables Added**: wind_u, wind_v, wind_w, tke, pcm_transpiration, pcm_heatrate

---

## Summary

✅ **Question 1: Does the script support these new variables?**
**YES** - The code is fully generalized. Variable definitions added to config.

⚠️ **Question 2: Will new variables merge with existing cache or overwrite?**
**SHOULD MERGE** - But merge might fail silently → **VERIFY BEFORE PRODUCTION RUN**

---

## Variables Added to Config

### 3D Atmospheric Variables (av_3d) - Terrain-Following Enabled

**wind_u** - U-component wind velocity
```yaml
palm_name: "u"
file_type: "av_3d"
units: m/s
terrain_following: true
```

**wind_v** - V-component wind velocity
```yaml
palm_name: "v"
file_type: "av_3d"
units: m/s
terrain_following: true
```

**wind_w** - W-component wind velocity (vertical)
```yaml
palm_name: "w"
file_type: "av_3d"
units: m/s
terrain_following: true
```

**tke** - Turbulent Kinetic Energy
```yaml
palm_name: "e"
file_type: "av_3d"
units: m²/s²
terrain_following: true
```

### XY Surface Variables (av_xy) - Surface Extraction

**pcm_transpiration** - Plant Canopy Model Transpiration
```yaml
palm_name: "pcm_transpiration*_xy"
file_type: "av_xy"
units: W/m²
wildcard: true
terrain_following: false
```

**pcm_heatrate** - Plant Canopy Model Heat Rate
```yaml
palm_name: "pcm_heatrate*_xy"
file_type: "av_xy"
units: K/s
wildcard: true
terrain_following: false
```

---

## Cache Merge Behavior Analysis

### How It Should Work

**Designed behavior** (terrain_transect.py lines 1851-1891):
```python
# For each variable:
1. Extract and process variable data
2. Check if cache file exists for this case/domain
3. If exists:
   - Load all existing variables from cache
   - Add new variable to dictionary (or update if already exists)
   - Save merged dictionary (all old + new variable)
4. If not exists:
   - Save just the new variable
```

**Expected result**:
- Run 1: Extract `temperature` → Save `{temperature}`
- Run 2: Extract `wind_u` → Load `{temperature}`, merge → Save `{temperature, wind_u}`
- Run 3: Extract `wind_v` → Load `{temperature, wind_u}`, merge → Save `{temperature, wind_u, wind_v}`

### Potential Issue (Cache Merge Bug)

**Problem code** (lines 1887-1890):
```python
try:
    existing_cache = self._load_surface_data(...)
    # Merge existing with new
    surface_data_dict[existing_var] = existing_data
except Exception as e:
    self.logger.warning(f"Could not merge: {e}")
    # Continue with just new variable ← OVERWRITES!
```

**If merge fails**:
- Exception is caught and logged as warning
- Execution continues with ONLY new variable
- File is overwritten, losing all previous variables

**Possible failure causes**:
1. Cache file corruption
2. Validation failure (grid size mismatch, domain type mismatch)
3. Variable name mismatch (wildcard expansion issues)
4. NetCDF read errors

---

## CRITICAL: Verify Cache Merge Before Production

### Step 1: Test with 2 Variables on Single Case

**Create test config** (or modify existing):
```yaml
# Use only ONE spacing/age combination for testing
data:
  spacings: [10]
  ages: [20]

# Test with just 2 av_3d variables
plots:
  figures:
    fig_6:
      variables: ["temperature", "wind_u"]  # Just 2 for testing

      plot_matrix:
        domains: ["child"]  # Just one domain
        comparisons: ["age"]

age_comparison:
  constant_spacing: [10]
  varying_ages: [20]
```

**Set cache mode**:
```yaml
terrain_following:
  mask_cache:
    mode: "save"  # Force save/merge
```

### Step 2: Run and Monitor

```bash
# Clean existing cache first (IMPORTANT!)
rm -rf cache/terrain_masks/thf_forest_lad_spacing_10m_age_20yrs*
rm -rf cache/surface_data/thf_forest_lad_spacing_10m_age_20yrs*

# Run test
python -m palmplot_thf palmplot_config_multivar_test.yaml
```

### Step 3: Check Cache Files

**After run completes, verify BOTH variables are in cache**:

```bash
# Check av_3d cache (should have temperature AND wind_u)
ncdump -h cache/terrain_masks/thf_forest_lad_spacing_10m_age_20yrs_child_terrain_mask.nc

# Look for data variables section:
# Expected output:
# variables:
#   float ta(ku_above_surf, y, x) ;  ← temperature
#   float u(ku_above_surf, y, x) ;   ← wind_u
```

**Expected**: 2 variables (ta, u)
**If merge failed**: Only 1 variable (whichever ran last)

### Step 4: Check Logs for Merge Messages

```bash
# Check if merge was attempted
grep -i "merge" logs/*.log

# Check for merge failures
grep -i "could not merge" logs/*.log
```

**Expected output**:
```
Found existing cache file, will merge variables
Keeping existing variable 'ta' in cache
Multi-variable cache: 2 total variables
```

**If merge failed**:
```
Could not merge with existing cache: [error message]
```

---

## Safe Production Run Strategy

### Option A: Clean Cache (Safest, but recomputes everything)

**Advantages**:
- Avoids merge issues entirely
- Guarantees all variables processed together
- Clean slate

**Disadvantages**:
- Recomputes all terrain masks (slow for 16 scenarios × 2 domains)
- Loses any previously cached data

```bash
# Backup existing cache (optional)
cp -r cache/terrain_masks cache/terrain_masks_backup_$(date +%Y%m%d)
cp -r cache/surface_data cache/surface_data_backup_$(date +%Y%m%d)

# Clean cache
rm -rf cache/terrain_masks/*
rm -rf cache/surface_data/*

# Run with all variables
python -m palmplot_thf palmplot_config_multivar_test.yaml
```

### Option B: Incremental Merge (Faster, but risky if merge fails)

**Advantages**:
- Keeps existing cached data
- Only computes new variables
- Much faster

**Disadvantages**:
- Relies on merge working correctly
- Risk of losing existing data if merge fails

```bash
# FIRST: Verify merge works (Step 1-4 above)
# ONLY proceed if merge verification passed!

# Run with new variables added
python -m palmplot_thf palmplot_config_multivar_test.yaml

# IMMEDIATELY check logs for merge warnings:
grep -i "could not merge" logs/*.log

# If ANY merge failures, STOP and investigate before continuing
```

**Monitor during run**:
```bash
# In another terminal, watch for merge messages:
tail -f logs/palmplot_*.log | grep -i merge
```

### Option C: Manual Variable Groups (Most Conservative)

Process variables in groups, verifying cache after each group:

**Group 1: Existing variables** (already cached)
```yaml
variables: ["temperature", "utci", "relative_humidity", ...]
```
Run → Verify cache has all variables

**Group 2: Add wind components**
```yaml
variables: ["temperature", ..., "wind_u", "wind_v", "wind_w"]
```
Run → Verify cache merged correctly → Check `ncdump -h`

**Group 3: Add TKE**
```yaml
variables: ["temperature", ..., "wind_u", "wind_v", "wind_w", "tke"]
```
Run → Verify cache merged correctly

**Group 4: Add PCM variables**
```yaml
variables: ["temperature", ..., "tke", "pcm_transpiration", "pcm_heatrate"]
```
Run → Verify final cache

---

## Verification Checklist

After each run, verify:

### ✅ Cache File Counts
```bash
# Should have files for each case/domain combination
ls -lh cache/terrain_masks/
ls -lh cache/surface_data/

# Expected: 32 files (16 scenarios × 2 domains)
```

### ✅ Variable Counts in Each File

**av_3d files should have 13 variables**:
```bash
for f in cache/terrain_masks/*_terrain_mask.nc; do
  echo "=== $f ==="
  ncdump -h "$f" | grep -A 50 "variables:" | grep "float\|double" | wc -l
done
```
Expected: 13 (ta, q, qv, rh, theta, wspeed, wdir, p, ti, u, v, w, e)

**av_xy files should have 12 variables**:
```bash
for f in cache/surface_data/*_surface_data.nc; do
  echo "=== $f ==="
  ncdump -h "$f" | grep -A 50 "variables:" | grep "float\|double" | wc -l
done
```
Expected: 12 (utci, pet, rad_net, rad_sw_in, rad_sw_out, rad_lw_in, rad_lw_out, tsurf, shf, ghf, pcm_transpiration, pcm_heatrate)

### ✅ Log Messages

Check for:
- ✅ No "Could not merge" warnings
- ✅ "Multi-variable cache: N total variables" (N increasing)
- ✅ No exceptions or errors during cache save/load

### ✅ File Sizes

Cache files should grow as variables are added:
```bash
# After adding new variables, files should be larger
ls -lh cache/terrain_masks/ | sort -k5 -h
```

If files DECREASE in size → merge failed, data lost!

---

## Troubleshooting

### Problem: Only last variable in cache

**Symptoms**:
```bash
ncdump -h cache/.../file.nc
# Shows only 1 variable instead of expected N
```

**Cause**: Merge failed, each variable overwrites previous

**Solution**:
1. Check logs: `grep "Could not merge" logs/*.log`
2. Identify error message
3. Fix root cause (validation settings, file permissions, etc.)
4. Re-run with clean cache

### Problem: Merge warnings in log

**Symptoms**:
```
WARNING - Could not merge with existing cache: [error]
```

**Actions**:
1. **STOP immediately** - don't continue processing
2. Identify error from log message
3. Check cache file: `ncdump -h cache/.../file.nc`
4. Verify grid sizes match: `ncdump -h` shows nx, ny
5. Check file permissions
6. If unfixable: Clean cache and restart

### Problem: Different variable counts across files

**Symptoms**:
```bash
# Case 1 has 13 variables
# Case 2 has 8 variables
# Case 3 has 5 variables
```

**Cause**: Merge worked for some cases but failed for others

**Solution**:
1. Identify which cases failed: Check logs for specific case names
2. Delete ONLY failed cases' cache files
3. Re-run to regenerate those specific files

---

## Recommendation

**SAFEST APPROACH**:

1. **Test merge first** with 2 variables, 1 case (10 minutes)
2. If merge works: **Clean cache** and run with all variables (slow but safe)
3. **Verify** variable counts in cache files after completion
4. **Future runs** can use incremental merge (if verified working)

**FASTEST APPROACH** (if confident):

1. **Backup cache** (just in case)
2. **Run with new variables** (merge should work)
3. **Monitor logs** for merge warnings during run
4. **Verify cache** immediately after completion
5. **Rollback from backup** if merge failed

---

## Files Modified

1. `palmplot_config_multivar_test.yaml`:
   - Lines 162-210: Added 4 av_3d variables (wind_u, wind_v, wind_w, tke)
   - Lines 346-371: Added 2 av_xy variables (pcm_transpiration, pcm_heatrate)
   - Line 424: Added new variables to processing list

## Current Configuration

**Total variables**: 23
- **av_3d** (terrain-following): 13 variables
  - temperature, humidity_q, humidity_qv, relative_humidity, potential_temperature
  - wind_speed, wind_direction, pressure, turbulence_intensity
  - wind_u, wind_v, wind_w, tke

- **av_xy** (surface): 10 variables
  - utci, pet, radiation_net, radiation_sw_in, radiation_sw_out
  - radiation_lw_in, radiation_lw_out, surface_temperature
  - sensible_heat_flux, ground_heat_flux, pcm_transpiration, pcm_heatrate

**Scenarios**: 16 (spacings: 10, 15, 20, 25; ages: 20, 40, 60, 80)

**Domains**: 2 (parent, child)

**Expected cache files**: 32 total
- 16 terrain_mask files (av_3d)
- 16 surface_data files (av_xy)

---

**Prepared**: 2025-11-08
**Status**: Ready to test
**Next step**: Run merge verification test before production
