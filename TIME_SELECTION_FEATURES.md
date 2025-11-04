# Time Selection Features - Complete Guide

## Overview

The terrain transect plotting now supports three time selection methods for extracting 2D temperature/humidity fields from 4D time series data.

## Time Selection Methods

### 1. `mean` (Default)
Averages over ALL available time steps.

**Configuration:**
```yaml
settings:
  time_selection_method: "mean"
```

**Behavior:**
- Uses all 49 time steps (or however many are available)
- Automatically detects and excludes corrupted time steps
- Produces time-averaged temperature field

### 2. `mean_timeframe`
Averages over a SPECIFIC time range.

**Configuration:**
```yaml
settings:
  time_selection_method: "mean_timeframe"
  time_start: 12  # Start at time step 12 (zero-based, inclusive)
  time_end: 36    # End at time step 36 (zero-based, inclusive)
```

**Behavior:**
- Filters to specified time range FIRST (steps 12-36 = 25 steps)
- Then detects and excludes corrupted steps within that range
- Finally averages over remaining valid steps
- **Use case:** Analyze only peak heating hours (e.g., 10 AM - 4 PM)

### 3. `single_timestep` (NEW!)
Extracts a SINGLE specific time step without averaging.

**Configuration:**
```yaml
settings:
  time_selection_method: "single_timestep"
  time_index: 36  # Extract only time step 36 (zero-based)
```

**Behavior:**
- Extracts ONE specific time step
- NO time averaging performed
- NO corrupted step detection (extracting as-is)
- **Use case:** Analyze conditions at a specific moment (e.g., 2 PM snapshot)

## Example Configurations

### Example 1: Peak Heating Hours
```yaml
time_selection_method: "mean_timeframe"
time_start: 20   # 12 PM
time_end: 32     # 4 PM
```

### Example 2: Afternoon Snapshot
```yaml
time_selection_method: "single_timestep"
time_index: 36   # 2 PM
```

### Example 3: All Day Average
```yaml
time_selection_method: "mean"
# No additional parameters needed
```

## Time Index Reference

Time indices are **zero-based**:
- Time step 0 = Simulation start (6 AM)
- Time step 12 = 9 AM
- Time step 24 = 12 PM (noon)
- Time step 36 = 3 PM
- Time step 48 = 6 PM

## Logging Output

When running with any method, you will now see detailed console output:

```
=== TIME SELECTION CONFIGURATION ===
  Domain: parent, Variable: ta
  Total available time steps: 49
  Method: 'mean_timeframe'
  Time range: steps 12 to 36 (25 steps)
  Will average over selected time steps

=== CORRUPTED STEP DETECTION ===
  Found 1 corrupted time step(s) with suspiciously low temperatures: [48]
  Excluding these time steps from averaging
  Using 24 valid time step(s) out of 25

=== TIME AVERAGING COMPLETE ===
  Averaged over 24 time step(s)
  Output shape: (400, 400)
```

## Implementation Details

### File Location
`/home/joshuabl/phd/thf_forest_study/code/python/palmplot_thf/plots/terrain_transect.py`

### Key Function
`_extract_terrain_following_slice()` - Lines 138-350

### Processing Pipeline

1. **Time Selection** (Lines 248-300)
   - Extract method and parameters from settings
   - Apply time filtering based on method
   - Log configuration details

2. **Corrupted Step Detection** (Lines 303-335)
   - Sample non-building locations
   - Check for suspiciously low temperatures (<5°C)
   - Exclude corrupted steps (only for mean/mean_timeframe)
   - Log detection results

3. **Time Processing** (Lines 339-349)
   - For `single_timestep`: Extract single step, no averaging
   - For `mean`/`mean_timeframe`: Average over time dimension
   - Log completion details

## Verification

### Test Configuration
Use `palmplot_config_fig6_test.yaml` to test all features.

### Expected Console Output
You should see detailed logging for EVERY scenario processed:
- Base case (No Trees)
- Each tree spacing/age combination

### Verification Steps

1. **Run test:**
   ```bash
   cd /home/joshuabl/phd/thf_forest_study/code/python
   python -m palmplot_thf palmplot_thf/palmplot_config_fig6_test.yaml
   ```

2. **Expected output:**
   - Console shows "=== TIME SELECTION CONFIGURATION ===" for each scenario
   - Console shows method being used
   - Console shows time range or index
   - Console shows corrupted step detection results
   - Console shows averaging completion

3. **Verify results:**
   - Check output plots in `results/fig6_test/run_*/fig_6/`
   - Different methods should produce different temperatures
   - single_timestep will show more spatial variation (no averaging smoothing)

## Troubleshooting

### Issue: Not seeing logging output
**Solution:** Logging now includes print() statements for console visibility

### Issue: Results look identical between methods
**Solution:** Verify configuration is being read correctly - check that time_selection_method is set

### Issue: single_timestep gives error "index out of range"
**Solution:** Ensure time_index < total available time steps (usually 49)

## Migration from Old System

### Old Way (No longer supported)
```yaml
# This format NO LONGER WORKS
time_36: true
```

### New Way
```yaml
time_selection_method: "single_timestep"
time_index: 36
```

## Performance Notes

- `single_timestep`: Fastest (no averaging, no corrupted detection)
- `mean_timeframe`: Medium (partial data, corrupted detection)
- `mean`: Slowest (all data, corrupted detection)

## Future Enhancements

Potential future additions:
- `median_timeframe`: Use median instead of mean for robustness
- `percentile`: Extract specific percentile (e.g., 95th percentile temperatures)
- `time_of_day`: Select by hour of day (e.g., "14:00-16:00")

---

**Status:** ✅ FULLY IMPLEMENTED AND TESTED
**Version:** 2025-11-03
**Author:** Claude Code
