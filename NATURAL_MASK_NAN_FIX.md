# Natural Mask NaN Handling Fix - November 17, 2025

## Problem

After fixing the building mask inversion, natural_mask mode produced **all NaN values** resulting in blank plots:

```
Max building height: nanm  <-- NaN in building data!
Level  0: filled 0 cells | Total: 0 (0.0%)
Level  5: filled 0 cells | Total: 0 (0.0%)
...
Final coverage: 0/40000 cells (0.0%) filled, 40000 remaining NaN
```

**Error**: `ValueError: Terrain-following extraction produced all NaN`

## Root Cause

**File**: `plots/terrain_transect.py:2917`

The fix for the inversion bug used:
```python
# BROKEN - fails when building_heights_2d contains NaN
not_building = building_heights_2d == 0
```

### Why This Failed

When `building_heights_2d` contains **NaN values**:
- `NaN == 0` **always returns False** (NaN comparisons are always False in NumPy)
- So `not_building = False` for ALL cells (both buildings AND open areas)
- `fillable_mask = ... & False` → Nothing is filled
- Result: All 40,000 cells remain NaN

### Why Does building_heights_2d Contain NaN?

PALM static files may have NaN values in `buildings_2d` for:
- Uninitialized regions
- Domain boundaries
- Cells outside the simulation area

## The Fix

**File**: `plots/terrain_transect.py:2916`

Changed from equality check to proper inequality:

```python
# CORRECT - handles NaN properly
not_building = ~(building_heights_2d > 0)
```

### Why This Works

**Boolean logic with NaN handling**:

| building_heights_2d | Value > 0 | ~(Value > 0) | Result |
|---------------------|-----------|--------------|---------|
| 15 (building)       | True      | False        | Masked ✓ |
| 0 (open area)       | False     | True         | Allowed ✓ |
| NaN (undefined)     | False     | True         | Allowed ✓ |
| -1 (invalid)        | False     | True         | Allowed ✓ |

**Key insight**: `NaN > 0` returns **False** (not True!), so:
- `~False = True` → NaN cells are allowed (not masked)
- This treats NaN as "no building", which is correct

## Implementation

**File**: `plots/terrain_transect.py`

**Lines 2905-2922** (final corrected version):
```python
elif buildings_mask and buildings_mask_mode == 'natural_mask' and building_heights_2d is not None:
    # NATURAL MODE: Mask cells where buildings exist
    # buildings_2d contains:
    # - 0 where there are NO buildings (open areas)
    # - >0 where there ARE buildings (building footprint)
    # - NaN where undefined (treat as open areas)

    # Mask only where buildings exist (buildings_2d > 0)
    # Allow where: no buildings (==0), NaN, or <=0
    # Use ~(buildings_2d > 0) to handle NaN correctly
    # (NaN > 0 returns False, so ~False = True, allowing NaN cells)
    not_building = ~(building_heights_2d > 0)

    # Also respect fill values (primary masking)
    # The combination ensures:
    # - Fill values always masked (has_valid_data check)
    # - Building locations masked (buildings_2d > 0)
    # - Open areas and NaN cells allowed
```

## Comparison of Approaches

### Approach 1: Equality Check (BROKEN)
```python
not_building = building_heights_2d == 0
```
- **Building (15m)**: `15 == 0` → False → masked ✓
- **Open (0m)**: `0 == 0` → True → allowed ✓
- **NaN**: `NaN == 0` → **False** → masked ✗ WRONG!

### Approach 2: Inequality with Negation (CORRECT)
```python
not_building = ~(building_heights_2d > 0)
```
- **Building (15m)**: `~(15 > 0)` → ~True → False → masked ✓
- **Open (0m)**: `~(0 > 0)` → ~False → True → allowed ✓
- **NaN**: `~(NaN > 0)` → ~False → True → allowed ✓

### Approach 3: Alternative with np.isnan (Also Works)
```python
not_building = (building_heights_2d == 0) | np.isnan(building_heights_2d)
```
- **Building (15m)**: `(False | False)` → False → masked ✓
- **Open (0m)**: `(True | False)` → True → allowed ✓
- **NaN**: `(False | True)` → True → allowed ✓

*Approach 2 is simpler and more elegant*

## Testing

After fix:
```bash
python -m palmplot_thf palmplot_config_fig3_test.yaml
```

**Expected output**:
```
Level 21 (39.00m): filled 33333 cells | Total: 33333 (83.3%)
Final coverage: 33333/40000 cells (83.3%) filled
```

Buildings (16.7%) properly masked, atmosphere (83.3%) filled with data.

## Expected Visual Results

With `buildings_mask_mode: 'natural_mask'`:
- ✅ **Buildings**: White/NaN (masked)
- ✅ **Atmosphere**: Colored temperature values (filled)
- ✅ **NaN regions** (if any): Treated as open areas (filled if valid data exists)

## Key Lessons

### NumPy NaN Comparison Behavior
```python
np.nan == 0      # False (not True!)
np.nan != 0      # True
np.nan > 0       # False
np.nan < 0       # False
np.nan == np.nan # False (NaN never equals itself!)
```

**Best practice**: Use `np.isnan()` for explicit NaN checks, or use inequality comparisons that naturally handle NaN as False.

### Why ~(x > 0) Works
- Captures "positive values" (buildings)
- Automatically handles NaN, 0, and negative values as "not positive"
- Single, simple expression

## Backward Compatibility

- ✅ Fully backward compatible
- ✅ Handles both clean data (no NaN) and NaN-containing data
- ✅ Static_mask mode unchanged
- ✅ No configuration changes required

---

**Implementation Status**: ✅ Complete
**Date**: November 17, 2025
**Bug**: All NaN output due to improper NaN comparison
**Fix**: Changed `building_heights_2d == 0` to `~(building_heights_2d > 0)`
**Files Modified**: `plots/terrain_transect.py:2916`
**Related**: Fix for building mask inversion (earlier today)
