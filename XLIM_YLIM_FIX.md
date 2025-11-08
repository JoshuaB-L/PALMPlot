# X-Axis Extent Matching Fix

**Date**: 2025-11-08
**Issue**: 2D slice x-axis not extending to match line plot above despite setting extent
**Status**: ✅ Fixed

---

## Problem Description

### Observed Behavior

After fixing the aspect ratio distortion, a new issue appeared:
- ✅ 2D slice maintained square aspect ratio (grid cells are square)
- ❌ 2D slice x-axis did NOT extend to match the line plot x-axis
- ❌ Features in line plot were not aligned with features in 2D slice

**Visual symptom**: The 2D slice appeared narrower than the line plot above it, even though both had the same x-axis labels (0-400m).

### Example

**Line plot**: X-axis spans from 0m to 398m (full transect length)
**2D slice**: X-axis SHOULD span 0m to 398m, but was only showing ~200m to ~398m
**Result**: Features didn't align vertically between the two panels

---

## Root Cause Analysis

### The Issue

When `aspect='equal'` is set on an axes, matplotlib enforces square pixels (1:1 data coordinate scaling). However, matplotlib can **adjust the visible axis limits** to maintain this aspect ratio while fitting the axes into the allocated figure space.

**The problem flow**:
1. We set `extent=[0, 398, 160, 242]` in `imshow()` → defines data coordinates
2. We set `aspect='equal'` → enforces square pixels
3. matplotlib tries to fit the axes in the allocated space
4. To maintain square pixels, matplotlib **shrinks the visible x-range**
5. Result: Only part of the x-extent is shown, not the full range

### Why This Happens

With `aspect='equal'`:
- 1 meter in x-direction = 1 meter in y-direction (in screen space)
- Y-extent: 242 - 160 = 82m
- X-extent: 398 - 0 = 398m
- Width/height ratio: 398/82 ≈ 4.85

If the allocated axes space has a different width/height ratio, matplotlib adjusts the visible limits to maintain the aspect ratio.

**Example**:
- Allocated axes space: 800 pixels wide × 200 pixels tall → ratio 4.0
- Required ratio for data: 398/82 = 4.85
- matplotlib shrinks the visible x-range to make the ratio match the available space

---

## The Solution

### Fix: Explicitly Set Axis Limits

After setting `aspect='equal'`, we must **explicitly set the x and y limits** to match the extent. This forces matplotlib to show the full extent regardless of the allocated axes space.

**Code change**:
```python
ax_map.set_aspect('equal')  # Ensure square aspect ratio

# THE FIX: Explicitly set limits to match extent
# Without this, aspect='equal' may shrink the visible range
ax_map.set_xlim(extent_x_min, extent_x_max)
ax_map.set_ylim(extent_y_min, extent_y_max)
```

### Why This Works

- `set_xlim()` and `set_ylim()` **override** matplotlib's automatic limit adjustment
- The axes will now show exactly the specified range
- `aspect='equal'` is still enforced (square pixels)
- matplotlib will adjust the axes **size** (not limits) to accommodate both constraints

**Result**: Full extent is shown with square aspect ratio maintained.

---

## Technical Details

### Before Fix (Broken)

```python
im = ax_map.imshow(
    xy_slice_plot,
    extent=[extent_x_min, extent_x_max, extent_y_min, extent_y_max],
    aspect='auto'
)
ax_map.set_aspect('equal')  # Square pixels
# ❌ No explicit limit setting → matplotlib adjusts limits automatically
```

**Result**:
- matplotlib shrinks x-limits to fit allocated space
- Only partial x-extent shown
- Misalignment with line plot

### After Fix (Correct)

```python
im = ax_map.imshow(
    xy_slice_plot,
    extent=[extent_x_min, extent_x_max, extent_y_min, extent_y_max],
    aspect='auto'
)
ax_map.set_aspect('equal')  # Square pixels
ax_map.set_xlim(extent_x_min, extent_x_max)  # ✓ Force full x-extent
ax_map.set_ylim(extent_y_min, extent_y_max)  # ✓ Force full y-extent
```

**Result**:
- Full extent shown (0 to 398m in x-direction)
- Perfect alignment with line plot
- Square aspect ratio maintained

---

## Verification

### Test Results

Created test script `test_xlim_fix.py` that verifies:
- ✅ X-limits match extent: `(0.0, 398.0)` ← Perfect match!
- ✅ Y-limits match extent: `(160.0, 242.0)` ← Perfect match!
- ✅ Red dotted lines at extent edges align in both panels
- ✅ Grid cells remain square

**Test output**:
```
Actual 2D slice limits:
  - X: (0.0, 398.0)
  - Y: (160.0, 242.0)
  ✓ X-limits match extent perfectly!
```

---

## Impact

### What's Fixed

✅ **X-axis extent matching** - 2D slice now extends to match line plot perfectly
✅ **Feature alignment** - Buildings, trees, and other features align vertically
✅ **Correct spatial reference** - Full transect shown, not cropped
✅ **Square aspect ratio** - Still maintained (from previous fix)

### What's Unchanged

✅ **Colorbar positioning** - Still works correctly
✅ **Transect width cropping** - Still works as configured
✅ **Aspect ratio preservation** - Still enforces square pixels

---

## Files Modified

**plots/terrain_transect.py** (lines 3488-3491):

Added two lines after `set_aspect('equal')`:
```python
ax_map.set_xlim(extent_x_min, extent_x_max)
ax_map.set_ylim(extent_y_min, extent_y_max)
```

**Total changes**: 2 lines added

---

## Related Fixes

This is the **third iteration** of the transect visualization fixes:

1. **Aspect ratio preservation** - Added `aspect='equal'` to prevent distortion
2. **Colorbar positioning** - Manual positioning to avoid compressing axes
3. **Extent matching** (this fix) - Explicit limits to ensure full extent shown

All three work together to provide:
- ✅ Square aspect ratio (correct proportions)
- ✅ Full width extent (matches line plot)
- ✅ Proper colorbar (below x-axis, doesn't compress plot)

---

## Summary

**Root cause**: `aspect='equal'` allows matplotlib to adjust visible limits to fit allocated space

**Solution**: Explicitly set `xlim` and `ylim` to match extent

**Result**: Perfect alignment between line plot and 2D slice with square aspect ratio maintained

**Lines added**: 2 (simple but critical fix)

---

**Status**: ✅ Fixed and tested
**Confidence**: High - verified with test script
**Testing**: Visual and numerical verification complete
**Impact**: Critical - ensures correct feature alignment
