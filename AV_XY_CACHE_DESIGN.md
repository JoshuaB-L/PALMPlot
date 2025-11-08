# AV_XY Surface Variable Caching - Design Document

**Date**: 2025-11-08
**Purpose**: Extend caching system to support av_xy (2D surface) variables
**Author**: PALMPlot Development Team

---

## Overview

This document describes the design for extending the terrain mask caching system to support av_xy (2D surface) variables. The implementation will mirror the existing av_3d caching architecture while accounting for the different data structure of surface variables.

### Goals

1. **Cache multiple av_xy variables** in a single NetCDF file per domain
2. **Separate cache files** for av_xy variables (distinct from av_3d terrain masks)
3. **Seamless integration** with existing code architecture
4. **User configuration** for which variables to cache
5. **Robust I/O** with validation and error handling

---

## Data Structure Comparison

### AV_3D Variables (3D Atmospheric)
```
Structure: [ku_above_surf, y, x]
- Requires terrain-following extraction
- Multiple vertical levels (e.g., 0-20 levels above surface)
- Source level tracking (which zu_3d index was used)
- Complex extraction with bottom-up filling algorithm

Cache file: {case_name}_{domain}_terrain_mask.nc
Variables: ta, rh, q (multiple 3D atmospheric variables)
```

### AV_XY Variables (2D Surface)
```
Structure: [y, x]
- Direct surface extraction (zu1_xy[0] or zu_xy[0])
- Single surface level only
- No terrain-following needed
- Time-averaged from [time, y, x] → [y, x]

Cache file: {case_name}_{domain}_surface_data.nc
Variables: bio_utci_xy, rad_net_xy, rad_lw_in_xy (multiple 2D surface variables)
```

---

## Architecture Design

### 1. New I/O Classes

Create `SurfaceDataWriter` and `SurfaceDataReader` classes in `core/surface_data_io.py`:

```python
class SurfaceDataWriter:
    """
    Writes time-averaged surface data (av_xy variables) to NetCDF files.

    Similar to TerrainMaskWriter but for 2D surface data.
    """

    def write_surface_data(self,
                          output_path: Path,
                          surface_data: Dict[str, np.ndarray],
                          coordinates: Dict[str, np.ndarray],
                          metadata: Dict,
                          compression: Optional[Dict] = None) -> None:
        """
        Write time-averaged surface data to NetCDF file.

        Args:
            output_path: Path to output NetCDF file
            surface_data: Dict of variable_name -> 2D array [y, x]
            coordinates: Dict of coordinate arrays (x, y)
            metadata: Dict of metadata (domain info, time averaging settings, etc.)
            compression: Optional compression settings
        """
        pass

class SurfaceDataReader:
    """
    Reads time-averaged surface data from NetCDF files.

    Similar to TerrainMaskReader but for 2D surface data.
    """

    def read_surface_data(self,
                         input_path: Path,
                         variables: Optional[List[str]] = None,
                         validate: bool = True) -> Dict:
        """
        Read time-averaged surface data from NetCDF file.

        Returns:
            Dictionary containing:
                - 'surface_data': Dict of variable_name -> array [y, x]
                - 'coordinates': Dict of coordinate arrays
                - 'metadata': Dict of metadata
        """
        pass
```

### 2. NetCDF Structure

#### File Naming Convention
```
av_3d cache:  {case_name}_{domain}_terrain_mask_offset{offsets}.nc
av_xy cache:  {case_name}_{domain}_surface_data.nc

Examples:
av_3d:  thf_forest_lad_spacing_10m_age_20yrs_parent_terrain_mask_offset0.nc
av_xy:  thf_forest_lad_spacing_10m_age_20yrs_parent_surface_data.nc
```

#### NetCDF Dimensions
```netcdf
dimensions:
    time = 1 ;           // Single time point (after averaging)
    y = 400 ;            // Grid size y (parent) or 200 (child)
    x = 400 ;            // Grid size x (parent) or 200 (child)
```

#### NetCDF Variables
```netcdf
variables:
    // Data variables - one per av_xy variable
    float bio_utci_xy(time, y, x) ;
        bio_utci_xy:_FillValue = -999999.f ;
        bio_utci_xy:units = "degC" ;
        bio_utci_xy:long_name = "Universal Thermal Climate Index" ;

    float rad_net_xy(time, y, x) ;
        rad_net_xy:_FillValue = -999999.f ;
        rad_net_xy:units = "W/m2" ;
        rad_net_xy:long_name = "Net Radiation" ;

    // Coordinates
    double x(x) ;
        x:units = "meters" ;
        x:axis = "X" ;
    double y(y) ;
        y:units = "meters" ;
        y:axis = "Y" ;
    double time(time) ;
        time:units = "seconds" ;
        time:axis = "T" ;
```

#### Global Attributes
```netcdf
:title = "Time-averaged surface data for {case_name}" ;
:Conventions = "CF-1.7" ;
:creation_time = "2025-11-08 12:00:00" ;
:data_content = "surface_data_time_averaged" ;
:version = 1 ;

// Time averaging info
:time_averaging_method = "mean" ;  // or "median", "specific_time"
:time_steps_used = 48 ;
:time_steps_total = 49 ;
:time_steps_corrupted = 1 ;

// Domain information
:domain_type = "parent" ;  // or "child"
:grid_size_x = 400 ;
:grid_size_y = 400 ;
:resolution = 10.0 ;

// Software information
:source = "PALMPlot surface_data_io module" ;
:author = "PALMPlot" ;
```

---

## Implementation Plan

### Phase 1: Create I/O Module (`surface_data_io.py`)

**Location**: `core/surface_data_io.py`

**Classes**:
1. `SurfaceDataWriter` - Writes NetCDF files with multi-variable 2D surface data
2. `SurfaceDataReader` - Reads NetCDF files with validation
3. Helper functions:
   - `generate_surface_data_filename(case_name, domain_type)`
   - `find_existing_surface_data_file(cache_dir, case_name, domain_type)`
   - `validate_surface_data_file(ds)`

**Features**:
- Multi-variable support (multiple av_xy variables in one file)
- Compression support (zlib with configurable level)
- Validation (grid size, domain type, variable completeness)
- CF-1.7 compliant metadata

### Phase 2: Integrate into `terrain_transect.py`

**New Methods**:
```python
def _get_surface_data_cache_path(self, case_name, domain_type, settings):
    """Get path for surface data cache file."""
    pass

def _save_surface_data(self, case_name, domain_type, surface_data_dict,
                      coordinates, metadata, settings):
    """Save time-averaged surface data to cache."""
    pass

def _load_surface_data(self, case_name, domain_type, required_variables,
                      settings, expected_grid_size):
    """Load time-averaged surface data from cache."""
    pass
```

**Integration Points**:

1. **After XY variable extraction** (around line 1576 in `terrain_transect.py`):
```python
# After time averaging is complete for XY variable
if cache_enabled and cache_mode in ['save', 'update']:
    # Check if this is a cacheable variable
    if variable in cacheable_xy_vars:
        # Merge with existing cache or create new
        self._save_surface_data(...)
```

2. **Before XY variable extraction** (check cache first):
```python
# Before extracting XY variable
if cache_enabled and cache_mode in ['load', 'update']:
    cached_data = self._load_surface_data(...)
    if cached_data and variable in cached_data['surface_data']:
        # Use cached data
        return cached_data['surface_data'][variable], var_name, False
```

### Phase 3: Configuration Schema

**Add to `palmplot_config.yaml`**:

```yaml
plots:
  figures:
    fig_6:
      settings:
        terrain_following:
          # Existing av_3d cache settings
          mask_cache:
            enabled: true
            mode: "save"
            cache_directory: "./cache/terrain_masks"
            parent:
              variables: ["temperature", "relative_humidity"]
            child:
              variables: ["temperature"]

          # NEW: av_xy surface data cache settings
          surface_data_cache:
            enabled: true
            mode: "save"  # Options: "save", "load", "update", "disabled"
            cache_directory: "./cache/surface_data"

            # Compression settings
            compression:
              enabled: true
              level: 4

            # Validation settings
            validation:
              check_grid_size: true
              check_domain_type: true
              max_age_days: 30
              on_mismatch: "recompute"  # Options: "error", "warn", "recompute"

            # Per-domain variable specification
            parent:
              # Which av_xy variables to cache for parent domain
              # Options:
              #   - "auto": Cache all av_xy variables enabled in plots
              #   - List: ["utci", "radiation_net"] for specific variables
              variables: "auto"

            child:
              # Which av_xy variables to cache for child domain
              variables: "auto"
```

### Phase 4: Cache Merging Logic

**Multi-Variable Support** (similar to av_3d):

```python
def _save_surface_data(...):
    # Start with new variable
    surface_data_dict = {var_name: data_2d}

    # Check if cache file already exists
    cache_path = self._get_surface_data_cache_path(case_name, domain_type, settings)

    if cache_path.exists():
        self.logger.info("Found existing surface data cache, will merge variables")
        try:
            # Load existing variables
            existing_cache = self._load_surface_data(...)

            if existing_cache and 'surface_data' in existing_cache:
                # Merge existing variables with new variable
                for existing_var, existing_data in existing_cache['surface_data'].items():
                    if existing_var != var_name:
                        surface_data_dict[existing_var] = existing_data
                        self.logger.info(f"  Keeping existing variable '{existing_var}'")
                    else:
                        self.logger.info(f"  Updating variable '{var_name}'")

                self.logger.info(f"Multi-variable cache: {len(surface_data_dict)} total")
        except Exception as e:
            self.logger.warning(f"Could not merge with existing cache: {e}")

    # Write cache (now with multiple variables)
    writer.write_surface_data(...)
```

---

## Data Flow

### Saving Flow (av_xy)
```
1. Extract av_xy variable (e.g., "utci")
   ↓
2. Variable lookup in metadata: palm_name="bio_utci*_xy", file_type="av_xy"
   ↓
3. Dataset selection: av_xy_n02 (child domain)
   ↓
4. Variable discovery: Wildcard match → "bio_utci_xy" found
   ↓
5. Extract surface level: zu1_xy[0] → [time, y, x]
   ↓
6. Time averaging with corruption detection → [y, x]
   ↓
7. Check cache configuration:
   - surface_data_cache.enabled = true?
   - surface_data_cache.mode in ["save", "update"]?
   - variable in cacheable_xy_vars?
   ↓
8. If YES to all:
   a. Check if cache file exists
   b. If exists: Load existing variables
   c. Merge new variable with existing variables
   d. Write combined cache file
   ↓
9. Log: "✓ Surface data saved to cache: 3 variable(s) - ['bio_utci_xy', 'rad_net_xy', 'rad_lw_in_xy']"
```

### Loading Flow (av_xy)
```
1. Variable requested: "utci"
   ↓
2. Check cache configuration:
   - surface_data_cache.enabled = true?
   - surface_data_cache.mode in ["load", "update"]?
   ↓
3. If YES:
   a. Get cache path for case_name + domain
   b. Check if cache file exists
   c. If exists: Load cache file
   d. Validate cache (grid size, domain type)
   e. Check if requested variable in cache
   ↓
4. If variable found in cache:
   - Return cached data [y, x]
   - Skip extraction completely
   ↓
5. If NOT found or cache invalid:
   - Perform normal extraction
   - Optionally save to cache (if mode="update")
```

---

## Error Handling

### Cache File Issues

**Grid Size Mismatch**:
```python
if cached_nx != expected_nx or cached_ny != expected_ny:
    if on_mismatch == 'error':
        raise ValueError("Grid size mismatch")
    elif on_mismatch == 'warn':
        logger.warning("Grid size mismatch, using anyway")
        return cached_data
    else:  # recompute
        logger.warning("Grid size mismatch, will recompute")
        return None
```

**Domain Type Mismatch**:
```python
if cached_domain != expected_domain:
    logger.warning(f"Domain mismatch: {cached_domain} != {expected_domain}")
    return None
```

**Missing Variables**:
```python
if variable not in cached_data['surface_data']:
    logger.info(f"Variable '{variable}' not in cache, will extract")
    return None
```

**Corrupted Cache File**:
```python
try:
    cached_data = reader.read_surface_data(cache_path)
except Exception as e:
    logger.error(f"Failed to read cache: {e}")
    if on_mismatch == 'error':
        raise
    else:
        logger.warning("Will recompute due to cache error")
        return None
```

---

## Testing Strategy

### Unit Tests

1. **Test `SurfaceDataWriter`**:
   - Single variable write
   - Multi-variable write
   - Compression enabled/disabled
   - Metadata completeness

2. **Test `SurfaceDataReader`**:
   - Variable loading (all vs specific)
   - Validation (grid size, domain)
   - Error handling (missing file, corrupted data)

3. **Test Cache Merging**:
   - Add variable to existing cache
   - Update existing variable
   - Multiple sequential adds

### Integration Tests

1. **Full Workflow**:
   ```bash
   # Test 1: Save mode
   - Configure: mode="save", variables=["utci", "radiation_net"]
   - Run: Extract 2 variables
   - Verify: Both variables in cache file

   # Test 2: Load mode
   - Configure: mode="load"
   - Run: Request cached variable
   - Verify: Data loaded from cache (no extraction)

   # Test 3: Update mode
   - Configure: mode="update"
   - Run: Request 3rd variable (not in cache)
   - Verify: New variable added to existing cache
   ```

2. **Error Scenarios**:
   - Cache file deleted between runs
   - Grid size change
   - Domain change
   - Corrupted cache file

---

## Performance Considerations

### Cache Benefits (av_xy)

**Time Savings**:
- Extraction + time averaging: ~5-10 seconds per variable
- Cache load: ~0.5 seconds per variable
- **Speedup**: ~10-20x for cached variables

**Disk Usage**:
- Single av_xy variable (400×400): ~640 KB uncompressed
- With compression (level 4): ~100-200 KB
- 5 variables: ~0.5-1 MB per domain per scenario

**Example** (9 plots with 3 av_xy variables each):
- Without cache: 27 extractions × 7 sec = ~3 minutes
- With cache: 3 initial + 24 loads × 0.5 sec = ~30 seconds
- **Total speedup**: ~6x

---

## Backward Compatibility

### Existing Code
- No changes to av_3d caching
- av_xy caching is **optional** (disabled by default)
- If av_xy caching disabled, behavior unchanged

### Config
- Old configs without `surface_data_cache` section: av_xy caching disabled
- New configs: av_xy caching explicitly controlled

---

## Implementation Checklist

### Phase 1: I/O Module
- [ ] Create `core/surface_data_io.py`
- [ ] Implement `SurfaceDataWriter` class
- [ ] Implement `SurfaceDataReader` class
- [ ] Implement helper functions
- [ ] Add unit tests

### Phase 2: Integration
- [ ] Add `_get_surface_data_cache_path()` to `terrain_transect.py`
- [ ] Add `_save_surface_data()` to `terrain_transect.py`
- [ ] Add `_load_surface_data()` to `terrain_transect.py`
- [ ] Integrate save logic after XY extraction
- [ ] Integrate load logic before XY extraction
- [ ] Add cache merging logic

### Phase 3: Configuration
- [ ] Update config schema in `core/config_handler.py`
- [ ] Add `surface_data_cache` section to schema
- [ ] Update example config files
- [ ] Add config validation tests

### Phase 4: Testing
- [ ] Test single variable caching
- [ ] Test multi-variable caching
- [ ] Test cache merging
- [ ] Test cache loading
- [ ] Test error handling
- [ ] Performance benchmarks

### Phase 5: Documentation
- [ ] Update `CLAUDE.md` with av_xy caching info
- [ ] Create user guide for cache configuration
- [ ] Add inline code comments
- [ ] Update README if needed

---

## Success Criteria

✅ **Multiple av_xy variables** can be cached in single NetCDF file
✅ **Separate cache files** for av_xy (distinct from av_3d)
✅ **Per-domain configuration** (parent/child variables specified independently)
✅ **Cache merging** works (add variables incrementally)
✅ **Validation** catches incompatible caches
✅ **Error handling** gracefully falls back to extraction
✅ **Performance** improves (10-20x speedup for cached variables)
✅ **Backward compatible** (old configs still work)
✅ **Code quality** matches existing av_3d implementation

---

**Status**: Design complete, ready for implementation
**Estimated Time**: ~8 hours total (2h I/O + 3h integration + 2h config + 1h testing)
