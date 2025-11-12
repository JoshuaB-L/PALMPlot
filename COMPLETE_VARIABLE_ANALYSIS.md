# Complete PALM Variable Analysis

**Date**: 2025-11-08
**Source Files**:
- av_3d_N02_merged.nc (36 variables total)
- av_xy_N02_merged.nc (19 variables total)

---

## Variables Currently in Config

### av_3d (13 variables) ✓
- temperature (ta)
- humidity_q (q)
- humidity_qv (qv)
- relative_humidity (rh)
- potential_temperature (theta)
- wind_speed (wspeed)
- wind_direction (wdir)
- pressure (p)
- turbulence_intensity (ti)
- wind_u (u)
- wind_v (v)
- wind_w (w)
- tke (e)

### av_xy (10 variables) ✓
- utci (bio_utci*_xy)
- pet (bio_pet*_xy)
- radiation_net (rad_net*_xy)
- radiation_sw_in (rad_sw_in*_xy)
- radiation_sw_out (rad_sw_out*_xy)
- radiation_lw_in (rad_lw_in*_xy)
- radiation_lw_out (rad_lw_out*_xy)
- surface_temperature (tsurf*_xy)
- sensible_heat_flux (shf*_xy)
- ground_heat_flux (ghf*_xy)

### ❌ INCORRECTLY DEFINED
- pcm_transpiration - Defined as av_xy, but NO such variable in av_xy file!
- pcm_heatrate - Defined as av_xy, but actually in av_3d with zpc_3d coordinate!

---

## Missing Variables - av_3d (15 variables)

### Soil Variables (zs_3d coordinate - 8 levels)

**m_soil** - Soil Moisture
- PALM name: m_soil
- z_coordinate: zs_3d (8 soil layers)
- Units: m³/m³
- terrain_following: false (below ground)

**t_soil** - Soil Temperature
- PALM name: t_soil
- z_coordinate: zs_3d
- Units: K → degC
- terrain_following: false (below ground)

### Plant Canopy Model Variables (zpc_3d coordinate - 15 levels)

**pcm_transpirationrate** - Transpiration Rate
- PALM name: pcm_transpirationrate
- z_coordinate: zpc_3d (15 canopy layers)
- Units: kg/kg/s
- terrain_following: true (canopy follows terrain)

**pcm_transpirationvolume** - Transpiration Volume
- PALM name: pcm_transpirationvolume
- z_coordinate: zpc_3d
- Units: m³/m³/s
- terrain_following: true

**pcm_transpirationvolume_pot** - Potential Transpiration Volume
- PALM name: pcm_transpirationvolume_pot
- z_coordinate: zpc_3d
- Units: m³/m³/s
- terrain_following: true

**pcm_heatrate** - Plant Canopy Heat Rate (FIX!)
- PALM name: pcm_heatrate
- z_coordinate: zpc_3d (NOT zu1_xy!)
- Units: K/s
- terrain_following: true
- **Currently wrongly defined as av_xy**

**pcm_latentrate** - Plant Canopy Latent Heat Rate
- PALM name: pcm_latentrate
- z_coordinate: zpc_3d
- Units: K/s
- terrain_following: true

### Radiation Transfer Model Variables (zu_3d coordinate - 52 levels)

**rtm_rad_pc_insw** - Absorbed Shortwave Radiation by Plant Canopy
- PALM name: rtm_rad_pc_insw
- z_coordinate: zu_3d
- Units: W/m³
- terrain_following: true

**rtm_rad_pc_inlw** - Absorbed Longwave Radiation by Plant Canopy
- PALM name: rtm_rad_pc_inlw
- z_coordinate: zu_3d
- Units: W/m³
- terrain_following: true

**rtm_rad_pc_inswdir** - Direct Shortwave Radiation
- PALM name: rtm_rad_pc_inswdir
- z_coordinate: zu_3d
- Units: W/m³
- terrain_following: true

**rtm_rad_pc_inswdif** - Diffuse Shortwave Radiation
- PALM name: rtm_rad_pc_inswdif
- z_coordinate: zu_3d
- Units: W/m³
- terrain_following: true

**rtm_rad_pc_inswref** - Reflected Shortwave Radiation
- PALM name: rtm_rad_pc_inswref
- z_coordinate: zu_3d
- Units: W/m³
- terrain_following: true

**rtm_mrt** - Mean Radiant Temperature
- PALM name: rtm_mrt
- z_coordinate: zu_3d
- Units: K → degC
- terrain_following: true

**rtm_mrt_lw** - Longwave Component of Mean Radiant Temperature
- PALM name: rtm_mrt_lw
- z_coordinate: zu_3d
- Units: W/m²
- terrain_following: true

**rtm_mrt_sw** - Shortwave Component of Mean Radiant Temperature
- PALM name: rtm_mrt_sw
- z_coordinate: zu_3d
- Units: W/m²
- terrain_following: true

---

## Missing Variables - av_xy (9 variables)

All use zu1_xy coordinate, wildcard=true, terrain_following=false

**wspeed_10m** - Wind Speed at 10m
- PALM name: wspeed_10m*_xy
- Units: m/s

**theta_2m** - Potential Temperature at 2m
- PALM name: theta_2m*_xy
- Units: K → degC

**us** - Friction Velocity
- PALM name: us*_xy
- Units: m/s

**z0** - Roughness Length
- PALM name: z0*_xy
- Units: m

**ol** - Obukhov Length
- PALM name: ol*_xy
- Units: m

**m_liq** - Liquid Water Content
- PALM name: m_liq*_xy
- Units: m

**r_s** - Surface Resistance
- PALM name: r_s*_xy
- Units: s/m

**r_a** - Aerodynamic Resistance
- PALM name: r_a*_xy
- Units: s/m

**qsws** - Surface Moisture Flux
- PALM name: qsws*_xy
- Units: W/m²

---

## Summary

**Currently supported**: 23 variables
**Missing**: 24 variables
**Incorrectly defined**: 2 variables

**After fixes**: 47 total variables
- av_3d: 28 variables (13 current + 15 missing)
- av_xy: 19 variables (10 current - 2 wrong + 9 missing + 2 fixed)

---

## Code Support Analysis

### ✅ Already Supported

The code ALREADY supports all these variables! The `VariableMetadata` class is fully generalized:

1. **Multiple z-coordinates**: Config specifies `z_coordinate` per variable
   - zu_3d (atmospheric)
   - zw_3d (vertical wind)
   - zs_3d (soil)
   - zpc_3d (plant canopy)
   - zu1_xy (surface)

2. **Wildcard expansion**: Handles `*` in variable names (e.g., `bio_utci*_xy`)

3. **Unit conversions**: Supports all needed conversions:
   - kelvin_to_celsius ✓
   - multiply_1000 ✓
   - divide_100 ✓
   - none ✓

### ⚠️ Potential Code Issue: Different z-coordinates

The terrain-following extraction code might assume `zu_3d` for all 3D variables. Need to verify:
- Does it handle `zs_3d` (8 levels)?
- Does it handle `zpc_3d` (15 levels)?

**Location to check**: `plots/terrain_transect.py` - `_extract_terrain_following()` method

If code hardcodes `zu_3d`, will need to update to use `var_metadata.get_z_coordinate(variable)`

---

## Implementation Plan

### Step 1: Fix Incorrect Variables
- Remove pcm_transpiration from av_xy (doesn't exist)
- Remove pcm_heatrate from av_xy (it's actually av_3d)

### Step 2: Add Soil Variables (av_3d)
- m_soil (zs_3d)
- t_soil (zs_3d)

### Step 3: Add PCM Variables (av_3d)
- pcm_transpirationrate (zpc_3d)
- pcm_transpirationvolume (zpc_3d)
- pcm_transpirationvolume_pot (zpc_3d)
- pcm_heatrate (zpc_3d) - moved from av_xy
- pcm_latentrate (zpc_3d)

### Step 4: Add RTM Variables (av_3d)
- 8 radiation and MRT variables (zu_3d)

### Step 5: Add Surface Variables (av_xy)
- 9 missing surface variables (zu1_xy)

### Step 6: Verify Code Handles All z-coordinates
- Check terrain-following extraction
- Check surface extraction
- Test with all coordinate types

---

**Analysis complete**: Ready to implement
**Total additions needed**: 24 variables + 2 fixes = 26 changes
