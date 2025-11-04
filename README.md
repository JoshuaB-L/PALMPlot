# PALMPlot - PALM-LES Visualization Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for visualizing PALM-LES (Parallelized Large-Eddy Simulation Model) simulation data, specifically analyzing urban microclimate effects of tree scenarios at Berlin's Tempelhof Airport (THF).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Processing](#data-processing)
- [Plotting Modules](#plotting-modules)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

PALMPlot processes NetCDF output files from PALM simulations and generates publication-quality visualization plots comparing urban forest scenarios. The package was developed to analyze the cooling effects of different tree ages (20, 40, 60, 80 years) and spacings (10, 15, 20, 25m) on urban microclimate.

### Study Area

- **Location**: Tempelhofer Feld (THF), Berlin, Germany
- **Tree Species**: Tilia (Linden trees)
- **Domains**:
  - Parent domain: 400×400m at 10m resolution
  - Child domain (N02): 200×200m at 2m resolution (nested in center of parent)

### Simulation Details

- **Model**: PALM v21.10
- **Period**: 2018-08-07 (summer day)
- **Scenarios**: 16 tree configurations + 1 base case
- **Output**: Time-averaged 3D temperature and humidity fields

## Features

### Core Capabilities

- ✅ **Multi-domain support**: Handles both parent (10m) and child (2m) resolution domains
- ✅ **Flexible data extraction**: Two methods for extracting transect data
  - `slice_2d`: Extract full 2D spatial slice with map visualization
  - `transect_direct`: Memory-efficient 1D extraction (~400× less memory)
- ✅ **Advanced time selection**: Three methods for temporal analysis
  - `mean`: Average over all time steps
  - `mean_timeframe`: Average over specific time range
  - `single_timestep`: Extract single snapshot
- ✅ **Automatic quality control**: Detects and excludes corrupted time steps
- ✅ **Publication-quality output**: High-DPI plots in PNG, PDF, SVG formats
- ✅ **Comprehensive logging**: Detailed console and file logging

### Visualization Types

1. **Tree Density Scenarios** (fig_1): Grid visualization of tree arrangements
2. **Temperature Dynamics** (fig_2): Time series and diurnal cycles
3. **Spatial Cooling Patterns** (fig_3): Horizontal temperature maps
4. **Vertical Profiles** (fig_4): Cross-sections and height profiles
5. **Cooling Relationships** (fig_5): Age-density-cooling surface plots
6. **Terrain Transects** (fig_6): Transect analysis above urban terrain

## Installation

### Prerequisites

```bash
# System requirements
Python >= 3.8
Git
```

### Dependencies

```bash
# Core dependencies
numpy >= 1.20.0
xarray >= 0.19.0
matplotlib >= 3.4.0
pandas >= 1.3.0
scipy >= 1.7.0
pyyaml >= 5.4.0
schema >= 0.7.4
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/JoshuaB-L/PALMPlot.git
cd PALMPlot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Run with configuration file
python -m palmplot_thf palmplot_config.yaml

# Validate configuration only
python -m palmplot_thf palmplot_config.yaml --validate-only

# List available plots
python -m palmplot_thf palmplot_config.yaml --list-plots
```

### Example Configuration

Create `palmplot_config.yaml`:

```yaml
# General Settings
general:
  project_name: "THF Forest Study"
  parallel_processing: false

# Data Paths
data:
  simulation_base_path: "/path/to/simulations"
  spacings: [10, 15, 20, 25]
  ages: [20, 40, 60, 80]

  domains:
    parent:
      resolution: 10.0
      grid_size: [400, 400]
    child:
      resolution: 2.0
      grid_size: [200, 200]

# Output Settings
output:
  base_directory: "./results"
  formats: ["png", "pdf"]
  dpi: 300

# Plot Configuration
plots:
  figures:
    fig_6:
      enabled: true
      plot_types:
        ta_parent_age: true
        ta_child_age: true
      settings:
        extraction_method: "slice_2d"  # or "transect_direct"
        time_selection_method: "mean"   # or "single_timestep", "mean_timeframe"
        transect_axis: "x"
        transect_location: 100
```

## Configuration

### Configuration Structure

```yaml
general:           # Project metadata and execution settings
data:             # Input data paths and domain specifications
output:           # Output directory, formats, and file naming
plots:            # Plot types and visualization settings
analysis:         # Analysis parameters and thresholds
logging:          # Logging configuration
performance:      # Performance tuning options
validation:       # Data validation settings
```

### Key Configuration Options

#### Extraction Method

Choose between two data extraction approaches:

**`slice_2d` (default)**:
- Extracts full 2D spatial slice
- Creates two-panel plot (transect + map)
- Higher memory usage (~31 MB per scenario)
- Provides spatial context visualization

**`transect_direct`**:
- Extracts 1D transect directly from 4D data
- Creates single-panel plot (transect only)
- ~400× more memory efficient (~78 KB per scenario)
- Faster processing

```yaml
settings:
  extraction_method: "slice_2d"  # or "transect_direct"
```

#### Time Selection Method

Control temporal averaging:

**`mean`**: Average over all time steps (default)

```yaml
time_selection_method: "mean"
```

**`mean_timeframe`**: Average over specific time range

```yaml
time_selection_method: "mean_timeframe"
time_start: 12  # Zero-based index
time_end: 36
```

**`single_timestep`**: Extract single snapshot

```yaml
time_selection_method: "single_timestep"
time_index: 36
```

#### Height Extraction

The package extracts data at near-surface levels that balance data availability with capturing cooling effects:

- **Parent domain**: zu_3d[2] = 15m (35% coverage, shows open areas)
- **Child domain**: zu_3d[11] = 21m (68% coverage, above terrain+buildings)

These heights are optimized for urban areas with tall buildings (up to 40m).

```yaml
settings:
  terrain_mask_height_z: 0  # Offset from base extraction level
```

## Data Processing

### Data Flow Pipeline

1. **Configuration Loading**: YAML config validated via schema
2. **Data Loading**: NetCDF files loaded with xarray
   - Base case: `thf_base_2018080700`
   - Tree scenarios: 16 combinations (4 spacings × 4 ages)
3. **Quality Control**: Corrupted time steps detected and excluded
4. **Time Processing**: Apply selected time averaging method
5. **Height Extraction**: Extract at specified vertical level
6. **Spatial Processing**: Extract transect or 2D slice
7. **Visualization**: Generate publication-quality plots
8. **Output**: Save in multiple formats with organized directory structure

### Output Directory Structure

```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── fig_1/
    │   ├── fig_1a_plot_type.png
    │   └── fig_1a_plot_type.pdf
    ├── fig_2/
    │   ├── fig_2a_time_series.png
    │   └── fig_2b_diurnal_cycle.png
    └── fig_6/
        ├── fig_6a_ta_parent_age.png
        └── fig_6b_ta_child_age.png
```

## Plotting Modules

### BasePlotter

Abstract base class providing:
- Matplotlib configuration and styling
- Font fallback handling
- Common color schemes
- Utility functions for all plotters

### TerrainTransectPlotter (fig_6)

Generates terrain-following transect plots showing temperature variations along transect lines.

**Features**:
- Dual extraction methods (slice_2d / transect_direct)
- Three time selection methods
- Automatic corrupted step detection
- Building and LAD shading
- Adaptive plot layout

**Plot Types**:
- `ta_parent_age`: Parent domain temperature, varying ages
- `ta_parent_spacing`: Parent domain temperature, varying spacings
- `ta_child_age`: Child domain temperature, varying ages
- `ta_child_spacing`: Child domain temperature, varying spacings
- `qv_*`: Water vapor mixing ratio variants

**Example Output**:
- Upper panel: Temperature transect with scenario comparison
- Lower panel: 2D temperature map with transect location (if using `slice_2d`)

### Other Plotters

- **TreeDensityPlotter** (fig_1): Tree arrangement visualizations
- **TemperatureDynamicsPlotter** (fig_2): Temporal analysis
- **SpatialCoolingPlotter** (fig_3): Horizontal temperature fields
- **VerticalProfilePlotter** (fig_4): Vertical cross-sections
- **CoolingRelationshipPlotter** (fig_5): Statistical relationships

## Advanced Features

### Memory-Efficient Processing

For large datasets or limited memory:

```yaml
settings:
  extraction_method: "transect_direct"
performance:
  multiprocessing: false
  chunk_size: 100
```

### Custom Time Selection

Analyze specific periods:

```yaml
# Peak heating hours (12 PM - 4 PM)
time_selection_method: "mean_timeframe"
time_start: 20
time_end: 32

# Afternoon snapshot
time_selection_method: "single_timestep"
time_index: 36
```

### Parallel Processing

Process multiple scenarios simultaneously:

```yaml
general:
  parallel_processing: true
  n_workers: 4
```

### Publication Settings

High-quality output for papers:

```yaml
output:
  formats: ["png", "pdf", "svg"]
  dpi: 600
  transparent_background: true

plots:
  global_settings:
    publication_quality:
      enabled: true
      tight_layout: true
      high_contrast: true
```

## Troubleshooting

### Common Issues

#### 1. "All NaN" Errors

**Problem**: Extracted slice contains only NaN values

**Cause**: Extraction height is inside buildings or below terrain

**Solution**: Current heights (15m parent, 21m child) are optimized. If issues persist, increase `terrain_mask_height_z` offset.

#### 2. Minimal Temperature Differences

**Problem**: Tree scenarios show <0.5°C differences

**Cause**: Extracting at too high altitude

**Solution**: Verify extraction heights in console logs. Should see ~15m (parent) or ~21m (child), not 360m or 41m.

#### 3. Memory Errors

**Problem**: Out of memory errors during processing

**Solution**: Switch to direct extraction method:
```yaml
extraction_method: "transect_direct"
```

#### 4. Missing Scenarios

**Problem**: Warning about missing scenarios

**Solution**: Ensure `data.spacings` and `data.ages` in config match the scenarios used in plot settings.

### Debug Logging

Enable detailed logging:

```yaml
logging:
  level: "DEBUG"
  console_output: true
  log_file: "./logs/palmplot_debug.log"
```

Check logs for:
- `Height: X.XXm (zu_3d[Y])` - Verify correct extraction height
- `TIME SELECTION CONFIGURATION` - Confirm time averaging method
- `Data stats: min=X, max=Y` - Check temperature ranges

### Validation

Test configuration before running:

```bash
python -m palmplot_thf config.yaml --validate-only
```

## Project Structure

```
palmplot_thf/
├── __init__.py
├── __main__.py           # Entry point
├── core/
│   ├── config_handler.py # Configuration validation
│   ├── data_loader.py    # NetCDF data loading
│   └── ...
├── plots/
│   ├── base_plotter.py   # Abstract base class
│   ├── terrain_transect.py  # Transect plotting (fig_6)
│   ├── tree_density.py   # Tree scenarios (fig_1)
│   ├── temperature_dynamics.py  # Time series (fig_2)
│   ├── spatial_cooling.py    # Spatial patterns (fig_3)
│   ├── vertical_profile.py   # Vertical analysis (fig_4)
│   └── cooling_relationship.py  # Statistics (fig_5)
├── utils/
│   ├── output_manager.py  # Output organization
│   ├── figure_mapper.py   # Figure ID mapping
│   └── ...
├── palmplot_config.yaml   # Main configuration
├── palmplot_config_fig6_test.yaml  # Test configuration
├── TIME_SELECTION_FEATURES.md  # Time selection guide
├── EXTRACTION_METHODS.md   # Extraction methods guide
└── CLAUDE.md            # Development guidance

Documentation:
├── README.md            # This file
├── TIME_SELECTION_FEATURES.md  # Time selection detailed guide
└── EXTRACTION_METHODS.md       # Extraction methods detailed guide
```

## Key Implementation Details

### Height/Vertical Index Handling

- PALM uses `zu_3d` coordinate for vertical levels
- Extraction heights differ by domain due to grid resolution
- Always use `.isel(zu_3d=idx)` for height extraction
- First data levels:
  - Parent: zu_3d[2] = 15.00m
  - Child: zu_3d[11] = 21.00m

### Time Dimension Processing

- PALM outputs time as seconds since 2018-08-07 00:00:00
- `PALMDataLoader` converts to pandas datetime
- Time averaging uses `xarray.mean(dim='time')` before NumPy conversion
- Corrupted step detection excludes time steps with T < 5°C

### Domain Nesting

- Child domain (200×200m) nested in center of parent (400×400m)
- Child spans 100-300m in both x and y
- File naming: '_N02' suffix indicates child domain

### Orientation and Plotting

- Use `origin='lower'` in `imshow()` for correct spatial orientation
- Do NOT transpose temperature arrays
- Extent format: `[x_min, x_max, y_min, y_max]`

## Recent Bug Fixes (2025-11-03)

### Critical Height Extraction Fix

**Issue**: Code was extracting at incorrect heights:
- Old: 360m (parent), 41m (child) - way too high!
- Result: No visible tree cooling (<0.3°C differences)

**Fix**: Corrected to near-surface levels:
- New: 15m (parent), 21m (child) - appropriate heights
- Result: Clear cooling effects (~1-2°C differences)

**Impact**: Results now show realistic urban forest microclimate effects suitable for publication.

### Memory-Efficient Extraction

**Feature**: Added `transect_direct` extraction method
- ~400× less memory (78 KB vs 31 MB per scenario)
- Identical accuracy to full 2D slice method
- Ideal for batch processing or memory-constrained systems

## Performance

### Typical Processing Times

| Operation | Parent Domain | Child Domain |
|-----------|--------------|--------------|
| Load scenario | ~5-10s | ~3-8s |
| Extract slice (2D) | ~2-5s | ~1-3s |
| Extract direct (1D) | ~0.5-1s | ~0.3-0.8s |
| Generate plot | ~1-2s | ~1-2s |

### Memory Usage

| Method | Parent Domain | Child Domain |
|--------|--------------|--------------|
| slice_2d | ~31 MB | ~8 MB |
| transect_direct | ~78 KB | ~20 KB |

*Per scenario, for 49 time steps*

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions/classes
- Include type hints where appropriate
- Update documentation for new features
- Add tests for new functionality
- Ensure backward compatibility

## Citation

If you use PALMPlot in your research, please cite:

```bibtex
@software{palmplot2024,
  author = {Joshua B-L},
  title = {PALMPlot: PALM-LES Visualization Tool for Urban Microclimate Analysis},
  year = {2024},
  url = {https://github.com/JoshuaB-L/PALMPlot},
  version = {1.0.0}
}
```

## Acknowledgments

- PALM model development team at Leibniz University Hannover
- THF Berlin study collaborators
- xarray and matplotlib development communities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Author**: Joshua B-L
**Repository**: https://github.com/JoshuaB-L/PALMPlot
**Issues**: https://github.com/JoshuaB-L/PALMPlot/issues

## References

### PALM Model

- Maronga, B., et al. (2020). Overview of the PALM model system 6.0. Geoscientific Model Development, 13(3), 1335-1372.
- PALM website: https://palm.muk.uni-hannover.de/

### Related Publications

- [Add your publications using PALMPlot here]

---

**Last Updated**: 2025-11-03
**Version**: 1.0.0
**Status**: Production Ready
