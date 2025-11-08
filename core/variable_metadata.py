"""
Variable metadata handler for PALMPlot

This module provides the VariableMetadata class which manages variable
configurations, including PALM names, file types, unit conversions, and
visualization settings.
"""

import logging
import re
import numpy as np
from typing import Dict, Optional, Callable, List


class VariableMetadata:
    """
    Handles variable configuration and metadata for multi-variable support.

    Provides:
    - Variable configuration lookup
    - PALM variable name resolution (including wildcards)
    - File type determination (av_3d vs av_xy)
    - Unit conversion functions
    - Terrain-following applicability checks
    - Visualization metadata (colormaps, labels, units)
    """

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize variable metadata handler.

        Args:
            config: Full PALMPlot configuration dictionary
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.variables = config['data']['variables']

        # Define conversion functions
        self.conversions: Dict[str, Callable] = {
            'none': lambda x: x,
            'kelvin_to_celsius': lambda x: x - 273.15,
            'celsius_to_kelvin': lambda x: x + 273.15,
            'multiply_1000': lambda x: x * 1000,
            'divide_100': lambda x: x / 100,
            'divide_1000': lambda x: x / 1000,
            'pa_to_hpa': lambda x: x / 100,  # Alias for divide_100
        }

        self.logger.info(f"Initialized VariableMetadata with {len(self.variables)} variables")

    def get_variable_config(self, var_name: str) -> Dict:
        """
        Get full configuration for a variable.

        Args:
            var_name: Variable name (e.g., 'temperature', 'utci')

        Returns:
            Variable configuration dictionary

        Raises:
            KeyError: If variable not found
        """
        if var_name not in self.variables:
            available = ', '.join(sorted(self.variables.keys()))
            raise KeyError(
                f"Variable '{var_name}' not defined in configuration. "
                f"Available variables: {available}"
            )
        return self.variables[var_name]

    def get_palm_name(self, var_name: str) -> str:
        """
        Get PALM variable name for a configured variable.

        Args:
            var_name: Variable name (e.g., 'temperature' -> 'ta')

        Returns:
            PALM variable name (may include wildcard)
        """
        config = self.get_variable_config(var_name)
        return config['palm_name']

    def get_file_type(self, var_name: str) -> str:
        """
        Get file type for a variable.

        Args:
            var_name: Variable name

        Returns:
            File type: 'av_3d' or 'av_xy'
        """
        config = self.get_variable_config(var_name)
        return config['file_type']

    def get_z_coordinate(self, var_name: str) -> str:
        """
        Get expected vertical coordinate name for a variable.

        Args:
            var_name: Variable name

        Returns:
            Z-coordinate name (e.g., 'zu_3d', 'zu1_xy')
        """
        config = self.get_variable_config(var_name)
        return config['z_coordinate']

    def requires_terrain_following(self, var_name: str) -> bool:
        """
        Check if variable should use terrain-following extraction.

        Args:
            var_name: Variable name

        Returns:
            True if terrain-following should be applied
        """
        config = self.get_variable_config(var_name)
        return config.get('terrain_following', True)

    def is_wildcard(self, var_name: str) -> bool:
        """
        Check if variable uses wildcard pattern matching.

        Args:
            var_name: Variable name

        Returns:
            True if PALM name contains wildcard
        """
        config = self.get_variable_config(var_name)
        return config.get('wildcard', False)

    def find_variable_in_dataset(self, dataset, var_name: str) -> tuple[object, str]:
        """
        Find variable in xarray dataset, handling wildcards.

        Args:
            dataset: xarray Dataset
            var_name: Variable name from config

        Returns:
            Tuple of (xarray DataArray, actual PALM variable name)

        Raises:
            KeyError: If variable not found
        """
        config = self.get_variable_config(var_name)
        palm_name = config['palm_name']
        is_wildcard = config.get('wildcard', False)

        if is_wildcard:
            # Handle wildcard pattern (e.g., "bio_utci*_xy")
            pattern = palm_name.replace('*', '.*')
            matches = [v for v in dataset.data_vars if re.match(f"^{pattern}$", v)]

            if not matches:
                available = ', '.join(sorted(dataset.data_vars.keys())[:20])
                raise KeyError(
                    f"No variables match pattern '{palm_name}' in dataset. "
                    f"Available variables: {available}..."
                )

            if len(matches) > 1:
                self.logger.warning(
                    f"Multiple matches for '{palm_name}': {matches}. Using first: {matches[0]}"
                )

            actual_name = matches[0]
            return dataset[actual_name], actual_name
        else:
            # Direct lookup
            if palm_name not in dataset:
                available = ', '.join(sorted(dataset.data_vars.keys())[:20])
                raise KeyError(
                    f"Variable '{palm_name}' not found in dataset. "
                    f"Available variables: {available}..."
                )
            return dataset[palm_name], palm_name

    def convert_units(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        Apply unit conversion to data.

        Args:
            data: Input data array
            var_name: Variable name

        Returns:
            Converted data array
        """
        config = self.get_variable_config(var_name)
        conversion_name = config.get('conversion', 'none')

        if conversion_name not in self.conversions:
            self.logger.warning(
                f"Unknown conversion '{conversion_name}' for variable '{var_name}'. "
                f"Available: {list(self.conversions.keys())}. Using 'none'."
            )
            conversion_name = 'none'

        conversion_func = self.conversions[conversion_name]

        # Apply conversion
        converted = conversion_func(data)

        if conversion_name != 'none':
            units_in = config.get('units_in', '')
            units_out = config.get('units_out', '')
            self.logger.debug(
                f"Converted {var_name} from {units_in} to {units_out} using '{conversion_name}'"
            )

        return converted

    def get_display_label(self, var_name: str) -> str:
        """
        Get display label for a variable.

        Args:
            var_name: Variable name

        Returns:
            Display label
        """
        config = self.get_variable_config(var_name)
        return config.get('label', var_name.replace('_', ' ').title())

    def get_units(self, var_name: str, output: bool = True) -> str:
        """
        Get units for a variable.

        Args:
            var_name: Variable name
            output: If True, return output units; if False, return input units

        Returns:
            Unit string
        """
        config = self.get_variable_config(var_name)
        key = 'units_out' if output else 'units_in'
        return config.get(key, '')

    def get_colormap(self, var_name: str) -> str:
        """
        Get colormap for a variable.

        Args:
            var_name: Variable name

        Returns:
            Matplotlib colormap name
        """
        config = self.get_variable_config(var_name)
        return config.get('colormap', 'viridis')

    def get_value_range(self, var_name: str):
        """
        Get value range for a variable.

        Args:
            var_name: Variable name

        Returns:
            'auto' or [min, max] list
        """
        config = self.get_variable_config(var_name)
        return config.get('value_range', 'auto')

    def get_plot_metadata(self, var_name: str) -> Dict:
        """
        Get all plotting metadata for a variable.

        Args:
            var_name: Variable name

        Returns:
            Dictionary with label, units, colormap, value_range
        """
        config = self.get_variable_config(var_name)
        return {
            'label': self.get_display_label(var_name),
            'units': self.get_units(var_name, output=True),
            'colormap': self.get_colormap(var_name),
            'value_range': self.get_value_range(var_name),
            'var_name': var_name,
            'palm_name': config['palm_name']
        }

    def get_enabled_3d_variables(self, enabled_var_names: List[str]) -> List[str]:
        """
        Filter variable list to only 3D variables (for terrain-following caching).

        Args:
            enabled_var_names: List of variable names to filter

        Returns:
            List of 3D variable names (terrain_following=True)
        """
        result = []
        for var_name in enabled_var_names:
            try:
                if self.requires_terrain_following(var_name):
                    result.append(var_name)
            except KeyError:
                self.logger.warning(f"Variable '{var_name}' not found in configuration")
                continue
        return result

    def validate_variable(self, var_name: str) -> bool:
        """
        Validate that a variable is properly configured.

        Args:
            var_name: Variable name

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = self.get_variable_config(var_name)
        except KeyError:
            return False

        # Check required fields
        required = [
            'palm_name', 'file_type', 'z_coordinate', 'units_in',
            'units_out', 'conversion', 'colormap', 'label', 'terrain_following'
        ]

        missing = [f for f in required if f not in config]
        if missing:
            raise ValueError(
                f"Variable '{var_name}' missing required fields: {missing}"
            )

        # Validate file_type
        if config['file_type'] not in ['av_3d', 'av_xy']:
            raise ValueError(
                f"Variable '{var_name}' has invalid file_type: {config['file_type']}"
            )

        # Validate conversion
        if config['conversion'] not in self.conversions:
            self.logger.warning(
                f"Variable '{var_name}' uses unknown conversion: {config['conversion']}"
            )

        return True
