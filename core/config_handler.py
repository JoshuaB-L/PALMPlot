"""
PALMPlot Configuration Handler
Manages configuration loading, validation, and access
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from schema import Schema, And, Or, Optional as SchemaOptional, SchemaError


class ConfigHandler:
    """Handles configuration loading and validation for PALMPlot"""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration handler
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = None
        self._schema = self._create_config_schema()
        
    def _create_config_schema(self) -> Schema:
        """Create configuration validation schema"""
        return Schema({
            'general': {
                'project_name': str,
                'description': str,
                'stop_on_error': bool,
                'parallel_processing': bool,
                'n_workers': And(int, lambda n: n > 0)
            },
            'data': {
                'simulation_base_path': str,
                'tree_locations_path': str,
                'lad_wad_data_path': str,
                'spacings': [And(int, lambda n: n > 0)],
                'ages': [And(int, lambda n: n > 0)],
                'domains': {
                    'parent': {
                        'resolution': And(float, lambda n: n > 0),
                        'grid_size': [And(int, lambda n: n > 0)],
                        'first_data_level': And(int, lambda n: n >= 0)
                    },
                    'child': {
                        'resolution': And(float, lambda n: n > 0),
                        'grid_size': [And(int, lambda n: n > 0)],
                        'first_data_level': And(int, lambda n: n >= 0)
                    }
                },
                'file_patterns': dict,
                # New comprehensive variable definitions
                'variables': {
                    str: {  # Variable name (e.g., 'temperature', 'utci')
                        'palm_name': str,  # PALM variable name (e.g., 'ta', 'bio_utci*_xy')
                        'file_type': Or('av_3d', 'av_xy'),  # File type
                        'z_coordinate': str,  # Vertical coordinate (e.g., 'zu_3d', 'zu1_xy')
                        'units_in': str,  # Input units
                        'units_out': str,  # Output units
                        'conversion': str,  # Conversion function name
                        'colormap': str,  # Matplotlib colormap
                        'value_range': Or('auto', list),  # [min, max] or 'auto'
                        'label': str,  # Display label
                        'terrain_following': bool,  # Use terrain-following extraction?
                        SchemaOptional('wildcard'): bool  # Is palm_name a wildcard pattern?
                    }
                }
            },
            'output': {
                'base_directory': str,
                'formats': [Or('png', 'pdf', 'svg', 'eps')],
                'dpi': And(int, lambda n: n > 0),
                'transparent_background': bool,
                'file_naming': {
                    'prefix': str,
                    'include_timestamp': bool,
                    'separator': str
                }
            },
            'plots': {
                'global_settings': dict,
                # Accept either 'slides' (legacy) or 'figures' (new)
                SchemaOptional('slides'): dict,
                SchemaOptional('figures'): dict
            },
            'analysis': dict,
            'logging': {
                'level': Or('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                'log_file': str,
                'console_output': bool,
                'log_format': str
            },
            'performance': {
                'chunk_size': And(int, lambda n: n > 0),
                'cache_data': bool,
                'compress_output': bool,
                'multiprocessing': bool
            },
            'validation': {
                'check_data_completeness': bool,
                'validate_units': bool,
                'check_missing_values': bool,
                'warn_on_outliers': bool,
                'outlier_threshold': And(float, lambda n: n > 0)
            }
        })
        
    def load_config(self) -> Dict:
        """
        Load and validate configuration from YAML file
        
        Returns:
            Validated configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Validate configuration
            self._validate_config()
            
            # Expand paths
            self._expand_paths()
            
            # Set defaults
            self._set_defaults()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")
            
    def _validate_config(self):
        """Validate configuration against schema"""
        try:
            self._schema.validate(self.config)
        except SchemaError as e:
            raise ValueError(f"Configuration validation error: {str(e)}")
            
        # Additional custom validations
        self._validate_paths()
        self._validate_variables()
        self._validate_plot_settings()
        
    def _validate_paths(self):
        """Validate that required paths exist"""
        # Check simulation base path
        sim_path = Path(self.config['data']['simulation_base_path'])
        if not sim_path.exists():
            raise ValueError(f"Simulation base path does not exist: {sim_path}")
            
        # Check tree locations path
        tree_path = Path(self.config['data']['tree_locations_path'])
        if not tree_path.exists():
            self.logger.warning(f"Tree locations path does not exist: {tree_path}")
            
        # Create output directory if it doesn't exist
        output_path = Path(self.config['output']['base_directory'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create log directory if needed
        log_file = Path(self.config['logging']['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def _validate_variables(self):
        """Validate variable definitions"""
        variables = self.config['data']['variables']

        if not variables:
            raise ValueError("No variables defined in data.variables section")

        # Validate each variable definition
        for var_name, var_config in variables.items():
            # Check required fields
            required_fields = [
                'palm_name', 'file_type', 'z_coordinate', 'units_in',
                'units_out', 'conversion', 'colormap', 'value_range',
                'label', 'terrain_following'
            ]

            missing_fields = [f for f in required_fields if f not in var_config]
            if missing_fields:
                raise ValueError(
                    f"Variable '{var_name}' missing required fields: {missing_fields}"
                )

            # Validate file_type
            if var_config['file_type'] not in ['av_3d', 'av_xy']:
                raise ValueError(
                    f"Variable '{var_name}' has invalid file_type: {var_config['file_type']}. "
                    "Must be 'av_3d' or 'av_xy'"
                )

            # Validate terrain_following for xy variables
            if var_config['file_type'] == 'av_xy' and var_config['terrain_following']:
                self.logger.warning(
                    f"Variable '{var_name}' is from av_xy file but has terrain_following=true. "
                    "This is unusual - xy variables are typically surface-only."
                )

            # Validate conversion function names
            valid_conversions = [
                'none', 'kelvin_to_celsius', 'multiply_1000', 'divide_100'
            ]
            if var_config['conversion'] not in valid_conversions:
                self.logger.warning(
                    f"Variable '{var_name}' uses unknown conversion: {var_config['conversion']}. "
                    f"Known conversions: {valid_conversions}"
                )

            # Validate value_range
            value_range = var_config['value_range']
            if value_range != 'auto':
                if not isinstance(value_range, list) or len(value_range) != 2:
                    raise ValueError(
                        f"Variable '{var_name}' has invalid value_range. "
                        "Must be 'auto' or [min, max] list"
                    )
                if value_range[0] >= value_range[1]:
                    raise ValueError(
                        f"Variable '{var_name}' has invalid value_range: min >= max"
                    )

        self.logger.info(f"Validated {len(variables)} variable definitions")

    def _validate_plot_settings(self):
        """Validate plot-specific settings for both slides and figures"""
        plots_config = self.config['plots']

        # Ensure at least one of slides or figures is present
        if 'slides' not in plots_config and 'figures' not in plots_config:
            raise ValueError(
                "Configuration must contain either 'plots.slides' or 'plots.figures' section"
            )

        # Get the appropriate section
        items_key = 'figures' if 'figures' in plots_config else 'slides'
        items = plots_config[items_key]

        for item_id, item_config in items.items():
            if not item_config['enabled']:
                continue

            # Check for old format (plot_types) and reject it
            if 'plot_types' in item_config:
                raise ValueError(
                    f"{item_id}: Old 'plot_types' format is no longer supported. "
                    f"Please migrate to new format with 'variables', 'plot_matrix', "
                    f"and optionally 'variable_overrides'. See documentation for details."
                )

            # Validate new format: variables and plot_matrix
            if 'variables' not in item_config or 'plot_matrix' not in item_config:
                raise ValueError(
                    f"{item_id}: New format requires both 'variables' and 'plot_matrix' sections. "
                    f"See documentation for migration guide."
                )

            # Validate variables list
            variables = item_config['variables']
            if not variables:
                raise ValueError(f"{item_id}: 'variables' list cannot be empty")

            # Check that all referenced variables exist in data.variables
            defined_vars = set(self.config['data']['variables'].keys())
            for var in variables:
                if var not in defined_vars:
                    raise ValueError(
                        f"{item_id}: Variable '{var}' not found in data.variables. "
                        f"Available: {sorted(defined_vars)}"
                    )

            # Validate plot_matrix
            plot_matrix = item_config['plot_matrix']
            if 'domains' not in plot_matrix or 'comparisons' not in plot_matrix:
                raise ValueError(
                    f"{item_id}: plot_matrix must contain 'domains' and 'comparisons'"
                )

            valid_domains = ['parent', 'child']
            for domain in plot_matrix['domains']:
                if domain not in valid_domains:
                    raise ValueError(
                        f"{item_id}: Invalid domain '{domain}'. Must be one of: {valid_domains}"
                    )

            valid_comparisons = ['age', 'spacing']
            for comp in plot_matrix['comparisons']:
                if comp not in valid_comparisons:
                    raise ValueError(
                        f"{item_id}: Invalid comparison '{comp}'. Must be one of: {valid_comparisons}"
                    )

            # Validate variable_overrides if present
            if 'variable_overrides' in item_config:
                overrides = item_config['variable_overrides']
                for var, var_overrides in overrides.items():
                    if var not in variables:
                        raise ValueError(
                            f"{item_id}: variable_overrides references '{var}' "
                            f"which is not in variables list"
                        )

                    # Validate override keys
                    valid_override_keys = ['domains', 'comparisons']
                    for key in var_overrides:
                        if key not in valid_override_keys:
                            raise ValueError(
                                f"{item_id}: Invalid override key '{key}' for variable '{var}'. "
                                f"Valid keys: {valid_override_keys}"
                            )

            self.logger.info(f"{item_id}: Validated {len(variables)} variables")

            # Validate terrain-following settings if extraction_method is 'terrain_following'
            if 'settings' in item_config:
                settings = item_config['settings']
                extraction_method = settings.get('extraction_method', 'slice_2d')

                if extraction_method == 'terrain_following':
                    self._validate_terrain_following_settings(item_id, settings)
                elif extraction_method not in ['slice_2d', 'transect_direct', 'terrain_following']:
                    raise ValueError(
                        f"{item_id}: Invalid extraction_method '{extraction_method}'. "
                        f"Must be one of: 'slice_2d', 'transect_direct', 'terrain_following'"
                    )

    def _validate_terrain_following_settings(self, item_id: str, settings: Dict):
        """
        Validate terrain-following specific settings.

        Args:
            item_id: Figure/slide identifier (for error messages)
            settings: Settings dictionary for the plot
        """
        if 'terrain_following' not in settings:
            self.logger.warning(
                f"{item_id}: extraction_method is 'terrain_following' but no "
                f"terrain_following settings found. Using defaults."
            )
            return

        tf_settings = settings['terrain_following']

        # Validate global output_mode
        output_mode = tf_settings.get('output_mode', '2d')
        if output_mode not in ['2d', '1d']:
            raise ValueError(
                f"{item_id}: Invalid terrain_following.output_mode '{output_mode}'. "
                f"Must be '2d' or '1d'"
            )

        # Validate global buildings_mask
        buildings_mask = tf_settings.get('buildings_mask', True)
        if not isinstance(buildings_mask, bool):
            raise ValueError(
                f"{item_id}: terrain_following.buildings_mask must be boolean (true/false)"
            )

        # Validate global start_z_index
        start_z_index = tf_settings.get('start_z_index', 0)
        if not isinstance(start_z_index, int) or start_z_index < 0:
            raise ValueError(
                f"{item_id}: terrain_following.start_z_index must be non-negative integer, "
                f"got {start_z_index}"
            )

        # Validate global max_z_index (optional)
        max_z_index = tf_settings.get('max_z_index', None)
        if max_z_index is not None:
            if not isinstance(max_z_index, int) or max_z_index < 0:
                raise ValueError(
                    f"{item_id}: terrain_following.max_z_index must be non-negative integer or null, "
                    f"got {max_z_index}"
                )
            if max_z_index < start_z_index:
                raise ValueError(
                    f"{item_id}: terrain_following.max_z_index ({max_z_index}) must be >= "
                    f"start_z_index ({start_z_index})"
                )

        # Validate global transect_z_offset (optional)
        transect_z_offset = tf_settings.get('transect_z_offset', None)
        if transect_z_offset is not None:
            if not isinstance(transect_z_offset, int):
                raise ValueError(
                    f"{item_id}: terrain_following.transect_z_offset must be integer or null, "
                    f"got {type(transect_z_offset).__name__}"
                )
            # Allow both positive and negative offsets, but warn if very large
            if abs(transect_z_offset) > 100:
                self.logger.warning(
                    f"{item_id}: terrain_following.transect_z_offset is very large "
                    f"({transect_z_offset}). This may result in out-of-bounds errors."
                )

        # Validate map_display_scenario (optional)
        map_display_scenario = tf_settings.get('map_display_scenario', 'first')
        if map_display_scenario is not None:
            if not isinstance(map_display_scenario, str):
                raise ValueError(
                    f"{item_id}: terrain_following.map_display_scenario must be string, "
                    f"got {type(map_display_scenario).__name__}"
                )
            # Log the configured setting
            self.logger.debug(
                f"{item_id}: 2D map will display scenario: '{map_display_scenario}'"
            )

        # Validate domain-specific settings if present
        for domain in ['parent', 'child']:
            if domain not in tf_settings:
                continue

            domain_settings = tf_settings[domain]
            if not isinstance(domain_settings, dict):
                raise ValueError(
                    f"{item_id}: terrain_following.{domain} must be a dictionary"
                )

            # Validate domain-specific start_z_index
            if 'start_z_index' in domain_settings:
                domain_start_z = domain_settings['start_z_index']
                if not isinstance(domain_start_z, int) or domain_start_z < 0:
                    raise ValueError(
                        f"{item_id}: terrain_following.{domain}.start_z_index must be "
                        f"non-negative integer, got {domain_start_z}"
                    )

            # Validate domain-specific max_z_index
            if 'max_z_index' in domain_settings:
                domain_max_z = domain_settings['max_z_index']
                if domain_max_z is not None:
                    if not isinstance(domain_max_z, int) or domain_max_z < 0:
                        raise ValueError(
                            f"{item_id}: terrain_following.{domain}.max_z_index must be "
                            f"non-negative integer or null, got {domain_max_z}"
                        )
                    # Check against domain-specific start_z_index if present
                    domain_start = domain_settings.get('start_z_index', start_z_index)
                    if domain_max_z < domain_start:
                        raise ValueError(
                            f"{item_id}: terrain_following.{domain}.max_z_index ({domain_max_z}) "
                            f"must be >= start_z_index ({domain_start})"
                        )

            # Validate domain-specific transect_axis
            if 'transect_axis' in domain_settings:
                domain_axis = domain_settings['transect_axis']
                if domain_axis not in ['x', 'y']:
                    raise ValueError(
                        f"{item_id}: terrain_following.{domain}.transect_axis must be 'x' or 'y', "
                        f"got '{domain_axis}'"
                    )

            # Validate domain-specific transect_location
            if 'transect_location' in domain_settings:
                domain_loc = domain_settings['transect_location']
                if not isinstance(domain_loc, int) or domain_loc < 0:
                    raise ValueError(
                        f"{item_id}: terrain_following.{domain}.transect_location must be "
                        f"non-negative integer, got {domain_loc}"
                    )

            # Validate domain-specific transect_width
            if 'transect_width' in domain_settings:
                domain_width = domain_settings['transect_width']
                if not isinstance(domain_width, int) or domain_width < 0:
                    raise ValueError(
                        f"{item_id}: terrain_following.{domain}.transect_width must be "
                        f"non-negative integer, got {domain_width}"
                    )

            # Validate domain-specific transect_z_offset
            if 'transect_z_offset' in domain_settings:
                domain_z_offset = domain_settings['transect_z_offset']
                if domain_z_offset is not None:
                    if not isinstance(domain_z_offset, int):
                        raise ValueError(
                            f"{item_id}: terrain_following.{domain}.transect_z_offset must be "
                            f"integer or null, got {type(domain_z_offset).__name__}"
                        )
                    # Allow both positive and negative offsets, but warn if very large
                    if abs(domain_z_offset) > 100:
                        self.logger.warning(
                            f"{item_id}: terrain_following.{domain}.transect_z_offset is very large "
                            f"({domain_z_offset}). This may result in out-of-bounds errors."
                        )

            self.logger.debug(
                f"{item_id}: Domain-specific terrain-following settings validated for '{domain}'"
            )

        # Validate PCM transect_z_offset settings (optional)
        if 'pcm_transect_z_offset' in tf_settings:
            pcm_config = tf_settings['pcm_transect_z_offset']

            if not isinstance(pcm_config, dict):
                raise ValueError(
                    f"{item_id}: terrain_following.pcm_transect_z_offset must be a dictionary"
                )

            # Validate enabled flag
            pcm_enabled = pcm_config.get('enabled', False)
            if not isinstance(pcm_enabled, bool):
                raise ValueError(
                    f"{item_id}: terrain_following.pcm_transect_z_offset.enabled must be boolean"
                )

            if pcm_enabled:
                # Validate mode setting
                if 'mode' not in pcm_config:
                    raise ValueError(
                        f"{item_id}: terrain_following.pcm_transect_z_offset.mode is required when enabled=true"
                    )

                mode = pcm_config['mode']
                valid_modes = ['upper', 'middle', 'lower', 'mean', 'individual']

                # Mode can be single string or list
                if isinstance(mode, str):
                    mode_list = [mode]
                elif isinstance(mode, list):
                    mode_list = mode
                    if len(mode_list) == 0:
                        raise ValueError(
                            f"{item_id}: terrain_following.pcm_transect_z_offset.mode list cannot be empty"
                        )
                else:
                    raise ValueError(
                        f"{item_id}: terrain_following.pcm_transect_z_offset.mode must be string or list, "
                        f"got {type(mode).__name__}"
                    )

                # Validate each mode
                for m in mode_list:
                    if m not in valid_modes:
                        raise ValueError(
                            f"{item_id}: Invalid PCM mode '{m}'. Must be one of: {valid_modes}"
                        )

                # Check mutual exclusivity: global modes vs individual
                has_global = any(m in ['upper', 'middle', 'lower', 'mean'] for m in mode_list)
                has_individual = 'individual' in mode_list

                if has_global and has_individual:
                    raise ValueError(
                        f"{item_id}: Cannot mix global modes (upper/middle/lower/mean) with individual mode"
                    )

                # Validate individual mode settings if present
                if has_individual:
                    if 'individual' not in pcm_config:
                        raise ValueError(
                            f"{item_id}: terrain_following.pcm_transect_z_offset.individual configuration "
                            f"required when mode='individual'"
                        )

                    individual_config = pcm_config['individual']
                    if not isinstance(individual_config, dict):
                        raise ValueError(
                            f"{item_id}: terrain_following.pcm_transect_z_offset.individual must be a dictionary"
                        )

                    # Validate domain-specific age configurations
                    for domain in ['parent', 'child']:
                        if domain not in individual_config:
                            self.logger.warning(
                                f"{item_id}: No PCM individual offsets configured for '{domain}' domain. "
                                f"Extraction will fail for this domain."
                            )
                            continue

                        domain_ages = individual_config[domain]
                        if not isinstance(domain_ages, dict):
                            raise ValueError(
                                f"{item_id}: terrain_following.pcm_transect_z_offset.individual.{domain} "
                                f"must be a dictionary"
                            )

                        # Validate age entries
                        valid_ages = ['20yrs', '40yrs', '60yrs', '80yrs']
                        for age_key, offset_value in domain_ages.items():
                            # Validate age key format
                            if age_key not in valid_ages:
                                self.logger.warning(
                                    f"{item_id}: PCM age key '{age_key}' not in expected format. "
                                    f"Expected: {valid_ages}"
                                )

                            # Validate offset value
                            if not isinstance(offset_value, int):
                                raise ValueError(
                                    f"{item_id}: PCM offset for {domain}.{age_key} must be integer, "
                                    f"got {type(offset_value).__name__}"
                                )

                            if offset_value < 0:
                                raise ValueError(
                                    f"{item_id}: PCM offset for {domain}.{age_key} must be non-negative, "
                                    f"got {offset_value}"
                                )

                            # Warn if offset is very large (PCM has typically 15 levels)
                            if offset_value > 14:
                                self.logger.warning(
                                    f"{item_id}: PCM offset for {domain}.{age_key} is {offset_value}. "
                                    f"Note: zpc_3d typically has ~15 levels (0-14). "
                                    f"This may result in out-of-bounds errors."
                                )

                # Validate cache_storage setting (optional)
                if 'cache_storage' in pcm_config:
                    cache_storage = pcm_config['cache_storage']
                    valid_storage = ['averaged', 'full_resolution']
                    if cache_storage not in valid_storage:
                        raise ValueError(
                            f"{item_id}: terrain_following.pcm_transect_z_offset.cache_storage must be "
                            f"one of: {valid_storage}, got '{cache_storage}'"
                        )

                self.logger.debug(
                    f"{item_id}: PCM transect_z_offset settings validated: enabled=true, mode={mode}"
                )
            else:
                self.logger.debug(
                    f"{item_id}: PCM transect_z_offset disabled"
                )

        self.logger.debug(
            f"{item_id}: Terrain-following settings validated: output_mode={output_mode}, "
            f"buildings_mask={buildings_mask}, start_z={start_z_index}, max_z={max_z_index}"
        )

        # NEW: Validate mask cache settings if present
        self._validate_terrain_mask_cache_settings(item_id, tf_settings)

    def _validate_terrain_mask_cache_settings(self, item_id: str, tf_settings: Dict):
        """
        Validate terrain mask cache configuration.

        Args:
            item_id: Figure/slide identifier (for error messages)
            tf_settings: terrain_following settings dictionary
        """
        if 'mask_cache' not in tf_settings:
            return  # Caching is optional

        cache_settings = tf_settings['mask_cache']

        # Validate enabled flag
        if not isinstance(cache_settings.get('enabled', True), bool):
            raise ValueError(f"{item_id}: mask_cache.enabled must be boolean")

        if not cache_settings.get('enabled', True):
            self.logger.debug(f"{item_id}: Terrain mask caching is disabled")
            return  # Disabled, skip further validation

        # Validate mode
        mode = cache_settings.get('mode', 'auto')
        if mode not in ['save', 'load', 'auto']:
            raise ValueError(
                f"{item_id}: mask_cache.mode must be 'save', 'load', or 'auto', "
                f"got '{mode}'"
            )

        # Validate cache_directory
        cache_dir = cache_settings.get('cache_directory')
        if not cache_dir or not isinstance(cache_dir, str):
            raise ValueError(
                f"{item_id}: mask_cache.cache_directory must be a non-empty string"
            )

        # Validate levels section
        levels = cache_settings.get('levels', {})
        if not isinstance(levels, dict):
            raise ValueError(f"{item_id}: mask_cache.levels must be a dictionary")

        max_levels = levels.get('max_levels', 20)
        if not isinstance(max_levels, int) or max_levels < 1:
            raise ValueError(
                f"{item_id}: mask_cache.levels.max_levels must be positive integer, "
                f"got {max_levels}"
            )

        # Validate offsets
        offsets = levels.get('offsets', [0])
        if isinstance(offsets, str):
            if offsets not in ['all'] and not offsets.startswith('range('):
                raise ValueError(
                    f"{item_id}: mask_cache.levels.offsets string must be 'all' or "
                    f"'range(...)', got '{offsets}'"
                )
        elif isinstance(offsets, list):
            if not all(isinstance(o, int) for o in offsets):
                raise ValueError(
                    f"{item_id}: mask_cache.levels.offsets list must contain only integers"
                )
            if any(o < 0 for o in offsets):
                raise ValueError(
                    f"{item_id}: mask_cache.levels.offsets cannot be negative"
                )
            if any(o >= max_levels for o in offsets):
                self.logger.warning(
                    f"{item_id}: Some offsets in mask_cache.levels.offsets exceed "
                    f"max_levels ({max_levels}). These may fail at runtime."
                )
        else:
            raise ValueError(
                f"{item_id}: mask_cache.levels.offsets must be list or string"
            )

        # Validate variables
        variables = cache_settings.get('variables', 'auto')
        if isinstance(variables, str):
            if variables not in ['auto', 'all']:
                raise ValueError(
                    f"{item_id}: mask_cache.variables string must be 'auto' or 'all', "
                    f"got '{variables}'"
                )
        elif not isinstance(variables, list):
            raise ValueError(
                f"{item_id}: mask_cache.variables must be string or list"
            )

        # Validate compression
        compression = cache_settings.get('compression', {})
        if not isinstance(compression, dict):
            raise ValueError(f"{item_id}: mask_cache.compression must be a dictionary")

        if compression.get('enabled', True):
            if not isinstance(compression.get('enabled'), bool):
                raise ValueError(f"{item_id}: mask_cache.compression.enabled must be boolean")

            level = compression.get('level', 4)
            if not isinstance(level, int) or not (1 <= level <= 9):
                raise ValueError(
                    f"{item_id}: mask_cache.compression.level must be integer 1-9, "
                    f"got {level}"
                )

        # Validate validation settings (meta!)
        validation = cache_settings.get('validation', {})
        if not isinstance(validation, dict):
            raise ValueError(f"{item_id}: mask_cache.validation must be a dictionary")

        on_mismatch = validation.get('on_mismatch', 'recompute')
        if on_mismatch not in ['error', 'warn', 'recompute']:
            raise ValueError(
                f"{item_id}: mask_cache.validation.on_mismatch must be "
                f"'error', 'warn', or 'recompute', got '{on_mismatch}'"
            )

        # Validate check flags
        for check_name in ['check_grid_size', 'check_domain_type', 'check_z_coordinate']:
            if check_name in validation:
                if not isinstance(validation[check_name], bool):
                    raise ValueError(
                        f"{item_id}: mask_cache.validation.{check_name} must be boolean"
                    )

        # Validate max_age_days
        if 'max_age_days' in validation:
            max_age = validation['max_age_days']
            if not isinstance(max_age, (int, float)) or max_age <= 0:
                raise ValueError(
                    f"{item_id}: mask_cache.validation.max_age_days must be positive number, "
                    f"got {max_age}"
                )

        self.logger.debug(
            f"{item_id}: Terrain mask cache settings validated: "
            f"mode={mode}, max_levels={max_levels}, offsets={offsets}"
        )

    def _expand_paths(self):
        """Expand relative paths to absolute paths"""
        # Expand user home directory
        path_fields = [
            ['data', 'simulation_base_path'],
            ['data', 'tree_locations_path'],
            ['data', 'lad_wad_data_path'],
            ['output', 'base_directory'],
            ['logging', 'log_file']
        ]
        
        for field_path in path_fields:
            value = self._get_nested(field_path)
            if value and isinstance(value, str):
                expanded = os.path.expanduser(value)
                expanded = os.path.abspath(expanded)
                self._set_nested(field_path, expanded)
                
    def _set_defaults(self):
        """Set default values for optional configuration items"""
        defaults = {
            'general': {
                'stop_on_error': False,
                'parallel_processing': True,
                'n_workers': 4
            },
            'output': {
                'transparent_background': False,
                'file_naming': {
                    'include_timestamp': False,
                    'separator': '_'
                }
            },
            'performance': {
                'cache_data': True,
                'compress_output': True,
                'multiprocessing': True
            }
        }
        
        self._merge_defaults(self.config, defaults)
        
    def _merge_defaults(self, config: Dict, defaults: Dict):
        """Recursively merge default values into configuration"""
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                self._merge_defaults(config[key], value)
                
    def _get_nested(self, keys: List[str]) -> Any:
        """Get nested configuration value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
        
    def _set_nested(self, keys: List[str], value: Any):
        """Set nested configuration value"""
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._get_nested(keys)
        return value if value is not None else default
        
    def get_enabled_slides(self) -> List[str]:
        """
        Get list of enabled slides/figures.

        Returns list IDs regardless of whether using 'slides' or 'figures' naming.
        """
        plots_config = self.config['plots']
        items_key = 'figures' if 'figures' in plots_config else 'slides'
        items = plots_config[items_key]
        return [item_id for item_id, config in items.items() if config['enabled']]

    def get_enabled_plots(self, item_id: str) -> List[str]:
        """
        Get list of enabled plots for a specific slide/figure.

        Args:
            item_id: Slide or figure identifier

        Returns:
            List of enabled plot type names
        """
        plots_config = self.config['plots']
        items_key = 'figures' if 'figures' in plots_config else 'slides'
        item_config = plots_config[items_key].get(item_id, {})
        plot_types = item_config.get('plot_types', {})
        return [plot_type for plot_type, enabled in plot_types.items() if enabled]
        
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
        self.logger.info(f"Configuration saved to: {save_path}")