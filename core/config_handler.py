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
                'variables': dict
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

            # Check that at least one plot type is enabled
            enabled_plots = [p for p, enabled in item_config['plot_types'].items() if enabled]
            if not enabled_plots:
                self.logger.warning(f"{item_id} is enabled but no plot types are selected")

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

            self.logger.debug(
                f"{item_id}: Domain-specific terrain-following settings validated for '{domain}'"
            )

        self.logger.debug(
            f"{item_id}: Terrain-following settings validated: output_mode={output_mode}, "
            f"buildings_mask={buildings_mask}, start_z={start_z_index}, max_z={max_z_index}"
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