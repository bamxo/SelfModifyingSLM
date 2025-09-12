"""
Configuration System for Neural Network Pruner

This module provides a comprehensive configuration system for pruning operations,
including support for YAML/JSON files, validation, and easy customization.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import copy

# Try to import YAML support, fall back gracefully if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class PruningThresholds:
    """Configuration for pruning thresholds."""
    
    # Activity-based thresholds
    firing_frequency_threshold: float = 0.01  # Minimum firing frequency to keep neuron
    mean_activation_threshold: float = 0.001  # Minimum mean activation magnitude
    
    # Magnitude-based thresholds  
    weight_magnitude_threshold: float = 0.1   # Minimum weight magnitude to keep neuron
    relative_magnitude_threshold: float = 0.05  # Relative to max magnitude in layer
    
    # Correlation thresholds
    correlation_threshold: float = 0.9  # Threshold for redundant neuron detection
    
    # Safety thresholds
    min_neurons_per_layer: int = 1      # Minimum neurons to keep in any layer
    max_sparsity_per_layer: float = 0.9  # Maximum fraction to prune from any layer


@dataclass
class StrategyConfig:
    """Configuration for pruning strategies."""
    
    # Strategy selection
    default_strategy: str = "magnitude"  # "magnitude", "structured", "gradual"
    
    # Magnitude strategy parameters
    magnitude_weight: float = 0.5        # Weight for magnitude in combined scoring
    activity_weight: float = 0.5         # Weight for activity in combined scoring
    
    # Structured strategy parameters
    structured_granularity: str = "neuron"  # "neuron", "channel", "filter"
    structured_group_size: int = 1       # Size of groups to prune together
    
    # Gradual strategy parameters
    gradual_epochs: int = 10             # Number of epochs for gradual pruning
    gradual_schedule: str = "polynomial" # "linear", "polynomial", "exponential"
    gradual_power: float = 3.0           # Power for polynomial schedule
    
    # Sparsity targets
    target_sparsity: float = 0.3         # Overall target sparsity ratio
    layer_specific_sparsity: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Configuration for logging and output."""
    
    # Logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "pruning.log"
    log_to_console: bool = True
    
    # Output verbosity
    verbose: bool = True                 # Detailed output during operations
    show_progress: bool = True           # Show progress bars/indicators
    save_intermediate_results: bool = True  # Save results at each step
    
    # Report generation
    generate_reports: bool = True        # Generate comprehensive reports
    save_visualizations: bool = True     # Save plots and heatmaps
    export_formats: List[str] = field(default_factory=lambda: ["json", "txt"])


@dataclass
class ValidationConfig:
    """Configuration for validation and safety checks."""
    
    # Pre-pruning validation
    validate_inputs: bool = True         # Validate recommendation inputs
    check_model_structure: bool = True   # Validate model structure before pruning
    
    # Post-pruning validation
    validate_outputs: bool = True        # Validate pruned model
    test_forward_pass: bool = True       # Test forward pass after pruning
    
    # Safety checks
    dry_run_first: bool = True          # Always do dry run before actual pruning
    backup_model: bool = True           # Backup original model state
    max_pruning_ratio: float = 0.8      # Maximum overall pruning ratio allowed


@dataclass
class PrunerConfig:
    """Main configuration class for the pruner system."""
    
    # Sub-configurations
    thresholds: PruningThresholds = field(default_factory=PruningThresholds)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Global settings
    random_seed: Optional[int] = 42      # For reproducible results
    device: str = "auto"                 # "cpu", "cuda", "auto"
    num_workers: int = 1                 # For parallel processing
    
    # Output paths
    output_dir: str = "pruning_outputs"  # Base directory for outputs
    experiment_name: str = "pruning_experiment"  # Name for this pruning experiment
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        errors = []
        
        # Validate thresholds
        if not 0.0 <= self.thresholds.firing_frequency_threshold <= 1.0:
            errors.append("firing_frequency_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.thresholds.max_sparsity_per_layer <= 1.0:
            errors.append("max_sparsity_per_layer must be between 0.0 and 1.0")
        
        if self.thresholds.min_neurons_per_layer < 1:
            errors.append("min_neurons_per_layer must be at least 1")
        
        # Validate strategy
        valid_strategies = ["magnitude", "structured", "gradual"]
        if self.strategy.default_strategy not in valid_strategies:
            errors.append(f"default_strategy must be one of {valid_strategies}")
        
        if not 0.0 <= self.strategy.target_sparsity <= 1.0:
            errors.append("target_sparsity must be between 0.0 and 1.0")
        
        # Validate weights sum to 1.0 (approximately)
        weight_sum = self.strategy.magnitude_weight + self.strategy.activity_weight
        if abs(weight_sum - 1.0) > 0.01:
            errors.append("magnitude_weight + activity_weight should sum to 1.0")
        
        # Validate logging
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        # Validate validation config
        if not 0.0 <= self.validation.max_pruning_ratio <= 1.0:
            errors.append("max_pruning_ratio must be between 0.0 and 1.0")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def save(self, filepath: Union[str, Path], format: str = "auto") -> str:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save the configuration
            format: Format to save in ("json", "yaml", "auto")
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format from extension
        if format == "auto":
            if filepath.suffix.lower() in [".yml", ".yaml"]:
                format = "yaml"
            else:
                format = "json"
        
        if format == "yaml":
            with open(filepath, 'w') as f:
                f.write(self.to_yaml())
        else:
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        
        return str(filepath)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PrunerConfig':
        """Create configuration from dictionary."""
        # Handle nested dictionaries
        if 'thresholds' in config_dict:
            config_dict['thresholds'] = PruningThresholds(**config_dict['thresholds'])
        if 'strategy' in config_dict:
            config_dict['strategy'] = StrategyConfig(**config_dict['strategy'])
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        if 'validation' in config_dict:
            config_dict['validation'] = ValidationConfig(**config_dict['validation'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PrunerConfig':
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'PrunerConfig':
        """Create configuration from YAML string."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PrunerConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            PrunerConfig instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Auto-detect format
        if filepath.suffix.lower() in [".yml", ".yaml"]:
            return cls.from_yaml(content)
        else:
            return cls.from_json(content)
    
    def update(self, **kwargs) -> 'PrunerConfig':
        """
        Create a new configuration with updated values.
        
        Args:
            **kwargs: Values to update
            
        Returns:
            New PrunerConfig instance with updates
        """
        config_dict = self.to_dict()
        
        # Handle nested updates
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like "thresholds.firing_frequency_threshold"
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return self.from_dict(config_dict)
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get parameters for the current strategy."""
        strategy = self.strategy.default_strategy
        
        base_params = {
            "dry_run": self.validation.dry_run_first,
            "target_sparsity": self.strategy.target_sparsity,
            "max_sparsity_per_layer": self.thresholds.max_sparsity_per_layer,
            "min_neurons_per_layer": self.thresholds.min_neurons_per_layer
        }
        
        if strategy == "magnitude":
            base_params.update({
                "magnitude_weight": self.strategy.magnitude_weight,
                "activity_weight": self.strategy.activity_weight,
                "magnitude_threshold": self.thresholds.weight_magnitude_threshold,
                "activity_threshold": self.thresholds.firing_frequency_threshold
            })
        elif strategy == "structured":
            base_params.update({
                "granularity": self.strategy.structured_granularity,
                "group_size": self.strategy.structured_group_size,
                "layer_sparsity": self.strategy.layer_specific_sparsity
            })
        elif strategy == "gradual":
            base_params.update({
                "num_epochs": self.strategy.gradual_epochs,
                "schedule_type": self.strategy.gradual_schedule,
                "schedule_power": self.strategy.gradual_power
            })
        
        return base_params
    
    def setup_logging(self) -> logging.Logger:
        """Set up logging based on configuration."""
        logger = logging.getLogger("pruner")
        logger.setLevel(getattr(logging, self.logging.log_level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.logging.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.logging.log_level))
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.logging.log_to_file:
            # Ensure log directory exists
            log_path = Path(self.logging.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(getattr(logging, self.logging.log_level))
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class ConfigManager:
    """Manager for handling multiple configurations and environment overrides."""
    
    def __init__(self, config: Optional[PrunerConfig] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Initial configuration (defaults to default config)
        """
        self.config = config or PrunerConfig()
        self.logger = self.config.setup_logging()
    
    def load_from_env(self, prefix: str = "PRUNER_") -> 'ConfigManager':
        """
        Override configuration values from environment variables.
        
        Args:
            prefix: Prefix for environment variables
            
        Returns:
            Self for chaining
        """
        env_overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse as appropriate type
                try:
                    # Try boolean
                    if value.lower() in ['true', 'false']:
                        env_overrides[config_key] = value.lower() == 'true'
                    # Try int
                    elif value.isdigit():
                        env_overrides[config_key] = int(value)
                    # Try float
                    elif '.' in value and value.replace('.', '').isdigit():
                        env_overrides[config_key] = float(value)
                    # Keep as string
                    else:
                        env_overrides[config_key] = value
                except ValueError:
                    env_overrides[config_key] = value
        
        if env_overrides:
            self.logger.info(f"Applying {len(env_overrides)} environment overrides")
            self.config = self.config.update(**env_overrides)
        
        return self
    
    def merge_configs(self, *configs: PrunerConfig) -> 'ConfigManager':
        """
        Merge multiple configurations, with later configs taking precedence.
        
        Args:
            *configs: Configurations to merge
            
        Returns:
            Self for chaining
        """
        merged_dict = self.config.to_dict()
        
        for config in configs:
            config_dict = config.to_dict()
            merged_dict.update(config_dict)
        
        self.config = PrunerConfig.from_dict(merged_dict)
        self.logger = self.config.setup_logging()
        
        return self
    
    def validate_for_model(self, model) -> List[str]:
        """
        Validate configuration against a specific model.
        
        Args:
            model: PyTorch model to validate against
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        try:
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Check if target sparsity is reasonable
            if self.config.strategy.target_sparsity > 0.8:
                warnings.append(
                    f"High target sparsity ({self.config.strategy.target_sparsity:.1%}) "
                    "may significantly impact model performance"
                )
            
            # Check layer-specific sparsity
            for layer_name, sparsity in self.config.strategy.layer_specific_sparsity.items():
                if sparsity > self.config.thresholds.max_sparsity_per_layer:
                    warnings.append(
                        f"Layer {layer_name} sparsity ({sparsity:.1%}) exceeds "
                        f"max_sparsity_per_layer ({self.config.thresholds.max_sparsity_per_layer:.1%})"
                    )
            
        except Exception as e:
            warnings.append(f"Error validating model: {e}")
        
        return warnings


def create_default_config() -> PrunerConfig:
    """Create a default configuration with sensible values."""
    return PrunerConfig()


def create_conservative_config() -> PrunerConfig:
    """Create a conservative configuration for safe pruning."""
    return PrunerConfig(
        thresholds=PruningThresholds(
            firing_frequency_threshold=0.001,  # Very low threshold
            max_sparsity_per_layer=0.5,       # Conservative pruning
            min_neurons_per_layer=2
        ),
        strategy=StrategyConfig(
            target_sparsity=0.1,               # Only 10% pruning
            magnitude_weight=0.7,              # Emphasize magnitude
            activity_weight=0.3
        ),
        validation=ValidationConfig(
            dry_run_first=True,
            max_pruning_ratio=0.3              # Maximum 30% overall
        )
    )


def create_aggressive_config() -> PrunerConfig:
    """Create an aggressive configuration for maximum compression."""
    return PrunerConfig(
        thresholds=PruningThresholds(
            firing_frequency_threshold=0.05,   # Higher threshold
            max_sparsity_per_layer=0.95,      # Allow heavy pruning
            min_neurons_per_layer=1
        ),
        strategy=StrategyConfig(
            target_sparsity=0.7,               # 70% pruning
            magnitude_weight=0.3,              # Emphasize activity
            activity_weight=0.7
        ),
        validation=ValidationConfig(
            max_pruning_ratio=0.9              # Allow up to 90%
        )
    )


def load_config_with_fallbacks(
    primary_path: Optional[str] = None,
    fallback_paths: Optional[List[str]] = None,
    create_if_missing: bool = True
) -> PrunerConfig:
    """
    Load configuration with fallback options.
    
    Args:
        primary_path: Primary configuration file path
        fallback_paths: List of fallback paths to try
        create_if_missing: Create default config if none found
        
    Returns:
        Loaded configuration
    """
    paths_to_try = []
    if primary_path:
        paths_to_try.append(primary_path)
    if fallback_paths:
        paths_to_try.extend(fallback_paths)
    
    # Add standard fallback locations
    paths_to_try.extend([
        "pruner_config.yaml",
        "pruner_config.json", 
        "config/pruner.yaml",
        "config/pruner.json",
        os.path.expanduser("~/.pruner/config.yaml")
    ])
    
    for path in paths_to_try:
        try:
            if Path(path).exists():
                return PrunerConfig.load(path)
        except Exception as e:
            logging.warning(f"Failed to load config from {path}: {e}")
    
    if create_if_missing:
        logging.info("No configuration file found, using default configuration")
        return create_default_config()
    else:
        raise FileNotFoundError("No valid configuration file found")
