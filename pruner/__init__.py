"""
Neural Network Pruner Package

High-performance neural network pruning system optimized for large models.
Provides comprehensive functionality for structured, magnitude-based, and gradual pruning
with seamless integration to the neuron tracker system.

Performance optimizations:
- Cached computations for repeated operations
- GPU-accelerated magnitude calculations
- Efficient batch processing for large models
- Memory-efficient weight copying and layer replacement
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Union
import logging

from .core import NeuronPruner
from .strategies import MagnitudePruner, StructuredPruner, GradualPruner
from .utils import PruningMetrics, PruningValidator
from .model_io import PrunedModelRepresentation, ModelSerializer, create_pruning_report
from .config import (
    PrunerConfig, ConfigManager, PruningThresholds, StrategyConfig, 
    LoggingConfig, ValidationConfig, create_default_config, 
    create_conservative_config, create_aggressive_config
)
from .workflow import (
    IntegratedTrackingPruningWorkflow, create_simple_workflow,
    verify_pruning_accuracy
)
from .test_functions import (
    load_and_validate_recommendations, 
    create_pruning_plan, 
    group_recommendations_by_layer,
    estimate_pruning_impact
)

__version__ = "1.0.0"
__all__ = ["NeuronPruner", "MagnitudePruner", "StructuredPruner", "GradualPruner", 
           "PruningMetrics", "PruningValidator", "PruningEngine",
           "PrunedModelRepresentation", "ModelSerializer", "create_pruning_report",
           "PrunerConfig", "ConfigManager", "create_default_config",
           "IntegratedTrackingPruningWorkflow", "create_simple_workflow",
           "load_and_validate_recommendations", "create_pruning_plan"]


class PruningEngine:
    """
    High-performance main interface for the neural network pruning system.
    
    Optimized for production use with large models, combining all pruning functionality
    into a single, efficient class with comprehensive configuration support.
    
    Features:
    - Lazy initialization of expensive components
    - Cached strategy instances for performance
    - Memory-efficient processing pipelines
    - Comprehensive error handling and logging
    """
    
    def __init__(self, tracker: Optional[Any] = None, config: Optional[PrunerConfig] = None, 
                 config_path: Optional[str] = None) -> None:
        """
        Initialize the high-performance pruning engine with lazy loading.
        
        Args:
            tracker: NeuronTracker instance for integration (optional)
            config: PrunerConfig instance (optional)
            config_path: Path to configuration file (optional)
        """
        # Initialize configuration efficiently
        if config_path:
            self.config_manager = ConfigManager(PrunerConfig.load(config_path))
        elif config:
            self.config_manager = ConfigManager(config)
        else:
            self.config_manager = ConfigManager(create_default_config())
        
        # Load environment overrides
        self.config_manager.load_from_env()
        
        # Set up logging
        self.logger = self.config_manager.config.setup_logging()
        
        # Store tracker reference
        self._tracker = tracker
        
        # Lazy initialization of expensive components (created on first use)
        self._core_pruner: Optional[NeuronPruner] = None
        self._magnitude_pruner: Optional[MagnitudePruner] = None
        self._structured_pruner: Optional[StructuredPruner] = None
        self._gradual_pruner: Optional[GradualPruner] = None
        self._metrics: Optional[PruningMetrics] = None
        self._validator: Optional[PruningValidator] = None
        
        self.logger.info("PruningEngine initialized with lazy loading")
        self.logger.debug(f"Strategy: {self.config_manager.config.strategy.default_strategy}")
        self.logger.debug(f"Target sparsity: {self.config_manager.config.strategy.target_sparsity}")
    
    @property
    def config(self) -> PrunerConfig:
        """Get the current configuration."""
        return self.config_manager.config
    
    @property
    def core_pruner(self) -> NeuronPruner:
        """Get core pruner with lazy initialization."""
        if self._core_pruner is None:
            self._core_pruner = NeuronPruner(self._tracker)
        return self._core_pruner
    
    @property
    def magnitude_pruner(self) -> MagnitudePruner:
        """Get magnitude pruner with lazy initialization."""
        if self._magnitude_pruner is None:
            self._magnitude_pruner = MagnitudePruner(self._tracker)
        return self._magnitude_pruner
    
    @property
    def structured_pruner(self) -> StructuredPruner:
        """Get structured pruner with lazy initialization."""
        if self._structured_pruner is None:
            self._structured_pruner = StructuredPruner(self._tracker)
        return self._structured_pruner
    
    @property
    def gradual_pruner(self) -> GradualPruner:
        """Get gradual pruner with lazy initialization."""
        if self._gradual_pruner is None:
            self._gradual_pruner = GradualPruner(self._tracker)
        return self._gradual_pruner
    
    @property
    def metrics(self) -> PruningMetrics:
        """Get metrics calculator with lazy initialization."""
        if self._metrics is None:
            self._metrics = PruningMetrics()
        return self._metrics
    
    @property
    def validator(self) -> PruningValidator:
        """Get validator with lazy initialization."""
        if self._validator is None:
            self._validator = PruningValidator()
        return self._validator
    
    def set_tracker(self, tracker: Any) -> None:
        """
        Set or update the tracker instance for all components.
        
        Args:
            tracker: NeuronTracker instance
        """
        self._tracker = tracker
        
        # Update existing components if they've been initialized
        if self._core_pruner is not None:
            self._core_pruner.set_tracker(tracker)
        if self._magnitude_pruner is not None:
            self._magnitude_pruner.set_tracker(tracker)
        if self._structured_pruner is not None:
            self._structured_pruner.set_tracker(tracker)
        if self._gradual_pruner is not None:
            self._gradual_pruner.set_tracker(tracker)
    
    def prune_by_recommendations(self, model, recommendations: Dict[str, Any], 
                               strategy: Optional[str] = None, dry_run: Optional[bool] = None, 
                               **kwargs) -> Dict[str, Any]:
        """
        Prune model based on tracker recommendations with optimized strategy selection.
        
        Args:
            model: PyTorch model to prune
            recommendations: Pruning recommendations from tracker
            strategy: Pruning strategy ("magnitude", "structured", "gradual")
            dry_run: Whether to simulate pruning (defaults to config setting)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Comprehensive pruning results
        """
        # Use configuration defaults if not specified
        strategy = strategy or self.config.strategy.default_strategy
        dry_run = dry_run if dry_run is not None else self.config.validation.dry_run_first
        
        self.logger.info(f"Starting {strategy} pruning ({'dry run' if dry_run else 'execution'})")
        
        # Validate model if configured
        if self.config.validation.validate_before_pruning:
            validation_issues = self.validator.validate_model_structure(model)
            if validation_issues:
                self.logger.warning(f"Model validation issues: {validation_issues}")
        
        # Load recommendations if path provided
        if isinstance(recommendations, str):
            try:
                recommendations_data = load_and_validate_recommendations(recommendations)
                if not recommendations_data["loaded"]:
                    return {"status": "error", "message": "Failed to load recommendations"}
                recommendations = recommendations_data["data"]
            except Exception as e:
                return {"status": "error", "message": f"Failed to load recommendations: {e}"}
        
        # Merge strategy-specific parameters from config
        strategy_params = self.config.get_strategy_params()
        merged_kwargs = {**strategy_params, **kwargs}
        
        # Execute pruning with selected strategy
        try:
            if strategy == "magnitude":
                results = self.magnitude_pruner.prune_by_recommendations(
                    model, recommendations, dry_run=dry_run, **merged_kwargs
                )
            elif strategy == "structured":
                results = self.structured_pruner.prune_by_recommendations(
                    model, recommendations, dry_run=dry_run, **merged_kwargs
                )
            elif strategy == "gradual":
                results = self.gradual_pruner.prune_by_recommendations(
                    model, recommendations, dry_run=dry_run, **merged_kwargs
                )
            else:
                return {"status": "error", "message": f"Unknown strategy: {strategy}"}
            
            # Add engine metadata
            results["engine_metadata"] = {
                "config_name": self.config.experiment_name,
                "strategy_used": strategy,
                "dry_run": dry_run,
                "configuration": strategy_params
            }
            
            self.logger.info(f"Pruning completed: {results.get('neurons_pruned', 0)} neurons affected")
            return results
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def prune_from_json(self, model, json_path: str, strategy: str = "magnitude", 
                       dry_run: bool = True) -> Dict[str, Any]:
        """
        Simplified interface for pruning from JSON recommendations.
        
        Args:
            model: PyTorch model to prune
            json_path: Path to JSON recommendations file
            strategy: Pruning strategy to use
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        return self.prune_by_recommendations(model, json_path, strategy, dry_run)
    
    def prune_by_magnitude(self, model, sparsity_ratio: float, dry_run: bool = True) -> Dict[str, Any]:
        """
        Direct magnitude-based pruning interface.
        
        Args:
            model: PyTorch model to prune
            sparsity_ratio: Fraction of neurons to prune
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        return self.magnitude_pruner.prune_by_magnitude(model, sparsity_ratio, dry_run)
    
    def prune_structured(self, model, layer_sparsity: Dict[str, float], 
                        dry_run: bool = True) -> Dict[str, Any]:
        """
        Direct structured pruning interface.
        
        Args:
            model: PyTorch model to prune
            layer_sparsity: Dictionary mapping layer names to sparsity ratios
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        return self.structured_pruner.prune_structured(model, layer_sparsity, dry_run)
    
    def create_gradual_schedule(self, initial_sparsity: float, final_sparsity: float, 
                               num_epochs: int) -> str:
        """
        Create a gradual pruning schedule.
        
        Args:
            initial_sparsity: Starting sparsity ratio
            final_sparsity: Target final sparsity ratio
            num_epochs: Number of epochs to spread pruning over
            
        Returns:
            Schedule identifier
        """
        return self.gradual_pruner.create_schedule(initial_sparsity, final_sparsity, num_epochs)
    
    def compute_metrics(self, model, original_model=None) -> Dict[str, Any]:
        """
        Compute comprehensive pruning metrics.
        
        Args:
            model: Current (possibly pruned) model
            original_model: Original model for comparison (optional)
            
        Returns:
            Dictionary with computed metrics
        """
        return self.metrics.compute_model_size(model)
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all pruning operations.
        
        Returns:
            Summary dictionary with statistics from all components
        """
        summary = {
            "engine_config": self.config.experiment_name,
            "components_initialized": {
                "core_pruner": self._core_pruner is not None,
                "magnitude_pruner": self._magnitude_pruner is not None,
                "structured_pruner": self._structured_pruner is not None,
                "gradual_pruner": self._gradual_pruner is not None
            }
        }
        
        # Add component-specific summaries if initialized
        if self._core_pruner is not None:
            summary["core_pruning"] = self._core_pruner.get_pruning_summary()
        
        if self._gradual_pruner is not None:
            active_schedules = len(self._gradual_pruner._schedules)
            summary["gradual_schedules"] = active_schedules
        
        return summary
    
    # Configuration management methods
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        updated_config = self.config.update(**kwargs)
        self.config_manager = ConfigManager(updated_config)
    
    def save_config(self, filepath: str, format: str = "auto") -> str:
        """Save current configuration to file."""
        return self.config.save(filepath, format)
    
    def load_config(self, filepath: str) -> None:
        """Load configuration from file."""
        new_config = PrunerConfig.load(filepath)
        self.config_manager = ConfigManager(new_config)
        self.logger.info(f"Configuration loaded from {filepath}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config_manager = ConfigManager(create_default_config())
        self.logger.info("Configuration reset to defaults")
    
    def set_conservative_mode(self) -> None:
        """Switch to conservative pruning configuration."""
        self.config_manager = ConfigManager(create_conservative_config())
        self.logger.info("Switched to conservative pruning mode")
    
    def set_aggressive_mode(self) -> None:
        """Switch to aggressive pruning configuration."""
        self.config_manager = ConfigManager(create_aggressive_config())
        self.logger.info("Switched to aggressive pruning mode")
    
    def clear_caches(self) -> None:
        """Clear all internal caches to free memory."""
        if self._core_pruner is not None:
            self._core_pruner.clear_cache()
        if self._magnitude_pruner is not None:
            self._magnitude_pruner.clear_cache()
        if self._structured_pruner is not None:
            self._structured_pruner.clear_cache()
        if self._gradual_pruner is not None:
            self._gradual_pruner.clear_schedules()
        
        self.logger.info("All caches cleared")
    
    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        try:
            self.clear_caches()
        except Exception:
            pass  # Ignore cleanup errors during deletion