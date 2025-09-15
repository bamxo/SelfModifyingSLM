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
from .strategies import MagnitudePruner, StructuredPruner, GradualPruner, TransformerStructuredPruner
from .utils import PruningMetrics, PruningValidator
from .model_io import PrunedModelRepresentation, ModelSerializer, create_pruning_report
from .config import (
    PrunerConfig, ConfigManager, PruningThresholds, StrategyConfig, 
    LoggingConfig, ValidationConfig, create_default_config, 
    create_conservative_config, create_aggressive_config,
    create_production_config, create_slm_transformer_config
)
from .engine import PruningEngine
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
__all__ = ["NeuronPruner", "MagnitudePruner", "StructuredPruner", "GradualPruner", "TransformerStructuredPruner",
           "PruningMetrics", "PruningValidator", "PruningEngine",
           "PrunedModelRepresentation", "ModelSerializer", "create_pruning_report",
           "PrunerConfig", "ConfigManager", "create_default_config", "create_production_config", "create_slm_transformer_config",
           "IntegratedTrackingPruningWorkflow", "create_simple_workflow",
           "load_and_validate_recommendations", "create_pruning_plan"]
