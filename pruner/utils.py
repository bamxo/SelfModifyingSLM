"""
Pruning Utilities

High-performance utility functions and classes for pruning operations,
including optimized metrics computation, validation, and helper functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from functools import lru_cache
import warnings


class PruningMetrics:
    """
    High-performance metrics calculator for pruning operations.
    
    Optimized for large models with efficient computation and caching.
    """
    
    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        self.logger = logging.getLogger(__name__)
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
    
    @lru_cache(maxsize=32)
    def compute_model_size(self, model_id: int) -> Dict[str, int]:
        """
        Compute model size metrics with caching.
        
        Args:
            model_id: Unique model identifier for caching
            
        Returns:
            Dictionary with cached size metrics
        """
        # This is a placeholder for cached computation
        return self._metrics_cache.get(str(model_id), {})
    
    def compute_model_size_direct(self, model: nn.Module) -> Dict[str, int]:
        """
        Compute comprehensive model size metrics with GPU acceleration.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with detailed size metrics
        """
        metrics = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "total_neurons": 0,
            "total_layers": 0,
            "memory_mb": 0.0
        }
        
        layer_details = {}
        
        with torch.no_grad():  # Optimize memory usage
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    # Efficient parameter counting
                    module_params = sum(p.numel() for p in module.parameters())
                    module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    
                    metrics["total_parameters"] += module_params
                    metrics["trainable_parameters"] += module_trainable
                    metrics["total_layers"] += 1
                    
                    # Efficient neuron counting
                    if isinstance(module, nn.Linear):
                        neurons = module.out_features
                    else:  # Conv layers
                        neurons = module.out_channels
                    
                    metrics["total_neurons"] += neurons
                    
                    # Memory estimation (approximate)
                    param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
                    metrics["memory_mb"] += param_memory / (1024 * 1024)
                    
                    layer_details[name] = {
                        "type": module.__class__.__name__,
                        "parameters": module_params,
                        "neurons": neurons,
                        "memory_mb": param_memory / (1024 * 1024)
                    }
        
        # Cache results for future use
        self._metrics_cache[str(id(model))] = {
            "metrics": metrics,
            "layer_details": layer_details
        }
        
        return metrics
    
    def compute_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute sparsity metrics efficiently across the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with sparsity metrics per layer and overall
        """
        sparsity_metrics = {}
        total_params = 0
        total_zeros = 0
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    # Efficient zero counting using vectorized operations
                    layer_zeros = 0
                    layer_params = 0
                    
                    for param in module.parameters():
                        # Use torch operations for efficiency
                        zeros = torch.sum(param.data == 0).item()
                        params = param.numel()
                        
                        layer_zeros += zeros
                        layer_params += params
                    
                    layer_sparsity = layer_zeros / layer_params if layer_params > 0 else 0.0
                    sparsity_metrics[name] = layer_sparsity
                    
                    total_zeros += layer_zeros
                    total_params += layer_params
        
        # Overall sparsity
        overall_sparsity = total_zeros / total_params if total_params > 0 else 0.0
        sparsity_metrics["overall"] = overall_sparsity
        
        return sparsity_metrics
    
    def compute_compression_ratio(self, original_model: nn.Module, 
                                 pruned_model: nn.Module) -> Dict[str, float]:
        """
        Compute compression ratios between original and pruned models.
        
        Args:
            original_model: Original model before pruning
            pruned_model: Model after pruning
            
        Returns:
            Dictionary with compression ratios
        """
        original_metrics = self.compute_model_size_direct(original_model)
        pruned_metrics = self.compute_model_size_direct(pruned_model)
        
        compression_ratios = {}
        
        for metric_name in ["total_parameters", "total_neurons", "memory_mb"]:
            original_value = original_metrics.get(metric_name, 0)
            pruned_value = pruned_metrics.get(metric_name, 0)
            
            if original_value > 0:
                ratio = pruned_value / original_value
                compression_ratios[f"{metric_name}_ratio"] = ratio
                compression_ratios[f"{metric_name}_reduction"] = 1.0 - ratio
            else:
                compression_ratios[f"{metric_name}_ratio"] = 1.0
                compression_ratios[f"{metric_name}_reduction"] = 0.0
        
        return compression_ratios
    
    def estimate_inference_speedup(self, original_model: nn.Module, pruned_model: nn.Module,
                                  sample_input: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
        """
        Estimate inference speedup from pruning with statistical measurement.
        
        Args:
            original_model: Original model
            pruned_model: Pruned model
            sample_input: Sample input tensor for timing
            num_runs: Number of timing runs for averaging
            
        Returns:
            Dictionary with timing and speedup metrics
        """
        device = sample_input.device
        
        def time_model(model: nn.Module, input_tensor: torch.Tensor, runs: int) -> float:
            """Time model inference with GPU synchronization."""
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_tensor)
            
            # Synchronize GPU if available
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Actual timing
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if start_time:
                start_time.record()
                
                with torch.no_grad():
                    for _ in range(runs):
                        _ = model(input_tensor)
                
                end_time.record()
                torch.cuda.synchronize()
                
                total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                # CPU timing
                import time
                start = time.time()
                
                with torch.no_grad():
                    for _ in range(runs):
                        _ = model(input_tensor)
                
                total_time = time.time() - start
            
            return total_time / runs  # Average time per run
        
        try:
            original_time = time_model(original_model, sample_input, num_runs)
            pruned_time = time_model(pruned_model, sample_input, num_runs)
            
            speedup = original_time / pruned_time if pruned_time > 0 else 1.0
            
            return {
                "original_inference_time": original_time,
                "pruned_inference_time": pruned_time,
                "speedup_ratio": speedup,
                "time_reduction": 1.0 - (pruned_time / original_time) if original_time > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"Inference timing failed: {e}")
            return {
                "original_inference_time": 0.0,
                "pruned_inference_time": 0.0,
                "speedup_ratio": 1.0,
                "time_reduction": 0.0,
                "error": str(e)
            }
    
    def clear_cache(self) -> None:
        """Clear metrics cache to free memory."""
        self._metrics_cache.clear()
        self.compute_model_size.cache_clear()


class PruningValidator:
    """
    High-performance validation utilities for pruning operations.
    
    Provides comprehensive validation with optimized checks for large models.
    """
    
    def __init__(self) -> None:
        """Initialize the validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_model_structure(self, model: nn.Module) -> List[str]:
        """
        Validate model structure for potential pruning issues.
        
        Args:
            model: PyTorch model to validate
            
        Returns:
            List of validation issues (empty if no issues)
        """
        issues = []
        
        # Check for layers with zero or very few neurons
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 0:
                    issues.append(f"Layer {name} has zero output features")
                elif module.out_features == 1:
                    issues.append(f"Layer {name} has only one output feature (risky for pruning)")
                
                if module.in_features == 0:
                    issues.append(f"Layer {name} has zero input features")
            
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                if module.out_channels == 0:
                    issues.append(f"Layer {name} has zero output channels")
                elif module.out_channels == 1:
                    issues.append(f"Layer {name} has only one output channel (risky for pruning)")
                
                if module.in_channels == 0:
                    issues.append(f"Layer {name} has zero input channels")
        
        # Check for parameter inconsistencies
        try:
            total_params = sum(p.numel() for p in model.parameters())
            if total_params == 0:
                issues.append("Model has no parameters")
        except Exception as e:
            issues.append(f"Error counting parameters: {e}")
        
        return issues
    
    def validate_pruning_plan(self, model: nn.Module, 
                            pruning_plan: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Validate a pruning plan before execution.
        
        Args:
            model: PyTorch model
            pruning_plan: Dictionary mapping layer names to neuron indices to prune
            
        Returns:
            Validation results with issues and recommendations
        """
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        for layer_name, neurons_to_prune in pruning_plan.items():
            # Check if layer exists
            layer = self._get_layer_by_name(model, layer_name)
            if layer is None:
                validation_results["issues"].append(f"Layer {layer_name} not found in model")
                validation_results["valid"] = False
                continue
            
            # Get layer size
            if isinstance(layer, nn.Linear):
                layer_size = layer.out_features
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                layer_size = layer.out_channels
            else:
                validation_results["warnings"].append(f"Unsupported layer type for {layer_name}")
                continue
            
            # Validate neuron indices
            invalid_indices = [idx for idx in neurons_to_prune if idx >= layer_size or idx < 0]
            if invalid_indices:
                validation_results["issues"].append(
                    f"Invalid neuron indices for {layer_name}: {invalid_indices} (layer size: {layer_size})"
                )
                validation_results["valid"] = False
            
            # Check pruning ratio
            pruning_ratio = len(neurons_to_prune) / layer_size
            if pruning_ratio >= 0.9:
                validation_results["warnings"].append(
                    f"High pruning ratio ({pruning_ratio:.1%}) for {layer_name} may cause instability"
                )
            elif pruning_ratio >= 1.0:
                validation_results["issues"].append(
                    f"Cannot prune all neurons from {layer_name}"
                )
                validation_results["valid"] = False
            
            # Check for duplicate indices
            if len(neurons_to_prune) != len(set(neurons_to_prune)):
                validation_results["warnings"].append(
                    f"Duplicate neuron indices in pruning plan for {layer_name}"
                )
        
        # Generate recommendations
        if validation_results["warnings"]:
            validation_results["recommendations"].append(
                "Consider reducing pruning ratios for layers with warnings"
            )
        
        if not validation_results["issues"] and not validation_results["warnings"]:
            validation_results["recommendations"].append("Pruning plan looks good for execution")
        
        return validation_results
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        Helper to get layer by name with error handling.
        
        Args:
            model: PyTorch model
            layer_name: Name of layer to find
            
        Returns:
            Layer module or None if not found
        """
        try:
            layer_parts = layer_name.split('.')
            current_module = model
            
            for part in layer_parts:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    try:
                        idx = int(part)
                        current_module = current_module[idx]
                    except (ValueError, IndexError, TypeError):
                        return None
            
            return current_module
            
        except Exception:
            return None
    
    def validate_recommendations(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tracker recommendations format and content.
        
        Args:
            recommendations: Recommendations dictionary from tracker
            
        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        # Check required fields
        required_fields = ["metadata", "prune", "statistics"]
        for field in required_fields:
            if field not in recommendations:
                validation["issues"].append(f"Missing required field: {field}")
                validation["valid"] = False
        
        if not validation["valid"]:
            return validation
        
        # Validate prune candidates
        prune_candidates = recommendations.get("prune", [])
        valid_candidates = 0
        
        for i, candidate in enumerate(prune_candidates):
            candidate_issues = []
            
            # Check required candidate fields
            required_candidate_fields = ["neuron_id", "layer_name", "local_index"]
            for field in required_candidate_fields:
                if field not in candidate:
                    candidate_issues.append(f"Missing field: {field}")
            
            # Validate data types
            if "neuron_id" in candidate and not isinstance(candidate["neuron_id"], int):
                candidate_issues.append("neuron_id must be integer")
            
            if "local_index" in candidate and not isinstance(candidate["local_index"], int):
                candidate_issues.append("local_index must be integer")
            
            if candidate_issues:
                validation["issues"].append(f"Candidate {i}: {', '.join(candidate_issues)}")
                validation["valid"] = False
            else:
                valid_candidates += 1
        
        validation["statistics"] = {
            "total_candidates": len(prune_candidates),
            "valid_candidates": valid_candidates,
            "invalid_candidates": len(prune_candidates) - valid_candidates
        }
        
        return validation


# Standalone utility functions
def optimize_layer_sizes(layer_sizes: List[int], target_compression: float) -> List[int]:
    """
    Optimize layer sizes to achieve target compression ratio.
    
    Args:
        layer_sizes: Current layer sizes
        target_compression: Target compression ratio (0.0 to 1.0)
        
    Returns:
        Optimized layer sizes
    """
    if not 0.0 <= target_compression <= 1.0:
        raise ValueError("Target compression must be between 0.0 and 1.0")
    
    total_neurons = sum(layer_sizes)
    target_neurons = int(total_neurons * (1.0 - target_compression))
    
    # Proportional reduction
    compression_factor = target_neurons / total_neurons
    
    optimized_sizes = []
    for size in layer_sizes:
        new_size = max(1, int(size * compression_factor))  # Ensure at least 1 neuron
        optimized_sizes.append(new_size)
    
    return optimized_sizes


def estimate_memory_savings(original_params: int, pruned_params: int, 
                           dtype_size: int = 4) -> Dict[str, float]:
    """
    Estimate memory savings from parameter reduction.
    
    Args:
        original_params: Number of parameters in original model
        pruned_params: Number of parameters in pruned model
        dtype_size: Size of parameter dtype in bytes (default: 4 for float32)
        
    Returns:
        Dictionary with memory savings estimates
    """
    original_memory = original_params * dtype_size
    pruned_memory = pruned_params * dtype_size
    memory_saved = original_memory - pruned_memory
    
    return {
        "original_memory_mb": original_memory / (1024 * 1024),
        "pruned_memory_mb": pruned_memory / (1024 * 1024),
        "memory_saved_mb": memory_saved / (1024 * 1024),
        "memory_reduction_ratio": memory_saved / original_memory if original_memory > 0 else 0.0
    }