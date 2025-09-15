"""
Pruning Strategies

High-performance pruning strategies for neural networks, including magnitude-based,
structured, and gradual pruning approaches optimized for large models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import math
import logging
from functools import lru_cache
from .filter_pruning import FilterPruner


class MagnitudePruner:
    """
    High-performance magnitude-based pruning strategy.
    
    Prunes neurons/weights based on their absolute magnitude values,
    optimized for large models with efficient batch processing.
    """
    
    def __init__(self, tracker: Optional[Any] = None) -> None:
        """
        Initialize magnitude pruner.
        
        Args:
            tracker: NeuronTracker instance for integration (optional)
        """
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        self._magnitude_cache: Dict[str, torch.Tensor] = {}
    
    def set_tracker(self, tracker: Any) -> None:
        """Set or update the tracker instance."""
        self.tracker = tracker
    
    @lru_cache(maxsize=32)
    def compute_neuron_magnitudes(self, model_id: int, layer_name: str) -> Optional[torch.Tensor]:
        """
        Compute magnitude scores with caching for performance.
        
        Args:
            model_id: Unique model identifier
            layer_name: Name of the layer
            
        Returns:
            Tensor of magnitude scores or None if not cached
        """
        cache_key = f"{model_id}_{layer_name}"
        return self._magnitude_cache.get(cache_key)
    
    def compute_neuron_magnitudes_direct(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute magnitude scores for all neurons in the model with GPU acceleration.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary mapping layer names to magnitude score tensors
        """
        magnitudes = {}
        
        with torch.no_grad():  # Optimize memory usage
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    
                    if isinstance(module, nn.Linear):
                        # L2 norm of weights for each output neuron (more stable than L1)
                        weight_norms = torch.norm(module.weight.data, p=2, dim=1)
                        
                        # Include bias if present
                        if module.bias is not None:
                            bias_contribution = torch.abs(module.bias.data)
                            # Combine weight and bias magnitudes
                            magnitudes[name] = weight_norms + 0.1 * bias_contribution
                        else:
                            magnitudes[name] = weight_norms
                    
                    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                        # For conv layers, compute magnitude per output channel
                        # Flatten spatial dimensions for norm calculation
                        weight_shape = module.weight.shape
                        weight_flat = module.weight.data.view(weight_shape[0], -1)
                        weight_norms = torch.norm(weight_flat, p=2, dim=1)
                        
                        if module.bias is not None:
                            bias_contribution = torch.abs(module.bias.data)
                            magnitudes[name] = weight_norms + 0.1 * bias_contribution
                        else:
                            magnitudes[name] = weight_norms
                    
                    # Cache for future use
                    self._magnitude_cache[f"{id(model)}_{name}"] = magnitudes[name]
        
        return magnitudes
    
    def prune_by_magnitude(self, model: nn.Module, sparsity_ratio: float, 
                          dry_run: bool = True, layer_wise: bool = False) -> Dict[str, Any]:
        """
        Prune neurons based on magnitude with optimized selection algorithm.
        
        Args:
            model: PyTorch model to prune
            sparsity_ratio: Fraction of neurons to prune (0.0 to 1.0)
            dry_run: If True, simulate pruning without modifying model
            layer_wise: If True, apply sparsity ratio per layer; if False, globally
            
        Returns:
            Pruning results dictionary
        """
        if not 0.0 <= sparsity_ratio <= 1.0:
            raise ValueError(f"Sparsity ratio must be between 0.0 and 1.0, got {sparsity_ratio}")
        
        # Compute magnitudes efficiently
        magnitudes = self.compute_neuron_magnitudes_direct(model)
        
        if not magnitudes:
            return {"status": "no_action", "message": "No prunable layers found"}
        
        if layer_wise:
            return self._prune_layer_wise(model, magnitudes, sparsity_ratio, dry_run)
        else:
            return self._prune_globally(model, magnitudes, sparsity_ratio, dry_run)
    
    def _prune_globally(self, model: nn.Module, magnitudes: Dict[str, torch.Tensor], 
                       sparsity_ratio: float, dry_run: bool) -> Dict[str, Any]:
        """
        Apply global magnitude-based pruning across all layers.
        
        Args:
            model: PyTorch model
            magnitudes: Precomputed magnitude scores
            sparsity_ratio: Fraction of neurons to prune globally
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        # Flatten all magnitudes for global threshold computation
        all_magnitudes = []
        magnitude_map = {}  # Maps global index to (layer_name, local_index)
        
        global_idx = 0
        for layer_name, layer_magnitudes in magnitudes.items():
            for local_idx, magnitude in enumerate(layer_magnitudes):
                all_magnitudes.append(magnitude.item())
                magnitude_map[global_idx] = (layer_name, local_idx)
                global_idx += 1
        
        if not all_magnitudes:
            return {"status": "no_action", "message": "No neurons found for pruning"}
        
        # Compute global threshold efficiently using torch
        all_magnitudes_tensor = torch.tensor(all_magnitudes, device=model.device if hasattr(model, 'device') else 'cpu')
        num_to_prune = int(len(all_magnitudes) * sparsity_ratio)
        
        if num_to_prune == 0:
            return {"status": "no_action", "message": "Sparsity ratio too small, no neurons to prune"}
        
        # Use torch.topk for efficient selection (k smallest)
        _, bottom_indices = torch.topk(all_magnitudes_tensor, num_to_prune, largest=False)
        
        # Group pruning candidates by layer
        layer_pruning_plan = defaultdict(list)
        for global_idx in bottom_indices.tolist():
            layer_name, local_idx = magnitude_map[global_idx]
            layer_pruning_plan[layer_name].append(local_idx)
        
        # Execute or simulate pruning
        if dry_run:
            return self._simulate_magnitude_pruning(layer_pruning_plan, magnitudes)
        else:
            return self._execute_magnitude_pruning(model, layer_pruning_plan, magnitudes)
    
    def _prune_layer_wise(self, model: nn.Module, magnitudes: Dict[str, torch.Tensor], 
                         sparsity_ratio: float, dry_run: bool) -> Dict[str, Any]:
        """
        Apply layer-wise magnitude-based pruning.
        
        Args:
            model: PyTorch model
            magnitudes: Precomputed magnitude scores
            sparsity_ratio: Fraction of neurons to prune per layer
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        layer_pruning_plan = {}
        
        for layer_name, layer_magnitudes in magnitudes.items():
            num_neurons = len(layer_magnitudes)
            num_to_prune = int(num_neurons * sparsity_ratio)
            
            if num_to_prune == 0 or num_to_prune >= num_neurons:
                continue
            
            # Get indices of neurons with lowest magnitudes
            _, bottom_indices = torch.topk(layer_magnitudes, num_to_prune, largest=False)
            layer_pruning_plan[layer_name] = bottom_indices.tolist()
        
        if dry_run:
            return self._simulate_magnitude_pruning(layer_pruning_plan, magnitudes)
        else:
            return self._execute_magnitude_pruning(model, layer_pruning_plan, magnitudes)
    
    def _simulate_magnitude_pruning(self, layer_pruning_plan: Dict[str, List[int]], 
                                   magnitudes: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Simulate magnitude-based pruning for analysis."""
        total_neurons_to_prune = sum(len(neurons) for neurons in layer_pruning_plan.values())
        
        results = {
            "status": "simulation",
            "strategy": "magnitude",
            "neurons_pruned": total_neurons_to_prune,
            "layers_affected": len(layer_pruning_plan),
            "layer_modifications": {},
            "magnitude_statistics": {}
        }
        
        for layer_name, neurons_to_prune in layer_pruning_plan.items():
            layer_magnitudes = magnitudes[layer_name]
            
            # Compute statistics for pruned neurons
            pruned_magnitudes = layer_magnitudes[neurons_to_prune]
            remaining_indices = [i for i in range(len(layer_magnitudes)) if i not in neurons_to_prune]
            remaining_magnitudes = layer_magnitudes[remaining_indices] if remaining_indices else torch.tensor([])
            
            results["layer_modifications"][layer_name] = {
                "neurons_removed": len(neurons_to_prune),
                "neurons_remaining": len(remaining_indices),
                "removal_indices": sorted(neurons_to_prune)
            }
            
            results["magnitude_statistics"][layer_name] = {
                "pruned_magnitude_mean": pruned_magnitudes.mean().item() if len(pruned_magnitudes) > 0 else 0.0,
                "pruned_magnitude_std": pruned_magnitudes.std().item() if len(pruned_magnitudes) > 0 else 0.0,
                "remaining_magnitude_mean": remaining_magnitudes.mean().item() if len(remaining_magnitudes) > 0 else 0.0,
                "magnitude_threshold": pruned_magnitudes.max().item() if len(pruned_magnitudes) > 0 else 0.0
            }
        
        return results
    
    def _execute_magnitude_pruning(self, model: nn.Module, layer_pruning_plan: Dict[str, List[int]], 
                                  magnitudes: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute actual magnitude-based pruning."""
        from .core import NeuronPruner
        
        # Use optimized core pruner for execution
        core_pruner = NeuronPruner(tracker=self.tracker)
        
        # Convert to the format expected by core pruner
        layer_pruning_dict = {name: indices for name, indices in layer_pruning_plan.items()}
        
        results = core_pruner._execute_pruning(model, layer_pruning_dict)
        results["strategy"] = "magnitude"
        
        # Add magnitude-specific statistics
        results["magnitude_statistics"] = {}
        for layer_name, neurons_pruned in layer_pruning_plan.items():
            if layer_name in magnitudes:
                layer_magnitudes = magnitudes[layer_name]
                pruned_magnitudes = layer_magnitudes[neurons_pruned]
                results["magnitude_statistics"][layer_name] = {
                    "pruned_magnitude_mean": pruned_magnitudes.mean().item(),
                    "pruned_magnitude_min": pruned_magnitudes.min().item(),
                    "pruned_magnitude_max": pruned_magnitudes.max().item()
                }
        
        return results
    
    def prune_by_recommendations(self, model: nn.Module, recommendations: Dict[str, Any], 
                               dry_run: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Prune based on tracker recommendations enhanced with magnitude analysis.
        
        Args:
            model: PyTorch model to prune
            recommendations: Tracker recommendations dictionary
            dry_run: Whether to simulate pruning
            **kwargs: Additional strategy parameters
            
        Returns:
            Enhanced pruning results
        """
        # Extract pruning candidates
        prune_candidates = recommendations.get("prune", [])
        if not prune_candidates:
            return {"status": "no_action", "message": "No pruning candidates in recommendations"}
        
        # Compute current magnitudes for enhanced analysis
        current_magnitudes = self.compute_neuron_magnitudes_direct(model)
        
        # Group candidates by layer for efficient processing
        layer_candidates = defaultdict(list)
        for candidate in prune_candidates:
            layer_name = candidate.get("layer_name")
            local_index = candidate.get("local_index")
            if layer_name and local_index is not None:
                layer_candidates[layer_name].append({
                    "local_index": local_index,
                    "neuron_id": candidate.get("neuron_id"),
                    "reason": candidate.get("reason", "unknown"),
                    "activity_score": candidate.get("firing_frequency", 0.0),
                    "magnitude_score": None  # Will be filled
                })
        
        # Enhance candidates with magnitude information
        enhanced_candidates = {}
        for layer_name, candidates in layer_candidates.items():
            if layer_name in current_magnitudes:
                layer_magnitudes = current_magnitudes[layer_name]
                
                for candidate in candidates:
                    local_idx = candidate["local_index"]
                    if local_idx < len(layer_magnitudes):
                        candidate["magnitude_score"] = layer_magnitudes[local_idx].item()
                
                # Sort by combined score (activity + magnitude)
                for candidate in candidates:
                    activity_weight = kwargs.get("activity_weight", 0.7)
                    magnitude_weight = kwargs.get("magnitude_weight", 0.3)
                    
                    # Normalize scores (lower is worse for both metrics)
                    activity_score = 1.0 - candidate["activity_score"]  # Lower activity = higher prune score
                    magnitude_score = 1.0 - (candidate["magnitude_score"] or 0.0)  # Lower magnitude = higher prune score
                    
                    candidate["combined_score"] = (activity_weight * activity_score + 
                                                 magnitude_weight * magnitude_score)
                
                enhanced_candidates[layer_name] = sorted(candidates, 
                                                       key=lambda x: x["combined_score"], 
                                                       reverse=True)
        
        # Create pruning plan based on enhanced analysis
        pruning_plan = {}
        for layer_name, candidates in enhanced_candidates.items():
            max_prune_ratio = kwargs.get("max_prune_ratio", 0.5)
            layer_size = len(current_magnitudes.get(layer_name, []))
            max_to_prune = int(layer_size * max_prune_ratio)
            
            # Select top candidates up to the limit
            selected_candidates = candidates[:min(len(candidates), max_to_prune)]
            pruning_plan[layer_name] = [c["local_index"] for c in selected_candidates]
        
        # Execute or simulate
        if dry_run:
            results = self._simulate_magnitude_pruning(pruning_plan, current_magnitudes)
        else:
            results = self._execute_magnitude_pruning(model, pruning_plan, current_magnitudes)
        
        # Add recommendation-specific metadata
        results["recommendation_metadata"] = {
            "total_candidates": len(prune_candidates),
            "enhanced_candidates": sum(len(candidates) for candidates in enhanced_candidates.values()),
            "selection_criteria": {
                "activity_weight": kwargs.get("activity_weight", 0.7),
                "magnitude_weight": kwargs.get("magnitude_weight", 0.3),
                "max_prune_ratio": kwargs.get("max_prune_ratio", 0.5)
            }
        }
        
        return results
    
    def clear_cache(self) -> None:
        """Clear magnitude computation cache to free memory."""
        self._magnitude_cache.clear()
        self.compute_neuron_magnitudes.cache_clear()


class FilterStructuredPruner:
    """
    Filter-based structured pruning strategy using neuron tracker recommendations.
    
    Removes entire filters/channels instead of individual weights to maintain
    architectural validity and prevent dimension mismatches.
    """
    
    def __init__(self, tracker: Optional[Any] = None):
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        self.filter_pruner = FilterPruner(self.logger)
        
    def set_tracker(self, tracker: Any) -> None:
        """Set tracker instance."""
        self.tracker = tracker
    
    def prune_by_recommendations(self, model: nn.Module, recommendations: Dict[str, Any], 
                               dry_run: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Apply filter-based structured pruning using neuron tracker recommendations.
        
        Args:
            model: PyTorch model to prune
            recommendations: Neuron tracker recommendations
            dry_run: Whether to simulate pruning
            **kwargs: Additional parameters
            
        Returns:
            Filter pruning results
        """
        self.logger.info("Starting filter-based structured pruning")
        
        # Convert neuron recommendations to filter-level pruning plan
        filter_plan = self.filter_pruner.convert_neuron_recommendations_to_filters(
            recommendations, model
        )
        
        if not filter_plan:
            return {
                "status": "no_action", 
                "message": "No valid filters to prune from recommendations"
            }
        
        # Log pruning plan
        total_filters = sum(len(filters) for filters in filter_plan.values())
        self.logger.info(f"Planning to prune {total_filters} filters from {len(filter_plan)} layers")
        
        for layer_name, filters in filter_plan.items():
            self.logger.info(f"  {layer_name}: {len(filters)} filters")
        
        # Execute filter pruning
        results = self.filter_pruner.prune_filters_structured(
            model, filter_plan, dry_run=dry_run
        )
        
        # Add strategy metadata
        results["strategy"] = "filter_structured"
        results["recommendation_metadata"] = {
            "total_candidates": len(recommendations.get("prune", [])),
            "converted_to_filters": total_filters,
            "conversion_rate": total_filters / max(1, len(recommendations.get("prune", [])))
        }
        
        return results
    
    def compute_filter_importance(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for filters/channels.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        importance_scores = {}
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    # For conv layers, compute L2 norm of each filter
                    weight = module.weight.data  # [out_channels, in_channels, ...]
                    
                    # Flatten each filter and compute L2 norm
                    num_filters = weight.shape[0]
                    filter_norms = []
                    
                    for i in range(num_filters):
                        filter_weights = weight[i].flatten()
                        filter_norm = torch.norm(filter_weights, p=2)
                        filter_norms.append(filter_norm)
                    
                    importance_scores[name] = torch.tensor(filter_norms, device=weight.device)
                
                elif isinstance(module, nn.Linear):
                    # For linear layers, compute L2 norm of each neuron's weights
                    weight = module.weight.data  # [out_features, in_features]
                    neuron_norms = torch.norm(weight, p=2, dim=1)
                    importance_scores[name] = neuron_norms
        
        return importance_scores
    
    def prune_by_importance(self, model: nn.Module, sparsity_ratio: float = 0.2, 
                          dry_run: bool = True) -> Dict[str, Any]:
        """
        Prune filters based on importance scores.
        
        Args:
            model: PyTorch model to prune
            sparsity_ratio: Fraction of filters to remove
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        # Compute importance scores
        importance_scores = self.compute_filter_importance(model)
        
        # Create pruning plan based on importance
        filter_plan = {}
        
        for layer_name, scores in importance_scores.items():
            num_filters = len(scores)
            num_to_remove = int(num_filters * sparsity_ratio)
            
            if num_to_remove > 0 and num_to_remove < num_filters:
                # Remove least important filters
                _, sorted_indices = torch.sort(scores)
                filters_to_remove = sorted_indices[:num_to_remove].tolist()
                filter_plan[layer_name] = filters_to_remove
        
        if not filter_plan:
            return {"status": "no_action", "message": "No filters to prune"}
        
        # Execute pruning
        results = self.filter_pruner.prune_filters_structured(
            model, filter_plan, dry_run=dry_run
        )
        
        results["strategy"] = "filter_structured_importance"
        results["sparsity_ratio"] = sparsity_ratio
        
        return results


class StructuredPruner:
    """
    High-performance structured pruning strategy.
    
    Removes entire channels/filters while maintaining regular structure
    for hardware acceleration compatibility.
    """
    
    def __init__(self, tracker: Optional[Any] = None) -> None:
        """
        Initialize structured pruner.
        
        Args:
            tracker: NeuronTracker instance for integration (optional)
        """
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        self._importance_cache: Dict[str, torch.Tensor] = {}
    
    def set_tracker(self, tracker: Any) -> None:
        """Set or update the tracker instance."""
        self.tracker = tracker
    
    def compute_channel_importance(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for channels/filters with GPU acceleration.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary mapping layer names to importance score tensors
        """
        importance_scores = {}
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    # For conv layers, use L2 norm of filters as importance
                    weight_shape = module.weight.shape
                    # Flatten spatial and input dimensions, keep output channel dimension
                    weight_flat = module.weight.data.view(weight_shape[0], -1)
                    filter_norms = torch.norm(weight_flat, p=2, dim=1)
                    importance_scores[name] = filter_norms
                    
                elif isinstance(module, nn.Linear):
                    # For linear layers, use L2 norm of neuron weights
                    neuron_norms = torch.norm(module.weight.data, p=2, dim=1)
                    importance_scores[name] = neuron_norms
                
                # Cache results
                self._importance_cache[f"{id(model)}_{name}"] = importance_scores.get(name, torch.empty(0))
        
        return importance_scores
    
    def prune_structured(self, model: nn.Module, layer_sparsity: Dict[str, float], 
                        dry_run: bool = True) -> Dict[str, Any]:
        """
        Apply structured pruning with specified sparsity per layer.
        
        Args:
            model: PyTorch model to prune
            layer_sparsity: Dictionary mapping layer names to sparsity ratios
            dry_run: Whether to simulate pruning
            
        Returns:
            Structured pruning results
        """
        # Validate sparsity ratios
        for layer_name, sparsity in layer_sparsity.items():
            if not 0.0 <= sparsity <= 0.9:  # Cap at 90% to prevent complete layer removal
                raise ValueError(f"Invalid sparsity {sparsity} for layer {layer_name}")
        
        # Compute importance scores
        importance_scores = self.compute_channel_importance(model)
        
        # Create structured pruning plan
        pruning_plan = {}
        for layer_name, sparsity in layer_sparsity.items():
            if layer_name not in importance_scores:
                self.logger.warning(f"Layer {layer_name} not found in importance scores")
                continue
            
            scores = importance_scores[layer_name]
            num_channels = len(scores)
            num_to_prune = int(num_channels * sparsity)
            
            if num_to_prune == 0 or num_to_prune >= num_channels:
                continue
            
            # Get indices of least important channels
            _, bottom_indices = torch.topk(scores, num_to_prune, largest=False)
            pruning_plan[layer_name] = bottom_indices.tolist()
        
        if dry_run:
            return self._simulate_structured_pruning(pruning_plan, importance_scores)
        else:
            return self.execute_structured_pruning(model, pruning_plan)
    
    def _simulate_structured_pruning(self, pruning_plan: Dict[str, List[int]], 
                                   importance_scores: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Simulate structured pruning for analysis."""
        total_channels_to_prune = sum(len(channels) for channels in pruning_plan.values())
        
        results = {
            "status": "simulation",
            "strategy": "structured",
            "channels_pruned": total_channels_to_prune,
            "layers_affected": len(pruning_plan),
            "layer_modifications": {},
            "importance_statistics": {}
        }
        
        for layer_name, channels_to_prune in pruning_plan.items():
            layer_importance = importance_scores[layer_name]
            
            # Compute statistics
            pruned_importance = layer_importance[channels_to_prune]
            remaining_indices = [i for i in range(len(layer_importance)) if i not in channels_to_prune]
            remaining_importance = layer_importance[remaining_indices] if remaining_indices else torch.tensor([])
            
            results["layer_modifications"][layer_name] = {
                "channels_removed": len(channels_to_prune),
                "channels_remaining": len(remaining_indices),
                "removal_indices": sorted(channels_to_prune)
            }
            
            results["importance_statistics"][layer_name] = {
                "pruned_importance_mean": pruned_importance.mean().item() if len(pruned_importance) > 0 else 0.0,
                "remaining_importance_mean": remaining_importance.mean().item() if len(remaining_importance) > 0 else 0.0,
                "importance_threshold": pruned_importance.max().item() if len(pruned_importance) > 0 else 0.0
            }
        
        return results
    
    def execute_structured_pruning(self, model: nn.Module, pruning_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute structured pruning with optimized layer replacement.
        
        Args:
            model: PyTorch model to modify
            pruning_plan: Dictionary with pruning specifications
            
        Returns:
            Execution results
        """
        from .core import NeuronPruner
        
        # Use core pruner for the actual model modification
        core_pruner = NeuronPruner(tracker=self.tracker)
        results = core_pruner._execute_pruning(model, pruning_plan)
        results["strategy"] = "structured"
        
        return results
    
    def prune_by_recommendations(self, model: nn.Module, recommendations: Dict[str, Any], 
                               dry_run: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Apply structured pruning based on tracker recommendations.
        
        Args:
            model: PyTorch model to prune
            recommendations: Tracker recommendations
            dry_run: Whether to simulate pruning
            **kwargs: Additional parameters
            
        Returns:
            Structured pruning results
        """
        # Extract recommendations and group by layer
        prune_candidates = recommendations.get("prune", [])
        layer_candidates = defaultdict(list)
        
        for candidate in prune_candidates:
            layer_name = candidate.get("layer_name")
            if layer_name:
                layer_candidates[layer_name].append(candidate)
        
        # Compute importance scores
        importance_scores = self.compute_channel_importance(model)
        
        # Create structured pruning plan
        default_sparsity = kwargs.get("default_sparsity", 0.2)
        max_sparsity = kwargs.get("max_sparsity", 0.5)
        
        pruning_plan = {}
        for layer_name, candidates in layer_candidates.items():
            if layer_name not in importance_scores:
                continue
            
            # Calculate adaptive sparsity based on number of candidates
            layer_size = len(importance_scores[layer_name])
            candidate_ratio = len(candidates) / layer_size
            adaptive_sparsity = min(max_sparsity, max(default_sparsity, candidate_ratio))
            
            num_to_prune = int(layer_size * adaptive_sparsity)
            if num_to_prune == 0 or num_to_prune >= layer_size:
                continue
            
            # Select least important channels
            scores = importance_scores[layer_name]
            _, bottom_indices = torch.topk(scores, num_to_prune, largest=False)
            pruning_plan[layer_name] = bottom_indices.tolist()
        
        if dry_run:
            return self._simulate_structured_pruning(pruning_plan, importance_scores)
        else:
            return self.execute_structured_pruning(model, pruning_plan)
    
    def clear_cache(self) -> None:
        """Clear importance computation cache."""
        self._importance_cache.clear()


class GradualPruner:
    """
    High-performance gradual pruning strategy.
    
    Implements progressive pruning over multiple epochs with optimized scheduling
    and minimal training disruption.
    """
    
    def __init__(self, tracker: Optional[Any] = None) -> None:
        """
        Initialize gradual pruner.
        
        Args:
            tracker: NeuronTracker instance for integration (optional)
        """
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._magnitude_pruner = MagnitudePruner(tracker)
    
    def set_tracker(self, tracker: Any) -> None:
        """Set or update the tracker instance."""
        self.tracker = tracker
        self._magnitude_pruner.set_tracker(tracker)
    
    def create_schedule(self, initial_sparsity: float, final_sparsity: float, 
                       num_epochs: int, schedule_type: str = "polynomial") -> str:
        """
        Create an optimized pruning schedule.
        
        Args:
            initial_sparsity: Starting sparsity ratio
            final_sparsity: Target final sparsity ratio
            num_epochs: Number of epochs to spread pruning over
            schedule_type: Type of schedule ("polynomial", "exponential", "linear")
            
        Returns:
            Unique schedule identifier
        """
        if not 0.0 <= initial_sparsity <= final_sparsity <= 1.0:
            raise ValueError("Invalid sparsity values")
        
        if num_epochs < 1:
            raise ValueError("Number of epochs must be positive")
        
        schedule_id = f"schedule_{len(self._schedules)}_{schedule_type}"
        
        # Pre-compute entire schedule for efficiency
        sparsity_values = []
        
        if schedule_type == "polynomial":
            # Polynomial decay (default power=3 for smoother transition)
            power = 3.0
            for epoch in range(num_epochs):
                progress = epoch / (num_epochs - 1) if num_epochs > 1 else 1.0
                sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (progress ** power)
                sparsity_values.append(sparsity)
        
        elif schedule_type == "exponential":
            # Exponential decay
            if num_epochs == 1:
                sparsity_values = [final_sparsity]
            else:
                decay_rate = math.log(final_sparsity / initial_sparsity) / (num_epochs - 1)
                for epoch in range(num_epochs):
                    sparsity = initial_sparsity * math.exp(decay_rate * epoch)
                    sparsity_values.append(sparsity)
        
        elif schedule_type == "linear":
            # Linear interpolation
            for epoch in range(num_epochs):
                progress = epoch / (num_epochs - 1) if num_epochs > 1 else 1.0
                sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress
                sparsity_values.append(sparsity)
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self._schedules[schedule_id] = {
            "initial_sparsity": initial_sparsity,
            "final_sparsity": final_sparsity,
            "num_epochs": num_epochs,
            "schedule_type": schedule_type,
            "sparsity_values": sparsity_values,
            "current_epoch": 0
        }
        
        self.logger.info(f"Created {schedule_type} pruning schedule: {schedule_id}")
        return schedule_id
    
    def get_current_sparsity(self, schedule_id: str, epoch: int) -> float:
        """
        Get sparsity for current epoch with bounds checking.
        
        Args:
            schedule_id: Schedule identifier
            epoch: Current epoch (0-indexed)
            
        Returns:
            Sparsity ratio for this epoch
        """
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule {schedule_id} not found")
        
        schedule = self._schedules[schedule_id]
        sparsity_values = schedule["sparsity_values"]
        
        # Clamp epoch to valid range
        epoch = max(0, min(epoch, len(sparsity_values) - 1))
        return sparsity_values[epoch]
    
    def prune_for_epoch(self, model: nn.Module, schedule_id: str, epoch: int, 
                       dry_run: bool = True, strategy: str = "magnitude") -> Dict[str, Any]:
        """
        Apply pruning for a specific epoch according to schedule.
        
        Args:
            model: PyTorch model to prune
            schedule_id: Pruning schedule identifier
            epoch: Current epoch
            dry_run: Whether to simulate pruning
            strategy: Underlying pruning strategy to use
            
        Returns:
            Epoch pruning results
        """
        current_sparsity = self.get_current_sparsity(schedule_id, epoch)
        
        if strategy == "magnitude":
            results = self._magnitude_pruner.prune_by_magnitude(
                model, current_sparsity, dry_run=dry_run, layer_wise=False
            )
        else:
            raise ValueError(f"Unsupported strategy for gradual pruning: {strategy}")
        
        # Add gradual pruning metadata
        results["gradual_metadata"] = {
            "schedule_id": schedule_id,
            "current_epoch": epoch,
            "current_sparsity": current_sparsity,
            "strategy": strategy
        }
        
        # Update schedule state
        if schedule_id in self._schedules:
            self._schedules[schedule_id]["current_epoch"] = epoch
        
        return results
    
    def execute_gradual_step(self, model: nn.Module, schedule_id: str, epoch: int) -> Dict[str, Any]:
        """
        Execute one step of gradual pruning with incremental approach.
        
        Args:
            model: PyTorch model
            schedule_id: Schedule identifier
            epoch: Current epoch
            
        Returns:
            Execution results
        """
        # Calculate incremental pruning needed
        current_sparsity = self.get_current_sparsity(schedule_id, epoch)
        
        # Use magnitude pruner for actual execution
        results = self._magnitude_pruner.prune_by_magnitude(
            model, current_sparsity, dry_run=False, layer_wise=False
        )
        
        results["execution_type"] = "gradual_step"
        results["schedule_metadata"] = {
            "schedule_id": schedule_id,
            "epoch": epoch,
            "target_sparsity": current_sparsity
        }
        
        return results
    
    def prune_by_recommendations(self, model: nn.Module, recommendations: Dict[str, Any], 
                               dry_run: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Create and execute gradual pruning based on recommendations.
        
        Args:
            model: PyTorch model
            recommendations: Tracker recommendations
            dry_run: Whether to simulate
            **kwargs: Additional parameters
            
        Returns:
            Gradual pruning results
        """
        # Extract parameters
        num_epochs = kwargs.get("num_epochs", 10)
        final_sparsity = kwargs.get("final_sparsity", 0.3)
        schedule_type = kwargs.get("schedule_type", "polynomial")
        
        # Create temporary schedule
        schedule_id = self.create_schedule(0.0, final_sparsity, num_epochs, schedule_type)
        
        # For recommendation-based gradual pruning, we simulate the full schedule
        results = {
            "status": "gradual_simulation" if dry_run else "gradual_execution",
            "strategy": "gradual",
            "schedule_id": schedule_id,
            "epoch_results": [],
            "final_sparsity": final_sparsity,
            "num_epochs": num_epochs
        }
        
        total_neurons_pruned = 0
        
        # Simulate or execute each epoch
        for epoch in range(num_epochs):
            epoch_sparsity = self.get_current_sparsity(schedule_id, epoch)
            
            if dry_run:
                # For simulation, just compute what would be pruned
                epoch_result = {
                    "epoch": epoch,
                    "target_sparsity": epoch_sparsity,
                    "simulated": True
                }
            else:
                # Actually execute pruning for this epoch
                epoch_result = self.execute_gradual_step(model, schedule_id, epoch)
                total_neurons_pruned += epoch_result.get("neurons_pruned", 0)
            
            results["epoch_results"].append(epoch_result)
        
        results["total_neurons_pruned"] = total_neurons_pruned
        
        return results
    
    def get_schedule_info(self, schedule_id: str) -> Dict[str, Any]:
        """Get information about a pruning schedule."""
        if schedule_id not in self._schedules:
            return {}
        
        schedule = self._schedules[schedule_id].copy()
        return schedule
    
    def clear_schedules(self) -> None:
        """Clear all pruning schedules to free memory."""
        self._schedules.clear()
        self._magnitude_pruner.clear_cache()
        self.logger.info("Cleared all gradual pruning schedules")


class TransformerStructuredPruner:
    """
    Specialized structured pruning for transformer architectures.
    Focuses on attention heads, embedding dimensions, and feed-forward layers.
    """
    
    def __init__(self, tracker: Optional[Any] = None) -> None:
        """
        Initialize transformer-specific structured pruner.
        
        Args:
            tracker: NeuronTracker instance for integration (optional)
        """
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        self._attention_cache: Dict[str, torch.Tensor] = {}
    
    def set_tracker(self, tracker: Any) -> None:
        """Set or update the tracker instance."""
        self.tracker = tracker
    
    def identify_transformer_components(self, model: nn.Module) -> Dict[str, List[str]]:
        """
        Identify transformer-specific components in the model.
        
        Args:
            model: PyTorch transformer model
            
        Returns:
            Dictionary mapping component types to layer names
        """
        components = {
            "attention_heads": [],
            "attention_projections": [],
            "feed_forward": [],
            "embeddings": [],
            "layer_norms": []
        }
        
        for name, module in model.named_modules():
            name_lower = name.lower()
            
            # Attention components
            if any(pattern in name_lower for pattern in ["attention", "attn", "self_attn"]):
                if any(proj in name_lower for proj in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
                    components["attention_projections"].append(name)
                elif "heads" in name_lower or "head" in name_lower:
                    components["attention_heads"].append(name)
            
            # Feed-forward components
            elif any(pattern in name_lower for pattern in ["fc", "linear", "dense", "mlp", "feed_forward", "ffn"]):
                if not any(skip in name_lower for skip in ["embed", "pos", "norm", "layer_norm"]):
                    components["feed_forward"].append(name)
            
            # Embedding layers
            elif any(pattern in name_lower for pattern in ["embed", "embedding", "token", "position"]):
                components["embeddings"].append(name)
            
            # Layer normalization
            elif any(pattern in name_lower for pattern in ["norm", "layernorm", "layer_norm"]):
                components["layer_norms"].append(name)
        
        # Log detected components
        for component_type, layers in components.items():
            if layers:
                self.logger.info(f"Detected {len(layers)} {component_type}: {layers[:3]}{'...' if len(layers) > 3 else ''}")
        
        return components
    
    def compute_attention_head_importance(self, model: nn.Module, data_loader, 
                                        num_batches: int = 3) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for attention heads based on output variance.
        
        Args:
            model: Transformer model
            data_loader: DataLoader for computing importance
            num_batches: Number of batches to use for computation
            
        Returns:
            Dictionary mapping attention layer names to importance scores
        """
        components = self.identify_transformer_components(model)
        attention_layers = components["attention_heads"] + components["attention_projections"]
        
        if not attention_layers:
            self.logger.warning("No attention layers detected for importance computation")
            return {}
        
        importance_scores = {}
        model.eval()
        
        # Hook storage for attention outputs
        attention_outputs = {}
        
        def create_attention_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Store attention output for analysis
                attention_outputs[layer_name] = output.detach()
            return hook
        
        # Register hooks for attention layers
        hooks = []
        for layer_name in attention_layers:
            layer = self._get_layer_by_name(model, layer_name)
            if layer and isinstance(layer, nn.Linear):
                hook = layer.register_forward_hook(create_attention_hook(layer_name))
                hooks.append(hook)
        
        try:
            batch_count = 0
            with torch.no_grad():
                for batch_data in data_loader:
                    if batch_count >= num_batches:
                        break
                    
                    # Handle different data formats
                    if isinstance(batch_data, (list, tuple)):
                        inputs = batch_data[0]
                    else:
                        inputs = batch_data
                    
                    if hasattr(inputs, 'to'):
                        inputs = inputs.to(next(model.parameters()).device)
                    
                    # Forward pass to collect attention outputs
                    try:
                        _ = model(inputs)
                    except Exception as e:
                        self.logger.warning(f"Forward pass failed for importance computation: {e}")
                        continue
                    
                    batch_count += 1
            
            # Compute importance scores based on attention output variance
            for layer_name, outputs in attention_outputs.items():
                if outputs.numel() > 0:
                    # For attention layers, compute variance across sequence dimension
                    if outputs.dim() >= 3:  # [batch, seq, hidden] or [batch, heads, seq, seq]
                        variance = torch.var(outputs, dim=1)  # Variance across sequence
                        if variance.dim() > 1:
                            importance = torch.mean(variance, dim=-1)  # Average across remaining dims
                        else:
                            importance = variance
                    else:
                        importance = torch.var(outputs, dim=0)  # Variance across batch
                    
                    importance_scores[layer_name] = importance
        
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        return importance_scores
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name with proper handling of nested modules."""
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
    
    def prune_attention_heads(self, model: nn.Module, head_pruning_ratio: float = 0.2,
                            dry_run: bool = True) -> Dict[str, Any]:
        """
        Prune attention heads based on importance scores.
        
        Args:
            model: Transformer model
            head_pruning_ratio: Fraction of attention heads to prune
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        components = self.identify_transformer_components(model)
        attention_projections = components["attention_projections"]
        
        if not attention_projections:
            return {"status": "no_action", "message": "No attention projections found"}
        
        # Compute importance scores (simplified for demo)
        pruning_plan = {}
        
        for layer_name in attention_projections:
            layer = self._get_layer_by_name(model, layer_name)
            if layer and isinstance(layer, nn.Linear):
                # For attention projections, assume head dimension organization
                # Typical: [hidden_size, num_heads * head_dim]
                out_features = layer.out_features
                
                # Estimate number of attention heads (common configurations)
                num_heads_candidates = [8, 12, 16, 32, 64]  # Common head counts
                estimated_heads = 8  # Default fallback
                
                for candidate in num_heads_candidates:
                    if out_features % candidate == 0:
                        estimated_heads = candidate
                        break
                
                head_dim = out_features // estimated_heads
                num_to_prune = int(estimated_heads * head_pruning_ratio)
                
                if num_to_prune > 0 and num_to_prune < estimated_heads:
                    # Prune least important heads (for demo, prune last heads)
                    heads_to_prune = list(range(estimated_heads - num_to_prune, estimated_heads))
                    
                    # Convert to neuron indices
                    neurons_to_prune = []
                    for head_idx in heads_to_prune:
                        start_idx = head_idx * head_dim
                        end_idx = start_idx + head_dim
                        neurons_to_prune.extend(range(start_idx, end_idx))
                    
                    pruning_plan[layer_name] = neurons_to_prune
        
        if not pruning_plan:
            return {"status": "no_action", "message": "No attention heads to prune"}
        
        # Execute or simulate pruning
        if dry_run:
            return self._simulate_transformer_pruning(pruning_plan)
        else:
            return self._execute_transformer_pruning(model, pruning_plan)
    
    def prune_feed_forward_layers(self, model: nn.Module, ff_pruning_ratio: float = 0.3,
                                dry_run: bool = True) -> Dict[str, Any]:
        """
        Prune feed-forward layers based on magnitude.
        
        Args:
            model: Transformer model
            ff_pruning_ratio: Fraction of FF neurons to prune
            dry_run: Whether to simulate pruning
            
        Returns:
            Pruning results
        """
        components = self.identify_transformer_components(model)
        ff_layers = components["feed_forward"]
        
        if not ff_layers:
            return {"status": "no_action", "message": "No feed-forward layers found"}
        
        # Use magnitude-based pruning for FF layers
        magnitude_pruner = MagnitudePruner(self.tracker)
        
        # Filter model to only include FF layers
        ff_model_dict = {}
        for layer_name in ff_layers:
            layer = self._get_layer_by_name(model, layer_name)
            if layer:
                ff_model_dict[layer_name] = layer
        
        if not ff_model_dict:
            return {"status": "no_action", "message": "No valid feed-forward layers found"}
        
        # Compute magnitudes for FF layers only
        magnitudes = {}
        with torch.no_grad():
            for layer_name, layer in ff_model_dict.items():
                if isinstance(layer, nn.Linear):
                    weight_norms = torch.norm(layer.weight.data, p=2, dim=1)
                    if layer.bias is not None:
                        bias_contribution = torch.abs(layer.bias.data)
                        magnitudes[layer_name] = weight_norms + 0.1 * bias_contribution
                    else:
                        magnitudes[layer_name] = weight_norms
        
        # Create pruning plan based on magnitude
        pruning_plan = {}
        for layer_name, layer_magnitudes in magnitudes.items():
            num_neurons = len(layer_magnitudes)
            num_to_prune = int(num_neurons * ff_pruning_ratio)
            
            if num_to_prune > 0 and num_to_prune < num_neurons:
                # Get indices of neurons with lowest magnitudes
                _, bottom_indices = torch.topk(layer_magnitudes, num_to_prune, largest=False)
                pruning_plan[layer_name] = bottom_indices.tolist()
        
        if dry_run:
            return self._simulate_transformer_pruning(pruning_plan)
        else:
            return self._execute_transformer_pruning(model, pruning_plan)
    
    def _simulate_transformer_pruning(self, pruning_plan: Dict[str, List[int]]) -> Dict[str, Any]:
        """Simulate transformer-specific pruning."""
        total_neurons_to_prune = sum(len(neurons) for neurons in pruning_plan.values())
        
        results = {
            "status": "simulation",
            "strategy": "transformer_structured",
            "neurons_pruned": total_neurons_to_prune,
            "layers_affected": len(pruning_plan),
            "transformer_specific": True,
            "layer_modifications": {}
        }
        
        for layer_name, neurons_to_prune in pruning_plan.items():
            results["layer_modifications"][layer_name] = {
                "neurons_removed": len(neurons_to_prune),
                "removal_indices": sorted(neurons_to_prune),
                "layer_type": self._classify_transformer_layer(layer_name)
            }
        
        return results
    
    def _execute_transformer_pruning(self, model: nn.Module, pruning_plan: Dict[str, List[int]]) -> Dict[str, Any]:
        """Execute transformer-specific pruning."""
        from .core import NeuronPruner
        
        # Use core pruner for actual execution
        core_pruner = NeuronPruner(tracker=self.tracker)
        results = core_pruner._execute_pruning(model, pruning_plan)
        
        # Add transformer-specific metadata
        results["strategy"] = "transformer_structured"
        results["transformer_specific"] = True
        
        # Classify layers
        for layer_name in pruning_plan.keys():
            if "layer_modifications" in results and layer_name in results["layer_modifications"]:
                results["layer_modifications"][layer_name]["layer_type"] = self._classify_transformer_layer(layer_name)
        
        return results
    
    def _classify_transformer_layer(self, layer_name: str) -> str:
        """Classify the type of transformer layer."""
        name_lower = layer_name.lower()
        
        if any(pattern in name_lower for pattern in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
            return "attention_projection"
        elif any(pattern in name_lower for pattern in ["attention", "attn"]):
            return "attention_other"
        elif any(pattern in name_lower for pattern in ["fc", "linear", "dense", "mlp", "feed_forward", "ffn"]):
            return "feed_forward"
        elif any(pattern in name_lower for pattern in ["embed", "embedding"]):
            return "embedding"
        else:
            return "other"
    
    def prune_by_recommendations(self, model: nn.Module, recommendations: Dict[str, Any], 
                               dry_run: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Apply transformer-specific pruning based on recommendations.
        
        Args:
            model: Transformer model
            recommendations: Tracker recommendations
            dry_run: Whether to simulate pruning
            **kwargs: Additional parameters
            
        Returns:
            Transformer pruning results
        """
        # Identify transformer components
        components = self.identify_transformer_components(model)
        
        # Extract recommendations by component type
        attention_candidates = []
        ff_candidates = []
        
        for candidate in recommendations.get("prune", []):
            layer_name = candidate.get("layer_name", "")
            if layer_name:
                layer_type = self._classify_transformer_layer(layer_name)
                if "attention" in layer_type:
                    attention_candidates.append(candidate)
                elif "feed_forward" in layer_type:
                    ff_candidates.append(candidate)
        
        # Apply specialized pruning strategies
        attention_ratio = kwargs.get("attention_pruning_ratio", 0.15)
        ff_ratio = kwargs.get("ff_pruning_ratio", 0.25)
        
        results = {
            "status": "completed",
            "strategy": "transformer_comprehensive",
            "attention_results": {},
            "ff_results": {},
            "total_neurons_pruned": 0
        }
        
        # Prune attention components
        if attention_candidates:
            attention_results = self.prune_attention_heads(model, attention_ratio, dry_run)
            results["attention_results"] = attention_results
            results["total_neurons_pruned"] += attention_results.get("neurons_pruned", 0)
        
        # Prune feed-forward components
        if ff_candidates:
            ff_results = self.prune_feed_forward_layers(model, ff_ratio, dry_run)
            results["ff_results"] = ff_results  
            results["total_neurons_pruned"] += ff_results.get("neurons_pruned", 0)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear transformer-specific caches."""
        self._attention_cache.clear()