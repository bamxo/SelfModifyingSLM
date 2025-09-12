"""
Unit Testable Functions

High-performance standalone functions for pruning operations that can be tested 
in isolation without requiring complex model setups. Optimized for large-scale operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import json
from pathlib import Path
import logging


# Core computational functions (optimized for performance)

def calculate_magnitude_scores_batch(layers: Dict[str, nn.Module]) -> Dict[str, np.ndarray]:
    """
    Calculate magnitude scores for multiple layers efficiently.
    
    Args:
        layers: Dictionary mapping layer names to layer modules
        
    Returns:
        Dictionary mapping layer names to magnitude score arrays
    """
    magnitude_scores = {}
    
    with torch.no_grad():  # Optimize memory usage
        for layer_name, layer in layers.items():
            if isinstance(layer, nn.Linear):
                # Vectorized L2 norm computation
                scores = torch.norm(layer.weight, dim=1, p=2)
                magnitude_scores[layer_name] = scores.cpu().numpy()
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                # Efficient tensor reshaping and norm computation
                weight_flat = layer.weight.view(layer.weight.size(0), -1)
                scores = torch.norm(weight_flat, dim=1, p=2)
                magnitude_scores[layer_name] = scores.cpu().numpy()
    
    return magnitude_scores


def identify_pruning_candidates(scores: np.ndarray, method: str = "threshold", 
                               **kwargs) -> np.ndarray:
    """
    Identify pruning candidates using various selection methods.
    
    Args:
        scores: Array of neuron scores (lower = more likely to prune)
        method: Selection method ("threshold", "percentile", "top_k")
        **kwargs: Method-specific parameters
        
    Returns:
        Array of indices to prune
    """
    if method == "threshold":
        threshold = kwargs.get("threshold", 0.1)
        return np.where(scores < threshold)[0]
    
    elif method == "percentile":
        percentile = kwargs.get("percentile", 10)  # Bottom 10%
        threshold = np.percentile(scores, percentile)
        return np.where(scores <= threshold)[0]
    
    elif method == "top_k":
        k = kwargs.get("k", 10)
        return np.argpartition(scores, k)[:k]
    
    else:
        raise ValueError(f"Unknown selection method: {method}")


def combine_scoring_criteria(magnitude_scores: np.ndarray, 
                           activity_scores: np.ndarray,
                           weights: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """
    Combine multiple scoring criteria efficiently using vectorized operations.
    
    Args:
        magnitude_scores: Array of magnitude scores (normalized)
        activity_scores: Array of activity scores (normalized)
        weights: Tuple of (magnitude_weight, activity_weight)
        
    Returns:
        Array of combined scores
    """
    if len(magnitude_scores) != len(activity_scores):
        raise ValueError("Score arrays must have same length")
    
    magnitude_weight, activity_weight = weights
    
    # Normalize scores to [0, 1] range for fair combination
    mag_normalized = (magnitude_scores - magnitude_scores.min()) / (magnitude_scores.max() - magnitude_scores.min() + 1e-8)
    act_normalized = (activity_scores - activity_scores.min()) / (activity_scores.max() - activity_scores.min() + 1e-8)
    
    # Invert for pruning (lower combined score = more likely to prune)
    combined = magnitude_weight * (1 - mag_normalized) + activity_weight * (1 - act_normalized)
    
    return combined


# Configuration and validation functions

def validate_layer_configuration(layer_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate layer configuration for pruning operations.
    
    Args:
        layer_config: Dictionary with layer configuration
        
    Returns:
        Validation results with issues and recommendations
    """
    validation = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    required_fields = ["layer_name", "neuron_count", "layer_type"]
    
    for field in required_fields:
        if field not in layer_config:
            validation["issues"].append(f"Missing required field: {field}")
            validation["valid"] = False
    
    if not validation["valid"]:
        return validation
    
    # Validate neuron count
    neuron_count = layer_config.get("neuron_count", 0)
    if neuron_count <= 0:
        validation["issues"].append("Neuron count must be positive")
        validation["valid"] = False
    elif neuron_count == 1:
        validation["warnings"].append("Layer has only one neuron - pruning may cause issues")
    
    # Validate layer type
    supported_types = ["Linear", "Conv1d", "Conv2d", "Unknown"]
    layer_type = layer_config.get("layer_type", "Unknown")
    if layer_type not in supported_types:
        validation["warnings"].append(f"Unsupported layer type: {layer_type}")
    
    # Add recommendations
    if neuron_count < 10:
        validation["recommendations"].append("Consider conservative pruning for small layers")
    
    return validation


def calculate_compression_metrics(original_neurons: int, pruned_neurons: int, 
                                original_params: int, pruned_params: int) -> Dict[str, float]:
    """
    Calculate comprehensive compression metrics efficiently.
    
    Args:
        original_neurons: Number of neurons before pruning
        pruned_neurons: Number of neurons after pruning
        original_params: Number of parameters before pruning
        pruned_params: Number of parameters after pruning
        
    Returns:
        Dictionary with compression metrics
    """
    if original_neurons <= 0 or original_params <= 0:
        return {
            "neuron_compression_ratio": 0.0,
            "parameter_compression_ratio": 0.0,
            "neuron_reduction_percentage": 0.0,
            "parameter_reduction_percentage": 0.0
        }
    
    neuron_ratio = pruned_neurons / original_neurons
    param_ratio = pruned_params / original_params
    
    return {
        "neuron_compression_ratio": neuron_ratio,
        "parameter_compression_ratio": param_ratio,
        "neuron_reduction_percentage": (1.0 - neuron_ratio) * 100,
        "parameter_reduction_percentage": (1.0 - param_ratio) * 100,
        "neurons_removed": original_neurons - pruned_neurons,
        "parameters_removed": original_params - pruned_params
    }


def simulate_pruning_effect(layer_sizes: List[int], pruning_plan: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    Simulate the effect of a pruning plan without actual model modification.
    
    Args:
        layer_sizes: List of original layer sizes
        pruning_plan: Dictionary mapping layer indices to neuron indices to prune
        
    Returns:
        Simulation results with projected compression and performance impact
    """
    if len(layer_sizes) == 0:
        return {"error": "No layers provided"}
    
    original_total = sum(layer_sizes)
    remaining_sizes = layer_sizes.copy()
    neurons_pruned = 0
    
    # Apply pruning plan
    for layer_idx_str, neuron_indices in pruning_plan.items():
        try:
            layer_idx = int(layer_idx_str)
            if 0 <= layer_idx < len(remaining_sizes):
                neurons_to_prune = len(neuron_indices)
                remaining_sizes[layer_idx] = max(1, remaining_sizes[layer_idx] - neurons_to_prune)
                neurons_pruned += neurons_to_prune
        except (ValueError, IndexError):
            continue
    
    pruned_total = sum(remaining_sizes)
    
    # Calculate compression metrics
    compression_ratio = pruned_total / original_total if original_total > 0 else 1.0
    reduction_ratio = 1.0 - compression_ratio
    
    # Estimate parameter reduction (simplified model)
    # Assumes fully connected layers with quadratic parameter scaling
    original_params = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
    pruned_params = sum(remaining_sizes[i] * remaining_sizes[i+1] for i in range(len(remaining_sizes)-1))
    param_compression = pruned_params / original_params if original_params > 0 else 1.0
    
    return {
        "original_neurons": original_total,
        "pruned_neurons": pruned_total,
        "neurons_removed": neurons_pruned,
        "neuron_compression_ratio": compression_ratio,
        "neuron_reduction_percentage": reduction_ratio * 100,
        "estimated_parameter_compression": param_compression,
        "estimated_parameter_reduction": (1.0 - param_compression) * 100,
        "layer_sizes_after_pruning": remaining_sizes,
        "pruning_feasible": all(size > 0 for size in remaining_sizes)
    }


# JSON and recommendation processing functions

def parse_tracker_recommendation(recommendation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate a single tracker recommendation with enhanced error handling.
    
    Args:
        recommendation: Single recommendation dictionary
        
    Returns:
        Parsed and validated recommendation
    """
    parsed = {
        "valid": True,
        "errors": [],
        "neuron_id": None,
        "layer_name": None,
        "local_index": None,
        "reason": "unknown",
        "confidence": 0.0,
        "metrics": {}
    }
    
    # Extract and validate required fields
    try:
        parsed["neuron_id"] = int(recommendation.get("neuron_id", -1))
        if parsed["neuron_id"] < 0:
            parsed["errors"].append("Invalid neuron_id")
            parsed["valid"] = False
    except (ValueError, TypeError):
        parsed["errors"].append("neuron_id must be an integer")
        parsed["valid"] = False
    
    parsed["layer_name"] = recommendation.get("layer_name", "")
    if not parsed["layer_name"]:
        parsed["errors"].append("Missing layer_name")
        parsed["valid"] = False
    
    try:
        parsed["local_index"] = int(recommendation.get("local_index", -1))
        if parsed["local_index"] < 0:
            parsed["errors"].append("Invalid local_index")
            parsed["valid"] = False
    except (ValueError, TypeError):
        parsed["errors"].append("local_index must be an integer")
        parsed["valid"] = False
    
    # Extract optional fields
    parsed["reason"] = recommendation.get("reason", "unknown")
    
    # Extract and validate metrics
    metrics_fields = ["firing_frequency", "mean_activation", "correlation_score"]
    for field in metrics_fields:
        if field in recommendation:
            try:
                parsed["metrics"][field] = float(recommendation[field])
            except (ValueError, TypeError):
                parsed["errors"].append(f"Invalid {field} value")
    
    # Calculate confidence based on available metrics
    if parsed["metrics"]:
        # Simple confidence calculation based on how "extreme" the metrics are
        firing_freq = parsed["metrics"].get("firing_frequency", 0.5)
        parsed["confidence"] = 1.0 - firing_freq  # Lower firing = higher confidence for pruning
    
    return parsed


def group_recommendations_by_layer(recommendations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group recommendations by layer name efficiently.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Dictionary mapping layer names to lists of recommendations
    """
    grouped = {}
    
    for rec in recommendations:
        layer_name = rec.get("layer_name")
        if layer_name:
            if layer_name not in grouped:
                grouped[layer_name] = []
            grouped[layer_name].append(rec)
    
    # Sort recommendations within each layer by local_index for consistency
    for layer_name in grouped:
        grouped[layer_name].sort(key=lambda x: x.get("local_index", 0))
    
    return grouped


def create_pruning_plan(grouped_recommendations: Dict[str, List[Dict[str, Any]]], 
                       max_prune_ratio: float = 0.5) -> Dict[str, List[int]]:
    """
    Create an optimized pruning plan from grouped recommendations.
    
    Args:
        grouped_recommendations: Recommendations grouped by layer
        max_prune_ratio: Maximum fraction of neurons to prune per layer
        
    Returns:
        Dictionary mapping layer names to lists of neuron indices to prune
    """
    pruning_plan = {}
    
    for layer_name, layer_recommendations in grouped_recommendations.items():
        if not layer_recommendations:
            continue
        
        # Extract valid recommendations
        valid_recs = []
        for rec in layer_recommendations:
            parsed = parse_tracker_recommendation(rec)
            if parsed["valid"]:
                valid_recs.append({
                    "local_index": parsed["local_index"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"]
                })
        
        if not valid_recs:
            continue
        
        # Sort by confidence (higher confidence = more likely to prune)
        valid_recs.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Apply pruning ratio limit
        # Estimate layer size from maximum local_index (rough approximation)
        max_local_index = max(rec["local_index"] for rec in valid_recs)
        estimated_layer_size = max_local_index + 1
        max_to_prune = int(estimated_layer_size * max_prune_ratio)
        
        # Select top candidates up to the limit
        selected_indices = [rec["local_index"] for rec in valid_recs[:max_to_prune]]
        
        if selected_indices:
            pruning_plan[layer_name] = sorted(selected_indices)
    
    return pruning_plan


def estimate_pruning_impact(pruning_plan: Dict[str, List[int]], 
                          layer_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Estimate the impact of a pruning plan on model performance and size.
    
    Args:
        pruning_plan: Dictionary mapping layer names to neuron indices to prune
        layer_configs: Dictionary mapping layer names to configuration info
        
    Returns:
        Impact estimation results
    """
    impact_estimate = {
        "total_neurons_to_prune": 0,
        "total_original_neurons": 0,
        "overall_compression_ratio": 1.0,
        "layer_impacts": {},
        "estimated_speedup": 1.0,
        "estimated_memory_reduction": 0.0,
        "risk_assessment": "low"
    }
    
    total_original = 0
    total_to_prune = 0
    high_impact_layers = 0
    
    for layer_name, neuron_indices in pruning_plan.items():
        layer_config = layer_configs.get(layer_name, {})
        original_size = layer_config.get("neuron_count", len(neuron_indices) * 2)  # Fallback estimate
        
        neurons_to_prune = len(neuron_indices)
        pruning_ratio = neurons_to_prune / original_size if original_size > 0 else 0.0
        
        impact_estimate["layer_impacts"][layer_name] = {
            "original_neurons": original_size,
            "neurons_to_prune": neurons_to_prune,
            "pruning_ratio": pruning_ratio,
            "remaining_neurons": original_size - neurons_to_prune
        }
        
        total_original += original_size
        total_to_prune += neurons_to_prune
        
        if pruning_ratio > 0.3:  # High impact threshold
            high_impact_layers += 1
    
    # Calculate overall metrics
    if total_original > 0:
        overall_compression = (total_original - total_to_prune) / total_original
        impact_estimate["overall_compression_ratio"] = overall_compression
        impact_estimate["estimated_compression"] = 1.0 - overall_compression
        
        # Rough speedup estimation (simplified model)
        impact_estimate["estimated_speedup"] = 1.0 / (overall_compression + 0.1)  # Avoid division by zero
        
        # Memory reduction estimation
        impact_estimate["estimated_memory_reduction"] = (1.0 - overall_compression) * 100
    
    impact_estimate["total_neurons_to_prune"] = total_to_prune
    impact_estimate["total_original_neurons"] = total_original
    
    # Risk assessment
    if high_impact_layers > len(pruning_plan) * 0.5:
        impact_estimate["risk_assessment"] = "high"
    elif high_impact_layers > 0:
        impact_estimate["risk_assessment"] = "medium"
    else:
        impact_estimate["risk_assessment"] = "low"
    
    return impact_estimate


def load_and_validate_recommendations(json_path: str) -> Dict[str, Any]:
    """
    Load and validate recommendations from JSON file with comprehensive error handling.
    
    Args:
        json_path: Path to JSON recommendations file
        
    Returns:
        Dictionary with loading results and validation info
    """
    result = {
        "loaded": False,
        "data": None,
        "validation": {
            "valid": False,
            "issues": [],
            "statistics": {}
        },
        "file_info": {
            "path": json_path,
            "exists": False,
            "size_mb": 0.0
        }
    }
    
    try:
        json_file = Path(json_path)
        result["file_info"]["exists"] = json_file.exists()
        
        if not json_file.exists():
            result["validation"]["issues"].append(f"File not found: {json_path}")
            return result
        
        result["file_info"]["size_mb"] = json_file.stat().st_size / (1024 * 1024)
        
        # Load JSON with error handling
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        result["loaded"] = True
        result["data"] = data
        
        # Validate structure
        required_top_level = ["metadata", "prune", "statistics"]
        validation_issues = []
        
        for field in required_top_level:
            if field not in data:
                validation_issues.append(f"Missing top-level field: {field}")
        
        if not validation_issues:
            # Validate prune candidates
            prune_candidates = data.get("prune", [])
            valid_candidates = 0
            
            for i, candidate in enumerate(prune_candidates):
                parsed = parse_tracker_recommendation(candidate)
                if parsed["valid"]:
                    valid_candidates += 1
                else:
                    validation_issues.extend([f"Candidate {i}: {error}" for error in parsed["errors"]])
            
            result["validation"]["statistics"] = {
                "total_candidates": len(prune_candidates),
                "valid_candidates": valid_candidates,
                "invalid_candidates": len(prune_candidates) - valid_candidates,
                "validation_rate": valid_candidates / len(prune_candidates) if prune_candidates else 0.0
            }
            
            result["validation"]["valid"] = len(validation_issues) == 0
        
        result["validation"]["issues"] = validation_issues
        
    except json.JSONDecodeError as e:
        result["validation"]["issues"].append(f"Invalid JSON format: {e}")
    except Exception as e:
        result["validation"]["issues"].append(f"Error loading file: {e}")
    
    return result


# Performance optimization utilities

def optimize_pruning_order(pruning_plan: Dict[str, List[int]], 
                          dependency_graph: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Optimize the order of layer pruning to minimize dependency issues.
    
    Args:
        pruning_plan: Dictionary mapping layer names to neuron indices
        dependency_graph: Optional dependency information between layers
        
    Returns:
        Optimized list of layer names in pruning order
    """
    if dependency_graph is None:
        # Simple alphabetical ordering if no dependency info
        return sorted(pruning_plan.keys())
    
    # Topological sort based on dependencies (simplified)
    ordered_layers = []
    remaining_layers = set(pruning_plan.keys())
    
    while remaining_layers:
        # Find layers with no dependencies or all dependencies already processed
        ready_layers = []
        for layer in remaining_layers:
            deps = dependency_graph.get(layer, [])
            if not deps or all(dep in ordered_layers or dep not in remaining_layers for dep in deps):
                ready_layers.append(layer)
        
        if not ready_layers:
            # Circular dependency or other issue - just add remaining layers
            ready_layers = list(remaining_layers)
        
        # Sort ready layers for consistency
        ready_layers.sort()
        ordered_layers.extend(ready_layers)
        remaining_layers -= set(ready_layers)
    
    return ordered_layers