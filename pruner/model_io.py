"""
Model Input/Output Operations

This module provides functionality for loading, saving, and representing pruned models,
including serialization of pruning configurations and model states.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import pickle
import os
from pathlib import Path
import time


class PrunedModelRepresentation:
    """
    Represents a pruned model with its configuration and metadata.
    """
    
    def __init__(self, model: nn.Module, pruning_config: Dict[str, Any]):
        """
        Initialize pruned model representation.
        
        Args:
            model: The pruned PyTorch model
            pruning_config: Configuration dict from pruning operations
        """
        self.model = model
        self.pruning_config = pruning_config
        self.creation_time = time.time()
        
        # Extract model structure
        self.model_structure = self._extract_model_structure(model)
        
    def _extract_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model structure information."""
        structure = {
            "layers": {},
            "total_parameters": 0,
            "total_neurons": 0
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                layer_info = {
                    "type": module.__class__.__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                }
                
                if isinstance(module, nn.Linear):
                    layer_info.update({
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "neurons": module.out_features
                    })
                elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    layer_info.update({
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "neurons": module.out_channels,
                        "kernel_size": module.kernel_size
                    })
                
                structure["layers"][name] = layer_info
                structure["total_parameters"] += layer_info["parameters"]
                structure["total_neurons"] += layer_info.get("neurons", 0)
        
        return structure
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pruned model."""
        pruning_meta = self.pruning_config.get("pruning_metadata", {})
        
        return {
            "model_summary": {
                "total_parameters": self.model_structure["total_parameters"],
                "total_neurons": self.model_structure["total_neurons"],
                "layer_count": len(self.model_structure["layers"])
            },
            "pruning_summary": {
                "neurons_pruned": pruning_meta.get("total_neurons_pruned", 0),
                "layers_affected": pruning_meta.get("layers_affected", 0),
                "pruning_status": pruning_meta.get("pruning_status", "unknown")
            },
            "creation_time": self.creation_time,
            "has_original_config": "original_model_config" in self.pruning_config
        }
    
    def compare_with_original(self) -> Optional[Dict[str, Any]]:
        """Compare with original model configuration if available."""
        if "original_model_config" not in self.pruning_config:
            return None
        
        original_config = self.pruning_config["original_model_config"]
        pruned_config = self.pruning_config.get("pruned_model_config", {})
        
        comparison = {
            "layer_changes": {},
            "total_reduction": {
                "neurons": 0,
                "parameters": 0
            }
        }
        
        for layer_name, orig_info in original_config.items():
            if layer_name in pruned_config:
                pruned_info = pruned_config[layer_name]
                
                orig_neurons = orig_info.get("neuron_count", 0)
                pruned_neurons = pruned_info.get("neuron_count", 0)
                
                comparison["layer_changes"][layer_name] = {
                    "original_neurons": orig_neurons,
                    "pruned_neurons": pruned_neurons,
                    "neurons_removed": orig_neurons - pruned_neurons,
                    "reduction_ratio": (orig_neurons - pruned_neurons) / orig_neurons if orig_neurons > 0 else 0
                }
                
                comparison["total_reduction"]["neurons"] += orig_neurons - pruned_neurons
        
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (without model weights)."""
        return {
            "model_structure": self.model_structure,
            "pruning_config": self.pruning_config,
            "creation_time": self.creation_time,
            "summary": self.get_summary()
        }


class ModelSerializer:
    """
    Handles serialization and deserialization of pruned models.
    """
    
    @staticmethod
    def save_pruned_model(model_repr: PrunedModelRepresentation, 
                         save_path: str, 
                         include_weights: bool = True,
                         format: str = "torch") -> Dict[str, str]:
        """
        Save pruned model representation to disk.
        
        Args:
            model_repr: PrunedModelRepresentation to save
            save_path: Base path for saving (without extension)
            include_weights: Whether to save model weights
            format: Save format ("torch", "json", "pickle")
            
        Returns:
            Dictionary with paths of saved files
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save model configuration and metadata as JSON
        config_path = save_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(model_repr.to_dict(), f, indent=2, default=str)
        saved_files["config"] = str(config_path)
        
        # Save model weights if requested
        if include_weights and model_repr.model is not None:
            if format == "torch":
                weights_path = save_path.with_suffix('.pth')
                torch.save(model_repr.model.state_dict(), weights_path)
                saved_files["weights"] = str(weights_path)
            elif format == "pickle":
                weights_path = save_path.with_suffix('.pkl')
                with open(weights_path, 'wb') as f:
                    pickle.dump(model_repr.model.state_dict(), f)
                saved_files["weights"] = str(weights_path)
        
        # Save complete model if torch format
        if format == "torch" and model_repr.model is not None:
            model_path = save_path.with_suffix('.model.pth')
            torch.save(model_repr.model, model_path)
            saved_files["complete_model"] = str(model_path)
        
        return saved_files
    
    @staticmethod
    def load_pruned_model_config(config_path: str) -> Dict[str, Any]:
        """
        Load pruned model configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_pruned_model(config_path: str, 
                         weights_path: Optional[str] = None,
                         model_class: Optional[type] = None) -> PrunedModelRepresentation:
        """
        Load pruned model from saved files.
        
        Args:
            config_path: Path to JSON configuration file
            weights_path: Path to weights file (optional)
            model_class: Model class for reconstruction (optional)
            
        Returns:
            PrunedModelRepresentation
        """
        # Load configuration
        config = ModelSerializer.load_pruned_model_config(config_path)
        
        model = None
        if weights_path and model_class:
            # Reconstruct model and load weights
            # This is simplified - in practice you'd need more info to reconstruct the model
            model = model_class()
            if weights_path.endswith('.pth'):
                model.load_state_dict(torch.load(weights_path))
            elif weights_path.endswith('.pkl'):
                with open(weights_path, 'rb') as f:
                    model.load_state_dict(pickle.load(f))
        
        return PrunedModelRepresentation(model, config.get("pruning_config", {}))


def create_pruning_report(model_before: nn.Module, 
                         model_after: nn.Module,
                         pruning_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive pruning report comparing before and after models.
    
    Args:
        model_before: Original model before pruning
        model_after: Model after pruning
        pruning_results: Results from pruning operation
        
    Returns:
        Comprehensive report dictionary
    """
    def get_model_stats(model):
        total_params = sum(p.numel() for p in model.parameters())
        total_neurons = 0
        layer_info = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                total_neurons += module.out_features
                layer_info[name] = {
                    "type": "Linear",
                    "neurons": module.out_features,
                    "parameters": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                }
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                total_neurons += module.out_channels
                layer_info[name] = {
                    "type": module.__class__.__name__,
                    "neurons": module.out_channels,
                    "parameters": sum(p.numel() for p in module.parameters())
                }
        
        return {
            "total_parameters": total_params,
            "total_neurons": total_neurons,
            "layers": layer_info
        }
    
    before_stats = get_model_stats(model_before)
    after_stats = get_model_stats(model_after)
    
    # Calculate reductions
    param_reduction = before_stats["total_parameters"] - after_stats["total_parameters"]
    neuron_reduction = before_stats["total_neurons"] - after_stats["total_neurons"]
    
    param_reduction_pct = (param_reduction / before_stats["total_parameters"]) * 100 if before_stats["total_parameters"] > 0 else 0
    neuron_reduction_pct = (neuron_reduction / before_stats["total_neurons"]) * 100 if before_stats["total_neurons"] > 0 else 0
    
    # Layer-wise comparison
    layer_comparison = {}
    for layer_name in before_stats["layers"]:
        if layer_name in after_stats["layers"]:
            before_layer = before_stats["layers"][layer_name]
            after_layer = after_stats["layers"][layer_name]
            
            layer_comparison[layer_name] = {
                "before": before_layer,
                "after": after_layer,
                "neurons_removed": before_layer["neurons"] - after_layer["neurons"],
                "parameters_removed": before_layer["parameters"] - after_layer["parameters"]
            }
    
    report = {
        "timestamp": time.time(),
        "pruning_operation": pruning_results,
        "model_comparison": {
            "before": before_stats,
            "after": after_stats,
            "reductions": {
                "parameters": {
                    "absolute": param_reduction,
                    "percentage": param_reduction_pct
                },
                "neurons": {
                    "absolute": neuron_reduction,
                    "percentage": neuron_reduction_pct
                }
            }
        },
        "layer_comparison": layer_comparison,
        "compression_ratio": param_reduction_pct / 100,
        "estimated_speedup": min(param_reduction_pct / 50, 2.0),  # Rough estimate
        "estimated_memory_savings_mb": (param_reduction * 4) / (1024 * 1024)  # Assuming float32
    }
    
    return report


def export_to_json(data: Dict[str, Any], filepath: str, pretty: bool = True) -> str:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export
        filepath: Output file path
        pretty: Whether to format JSON prettily
        
    Returns:
        Path to saved file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2, default=str)
        else:
            json.dump(data, f, default=str)
    
    return filepath


def load_tracker_recommendations(json_path: str) -> Dict[str, Any]:
    """
    Load tracker recommendations from JSON file.
    
    Args:
        json_path: Path to recommendations JSON file
        
    Returns:
        Recommendations dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def validate_pruning_inputs(recommendations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that pruning inputs are correctly formatted.
    
    Args:
        recommendations: Recommendations dictionary
        
    Returns:
        Validation results
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required fields
    if "prune" not in recommendations:
        validation["errors"].append("Missing 'prune' field in recommendations")
        validation["valid"] = False
    
    if "metadata" not in recommendations:
        validation["warnings"].append("Missing 'metadata' field")
    
    # Validate prune candidates
    if "prune" in recommendations:
        prune_candidates = recommendations["prune"]
        if not isinstance(prune_candidates, list):
            validation["errors"].append("'prune' field must be a list")
            validation["valid"] = False
        else:
            for i, candidate in enumerate(prune_candidates):
                required_fields = ["neuron_id", "layer_name", "local_index"]
                for field in required_fields:
                    if field not in candidate:
                        validation["errors"].append(f"Prune candidate {i} missing field: {field}")
                        validation["valid"] = False
    
    return validation
