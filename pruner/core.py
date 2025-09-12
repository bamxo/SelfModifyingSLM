"""
Core Neuron Pruner

High-performance neuron pruning implementation optimized for large models.
Provides fundamental functionality to prune neurons in PyTorch neural networks
based on tracker analysis and optimization recommendations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from collections import defaultdict
import json
import logging
from functools import lru_cache
import warnings


class NeuronPruner:
    """
    High-performance core class for pruning neurons in PyTorch neural networks.
    
    Optimized for large models with efficient batch processing and minimal memory overhead.
    Works with the NeuronTracker to identify and remove neurons based on various criteria
    such as dead neurons, low activity, or optimization recommendations.
    
    Performance Features:
    - Cached layer lookups for repeated operations
    - Batch pruning for multiple layers
    - Memory-efficient weight copying
    - Optimized dependency tracking
    """
    
    def __init__(self, tracker: Optional[Any] = None) -> None:
        """
        Initialize the high-performance neuron pruner.
        
        Args:
            tracker: NeuronTracker instance for integration (optional)
        """
        self.tracker = tracker
        self._pruning_history: List[Dict[str, Any]] = []
        self._pruned_neurons: Dict[str, Set[int]] = defaultdict(set)
        self._original_model_state: Optional[Dict[str, torch.Tensor]] = None
        self._layer_cache: Dict[str, nn.Module] = {}  # Performance optimization
        self._dependency_graph: Dict[str, List[str]] = {}  # Track layer dependencies
        self.logger = logging.getLogger(__name__)
    
    def set_tracker(self, tracker: Any) -> None:
        """
        Set or update the tracker instance.
        
        Args:
            tracker: NeuronTracker instance
        """
        self.tracker = tracker
    
    def _validate_tracker(self) -> None:
        """Validate that tracker is available and has been used."""
        if self.tracker is None:
            warnings.warn("No tracker set. Some functionality may be limited.", UserWarning)
    
    @lru_cache(maxsize=128)
    def _get_layer_by_name(self, model_id: int, layer_name: str) -> Optional[nn.Module]:
        """
        Cached layer lookup for performance optimization.
        
        Args:
            model_id: Unique identifier for the model (id(model))
            layer_name: Name of the layer to retrieve
            
        Returns:
            The layer module or None if not found
        """
        cache_key = f"{model_id}_{layer_name}"
        if cache_key in self._layer_cache:
            return self._layer_cache[cache_key]
        
        # This is a placeholder - actual implementation would need the model reference
        return None
    
    def _get_layer_by_name_direct(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        Direct layer lookup without caching for immediate use.
        
        Args:
            model: PyTorch model
            layer_name: Name of the layer to retrieve
            
        Returns:
            The layer module or None if not found
        """
        try:
            # Handle nested layer names (e.g., "module.layer1.0")
            layer_parts = layer_name.split('.')
            current_module = model
            
            for part in layer_parts:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    # Try numeric indexing for sequential modules
                    try:
                        idx = int(part)
                        current_module = current_module[idx]
                    except (ValueError, IndexError, TypeError):
                        return None
            
            return current_module
            
        except Exception as e:
            self.logger.warning(f"Layer lookup failed for {layer_name}: {e}")
            return None
    
    def _create_pruned_layer(self, original_layer: nn.Module, neurons_to_keep: List[int]) -> nn.Module:
        """
        Create a new layer with specified neurons kept (others pruned).
        
        Args:
            original_layer: Original layer to prune
            neurons_to_keep: List of neuron indices to keep
            
        Returns:
            New layer with pruned neurons
        """
        if isinstance(original_layer, nn.Linear):
            return self._prune_linear_layer(original_layer, neurons_to_keep)
        elif isinstance(original_layer, nn.Conv2d):
            return self._prune_conv2d_layer(original_layer, neurons_to_keep)
        elif isinstance(original_layer, nn.Conv1d):
            return self._prune_conv1d_layer(original_layer, neurons_to_keep)
        elif isinstance(original_layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return self._prune_batchnorm_layer(original_layer, neurons_to_keep)
        else:
            raise ValueError(f"Unsupported layer type for pruning: {type(original_layer)}")
    
    def _prune_linear_layer(self, layer: nn.Linear, neurons_to_keep: List[int]) -> nn.Linear:
        """
        Prune a Linear layer by keeping specified output neurons.
        
        Args:
            layer: Original Linear layer
            neurons_to_keep: Indices of neurons to keep
            
        Returns:
            New Linear layer with pruned outputs
        """
        device = layer.weight.device
        dtype = layer.weight.dtype
        
        new_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=len(neurons_to_keep),
            bias=layer.bias is not None,
            device=device,
            dtype=dtype
        )
        
        # Efficiently copy weights using advanced indexing
        neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=device)
        new_layer.weight.data = layer.weight.data[neurons_to_keep_tensor].clone()
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[neurons_to_keep_tensor].clone()
        
        return new_layer
    
    def _prune_conv2d_layer(self, layer: nn.Conv2d, neurons_to_keep: List[int]) -> nn.Conv2d:
        """
        Prune a Conv2d layer by keeping specified output channels.
        
        Args:
            layer: Original Conv2d layer
            neurons_to_keep: Indices of channels to keep
            
        Returns:
            New Conv2d layer with pruned channels
        """
        device = layer.weight.device
        dtype = layer.weight.dtype
        
        new_layer = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=len(neurons_to_keep),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=min(layer.groups, len(neurons_to_keep)),  # Adjust groups if necessary
            bias=layer.bias is not None,
            device=device,
            dtype=dtype
        )
        
        # Efficiently copy weights
        neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=device)
        new_layer.weight.data = layer.weight.data[neurons_to_keep_tensor].clone()
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[neurons_to_keep_tensor].clone()
        
        return new_layer
    
    def _prune_conv1d_layer(self, layer: nn.Conv1d, neurons_to_keep: List[int]) -> nn.Conv1d:
        """
        Prune a Conv1d layer by keeping specified output channels.
        
        Args:
            layer: Original Conv1d layer
            neurons_to_keep: Indices of channels to keep
            
        Returns:
            New Conv1d layer with pruned channels
        """
        device = layer.weight.device
        dtype = layer.weight.dtype
        
        new_layer = nn.Conv1d(
            in_channels=layer.in_channels,
            out_channels=len(neurons_to_keep),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=min(layer.groups, len(neurons_to_keep)),
            bias=layer.bias is not None,
            device=device,
            dtype=dtype
        )
        
        neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=device)
        new_layer.weight.data = layer.weight.data[neurons_to_keep_tensor].clone()
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[neurons_to_keep_tensor].clone()
        
        return new_layer
    
    def _prune_batchnorm_layer(self, layer: nn.Module, neurons_to_keep: List[int]) -> nn.Module:
        """
        Prune a BatchNorm layer by keeping specified channels.
        
        Args:
            layer: Original BatchNorm layer (1d or 2d)
            neurons_to_keep: Indices of channels to keep
            
        Returns:
            New BatchNorm layer with pruned channels
        """
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        
        if isinstance(layer, nn.BatchNorm1d):
            new_layer = nn.BatchNorm1d(
                num_features=len(neurons_to_keep),
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                device=device,
                dtype=dtype
            )
        elif isinstance(layer, nn.BatchNorm2d):
            new_layer = nn.BatchNorm2d(
                num_features=len(neurons_to_keep),
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError(f"Unsupported BatchNorm type: {type(layer)}")
        
        neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=device)
        
        # Copy parameters if they exist
        if layer.affine:
            new_layer.weight.data = layer.weight.data[neurons_to_keep_tensor].clone()
            new_layer.bias.data = layer.bias.data[neurons_to_keep_tensor].clone()
        
        if layer.track_running_stats:
            new_layer.running_mean.data = layer.running_mean.data[neurons_to_keep_tensor].clone()
            new_layer.running_var.data = layer.running_var.data[neurons_to_keep_tensor].clone()
            new_layer.num_batches_tracked.data = layer.num_batches_tracked.data.clone()
        
        return new_layer
    
    def _build_dependency_graph(self, model: nn.Module) -> Dict[str, List[str]]:
        """
        Build a dependency graph of layers for efficient update ordering.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping layer names to their dependent layers
        """
        if hasattr(self, '_cached_dependency_graph'):
            return self._cached_dependency_graph
        
        dependency_graph = defaultdict(list)
        layer_names = [name for name, _ in model.named_modules() 
                      if isinstance(_, (nn.Linear, nn.Conv2d, nn.Conv1d))]
        
        # Build sequential dependencies (simplified heuristic)
        for i, layer_name in enumerate(layer_names[:-1]):
            next_layer_name = layer_names[i + 1]
            dependency_graph[layer_name].append(next_layer_name)
        
        self._cached_dependency_graph = dict(dependency_graph)
        return self._cached_dependency_graph
    
    def prune_neurons_by_ids(self, model: nn.Module, neuron_ids_to_prune: List[int], 
                           layer_mapping: Optional[Dict[int, Tuple[str, int]]] = None,
                           dry_run: bool = True) -> Dict[str, Any]:
        """
        Prune neurons by their global IDs with optimized batch processing.
        
        Args:
            model: PyTorch model to prune
            neuron_ids_to_prune: List of global neuron IDs to prune
            layer_mapping: Mapping from global neuron IDs to (layer_name, local_index)
            dry_run: If True, simulate pruning without modifying the model
            
        Returns:
            Dictionary with pruning results and statistics
        """
        if not neuron_ids_to_prune:
            return {"status": "no_action", "message": "No neurons to prune"}
        
        # Build layer mapping if not provided
        if layer_mapping is None:
            layer_mapping = self._build_layer_mapping(model)
        
        # Group neurons by layer for batch processing
        layer_pruning_plan = defaultdict(list)
        for neuron_id in neuron_ids_to_prune:
            if neuron_id in layer_mapping:
                layer_name, local_index = layer_mapping[neuron_id]
                layer_pruning_plan[layer_name].append(local_index)
        
        if dry_run:
            return self._simulate_pruning(layer_pruning_plan)
        else:
            return self._execute_pruning(model, layer_pruning_plan)
    
    def _build_layer_mapping(self, model: nn.Module) -> Dict[int, Tuple[str, int]]:
        """
        Build mapping from global neuron IDs to (layer_name, local_index).
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping global neuron IDs to layer information
        """
        layer_mapping = {}
        global_neuron_id = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                num_neurons = module.out_features
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                num_neurons = module.out_channels
            else:
                continue
            
            for local_idx in range(num_neurons):
                layer_mapping[global_neuron_id] = (name, local_idx)
                global_neuron_id += 1
        
        return layer_mapping
    
    def _simulate_pruning(self, layer_pruning_plan: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Simulate pruning operation for performance and validation testing.
        
        Args:
            layer_pruning_plan: Dictionary mapping layer names to lists of neuron indices to prune
            
        Returns:
            Simulation results
        """
        total_neurons_to_prune = sum(len(neurons) for neurons in layer_pruning_plan.values())
        
        results = {
            "status": "simulation",
            "neurons_pruned": total_neurons_to_prune,
            "layers_affected": len(layer_pruning_plan),
            "layer_modifications": {},
            "simulation_time": 0.0
        }
        
        for layer_name, neurons_to_prune in layer_pruning_plan.items():
            results["layer_modifications"][layer_name] = {
                "neurons_removed": len(neurons_to_prune),
                "neurons_to_remove": sorted(neurons_to_prune),
                "pruning_action": "simulated"
            }
        
        return results
    
    def _execute_pruning(self, model: nn.Module, layer_pruning_plan: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Execute actual pruning with optimized batch processing.
        
        Args:
            model: PyTorch model to modify
            layer_pruning_plan: Dictionary mapping layer names to neuron indices to prune
            
        Returns:
            Pruning execution results
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Store original state for potential restoration
        if self._original_model_state is None:
            self._original_model_state = {name: param.clone() for name, param in model.state_dict().items()}
        
        results = {
            "status": "completed",
            "neurons_pruned": 0,
            "layers_affected": 0,
            "layer_modifications": {},
            "pruning_time": 0.0
        }
        
        layers_needing_update = {}
        
        # Process each layer in the pruning plan
        for layer_name, neurons_to_prune in layer_pruning_plan.items():
            if not neurons_to_prune:
                continue
            
            layer = self._get_layer_by_name_direct(model, layer_name)
            if layer is None:
                self.logger.warning(f"Layer {layer_name} not found, skipping")
                continue
            
            # Calculate neurons to keep (more efficient than tracking removals)
            if isinstance(layer, nn.Linear):
                total_neurons = layer.out_features
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                total_neurons = layer.out_channels
            else:
                self.logger.warning(f"Unsupported layer type {type(layer)} for {layer_name}")
                continue
            
            neurons_to_keep = [i for i in range(total_neurons) if i not in neurons_to_prune]
            
            if not neurons_to_keep:
                self.logger.warning(f"Cannot prune all neurons from layer {layer_name}")
                continue
            
            # Create pruned layer
            try:
                new_layer = self._create_pruned_layer(layer, neurons_to_keep)
                self._replace_layer_in_model(model, layer_name, new_layer)
                
                # Track pruning for dependent layer updates
                neurons_removed = len(neurons_to_prune)
                layers_needing_update[layer_name] = neurons_removed
                
                # Update statistics
                results["neurons_pruned"] += neurons_removed
                results["layers_affected"] += 1
                results["layer_modifications"][layer_name] = {
                    "neurons_removed": neurons_removed,
                    "neurons_remaining": len(neurons_to_keep),
                    "pruning_action": "executed"
                }
                
                # Track pruned neurons for this session
                self._pruned_neurons[layer_name].update(neurons_to_prune)
                
            except Exception as e:
                self.logger.error(f"Failed to prune layer {layer_name}: {e}")
                continue
        
        # Update dependent layers efficiently
        if layers_needing_update:
            self._update_dependent_layers(model, layers_needing_update)
        
        # Record timing
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            results["pruning_time"] = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        # Add to pruning history
        self._pruning_history.append({
            "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            "layer_modifications": results["layer_modifications"].copy(),
            "total_neurons_pruned": results["neurons_pruned"]
        })
        
        return results
    
    def _replace_layer_in_model(self, model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
        """
        Replace a layer in the model with optimized attribute setting.
        
        Args:
            model: PyTorch model
            layer_name: Name of the layer to replace
            new_layer: New layer to install
        """
        layer_parts = layer_name.split('.')
        parent_module = model
        
        # Navigate to parent module
        for part in layer_parts[:-1]:
            if hasattr(parent_module, part):
                parent_module = getattr(parent_module, part)
            else:
                try:
                    idx = int(part)
                    parent_module = parent_module[idx]
                except (ValueError, IndexError, TypeError):
                    raise ValueError(f"Cannot navigate to layer {layer_name}")
        
        # Replace the final layer
        final_part = layer_parts[-1]
        if hasattr(parent_module, final_part):
            setattr(parent_module, final_part, new_layer)
        else:
            try:
                idx = int(final_part)
                parent_module[idx] = new_layer
            except (ValueError, IndexError, TypeError):
                raise ValueError(f"Cannot replace layer {layer_name}")
    
    def _update_dependent_layers(self, model: nn.Module, layers_pruned: Dict[str, int]) -> None:
        """
        Update layers that depend on pruned layers with optimized batch processing.
        
        Args:
            model: PyTorch model
            layers_pruned: Dictionary mapping pruned layer names to number of neurons removed
        """
        dependency_graph = self._build_dependency_graph(model)
        
        for pruned_layer_name, neurons_removed in layers_pruned.items():
            dependent_layers = dependency_graph.get(pruned_layer_name, [])
            
            for dependent_layer_name in dependent_layers:
                dependent_layer = self._get_layer_by_name_direct(model, dependent_layer_name)
                if dependent_layer and self._layer_needs_input_update(dependent_layer):
                    self._update_layer_input_dim(model, dependent_layer_name, dependent_layer, neurons_removed)
    
    def _layer_needs_input_update(self, layer: nn.Module) -> bool:
        """Check if layer type requires input dimension updates after pruning."""
        return isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv1d))
    
    def _update_layer_input_dim(self, model: nn.Module, layer_name: str, 
                               layer: nn.Module, input_reduction: int) -> None:
        """
        Update a layer's input dimensions after upstream pruning.
        
        Args:
            model: PyTorch model
            layer_name: Name of the layer to update
            layer: The layer module to update
            input_reduction: Number of input features/channels removed
        """
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        
        if isinstance(layer, nn.Linear):
            new_in_features = layer.in_features - input_reduction
            if new_in_features <= 0:
                self.logger.warning(f"Cannot reduce input features to {new_in_features} for {layer_name}")
                return
            
            new_layer = nn.Linear(
                in_features=new_in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                device=device,
                dtype=dtype
            )
            
            # Copy weights (truncate input dimension)
            new_layer.weight.data = layer.weight.data[:, :new_in_features].clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            
            self._replace_layer_in_model(model, layer_name, new_layer)
            
        elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            new_in_channels = layer.in_channels - input_reduction
            if new_in_channels <= 0:
                self.logger.warning(f"Cannot reduce input channels to {new_in_channels} for {layer_name}")
                return
            
            if isinstance(layer, nn.Conv2d):
                new_layer = nn.Conv2d(
                    in_channels=new_in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=min(layer.groups, new_in_channels),
                    bias=layer.bias is not None,
                    device=device,
                    dtype=dtype
                )
            else:  # Conv1d
                new_layer = nn.Conv1d(
                    in_channels=new_in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=min(layer.groups, new_in_channels),
                    bias=layer.bias is not None,
                    device=device,
                    dtype=dtype
                )
            
            # Copy weights (truncate input channel dimension)
            new_layer.weight.data = layer.weight.data[:, :new_in_channels].clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            
            self._replace_layer_in_model(model, layer_name, new_layer)
    
    def prune_from_json_recommendations(self, model: nn.Module, 
                                      json_path: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Prune model based on JSON recommendations with optimized loading.
        
        Args:
            model: PyTorch model to prune
            json_path: Path to JSON file with pruning recommendations
            dry_run: If True, simulate pruning without modifying model
            
        Returns:
            Pruning results dictionary
        """
        try:
            with open(json_path, 'r') as f:
                recommendations = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load recommendations: {e}"}
        
        # Extract pruning candidates
        prune_candidates = recommendations.get("prune", [])
        if not prune_candidates:
            return {"status": "no_action", "message": "No pruning candidates found"}
        
        # Build efficient neuron ID list
        neuron_ids_to_prune = [candidate.get("neuron_id") for candidate in prune_candidates
                              if candidate.get("neuron_id") is not None]
        
        if not neuron_ids_to_prune:
            return {"status": "error", "message": "No valid neuron IDs found in recommendations"}
        
        # Execute pruning
        return self.prune_neurons_by_ids(model, neuron_ids_to_prune, dry_run=dry_run)
    
    def restore_model(self, model: nn.Module) -> bool:
        """
        Restore model to its original state before pruning.
        
        Args:
            model: PyTorch model to restore
            
        Returns:
            True if restoration successful, False otherwise
        """
        if self._original_model_state is None:
            self.logger.warning("No original model state saved for restoration")
            return False
        
        try:
            model.load_state_dict(self._original_model_state)
            self._pruned_neurons.clear()
            self._layer_cache.clear()
            self.logger.info("Model restored to original state")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore model: {e}")
            return False
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all pruning operations.
        
        Returns:
            Dictionary with pruning statistics and history
        """
        total_neurons_pruned = sum(len(neurons) for neurons in self._pruned_neurons.values())
        
        return {
            "total_neurons_pruned": total_neurons_pruned,
            "layers_affected": len(self._pruned_neurons),
            "pruning_sessions": len(self._pruning_history),
            "layer_details": {
                layer_name: {
                    "neurons_pruned": len(neurons),
                    "pruned_indices": sorted(list(neurons))
                }
                for layer_name, neurons in self._pruned_neurons.items()
            },
            "pruning_history": self._pruning_history.copy()
        }
    
    def clear_cache(self) -> None:
        """Clear internal caches to free memory."""
        self._layer_cache.clear()
        if hasattr(self, '_cached_dependency_graph'):
            delattr(self, '_cached_dependency_graph')
        self._get_layer_by_name.cache_clear()