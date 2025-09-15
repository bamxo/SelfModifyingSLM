"""
Filter-Based Structured Pruning

Implements intelligent filter/channel removal based on neuron tracker recommendations.
Maintains architectural validity by removing entire filters and adjusting dependent layers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import logging
import numpy as np


class FilterPruner:
    """
    Filter-based structured pruning that removes entire filters/channels.
    
    This approach maintains architectural validity and prevents dimension mismatches
    by ensuring that when filters are removed from a layer, all dependent layers
    are automatically adjusted to match the new dimensions.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.pruning_history = []
        
    def convert_neuron_recommendations_to_filters(self, 
                                                 recommendations: Dict[str, Any], 
                                                 model: nn.Module) -> Dict[str, List[int]]:
        """
        Convert neuron tracker recommendations to filter-level pruning plan.
        
        Args:
            recommendations: Neuron tracker recommendations
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping layer names to lists of filter indices to remove
        """
        filter_plan = {}
        
        # Extract pruning candidates from neuron tracker
        prune_candidates = recommendations.get("prune", [])
        if not prune_candidates:
            return filter_plan
        
        # Group candidates by layer
        layer_candidates = defaultdict(list)
        for candidate in prune_candidates:
            layer_name = candidate.get("layer_name")
            local_index = candidate.get("local_index")
            
            if layer_name and local_index is not None:
                # For conv layers, local_index corresponds to output channel/filter
                layer_candidates[layer_name].append(local_index)
        
        # Convert to filter removal plan
        for layer_name, neuron_indices in layer_candidates.items():
            layer = self._get_layer_by_name(model, layer_name)
            
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                # For conv layers, each neuron corresponds to an output filter
                filters_to_remove = sorted(set(neuron_indices))
                
                # Validate filter indices
                max_filters = layer.out_channels
                valid_filters = [f for f in filters_to_remove if 0 <= f < max_filters]
                
                if valid_filters:
                    # Don't remove all filters
                    if len(valid_filters) < max_filters:
                        filter_plan[layer_name] = valid_filters
                        self.logger.info(f"Planning to remove {len(valid_filters)} filters from {layer_name}")
                    else:
                        self.logger.warning(f"Cannot remove all {max_filters} filters from {layer_name}")
            
            elif isinstance(layer, nn.Linear):
                # For linear layers, treat as neuron pruning (existing logic works)
                neurons_to_remove = sorted(set(neuron_indices))
                max_neurons = layer.out_features
                valid_neurons = [n for n in neurons_to_remove if 0 <= n < max_neurons]
                
                if valid_neurons and len(valid_neurons) < max_neurons:
                    filter_plan[layer_name] = valid_neurons
                    self.logger.info(f"Planning to remove {len(valid_neurons)} neurons from {layer_name}")
        
        return filter_plan
    
    def prune_filters_structured(self, 
                                model: nn.Module, 
                                filter_plan: Dict[str, List[int]], 
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute filter-based structured pruning.
        
        Args:
            model: PyTorch model to prune
            filter_plan: Dictionary mapping layer names to filter indices to remove
            dry_run: If True, simulate without modifying model
            
        Returns:
            Pruning results with statistics
        """
        if dry_run:
            return self._simulate_filter_pruning(model, filter_plan)
        
        self.logger.info("Starting filter-based structured pruning")
        
        results = {
            "status": "completed",
            "strategy": "filter_structured",
            "filters_removed": 0,
            "parameters_removed": 0,
            "layers_modified": 0,
            "layer_details": {},
            "dependency_updates": []
        }
        
        # Build dependency graph for the model
        dependency_graph = self._build_filter_dependency_graph(model)
        
        # Special handling for ResNet: coordinate skip connection pruning
        if self._is_resnet_model(model):
            original_plan = filter_plan.copy()
            filter_plan = self._coordinate_resnet_pruning(model, filter_plan)
            
            # CRITICAL: Validate coordination worked properly
            coordination_valid = self._validate_resnet_coordination(model, filter_plan)
            if not coordination_valid:
                self.logger.error("ResNet coordination validation failed - disabling ALL ResNet block pruning for safety")
                # Remove ALL ResNet block layers from pruning plan to prevent dimension mismatches
                filter_plan = {k: v for k, v in filter_plan.items() 
                              if not self._is_in_resnet_block(k)}
                self.logger.info(f"Fallback: Pruning plan reduced to {len(filter_plan)} non-ResNet layers")
        
        # Sort layers to handle in proper order (avoid dependency conflicts)
        sorted_layers = self._topological_sort_layers(model, filter_plan.keys())
        
        # Track channel changes for dependency updates
        channel_changes = {}  # layer_name -> (original_channels, new_channels)
        
        for layer_name in sorted_layers:
            if layer_name not in filter_plan:
                continue
                
            filters_to_remove = filter_plan[layer_name]
            if not filters_to_remove:
                continue
            
            layer = self._get_layer_by_name(model, layer_name)
            if layer is None:
                self.logger.warning(f"Layer {layer_name} not found, skipping")
                continue
            
            # Prune the layer
            pruning_result = self._prune_layer_filters(model, layer_name, layer, filters_to_remove)
            
            if pruning_result["success"]:
                # Update results
                results["filters_removed"] += pruning_result["filters_removed"]
                results["parameters_removed"] += pruning_result["parameters_removed"]
                results["layers_modified"] += 1
                results["layer_details"][layer_name] = pruning_result
                
                # Track channel changes for dependency updates
                if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                    original_channels = layer.out_channels
                    new_channels = original_channels - len(filters_to_remove)
                    channel_changes[layer_name] = (original_channels, new_channels)
        
        # Update dependent layers
        dependency_updates = self._update_dependent_layers_structured(
            model, channel_changes, dependency_graph
        )
        results["dependency_updates"] = dependency_updates
        
        # Final model validation
        validation_result = self._validate_model_structure(model)
        results["validation"] = validation_result
        
        # Store in history
        self.pruning_history.append({
            "type": "filter_structured",
            "results": results.copy(),
            "filter_plan": filter_plan.copy()
        })
        
        return results
    
    def _simulate_filter_pruning(self, 
                                model: nn.Module, 
                                filter_plan: Dict[str, List[int]]) -> Dict[str, Any]:
        """Simulate filter pruning to estimate results."""
        results = {
            "status": "simulation",
            "strategy": "filter_structured",
            "filters_removed": 0,
            "parameters_removed": 0,
            "layers_modified": len(filter_plan),
            "layer_details": {}
        }
        
        for layer_name, filters_to_remove in filter_plan.items():
            layer = self._get_layer_by_name(model, layer_name)
            if layer is None:
                continue
            
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                # Calculate parameter reduction
                if isinstance(layer, nn.Conv2d):
                    # Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
                    params_per_filter = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                else:  # Conv1d
                    # Conv1d: [out_channels, in_channels, kernel_size]
                    params_per_filter = layer.in_channels * layer.kernel_size[0]
                
                if layer.bias is not None:
                    params_per_filter += 1  # Add bias parameter
                
                filters_removed = len(filters_to_remove)
                params_removed = filters_removed * params_per_filter
                
                results["filters_removed"] += filters_removed
                results["parameters_removed"] += params_removed
                results["layer_details"][layer_name] = {
                    "filters_removed": filters_removed,
                    "parameters_removed": params_removed,
                    "original_filters": layer.out_channels,
                    "remaining_filters": layer.out_channels - filters_removed
                }
            
            elif isinstance(layer, nn.Linear):
                # Linear layer neuron removal
                neurons_removed = len(filters_to_remove)
                params_removed = neurons_removed * layer.in_features
                if layer.bias is not None:
                    params_removed += neurons_removed
                
                results["parameters_removed"] += params_removed
                results["layer_details"][layer_name] = {
                    "neurons_removed": neurons_removed,
                    "parameters_removed": params_removed,
                    "original_neurons": layer.out_features,
                    "remaining_neurons": layer.out_features - neurons_removed
                }
        
        return results
    
    def _prune_layer_filters(self, 
                           model: nn.Module, 
                           layer_name: str, 
                           layer: nn.Module, 
                           filters_to_remove: List[int]) -> Dict[str, Any]:
        """Remove specific filters from a layer."""
        result = {
            "success": False,
            "filters_removed": 0,
            "parameters_removed": 0,
            "error": None
        }
        
        try:
            if isinstance(layer, nn.Conv2d):
                new_layer = self._create_pruned_conv2d(layer, filters_to_remove)
            elif isinstance(layer, nn.Conv1d):
                new_layer = self._create_pruned_conv1d(layer, filters_to_remove)
            elif isinstance(layer, nn.Linear):
                new_layer = self._create_pruned_linear(layer, filters_to_remove)
            else:
                result["error"] = f"Unsupported layer type: {type(layer)}"
                return result
            
            # Replace layer in model
            self._replace_layer_in_model(model, layer_name, new_layer)
            
            # Calculate statistics
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                result["filters_removed"] = len(filters_to_remove)
                if isinstance(layer, nn.Conv2d):
                    params_per_filter = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                else:
                    params_per_filter = layer.in_channels * layer.kernel_size[0]
                
                if layer.bias is not None:
                    params_per_filter += 1
                
                result["parameters_removed"] = len(filters_to_remove) * params_per_filter
            
            elif isinstance(layer, nn.Linear):
                result["filters_removed"] = len(filters_to_remove)  # neurons in this case
                params_removed = len(filters_to_remove) * layer.in_features
                if layer.bias is not None:
                    params_removed += len(filters_to_remove)
                result["parameters_removed"] = params_removed
            
            result["success"] = True
            self.logger.info(f"Successfully pruned {len(filters_to_remove)} filters from {layer_name}")
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Failed to prune filters from {layer_name}: {e}")
        
        return result
    
    def _create_pruned_conv2d(self, layer: nn.Conv2d, filters_to_remove: List[int]) -> nn.Conv2d:
        """Create new Conv2d layer with specified filters removed."""
        filters_to_keep = [i for i in range(layer.out_channels) if i not in filters_to_remove]
        
        new_layer = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=len(filters_to_keep),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=min(layer.groups, len(filters_to_keep)),
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            device=layer.weight.device,
            dtype=layer.weight.dtype
        )
        
        # Copy weights for kept filters
        filters_to_keep_tensor = torch.tensor(filters_to_keep, device=layer.weight.device)
        new_layer.weight.data = layer.weight.data[filters_to_keep_tensor].clone()
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[filters_to_keep_tensor].clone()
        
        return new_layer
    
    def _create_pruned_conv1d(self, layer: nn.Conv1d, filters_to_remove: List[int]) -> nn.Conv1d:
        """Create new Conv1d layer with specified filters removed."""
        filters_to_keep = [i for i in range(layer.out_channels) if i not in filters_to_remove]
        
        new_layer = nn.Conv1d(
            in_channels=layer.in_channels,
            out_channels=len(filters_to_keep),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=min(layer.groups, len(filters_to_keep)),
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            device=layer.weight.device,
            dtype=layer.weight.dtype
        )
        
        # Copy weights for kept filters
        filters_to_keep_tensor = torch.tensor(filters_to_keep, device=layer.weight.device)
        new_layer.weight.data = layer.weight.data[filters_to_keep_tensor].clone()
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[filters_to_keep_tensor].clone()
        
        return new_layer
    
    def _create_pruned_linear(self, layer: nn.Linear, neurons_to_remove: List[int]) -> nn.Linear:
        """Create new Linear layer with specified neurons removed."""
        neurons_to_keep = [i for i in range(layer.out_features) if i not in neurons_to_remove]
        
        new_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=len(neurons_to_keep),
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype
        )
        
        # Copy weights for kept neurons
        neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=layer.weight.device)
        new_layer.weight.data = layer.weight.data[neurons_to_keep_tensor].clone()
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[neurons_to_keep_tensor].clone()
        
        return new_layer
    
    def _build_filter_dependency_graph(self, model: nn.Module) -> Dict[str, List[str]]:
        """
        Build dependency graph with comprehensive forward pass analysis.
        This captures ALL layer dependencies including complex architectures.
        """
        dependency_graph = defaultdict(list)
        
        try:
            # Use comprehensive forward analysis for all architectures
            self.logger.info("Using comprehensive forward pass dependency analysis")
            dependency_graph = self._build_comprehensive_dependency_graph(model)
            
            # Add ResNet-specific skip connection handling if applicable
            if self._is_resnet_model(model):
                self.logger.info("Adding ResNet-specific skip connection coordination")
                resnet_deps = self._build_resnet_skip_dependencies(model)
                # Merge ResNet skip dependencies
                for source, targets in resnet_deps.items():
                    dependency_graph[source].extend(targets)
                    dependency_graph[source] = list(set(dependency_graph[source]))  # Remove duplicates
        
        except Exception as e:
            self.logger.warning(f"Comprehensive analysis failed: {e}. Using fallback approach.")
            dependency_graph = self._build_conservative_dependency_graph(model)
        
        self.logger.info(f"Built dependency graph with {len(dependency_graph)} source layers")
        return dict(dependency_graph)
    
    def _is_resnet_model(self, model: nn.Module) -> bool:
        """Check if this is a ResNet model."""
        # Look for ResNet-specific patterns
        layer_names = [name for name, _ in model.named_modules()]
        resnet_patterns = ['layer1', 'layer2', 'layer3', 'layer4', 'downsample']
        
        return any(pattern in ' '.join(layer_names) for pattern in resnet_patterns)
    
    def _build_comprehensive_dependency_graph(self, model: nn.Module) -> Dict[str, List[str]]:
        """
        Build comprehensive dependency graph using forward pass analysis.
        Captures all conv/linear → conv/linear/batchnorm dependencies.
        """
        dependency_graph = defaultdict(list)
        
        # Get device for dummy input
        device = next(model.parameters()).device
        
        # Determine appropriate input size
        dummy_input = self._create_dummy_input(model, device)
        if dummy_input is None:
            return self._build_conservative_dependency_graph(model)
        
        # Track layer activations and their shapes
        layer_activations = {}
        layer_shapes = {}
        execution_order = []
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                execution_order.append(layer_name)
                if isinstance(output, torch.Tensor):
                    layer_shapes[layer_name] = output.shape
                    # Store input shapes for dependency analysis
                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        if isinstance(input[0], torch.Tensor):
                            layer_activations[layer_name] = input[0].shape
            return hook_fn
        
        # Register hooks for conv, linear, and batchnorm layers
        hooks = []
        target_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                target_layers[name] = module
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        try:
            # Run forward pass
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        # Build dependencies based on shape matching and execution order
        conv_linear_layers = {name: module for name, module in target_layers.items() 
                             if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear))}
        
        for i, source_name in enumerate(execution_order):
            if source_name not in conv_linear_layers:
                continue
                
            source_module = conv_linear_layers[source_name]
            source_output_shape = layer_shapes.get(source_name)
            
            if source_output_shape is None:
                continue
            
            # Find layers that use this layer's output
            for j in range(i + 1, len(execution_order)):
                target_name = execution_order[j]
                target_input_shape = layer_activations.get(target_name)
                
                if target_input_shape is None:
                    continue
                
                # Check if shapes match (indicating dependency)
                if self._shapes_indicate_dependency(source_output_shape, target_input_shape):
                    target_module = target_layers.get(target_name)
                    
                    # Only add valid dependencies
                    if isinstance(target_module, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                        dependency_graph[source_name].append(target_name)
                        
                        # For conv/linear layers, usually only the immediate next dependency matters
                        if isinstance(target_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                            break
        
        return dict(dependency_graph)
    
    def _create_dummy_input(self, model: nn.Module, device: torch.device) -> torch.Tensor:
        """Create appropriate dummy input for the model."""
        # Find first conv/linear layer to determine input requirements
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Assume CIFAR-10 like input for ResNet
                return torch.randn(1, module.in_channels, 32, 32, device=device)
            elif isinstance(module, nn.Conv1d):
                return torch.randn(1, module.in_channels, 100, device=device)
            elif isinstance(module, nn.Linear):
                return torch.randn(1, module.in_features, device=device)
        
        return None
    
    def _shapes_indicate_dependency(self, output_shape: torch.Size, input_shape: torch.Size) -> bool:
        """Check if output and input shapes indicate a dependency."""
        if len(output_shape) != len(input_shape):
            return False
        
        # For conv layers: [batch, channels, height, width]
        # For linear layers: [batch, features]
        
        if len(output_shape) >= 2:
            # Check if channel/feature dimension matches
            return output_shape[1] == input_shape[1] and output_shape[0] == input_shape[0]
        
        return False
    
    def _build_resnet_skip_dependencies(self, model: nn.Module) -> Dict[str, List[str]]:
        """Build ResNet-specific skip connection dependencies."""
        dependency_graph = defaultdict(list)
        all_modules = dict(model.named_modules())
        
        # Handle skip connections in ResNet blocks
        for layer_name, layer_module in all_modules.items():
            if isinstance(layer_module, (nn.Conv2d, nn.Conv1d)) and self._is_in_resnet_block(layer_name):
                skip_dependent_layers = self._find_skip_dependent_layers(layer_name, all_modules)
                dependency_graph[layer_name].extend(skip_dependent_layers)
        
        return dict(dependency_graph)
    
    def _build_resnet_dependency_graph(self, model: nn.Module) -> Dict[str, List[str]]:
        """Build dependency graph specifically for ResNet architecture."""
        dependency_graph = defaultdict(list)
        all_modules = dict(model.named_modules())
        
        # Get all conv layers in order to map forward dependencies
        conv_layers = [(name, module) for name, module in all_modules.items() 
                       if isinstance(module, (nn.Conv2d, nn.Conv1d))]
        
        for layer_name, layer_module in conv_layers:
            
            # 1. Find direct BatchNorm dependencies
            bn_name = self._find_corresponding_batchnorm(layer_name, all_modules)
            if bn_name:
                dependency_graph[layer_name].append(bn_name)
            
            # 2. Find forward path conv→conv dependencies 
            forward_dependent_convs = self._find_forward_conv_dependencies(layer_name, conv_layers)
            dependency_graph[layer_name].extend(forward_dependent_convs)
            
            # 3. For ResNet blocks, handle skip connection constraints
            if self._is_in_resnet_block(layer_name):
                skip_dependent_layers = self._find_skip_dependent_layers(layer_name, all_modules)
                dependency_graph[layer_name].extend(skip_dependent_layers)
        
        return dict(dependency_graph)
    
    def _find_corresponding_batchnorm(self, conv_name: str, all_modules: Dict[str, nn.Module]) -> str:
        """Find the BatchNorm layer that directly follows a conv layer."""
        # Common patterns: conv1 -> bn1, conv2 -> bn2
        if 'conv1' in conv_name:
            bn_name = conv_name.replace('conv1', 'bn1')
        elif 'conv2' in conv_name:
            bn_name = conv_name.replace('conv2', 'bn2')
        elif 'downsample.0' in conv_name:
            bn_name = conv_name.replace('downsample.0', 'downsample.1')
        else:
            return None
        
        if bn_name in all_modules and isinstance(all_modules[bn_name], (nn.BatchNorm1d, nn.BatchNorm2d)):
            return bn_name
        
        return None
    
    def _find_forward_conv_dependencies(self, source_layer: str, conv_layers: List[Tuple[str, nn.Module]]) -> List[str]:
        """
        Find conv layers that directly depend on the source layer's output.
        This is CRITICAL for ResNet where conv1 → layer1.0.conv1 connections exist.
        """
        dependent_convs = []
        
        # ResNet-specific forward dependencies
        if source_layer.endswith("conv1") and ("resnet.conv1" in source_layer or source_layer == "resnet.conv1"):
            # resnet.conv1 feeds into resnet.layer1.0.conv1
            dependent_convs.append("resnet.layer1.0.conv1")
            
        elif "layer1" in source_layer and "conv2" in source_layer:
            # layer1.X.conv2 feeds into layer2.0.conv1 (if it exists)
            if any("layer2.0.conv1" in name for name, _ in conv_layers):
                dependent_convs.append("resnet.layer2.0.conv1")
                
        elif "layer2" in source_layer and "conv2" in source_layer:
            # layer2.X.conv2 feeds into layer3.0.conv1 (if it exists)
            if any("layer3.0.conv1" in name for name, _ in conv_layers):
                dependent_convs.append("resnet.layer3.0.conv1")
                
        elif "layer3" in source_layer and "conv2" in source_layer:
            # layer3.X.conv2 feeds into layer4.0.conv1 (if it exists)
            if any("layer4.0.conv1" in name for name, _ in conv_layers):
                dependent_convs.append("resnet.layer4.0.conv1")
        
        # Within-block dependencies: conv1 → conv2
        if "conv1" in source_layer and not "downsample" in source_layer:
            # Replace conv1 with conv2 in the same block
            conv2_name = source_layer.replace("conv1", "conv2")
            if any(conv2_name == name for name, _ in conv_layers):
                dependent_convs.append(conv2_name)
        
        # Remove duplicates and filter valid layer names
        return list(set(dependent_convs))
    
    def _is_in_resnet_block(self, layer_name: str) -> bool:
        """Check if a layer is part of a ResNet block with skip connections."""
        # ResNet blocks are in layer1, layer2, layer3, layer4
        return any(block in layer_name for block in ['layer1', 'layer2', 'layer3', 'layer4'])
    
    def _find_skip_dependent_layers(self, layer_name: str, all_modules: Dict[str, nn.Module]) -> List[str]:
        """
        Find layers that are dependent due to ResNet skip connections.
        
        CRITICAL: In ResNet, if we prune channels from certain layers,
        we must also prune the same channels from layers in the skip path.
        """
        dependent_layers = []
        
        # Parse the layer name to understand the block structure
        # e.g., 'resnet.layer1.0.conv1' -> layer1, block 0, conv1
        parts = layer_name.split('.')
        if len(parts) < 4:
            return dependent_layers
        
        try:
            layer_group = parts[1]  # layer1, layer2, etc.
            block_idx = parts[2]    # 0, 1, etc.
            conv_type = parts[3]    # conv1, conv2, downsample
            
            # Skip connection rules for ResNet:
            # 1. If we prune conv2 in a block, we may need to adjust the skip connection
            # 2. If we prune downsample.0, we need to be very careful
            
            if conv_type == 'conv2':
                # Check if this block has a downsample (skip connection with projection)
                downsample_name = f"{parts[0]}.{layer_group}.{block_idx}.downsample.0"
                if downsample_name in all_modules:
                    # This block has a skip connection with projection
                    # We must prune the same channels from downsample
                    dependent_layers.append(downsample_name)
                    
                    # Also add the corresponding batch norm
                    downsample_bn = f"{parts[0]}.{layer_group}.{block_idx}.downsample.1"
                    if downsample_bn in all_modules:
                        dependent_layers.append(downsample_bn)
            
            elif 'downsample.0' in layer_name:
                # If we prune downsample, we need to prune the corresponding conv2
                conv2_name = f"{parts[0]}.{layer_group}.{block_idx}.conv2"
                if conv2_name in all_modules:
                    dependent_layers.append(conv2_name)
                    
                    # Also add corresponding batch norm
                    bn2_name = f"{parts[0]}.{layer_group}.{block_idx}.bn2"
                    if bn2_name in all_modules:
                        dependent_layers.append(bn2_name)
        
        except (IndexError, ValueError):
            # If parsing fails, return empty list
            pass
        
        return dependent_layers
    
    def _coordinate_resnet_pruning(self, model: nn.Module, filter_plan: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        Coordinate ResNet pruning to ensure skip connections remain valid.
        
        CRITICAL: In ResNet, ALL output channels within the same layer group must match
        because of residual additions between blocks. E.g., layer1.0.conv2 and layer1.1.conv2
        must have the same output channels as they both feed into layer2 connections.
        """
        coordinated_plan = filter_plan.copy()
        all_modules = dict(model.named_modules())
        
        # Group layers by ResNet layer group (layer1, layer2, layer3, layer4)
        layer_groups = {}  # layer_group -> {block_idx -> {'conv2': [], 'downsample': []}}
        
        for layer_name in filter_plan.keys():
            if self._is_in_resnet_block(layer_name):
                parts = layer_name.split('.')
                if len(parts) >= 4:
                    layer_group = parts[1]  # layer1, layer2, etc.
                    block_idx = parts[2]    # 0, 1, etc.
                    conv_type = parts[3]    # conv1, conv2, downsample
                    
                    if layer_group not in layer_groups:
                        layer_groups[layer_group] = {}
                    
                    block_key = f"{layer_group}.{block_idx}"
                    if block_key not in layer_groups[layer_group]:
                        layer_groups[layer_group][block_key] = {'conv2': [], 'downsample': []}
                    
                    if conv_type == 'conv2':
                        layer_groups[layer_group][block_key]['conv2'] = filter_plan[layer_name]
                    elif 'downsample.0' in layer_name:
                        layer_groups[layer_group][block_key]['downsample'] = filter_plan[layer_name]
        
        # Coordinate pruning within each layer group to ensure consistent output channels
        for layer_group, blocks_in_group in layer_groups.items():
            self.logger.info(f"Coordinating ResNet {layer_group} with {len(blocks_in_group)} blocks")
            
            # Collect all conv2 and downsample filters that need coordination
            all_conv2_filters = []
            all_downsample_filters = []
            conv2_layers = []
            downsample_layers = []
            
            for block_key, block_filters in blocks_in_group.items():
                if block_filters['conv2']:
                    all_conv2_filters.extend(block_filters['conv2'])
                    conv2_layers.append(f"resnet.{block_key}.conv2")
                
                if block_filters['downsample']:
                    all_downsample_filters.extend(block_filters['downsample'])
                    downsample_layers.append(f"resnet.{block_key}.downsample.0")
            
            # Find common filters that can be safely pruned across ALL layers in this group
            if all_conv2_filters or all_downsample_filters:
                all_filters_in_group = set(all_conv2_filters + all_downsample_filters)
                
                if all_conv2_filters and all_downsample_filters:
                    # Both conv2 and downsample layers exist - use intersection
                    common_filters = sorted(set(all_conv2_filters) & set(all_downsample_filters))
                elif all_conv2_filters:
                    # Only conv2 layers - check if any block has downsample
                    has_any_downsample = any(f"resnet.{block_key}.downsample.0" in all_modules 
                                           for block_key in blocks_in_group.keys())
                    if has_any_downsample:
                        # Cannot prune conv2 if any block has downsample that's not being pruned
                        self.logger.warning(f"Cannot prune conv2 in {layer_group} - some blocks have unpruned downsample")
                        common_filters = []
                    else:
                        common_filters = sorted(set(all_conv2_filters))
                else:
                    # Only downsample layers
                    common_filters = sorted(set(all_downsample_filters))
                
                if common_filters:
                    self.logger.info(f"Coordinating {layer_group}: pruning {len(common_filters)} common filters across {len(conv2_layers + downsample_layers)} layers")
                    
                    # Apply common filters to all layers in this group
                    for layer_name in conv2_layers + downsample_layers:
                        if layer_name in coordinated_plan:
                            coordinated_plan[layer_name] = common_filters
                else:
                    # No safe filters to prune - remove all layers in this group from plan
                    self.logger.warning(f"Cannot safely coordinate {layer_group} - removing all layers from pruning plan")
                    for layer_name in conv2_layers + downsample_layers:
                        coordinated_plan.pop(layer_name, None)
        
        return coordinated_plan
    
    def _validate_resnet_coordination(self, model: nn.Module, filter_plan: Dict[str, List[int]]) -> bool:
        """
        CRITICAL: Validate that ResNet coordination ensures skip connection compatibility.
        Returns False if any potential dimension mismatch is detected.
        """
        all_modules = dict(model.named_modules())
        
        # Group layers by ResNet layer groups
        layer_groups = {}  # layer_group -> {block_idx: {'conv2': filters, 'downsample': filters}}
        
        for layer_name, filters_to_remove in filter_plan.items():
            if self._is_in_resnet_block(layer_name):
                parts = layer_name.split('.')
                if len(parts) >= 4:
                    layer_group = parts[1]  # layer1, layer2, etc.
                    block_idx = parts[2]    # 0, 1, etc.
                    conv_type = parts[3]    # conv1, conv2, downsample
                    
                    if layer_group not in layer_groups:
                        layer_groups[layer_group] = {}
                    if block_idx not in layer_groups[layer_group]:
                        layer_groups[layer_group][block_idx] = {}
                    
                    if conv_type == 'conv2':
                        layer_groups[layer_group][block_idx]['conv2'] = set(filters_to_remove)
                    elif 'downsample.0' in layer_name:
                        layer_groups[layer_group][block_idx]['downsample'] = set(filters_to_remove)
        
        # Validate coordination within each layer group
        for layer_group, blocks in layer_groups.items():
            conv2_filter_sets = []
            downsample_filter_sets = []
            
            for block_idx, block_data in blocks.items():
                conv2_filters = block_data.get('conv2', set())
                downsample_filters = block_data.get('downsample', set())
                
                if conv2_filters:
                    conv2_filter_sets.append(conv2_filters)
                if downsample_filters:
                    downsample_filter_sets.append(downsample_filters)
                
                # CRITICAL CHECK: conv2 and downsample in same block must have identical filters
                if conv2_filters and downsample_filters:
                    if conv2_filters != downsample_filters:
                        self.logger.error(f"Coordination failed: {layer_group}.{block_idx} conv2 and downsample have different filters")
                        self.logger.error(f"  conv2: {sorted(conv2_filters)}")
                        self.logger.error(f"  downsample: {sorted(downsample_filters)}")
                        return False
            
            # CRITICAL CHECK: All conv2 layers within same layer group must have identical filters
            if len(conv2_filter_sets) > 1:
                reference_filters = conv2_filter_sets[0]
                for i, filter_set in enumerate(conv2_filter_sets[1:], 1):
                    if filter_set != reference_filters:
                        self.logger.error(f"Coordination failed: {layer_group} conv2 layers have different filters")
                        self.logger.error(f"  Block 0: {sorted(reference_filters)}")
                        self.logger.error(f"  Block {i}: {sorted(filter_set)}")
                        return False
        
        self.logger.info("ResNet coordination validation passed - all skip connections will be compatible")
        return True
    
    def _analyze_forward_dependencies(self, model: nn.Module) -> Dict[str, List[str]]:
        """Analyze model dependencies using forward pass tracing."""
        dependency_graph = defaultdict(list)
        
        # Create a dummy input to trace through the model
        device = next(model.parameters()).device
        
        # Determine input size from first conv layer
        first_conv = None
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                first_conv = module
                break
        
        if first_conv is None:
            return dict(dependency_graph)
        
        # Create appropriate dummy input
        if isinstance(first_conv, nn.Conv2d):
            dummy_input = torch.randn(1, first_conv.in_channels, 32, 32, device=device)
        else:
            dummy_input = torch.randn(1, first_conv.in_channels, 100, device=device)
        
        # Hook to track layer connections
        connections = {}
        layer_outputs = {}
        
        def hook_factory(name):
            def hook(module, input, output):
                # Record this layer's output shape
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = output.shape
                
                # Track which layers use this layer's output
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        # Find which layer produced this input
                        for producer_name, producer_shape in layer_outputs.items():
                            if (producer_name != name and 
                                inp.shape == producer_shape and
                                producer_name not in connections.get(name, [])):
                                
                                connections.setdefault(producer_name, []).append(name)
            return hook
        
        # Register hooks for all relevant layers
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                handle = module.register_forward_hook(hook_factory(name))
                handles.append(handle)
        
        # Run forward pass
        try:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()
        
        # Convert connections to dependency graph
        for source_layer, dependent_layers in connections.items():
            # Only include conv/linear -> conv/linear/batchnorm dependencies
            source_module = self._get_layer_by_name(model, source_layer)
            if isinstance(source_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                for dep_layer in dependent_layers:
                    dep_module = self._get_layer_by_name(model, dep_layer)
                    if isinstance(dep_module, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                        dependency_graph[source_layer].append(dep_layer)
        
        return dependency_graph
    
    def _build_conservative_dependency_graph(self, model: nn.Module) -> Dict[str, List[str]]:
        """Conservative approach: only map obvious conv->batchnorm connections."""
        dependency_graph = defaultdict(list)
        
        all_modules = list(model.named_modules())
        
        for i, (layer_name, layer_module) in enumerate(all_modules):
            if isinstance(layer_module, (nn.Conv2d, nn.Conv1d)):
                # Look for immediate BatchNorm after this conv
                for j in range(i + 1, min(i + 3, len(all_modules))):  # Check next 2 layers
                    next_name, next_module = all_modules[j]
                    if isinstance(next_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        # Simple heuristic: same parent path
                        if self._layers_are_connected(layer_name, next_name):
                            dependency_graph[layer_name].append(next_name)
                            break
        
        return dict(dependency_graph)
    
    def _layers_are_connected(self, layer1_name: str, layer2_name: str) -> bool:
        """Heuristic to determine if two layers are connected."""
        # Simple heuristic: if layer names suggest they're in sequence
        # This could be improved with actual forward pass analysis
        
        # Extract layer indices/numbers for comparison
        def extract_layer_info(name):
            parts = name.split('.')
            return parts
        
        parts1 = extract_layer_info(layer1_name)
        parts2 = extract_layer_info(layer2_name)
        
        # If they share a common parent (e.g., both in resnet.layer1)
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[:-1] == parts2[:-1]:  # Same parent module
                return True
        
        # Sequential naming patterns
        if len(parts1) >= 3 and len(parts2) >= 3:
            # For ResNet: resnet.layer1.0.conv1 -> resnet.layer1.0.bn1
            if parts1[:-1] == parts2[:-1]:
                return True
        
        return False
    
    def _topological_sort_layers(self, model: nn.Module, layer_names: List[str]) -> List[str]:
        """Sort layers in topological order to avoid dependency conflicts."""
        # Simple approach: sort by depth in module hierarchy
        def get_depth(name):
            return len(name.split('.'))
        
        return sorted(layer_names, key=get_depth)
    
    def _update_dependent_layers_structured(self, 
                                          model: nn.Module, 
                                          channel_changes: Dict[str, Tuple[int, int]], 
                                          dependency_graph: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Update layers that depend on pruned layers."""
        updates = []
        
        for pruned_layer, (original_channels, new_channels) in channel_changes.items():
            channels_removed = original_channels - new_channels
            dependent_layers = dependency_graph.get(pruned_layer, [])
            
            for dep_layer_name in dependent_layers:
                dep_layer = self._get_layer_by_name(model, dep_layer_name)
                if dep_layer is None:
                    continue
                
                update_result = self._update_dependent_layer(
                    model, dep_layer_name, dep_layer, channels_removed
                )
                
                if update_result["success"]:
                    updates.append({
                        "dependent_layer": dep_layer_name,
                        "pruned_layer": pruned_layer,
                        "channels_adjusted": channels_removed,
                        "layer_type": type(dep_layer).__name__
                    })
                    self.logger.info(f"Updated {dep_layer_name} input channels: -{channels_removed}")
        
        return updates
    
    def _update_dependent_layer(self, 
                              model: nn.Module, 
                              layer_name: str, 
                              layer: nn.Module, 
                              input_channel_reduction: int) -> Dict[str, Any]:
        """Update a single dependent layer to match reduced input channels."""
        result = {"success": False, "error": None}
        
        try:
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # BatchNorm: reduce num_features to match new input
                new_num_features = layer.num_features - input_channel_reduction
                if new_num_features <= 0:
                    result["error"] = f"Cannot reduce BatchNorm features to {new_num_features}"
                    return result
                
                # Create new BatchNorm with reduced features
                if isinstance(layer, nn.BatchNorm2d):
                    new_layer = nn.BatchNorm2d(
                        num_features=new_num_features,
                        eps=layer.eps,
                        momentum=layer.momentum,
                        affine=layer.affine,
                        track_running_stats=layer.track_running_stats,
                        device=next(layer.parameters()).device,
                        dtype=next(layer.parameters()).dtype
                    )
                else:  # BatchNorm1d
                    new_layer = nn.BatchNorm1d(
                        num_features=new_num_features,
                        eps=layer.eps,
                        momentum=layer.momentum,
                        affine=layer.affine,
                        track_running_stats=layer.track_running_stats,
                        device=next(layer.parameters()).device,
                        dtype=next(layer.parameters()).dtype
                    )
                
                # Copy parameters (keep first num_features)
                if layer.affine:
                    new_layer.weight.data = layer.weight.data[:new_num_features].clone()
                    new_layer.bias.data = layer.bias.data[:new_num_features].clone()
                
                if layer.track_running_stats:
                    new_layer.running_mean.data = layer.running_mean.data[:new_num_features].clone()
                    new_layer.running_var.data = layer.running_var.data[:new_num_features].clone()
                    new_layer.num_batches_tracked.data = layer.num_batches_tracked.data.clone()
                
                self._replace_layer_in_model(model, layer_name, new_layer)
                result["success"] = True
            
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                # Convolutional layer: reduce input channels
                new_in_channels = layer.in_channels - input_channel_reduction
                if new_in_channels <= 0:
                    result["error"] = f"Cannot reduce input channels to {new_in_channels}"
                    return result
                
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
                        padding_mode=layer.padding_mode,
                        device=layer.weight.device,
                        dtype=layer.weight.dtype
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
                        padding_mode=layer.padding_mode,
                        device=layer.weight.device,
                        dtype=layer.weight.dtype
                    )
                
                # Copy weights (keep first input channels)
                new_layer.weight.data = layer.weight.data[:, :new_in_channels].clone()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.clone()
                
                self._replace_layer_in_model(model, layer_name, new_layer)
                result["success"] = True
            
            elif isinstance(layer, nn.Linear):
                # Linear layer: reduce input features
                new_in_features = layer.in_features - input_channel_reduction
                if new_in_features <= 0:
                    result["error"] = f"Cannot reduce input features to {new_in_features}"
                    return result
                
                new_layer = nn.Linear(
                    in_features=new_in_features,
                    out_features=layer.out_features,
                    bias=layer.bias is not None,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype
                )
                
                # Copy weights (keep first input features)
                new_layer.weight.data = layer.weight.data[:, :new_in_features].clone()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.clone()
                
                self._replace_layer_in_model(model, layer_name, new_layer)
                result["success"] = True
            
            else:
                result["error"] = f"Unsupported layer type for dependency update: {type(layer)}"
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _validate_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """Validate model structure after pruning."""
        validation = {"status": "success", "issues": []}
        
        try:
            # Try a forward pass with dummy input
            model.eval()
            with torch.no_grad():
                # Determine input size based on model type
                if hasattr(model, 'resnet') or 'resnet' in str(model.__class__).lower():
                    # ResNet-like model expects image input
                    test_input = torch.randn(1, 3, 112, 112)
                else:
                    # Try to infer from first layer
                    first_layer = None
                    for module in model.modules():
                        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                            first_layer = module
                            break
                    
                    if isinstance(first_layer, nn.Conv2d):
                        test_input = torch.randn(1, first_layer.in_channels, 112, 112)
                    elif isinstance(first_layer, nn.Conv1d):
                        test_input = torch.randn(1, first_layer.in_channels, 100)
                    elif isinstance(first_layer, nn.Linear):
                        test_input = torch.randn(1, first_layer.in_features)
                    else:
                        test_input = torch.randn(1, 3, 112, 112)  # Default
                
                # Move to same device as model
                device = next(model.parameters()).device
                test_input = test_input.to(device)
                
                # Forward pass
                output = model(test_input)
                
                validation["output_shape"] = list(output.shape)
                validation["forward_pass"] = "success"
                
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"Forward pass failed: {str(e)}")
        
        return validation
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from model."""
        try:
            parts = layer_name.split('.')
            current = model
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    try:
                        idx = int(part)
                        current = current[idx]
                    except (ValueError, IndexError, TypeError):
                        return None
            
            return current
        except Exception:
            return None
    
    def _replace_layer_in_model(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """Replace layer in model."""
        parts = layer_name.split('.')
        parent = model
        
        # Navigate to parent
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                idx = int(part)
                parent = parent[idx]
        
        # Replace final layer
        final_part = parts[-1]
        if hasattr(parent, final_part):
            setattr(parent, final_part, new_layer)
        else:
            idx = int(final_part)
            parent[idx] = new_layer
