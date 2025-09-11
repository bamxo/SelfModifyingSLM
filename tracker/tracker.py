"""
Core Neuron Tracker

This module provides the fundamental functionality to enumerate and track all neurons 
in any PyTorch nn.Module, assigning unique IDs for tracking purposes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
import numpy as np
from collections import defaultdict


class NeuronTracker:
    """
    Core class to track and enumerate all neurons in a PyTorch neural network.
    
    This tracker assigns unique IDs to each neuron across all layers, enabling
    precise tracking during network modification operations.
    """
    
    def __init__(self):
        self.neuron_counter = 0
        self.layer_info = OrderedDict()
        self.neuron_map = {}  # Maps neuron_id to (layer_name, local_neuron_index)
        
        # Activation tracking
        self.hooks = []  # Store hook handles for cleanup
        self.activation_stats = defaultdict(dict)  # Maps neuron_id to stats dict
        self.batch_count = 0
        self.tracking_enabled = False
        
        # Dataset-level statistics
        self.dataset_stats = defaultdict(dict)  # Maps neuron_id to dataset stats
        self.activation_threshold = 0.01  # Threshold for considering neuron "active"
        self.total_samples = 0  # Total number of samples processed
        
        # Correlation analysis
        self.collect_correlations = False  # Whether to collect activation vectors for correlation
        self.activation_vectors = defaultdict(list)  # Maps neuron_id to list of activations
        self.correlation_threshold = 0.9  # Threshold for identifying redundant neurons
        self.max_samples_for_correlation = 2000  # Limit samples to manage memory
        
        # Training history logging
        self.training_history = defaultdict(lambda: {
            'mean': [], 'variance': [], 'frequency': [], 
            'loss': [], 'accuracy': []
        })  # Maps neuron_id to training history
        self.current_epoch = 0
        
    def _get_layer_neuron_count(self, layer: nn.Module, layer_name: str) -> int:
        """
        Calculate the number of neurons/output units for different layer types.
        
        Args:
            layer: The PyTorch layer module
            layer_name: Name of the layer
            
        Returns:
            Number of neurons in the layer
        """
        if isinstance(layer, nn.Linear):
            return layer.out_features
        elif isinstance(layer, nn.Conv2d):
            # For Conv2d, each filter produces one feature map
            return layer.out_channels
        elif isinstance(layer, nn.Conv1d):
            return layer.out_channels
        elif isinstance(layer, nn.BatchNorm1d):
            return layer.num_features
        elif isinstance(layer, nn.BatchNorm2d):
            return layer.num_features
        elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
            return layer.hidden_size
        elif hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'num_features'):
            return layer.num_features
        else:
            # For layers we don't recognize, return 0 (no trackable neurons)
            return 0
    
    def track_model(self, model: nn.Module, model_name: str = "Model") -> Dict[str, List[int]]:
        """
        Enumerate and track all neurons in the given model.
        
        Args:
            model: PyTorch model to track
            model_name: Name for the model (for display purposes)
            
        Returns:
            Dictionary mapping layer names to lists of neuron IDs
        """
        print(f"Tracking neurons in model: {model_name}")
        print("=" * 60)
        
        neuron_mapping = {}
        
        for layer_name, layer in model.named_modules():
            # Skip the root module and modules without trackable neurons
            if layer_name == "" or layer is model:
                continue
                
            neuron_count = self._get_layer_neuron_count(layer, layer_name)
            
            if neuron_count > 0:
                # Assign unique IDs to each neuron in this layer
                neuron_ids = list(range(self.neuron_counter, self.neuron_counter + neuron_count))
                
                # Store layer information
                self.layer_info[layer_name] = {
                    'type': layer.__class__.__name__,
                    'neuron_count': neuron_count,
                    'neuron_ids': neuron_ids,
                    'layer_module': layer
                }
                
                # Update neuron mapping
                for local_idx, neuron_id in enumerate(neuron_ids):
                    self.neuron_map[neuron_id] = (layer_name, local_idx)
                
                neuron_mapping[layer_name] = neuron_ids
                self.neuron_counter += neuron_count
                
                # Display information
                if neuron_count <= 10:
                    ids_display = str(neuron_ids)
                else:
                    ids_display = f"{neuron_ids[:5]}..."
                
                print(f"Layer: {layer_name}")
                print(f"   Type: {layer.__class__.__name__}")
                print(f"   Neurons: {neuron_count}")
                print(f"   IDs: {ids_display}")
                print()
            else:
                print(f"Layer: {layer_name} ({layer.__class__.__name__}) - No trackable neurons")
        
        total_neurons = sum(info['neuron_count'] for info in self.layer_info.values())
        total_layers = len(self.layer_info)
        
        print(f"Total trackable neurons: {total_neurons}")
        print(f"Total layers with neurons: {total_layers}")
        
        return neuron_mapping
    
    def get_neuron_info(self, neuron_id: int) -> Tuple[str, int]:
        """
        Get layer name and local index for a neuron ID.
        
        Args:
            neuron_id: The unique neuron ID
            
        Returns:
            Tuple of (layer_name, local_neuron_index)
        """
        if neuron_id not in self.neuron_map:
            raise ValueError(f"Neuron ID {neuron_id} not found")
        
        return self.neuron_map[neuron_id]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the tracked model.
        
        Returns:
            Dictionary with model statistics
        """
        total_neurons = sum(info['neuron_count'] for info in self.layer_info.values())
        
        layer_types = {}
        for info in self.layer_info.values():
            layer_type = info['type']
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        return {
            'total_neurons': total_neurons,
            'total_layers': len(self.layer_info),
            'layer_types': layer_types,
            'layers': {name: info['neuron_count'] for name, info in self.layer_info.items()}
        }
    
    def set_activation_threshold(self, threshold: float):
        """
        Set the activation threshold for determining if a neuron is "active".
        
        Args:
            threshold: Threshold value (typically 0.01 - 0.1)
        """
        self.activation_threshold = threshold
        print(f"Activation threshold set to {threshold}")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.clear_hooks()
