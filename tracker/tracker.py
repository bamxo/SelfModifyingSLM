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
        
        # Context-aware tracking for Pythia-160M
        self.layer_types = {}  # Maps layer_name to layer_type (mlp, attention, embedding, lm_head)
        self.layer_pruning_ratios = {
            'mlp': 0.30,      # 30% pruning for MLP layers
            'attention': 0.15, # 15% pruning for attention layers
            'embedding': 0.05, # 5% pruning for embedding layers
            'lm_head': 0.10   # 10% pruning for language modeling head
        }
        self.gradient_importance = {}  # Maps neuron_id to gradient-based importance score
        self.attention_patterns = {}  # Maps layer_name to attention pattern analysis
        self.pruned_neuron_counts = defaultdict(int)  # Maps layer_type to pruned count
        
    def _get_layer_neuron_count(self, layer: nn.Module, layer_name: str) -> int:
        """
        Calculate the number of neurons/output units for different layer types.
        Enhanced for Pythia-160M transformer architecture with context-aware tracking.
        
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
        # Pythia-160M specific layer types
        elif hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'num_features'):
            return layer.num_features
        elif hasattr(layer, 'hidden_size'):
            return layer.hidden_size
        elif hasattr(layer, 'embed_dim'):
            return layer.embed_dim
        elif hasattr(layer, 'num_heads'):
            # For multi-head attention, track each head separately
            return layer.num_heads * (layer.embed_dim if hasattr(layer, 'embed_dim') else layer.head_dim)
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
    
    def identify_layer_types(self, model: nn.Module) -> Dict[str, str]:
        """
        Identify layer types for context-aware pruning in Pythia-160M architecture.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping layer names to layer types
        """
        layer_types = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'embed' in name.lower():
                    layer_types[name] = 'embedding'
                elif 'lm_head' in name.lower() or 'head' in name.lower():
                    layer_types[name] = 'lm_head'
                elif 'mlp' in name.lower() or 'feed_forward' in name.lower():
                    layer_types[name] = 'mlp'
                elif 'attention' in name.lower() or 'attn' in name.lower():
                    layer_types[name] = 'attention'
                else:
                    layer_types[name] = 'mlp'  # Default for other linear layers
        
        self.layer_types = layer_types
        
        # Print summary
        print(f"Identified {len(layer_types)} layers for context-aware pruning:")
        for layer_type in ['embedding', 'attention', 'mlp', 'lm_head']:
            count = sum(1 for t in layer_types.values() if t == layer_type)
            if count > 0:
                print(f"  {layer_type}: {count} layers")
        
        return layer_types
    
    def get_layer_type(self, layer_name: str) -> str:
        """Get the layer type for a given layer name."""
        return self.layer_types.get(layer_name, 'mlp')
    
    def get_layer_pruning_ratio(self, layer_name: str) -> float:
        """Get the pruning ratio for a specific layer based on its type."""
        layer_type = self.get_layer_type(layer_name)
        return self.layer_pruning_ratios.get(layer_type, 0.1)
    
    def compute_gradient_importance(self, model: nn.Module, loss_fn, data_loader, num_batches: int = 10) -> Dict[int, float]:
        """
        Compute gradient-based importance scores for neurons.
        
        Args:
            model: PyTorch model
            loss_fn: Loss function
            data_loader: Data loader for computing gradients
            num_batches: Number of batches to process
            
        Returns:
            Dictionary mapping neuron_id to importance score
        """
        model.eval()
        gradient_scores = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                # Forward pass
                if isinstance(batch, dict):
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs.logits, batch['labels'])
                else:
                    outputs = model(batch)
                    loss = loss_fn(outputs, batch)
                
                # Compute gradients
                loss.backward()
                
                # Extract gradient magnitudes for each neuron
                neuron_id = 0
                for layer_name, layer in model.named_modules():
                    if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                        # Compute gradient magnitude for each output neuron
                        grad_magnitudes = torch.norm(layer.weight.grad, p=2, dim=1)
                        
                        for local_idx in range(layer.out_features):
                            if neuron_id not in gradient_scores:
                                gradient_scores[neuron_id] = 0.0
                            gradient_scores[neuron_id] += grad_magnitudes[local_idx].item()
                            neuron_id += 1
                
                # Clear gradients
                model.zero_grad()
        
        # Normalize scores
        if gradient_scores:
            max_score = max(gradient_scores.values())
            for neuron_id in gradient_scores:
                gradient_scores[neuron_id] /= max_score
        
        self.gradient_importance = gradient_scores
        return gradient_scores
    
    def analyze_attention_patterns(self, model: nn.Module, data_loader, num_batches: int = 5) -> Dict[str, Dict]:
        """
        Analyze attention patterns for code structure preservation.
        
        Args:
            model: PyTorch model
            data_loader: Data loader
            num_batches: Number of batches to analyze
            
        Returns:
            Dictionary with attention pattern analysis
        """
        attention_patterns = {}
        
        # Hook function to capture attention weights
        def attention_hook(module, input, output):
            layer_name = getattr(module, '_layer_name', 'unknown')
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_patterns[layer_name] = {
                    'attention_weights': output.attentions[-1].detach().cpu(),
                    'attention_entropy': self._compute_attention_entropy(output.attentions[-1]),
                    'attention_sparsity': self._compute_attention_sparsity(output.attentions[-1])
                }
        
        # Register hooks for attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                module._layer_name = name
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Process batches
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        self.attention_patterns = attention_patterns
        return attention_patterns
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # Flatten attention weights and compute entropy
        weights = attention_weights.view(-1)
        weights = torch.softmax(weights, dim=0)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        return entropy.item()
    
    def _compute_attention_sparsity(self, attention_weights: torch.Tensor) -> float:
        """Compute sparsity of attention weights."""
        # Count near-zero weights
        threshold = 0.01
        total_weights = attention_weights.numel()
        sparse_weights = (torch.abs(attention_weights) < threshold).sum().item()
        return sparse_weights / total_weights
    
    def track_pruned_neurons(self, layer_name: str, neuron_count: int):
        """Track pruned neuron counts per layer type."""
        layer_type = self.get_layer_type(layer_name)
        self.pruned_neuron_counts[layer_type] += neuron_count
    
    def get_context_aware_recommendations(self, model: nn.Module, current_pruning_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Generate context-aware pruning recommendations for Pythia-160M.
        
        Args:
            model: PyTorch model
            current_pruning_ratio: Current pruning ratio to apply
            
        Returns:
            Dictionary with pruning recommendations
        """
        recommendations = {
            'prune': [],
            'keep': [],
            'layer_analysis': {},
            'context_aware_info': {
                'layer_types': self.layer_types,
                'pruning_ratios': self.layer_pruning_ratios,
                'gradient_importance': len(self.gradient_importance),
                'attention_patterns': len(self.attention_patterns),
                'pruned_counts': dict(self.pruned_neuron_counts)
            }
        }
        
        neuron_id = 0
        
        for layer_name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                layer_type = self.get_layer_type(layer_name)
                target_ratio = self.layer_pruning_ratios.get(layer_type, 0.1) * current_pruning_ratio
                
                # Calculate neurons to prune for this layer
                total_neurons = layer.out_features
                neurons_to_prune = int(total_neurons * target_ratio)
                
                if neurons_to_prune > 0:
                    # Get neuron importance scores (combine magnitude and gradient)
                    importance_scores = []
                    
                    for local_idx in range(total_neurons):
                        # Magnitude-based importance
                        magnitude_score = torch.norm(layer.weight.data[local_idx]).item()
                        
                        # Gradient-based importance
                        gradient_score = self.gradient_importance.get(neuron_id, 0.0)
                        
                        # Combined score
                        combined_score = 0.7 * magnitude_score + 0.3 * gradient_score
                        importance_scores.append((local_idx, combined_score))
                        
                        neuron_id += 1
                    
                    # Sort by importance and select least important neurons for pruning
                    importance_scores.sort(key=lambda x: x[1])
                    
                    for local_idx, _ in importance_scores[:neurons_to_prune]:
                        global_neuron_id = neuron_id - total_neurons + local_idx
                        recommendations['prune'].append({
                            'neuron_id': global_neuron_id,
                            'layer_name': layer_name,
                            'local_index': local_idx,
                            'layer_type': layer_type,
                            'importance_score': importance_scores[local_idx][1],
                            'reason': f'Low importance in {layer_type} layer'
                        })
                    
                    # Layer analysis
                    recommendations['layer_analysis'][layer_name] = {
                        'layer_type': layer_type,
                        'total_neurons': total_neurons,
                        'neurons_to_prune': neurons_to_prune,
                        'pruning_ratio': target_ratio,
                        'avg_importance': np.mean([score for _, score in importance_scores])
                    }
        
        return recommendations
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.clear_hooks()
