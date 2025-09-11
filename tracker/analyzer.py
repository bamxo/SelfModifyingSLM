"""
Neuron Activity Analyzer

This module provides functionality for tracking neuron activations, computing statistics,
and performing correlation analysis for redundancy detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class ActivationAnalyzer:
    """
    Analyzes neuron activations and computes various statistics.
    """
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def _create_activation_hook(self, layer_name: str, layer_neuron_ids: List[int]):
        """
        Create a forward hook function for a specific layer.
        
        Args:
            layer_name: Name of the layer
            layer_neuron_ids: List of neuron IDs for this layer
            
        Returns:
            Hook function for the layer
        """
        def hook_fn(module, input, output):
            if not self.tracker.tracking_enabled:
                return
            
            # Convert output to tensor if needed and flatten batch dimension
            if isinstance(output, tuple):
                output = output[0]
            
            # Handle different output shapes
            if output.dim() == 4:  # Conv2d output: (batch, channels, height, width)
                # Average pool spatial dimensions to get per-channel activations
                activations = output.mean(dim=(2, 3))  # Shape: (batch, channels)
            elif output.dim() == 3:  # Conv1d or sequence output: (batch, channels, length)
                activations = output.mean(dim=2)  # Shape: (batch, channels)
            elif output.dim() == 2:  # Linear output: (batch, neurons)
                activations = output
            else:
                # For other cases, try to reshape to (batch, neurons)
                activations = output.view(output.size(0), -1)
            
            batch_size = activations.size(0)
            num_neurons = activations.size(1)
            
            # Process each neuron's activations
            for local_idx in range(min(num_neurons, len(layer_neuron_ids))):
                neuron_id = layer_neuron_ids[local_idx]
                neuron_activations = activations[:, local_idx]
                
                # Compute batch-level statistics
                mean_activation = neuron_activations.mean().item()
                variance_activation = neuron_activations.var().item()
                
                # Count non-zero activations (sparsity)
                active_count = (torch.abs(neuron_activations) > self.tracker.activation_threshold).sum().item()
                sparsity = active_count / batch_size
                
                # Store batch-level statistics
                if neuron_id not in self.tracker.activation_stats:
                    self.tracker.activation_stats[neuron_id] = {
                        'mean_sum': 0.0,
                        'variance_sum': 0.0,
                        'sparsity_sum': 0.0,
                        'batch_count': 0
                    }
                
                stats = self.tracker.activation_stats[neuron_id]
                stats['mean_sum'] += mean_activation
                stats['variance_sum'] += variance_activation
                stats['sparsity_sum'] += sparsity
                stats['batch_count'] += 1
                
                # Store dataset-level statistics (accumulating)
                if neuron_id not in self.tracker.dataset_stats:
                    self.tracker.dataset_stats[neuron_id] = {
                        'activation_sum': 0.0,
                        'activation_squared_sum': 0.0,
                        'active_samples': 0,
                        'total_samples': 0,
                        'max_activation': float('-inf'),
                        'min_activation': float('inf')
                    }

                dataset_stats = self.tracker.dataset_stats[neuron_id]
                dataset_stats['activation_sum'] += neuron_activations.sum().item()
                dataset_stats['activation_squared_sum'] += (neuron_activations ** 2).sum().item()
                dataset_stats['active_samples'] += active_count
                dataset_stats['total_samples'] += batch_size
                dataset_stats['max_activation'] = max(dataset_stats['max_activation'], neuron_activations.max().item())
                dataset_stats['min_activation'] = min(dataset_stats['min_activation'], neuron_activations.min().item())
                
                # Collect activation vectors for correlation analysis if enabled
                if self.tracker.collect_correlations and self.tracker.total_samples <= self.tracker.max_samples_for_correlation:
                    # Store individual activations for each sample in the batch
                    for sample_activation in neuron_activations:
                        self.tracker.activation_vectors[neuron_id].append(sample_activation.item())
        
        return hook_fn
    
    def register_activation_hooks(self, model: nn.Module):
        """
        Register forward hooks on all trackable layers.
        
        Args:
            model: PyTorch model to add hooks to
        """
        for layer_name, layer_info in self.tracker.layer_info.items():
            layer_module = layer_info['layer_module']
            layer_neuron_ids = layer_info['neuron_ids']
            
            hook_fn = self._create_activation_hook(layer_name, layer_neuron_ids)
            handle = layer_module.register_forward_hook(hook_fn)
            self.tracker.hooks.append(handle)
        
        print(f"Registered {len(self.tracker.hooks)} activation hooks")
    
    def start_tracking(self, enable_correlation_analysis=False):
        """
        Enable activation tracking.
        
        Args:
            enable_correlation_analysis: Whether to collect activation vectors for correlation analysis
        """
        self.tracker.tracking_enabled = True
        self.tracker.collect_correlations = enable_correlation_analysis
        self.tracker.activation_stats.clear()
        self.tracker.dataset_stats.clear()
        self.tracker.activation_vectors.clear()
        self.tracker.batch_count = 0
        self.tracker.total_samples = 0
        
        tracking_mode = "with correlation analysis" if enable_correlation_analysis else "standard"
        print(f"Activation tracking started ({tracking_mode})")
    
    def stop_tracking(self):
        """Disable activation tracking."""
        self.tracker.tracking_enabled = False
        print("Activation tracking stopped")
    
    def get_activation_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Get summary statistics for all tracked neurons.
        
        Returns:
            Dictionary mapping neuron_id to statistics
        """
        summary = {}
        
        for neuron_id, stats in self.tracker.activation_stats.items():
            if stats['batch_count'] > 0:
                summary[neuron_id] = {
                    'mean_activation': stats['mean_sum'] / stats['batch_count'],
                    'mean_variance': stats['variance_sum'] / stats['batch_count'],
                    'mean_sparsity': stats['sparsity_sum'] / stats['batch_count'],
                    'batches_processed': stats['batch_count']
                }
        
        return summary
    
    def get_dataset_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Compute comprehensive dataset-level statistics for each neuron.
        
        Returns:
            Dictionary mapping neuron_id to dataset statistics
        """
        statistics = {}
        
        for neuron_id, stats in self.tracker.dataset_stats.items():
            if stats['total_samples'] > 0:
                # Calculate dataset-level statistics
                mean_activation = stats['activation_sum'] / stats['total_samples']
                
                # Calculate standard deviation
                variance = (stats['activation_squared_sum'] / stats['total_samples']) - (mean_activation ** 2)
                std_deviation = variance ** 0.5 if variance >= 0 else 0.0
                
                # Calculate firing frequency
                firing_frequency = stats['active_samples'] / stats['total_samples']
                
                # Determine if neuron is dead
                is_dead = firing_frequency == 0.0
                
                statistics[neuron_id] = {
                    'mean_activation': mean_activation,
                    'std_deviation': std_deviation,
                    'firing_frequency': firing_frequency,
                    'is_dead': is_dead,
                    'max_activation': stats['max_activation'],
                    'min_activation': stats['min_activation'],
                    'total_samples': stats['total_samples']
                }
        
        return statistics
    
    def compute_layer_correlations(self, layer_name: str) -> Dict[Tuple[int, int], float]:
        """
        Compute Pearson correlations between all pairs of neurons in a layer.
        
        Args:
            layer_name: Name of the layer to analyze
            
        Returns:
            Dictionary mapping (neuron_id1, neuron_id2) to correlation coefficient
        """
        if not self.tracker.collect_correlations:
            raise RuntimeError("Correlation analysis not enabled. Use start_tracking(enable_correlation_analysis=True)")
        
        # Get neuron IDs for this layer
        if layer_name not in self.tracker.layer_info:
            raise ValueError(f"Layer {layer_name} not found")
        
        layer_neuron_ids = self.tracker.layer_info[layer_name]['neuron_ids']
        
        # Check if we have activation vectors for these neurons
        available_neurons = [nid for nid in layer_neuron_ids if nid in self.tracker.activation_vectors and len(self.tracker.activation_vectors[nid]) > 1]
        
        if len(available_neurons) < 2:
            return {}
        
        correlations = {}
        
        # Compute correlations between all pairs
        for i, neuron_id1 in enumerate(available_neurons):
            for neuron_id2 in available_neurons[i+1:]:
                activations1 = np.array(self.tracker.activation_vectors[neuron_id1])
                activations2 = np.array(self.tracker.activation_vectors[neuron_id2])
                
                # Ensure both vectors have the same length
                min_length = min(len(activations1), len(activations2))
                activations1 = activations1[:min_length]
                activations2 = activations2[:min_length]
                
                # Compute Pearson correlation
                if np.std(activations1) > 1e-8 and np.std(activations2) > 1e-8:  # Avoid division by zero
                    correlation = np.corrcoef(activations1, activations2)[0, 1]
                    if not np.isnan(correlation):
                        correlations[(neuron_id1, neuron_id2)] = correlation
        
        return correlations
    
    def find_redundant_neurons(self, correlation_threshold: Optional[float] = None) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Find pairs of neurons with high correlation (redundant neurons).
        
        Args:
            correlation_threshold: Minimum correlation to consider neurons redundant
            
        Returns:
            Dictionary mapping layer_name to list of (neuron_id1, neuron_id2, correlation) tuples
        """
        if correlation_threshold is None:
            correlation_threshold = self.tracker.correlation_threshold
        
        redundant_pairs = defaultdict(list)
        
        for layer_name in self.tracker.layer_info.keys():
            layer_correlations = self.compute_layer_correlations(layer_name)
            
            for (neuron_id1, neuron_id2), correlation in layer_correlations.items():
                if abs(correlation) >= correlation_threshold:
                    redundant_pairs[layer_name].append((neuron_id1, neuron_id2, correlation))
        
        return dict(redundant_pairs)
    
    def get_dead_neurons(self) -> Dict[str, List[int]]:
        """
        Get dead neurons (never activate) grouped by layer.
        
        Returns:
            Dictionary mapping layer_name to list of dead neuron IDs
        """
        dataset_stats = self.get_dataset_statistics()
        dead_neurons = defaultdict(list)
        
        for neuron_id, stats in dataset_stats.items():
            if stats['is_dead']:
                layer_name, _ = self.tracker.get_neuron_info(neuron_id)
                dead_neurons[layer_name].append(neuron_id)
        
        return dict(dead_neurons)
    
    def get_layer_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics for each layer.
        
        Returns:
            Dictionary with layer-wise statistics
        """
        dataset_stats = self.get_dataset_statistics()
        layer_summary = {}
        
        for layer_name, layer_info in self.tracker.layer_info.items():
            layer_neuron_ids = layer_info['neuron_ids']
            
            # Get statistics for neurons in this layer
            neuron_stats_list = []
            dead_count = 0
            
            for neuron_id in layer_neuron_ids:
                if neuron_id in dataset_stats:
                    stats = dataset_stats[neuron_id]
                    neuron_stats_list.append(stats)
                    if stats['is_dead']:
                        dead_count += 1
            
            if neuron_stats_list:
                total_neurons = len(layer_neuron_ids)
                dead_percentage = (dead_count / total_neurons) * 100
                
                layer_summary[layer_name] = {
                    'total_neurons': total_neurons,
                    'dead_neurons': dead_count,
                    'dead_percentage': dead_percentage,
                    'avg_firing_frequency': np.mean([stats['firing_frequency'] for stats in neuron_stats_list]),
                    'avg_mean_activation': np.mean([abs(stats['mean_activation']) for stats in neuron_stats_list]),
                    'avg_std_deviation': np.mean([stats['std_deviation'] for stats in neuron_stats_list])
                }
        
        return layer_summary
