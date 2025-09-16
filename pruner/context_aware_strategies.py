"""
Context-Aware Pruning Strategies for Pythia-160M

This module provides context-aware pruning strategies specifically designed for
Pythia-160M transformer architecture with layer-specific pruning ratios and
code structure preservation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import logging
from functools import lru_cache


class ContextAwarePruner:
    """
    Context-aware pruning strategy for Pythia-160M transformer architecture.
    
    Implements layer-specific pruning ratios and gradual scheduling optimized
    for code completion tasks and language modeling.
    """
    
    def __init__(self, tracker: Optional[Any] = None, enable_pruning: bool = True) -> None:
        """
        Initialize context-aware pruner.
        
        Args:
            tracker: NeuronTracker instance for integration
            enable_pruning: Boolean toggle for enabling/disabling pruning
        """
        self.tracker = tracker
        self.enable_pruning = enable_pruning
        self.logger = logging.getLogger(__name__)
        
        # Layer-specific pruning ratios for Pythia-160M
        # Balanced ratios for unstructured pruning (weight-level, not neuron-level)
        self.layer_pruning_ratios = {
            'mlp': 1.0,       # 100% of schedule ratio for MLP layers
            'attention': 0.8, # 80% of schedule ratio for attention layers
            'embedding': 0.0, # 0% - Skip embedding layers to avoid dimension issues
            'lm_head': 1.0,   # 100% of schedule ratio for LM head layers
            'other': 1.0      # 100% of schedule ratio for other layers
        }
        
        # Gradual pruning schedule - target 20% overall pruning
        self.pruning_schedule = [0.2, 0.2]  # 20% pruning (consistent)
        self.current_schedule_step = 0
        
        # Recovery training configuration
        self.recovery_epochs = 1  # 1 epoch for development
        
        # Pruning modes - use unstructured to avoid dimension issues
        self.structured_mode = False   # Structured pruning by default
        self.unstructured_mode = False
        
        # Memory optimization
        self.memory_efficient = True
        
        self.logger.info(f"ContextAwarePruner initialized (pruning: {'enabled' if enable_pruning else 'disabled'})")
    
    def set_tracker(self, tracker: Any) -> None:
        """Set or update the tracker instance."""
        self.tracker = tracker
    
    def set_pruning_enabled(self, enabled: bool) -> None:
        """Toggle pruning on/off."""
        self.enable_pruning = enabled
        self.logger.info(f"Pruning {'enabled' if enabled else 'disabled'}")
    
    def set_pruning_schedule(self, schedule: List[float]) -> None:
        """Set the gradual pruning schedule."""
        self.pruning_schedule = schedule
        self.current_schedule_step = 0
        self.logger.info(f"Pruning schedule set: {schedule}")
    
    def get_current_pruning_ratio(self) -> float:
        """Get current pruning ratio from schedule."""
        if not self.enable_pruning or not self.pruning_schedule:
            return 0.0
        
        if self.current_schedule_step >= len(self.pruning_schedule):
            return self.pruning_schedule[-1]  # Use final ratio
        
        return self.pruning_schedule[self.current_schedule_step]
    
    def advance_pruning_schedule(self) -> float:
        """Advance to next pruning ratio in schedule."""
        if self.current_schedule_step < len(self.pruning_schedule):
            self.current_schedule_step += 1
        
        current_ratio = self.get_current_pruning_ratio()
        self.logger.info(f"Advanced to pruning ratio: {current_ratio:.1%}")
        return current_ratio
    
    def identify_transformer_layers(self, model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Identify and categorize transformer layers for context-aware pruning.
        
        Args:
            model: PyTorch model (Pythia-160M)
            
        Returns:
            Dictionary with layer categorization
        """
        layer_info = {
            'embedding_layers': [],
            'attention_layers': [],
            'mlp_layers': [],
            'lm_head_layers': [],
            'other_layers': []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Handle both Linear and Embedding layers
                if isinstance(module, nn.Linear):
                    layer_data = {
                        'name': name,
                        'module': module,
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'total_params': module.in_features * module.out_features + (module.out_features if module.bias is not None else 0)
                    }
                else:  # nn.Embedding
                    layer_data = {
                        'name': name,
                        'module': module,
                        'in_features': module.num_embeddings,  # vocab_size
                        'out_features': module.embedding_dim,   # hidden_size
                        'total_params': module.num_embeddings * module.embedding_dim,
                        'bias': None  # Embedding layers don't have bias
                    }
                
                # Categorize layers based on name patterns (Pythia-160M specific)
                # Check LM head layers FIRST to avoid embed_out being categorized as embedding
                if 'lm_head' in name.lower() or 'head' in name.lower() or 'embed_out' in name.lower():
                    layer_info['lm_head_layers'].append(layer_data)
                elif 'embed' in name.lower():
                    layer_info['embedding_layers'].append(layer_data)
                elif ('attention' in name.lower() or 'attn' in name.lower() or 
                      'query_key_value' in name.lower()):
                    layer_info['attention_layers'].append(layer_data)
                elif ('mlp' in name.lower() or 'feed_forward' in name.lower() or 
                      'ff' in name.lower() or 'dense_4h_to_h' in name.lower() or 
                      'dense_h_to_4h' in name.lower()):
                    layer_info['mlp_layers'].append(layer_data)
                else:
                    # All other layers go to 'other' category
                    layer_info['other_layers'].append(layer_data)
        
        # Ensure embedding layer is found (debug)
        embedding_found = False
        for layer in layer_info['embedding_layers']:
            if 'embed_in' in layer['name']:
                embedding_found = True
                self.logger.info(f"Found embedding layer: {layer['name']} with {layer['out_features']} neurons")
                break
        
        if not embedding_found:
            self.logger.warning("Embedding layer not found in layer identification!")
        
        # Log layer statistics
        total_layers = sum(len(layers) for layers in layer_info.values())
        total_params = sum(
            sum(layer['total_params'] for layer in layers) 
            for layers in layer_info.values()
        )
        
        self.logger.info(f"Identified {total_layers} Linear layers ({total_params:,} total parameters)")
        for category, layers in layer_info.items():
            if layers:
                layer_count = len(layers)
                param_count = sum(layer['total_params'] for layer in layers)
                self.logger.info(f"  {category}: {layer_count} layers, {param_count:,} parameters")
        
        return layer_info
    
    def compute_layer_importance_scores(self, model: nn.Module, layer_info: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """
        Compute importance scores for each layer based on magnitude and gradient information.
        OPTIMIZED for development speed - simplified scoring.
        
        Args:
            model: PyTorch model
            layer_info: Layer categorization from identify_transformer_layers
            
        Returns:
            Dictionary with importance scores for each layer
        """
        importance_scores = {}
        
        # FAST DEVELOPMENT MODE: Skip gradient computation for speed
        self.logger.info("Using FAST importance scoring (magnitude only) for development speed")
        
        for category, layers in layer_info.items():
            for layer_data in layers:
                layer_name = layer_data['name']
                layer_module = layer_data['module']
                
                # FAST: Only magnitude-based importance (no gradient computation)
                weight_magnitude = torch.norm(layer_module.weight.data).item()
                # Handle bias for different layer types
                if isinstance(layer_module, nn.Embedding):
                    bias_magnitude = 0.0  # Embedding layers don't have bias
                else:
                    bias_magnitude = torch.norm(layer_module.bias.data).item() if layer_module.bias is not None else 0.0
                magnitude_score = weight_magnitude + 0.1 * bias_magnitude
                
                # FAST: Skip gradient computation for development speed
                gradient_score = 0.0
                
                # Combined importance score (simplified)
                combined_score = magnitude_score  # Just use magnitude for speed
                
                importance_scores[layer_name] = {
                    'magnitude_score': magnitude_score,
                    'gradient_score': gradient_score,
                    'combined_score': combined_score,
                    'category': category
                }
        
        return importance_scores
    
    def _get_layer_neuron_ids(self, layer_name: str) -> List[int]:
        """Get neuron IDs for a specific layer."""
        if not self.tracker or not hasattr(self.tracker, 'layer_info'):
            return []
        
        layer_info = self.tracker.layer_info.get(layer_name)
        if layer_info:
            return layer_info.get('neuron_ids', [])
        return []
    
    def generate_context_aware_pruning_plan(self, model: nn.Module, current_ratio: float = None) -> Dict[str, Any]:
        """
        Generate context-aware pruning plan for Pythia-160M.
        
        Args:
            model: PyTorch model to prune
            current_ratio: Current pruning ratio (if None, uses schedule)
            
        Returns:
            Dictionary with pruning plan
        """
        if not self.enable_pruning:
            return {
                'status': 'disabled',
                'message': 'Pruning is disabled',
                'pruning_plan': {}
            }
        
        if current_ratio is None:
            current_ratio = self.get_current_pruning_ratio()
        
        if current_ratio <= 0:
            return {
                'status': 'no_pruning',
                'message': f'No pruning needed (ratio: {current_ratio})',
                'pruning_plan': {}
            }
        
        self.logger.info(f"Generating context-aware pruning plan (ratio: {current_ratio:.1%})")
        
        # Identify layers
        layer_info = self.identify_transformer_layers(model)
        
        # Compute importance scores
        importance_scores = self.compute_layer_importance_scores(model, layer_info)
        
        # Generate pruning plan
        pruning_plan = {
            'global_ratio': current_ratio,
            'layer_plans': {},
            'total_neurons_to_prune': 0,
            'total_neurons_remaining': 0,
            'pruning_mode': 'structured' if self.structured_mode else 'unstructured'
        }
        
        for category, layers in layer_info.items():
            category_ratio = self.layer_pruning_ratios.get(category.replace('_layers', ''), 0.1)
            effective_ratio = category_ratio * current_ratio
            
            for layer_data in layers:
                layer_name = layer_data['name']
                layer_module = layer_data['module']
                
                # Calculate neurons to prune
                if isinstance(layer_module, nn.Embedding):
                    total_neurons = layer_module.embedding_dim
                else:
                    total_neurons = layer_module.out_features
                
                neurons_to_prune = max(1, int(total_neurons * effective_ratio))
                neurons_to_keep = total_neurons - neurons_to_prune
                
                # Safety check: don't prune more than 80% of any layer
                max_prune_ratio = 0.8
                max_neurons_to_prune = int(total_neurons * max_prune_ratio)
                if neurons_to_prune > max_neurons_to_prune:
                    neurons_to_prune = max_neurons_to_prune
                    neurons_to_keep = total_neurons - neurons_to_prune
                
                # Debug logging for embedding layers
                if 'embed' in layer_name.lower():
                    self.logger.info(f"DEBUG: {layer_name} - total_neurons={total_neurons}, effective_ratio={effective_ratio:.3f}, neurons_to_prune={neurons_to_prune}")
                
                # Ensure minimum neurons per layer
                min_neurons = 8  # Minimum for transformer layers
                if neurons_to_keep < min_neurons:
                    neurons_to_keep = min_neurons
                    neurons_to_prune = total_neurons - neurons_to_keep
                
                if neurons_to_prune > 0:
                    # Select neurons to prune based on importance
                    neurons_to_prune_indices = self._select_neurons_to_prune(
                        layer_module, neurons_to_prune, importance_scores.get(layer_name, {})
                    )
                    
                    pruning_plan['layer_plans'][layer_name] = {
                        'category': category,
                        'total_neurons': total_neurons,
                        'neurons_to_prune': neurons_to_prune,
                        'neurons_to_keep': neurons_to_keep,
                        'pruning_ratio': neurons_to_prune / total_neurons,
                        'neurons_to_prune_indices': neurons_to_prune_indices,
                        'importance_score': importance_scores.get(layer_name, {}).get('combined_score', 0.0)
                    }
                    
                    pruning_plan['total_neurons_to_prune'] += neurons_to_prune
                    pruning_plan['total_neurons_remaining'] += neurons_to_keep
        
        # Check if we have any layers to actually prune
        if pruning_plan['total_neurons_to_prune'] == 0:
            self.logger.warning("No safe layers found for pruning - all layers skipped to prevent dimension issues")
            pruning_plan['status'] = 'no_safe_layers'
            pruning_plan['message'] = 'No safe layers found for pruning'
        
        self.logger.info(f"Pruning plan generated: {pruning_plan['total_neurons_to_prune']} neurons to prune")
        return pruning_plan
    
    def _select_neurons_to_prune(self, layer_module: Union[nn.Linear, nn.Embedding], num_to_prune: int, importance_info: Dict) -> List[int]:
        """
        Select neurons to prune based on importance scores.
        
        Args:
            layer_module: Linear or Embedding layer module
            num_to_prune: Number of neurons to prune
            importance_info: Importance information for the layer
            
        Returns:
            List of neuron indices to prune
        """
        # Compute neuron importance scores
        neuron_scores = []
        
        # Handle different layer types
        if isinstance(layer_module, nn.Embedding):
            # For embedding layers, we prune embedding dimensions
            total_features = layer_module.embedding_dim
            for neuron_idx in range(total_features):
                # For embedding, use norm of the embedding dimension
                weight_magnitude = torch.norm(layer_module.weight.data[:, neuron_idx]).item()
                neuron_score = weight_magnitude
                neuron_scores.append((neuron_idx, neuron_score))
        else:
            # For linear layers, we prune output features
            total_features = layer_module.out_features
            for neuron_idx in range(total_features):
                # Weight magnitude for this neuron
                weight_magnitude = torch.norm(layer_module.weight.data[neuron_idx]).item()
                
                # Bias magnitude (if present)
                bias_magnitude = abs(layer_module.bias.data[neuron_idx].item()) if layer_module.bias is not None else 0.0
                
                # Combined score (lower = more likely to prune)
                neuron_score = weight_magnitude + 0.1 * bias_magnitude
                neuron_scores.append((neuron_idx, neuron_score))
        
        # Sort by score (ascending) and select least important neurons
        neuron_scores.sort(key=lambda x: x[1])
        neurons_to_prune = [idx for idx, _ in neuron_scores[:num_to_prune]]
        
        return neurons_to_prune
    
    def execute_context_aware_pruning(self, model: nn.Module, pruning_plan: Dict[str, Any], 
                                    dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute context-aware pruning based on the pruning plan.
        
        Args:
            model: PyTorch model to prune
            pruning_plan: Pruning plan from generate_context_aware_pruning_plan
            dry_run: If True, simulate pruning without modifying model
            
        Returns:
            Dictionary with pruning results
        """
        if not self.enable_pruning:
            return {
                'status': 'disabled',
                'message': 'Pruning is disabled',
                'neurons_pruned': 0,
                'layers_affected': 0
            }
        
        if pruning_plan.get('status') in ['disabled', 'no_pruning', 'no_safe_layers']:
            return {
                'status': pruning_plan['status'],
                'message': pruning_plan['message'],
                'neurons_pruned': 0,
                'layers_affected': 0
            }
        
        self.logger.info(f"Executing context-aware pruning ({'dry run' if dry_run else 'real'})")
        
        results = {
            'status': 'completed' if not dry_run else 'simulated',
            'neurons_pruned': 0,
            'layers_affected': 0,
            'layer_results': {},
            'pruning_mode': pruning_plan.get('pruning_mode', 'structured')
        }
        
        if dry_run:
            # Simulation mode
            for layer_name, plan in pruning_plan['layer_plans'].items():
                results['layer_results'][layer_name] = {
                    'neurons_pruned': plan['neurons_to_prune'],
                    'neurons_remaining': plan['neurons_to_keep'],
                    'action': 'simulated'
                }
                results['neurons_pruned'] += plan['neurons_to_prune']
                results['layers_affected'] += 1
        else:
            # Real pruning execution with transformer block consistency
            # First, collect all transformer blocks that need consistent pruning
            transformer_blocks = {}
            for layer_name, plan in pruning_plan['layer_plans'].items():
                if 'attention.dense' in layer_name or 'mlp.dense_4h_to_h' in layer_name:
                    block_layers = self._get_transformer_block_layers(layer_name)
                    block_key = block_layers[0].split('.')[2]  # Extract block number
                    if block_key not in transformer_blocks:
                        transformer_blocks[block_key] = []
                    transformer_blocks[block_key].append((layer_name, plan))
            
            # Process transformer blocks to ensure consistent pruning
            for block_key, block_plans in transformer_blocks.items():
                if len(block_plans) == 2:  # Both attention.dense and mlp.dense_4h_to_h
                    # Use the smaller pruning amount to ensure consistency
                    min_neurons_to_prune = min(plan['neurons_to_prune'] for _, plan in block_plans)
                    for layer_name, plan in block_plans:
                        plan['neurons_to_prune'] = min_neurons_to_prune
                        plan['neurons_to_keep'] = plan['total_neurons'] - min_neurons_to_prune
                        # Recalculate pruning indices
                        plan['neurons_to_prune_indices'] = self._select_neurons_to_prune(
                            self._get_layer_by_name(model, layer_name), 
                            min_neurons_to_prune, 
                            {}
                        )
            
            # Now execute pruning
            for layer_name, plan in pruning_plan['layer_plans'].items():
                try:
                    layer_module = self._get_layer_by_name(model, layer_name)
                    if layer_module is None:
                        self.logger.warning(f"Layer {layer_name} not found, skipping")
                        continue
                    
                    # Check if this layer is safe to prune
                    if not self._is_safe_to_prune(layer_name) or isinstance(layer_module, nn.Embedding):
                        skip_reason = "unsafe layer"
                        if isinstance(layer_module, nn.Embedding):
                            skip_reason = "embedding layer (dimension mismatch issues)"
                        elif 'query_key_value' in layer_name:
                            skip_reason = "attention QKV layer (complex tensor reshaping)"
                        elif 'dense_h_to_4h' in layer_name:
                            skip_reason = "MLP expansion layer (dimension expansion)"
                        elif 'embed_in' in layer_name:
                            skip_reason = "input embedding (dimension mismatch with LayerNorm)"
                        else:
                            skip_reason = "unsafe layer pattern"
                        
                        self.logger.info(f"Skipping {layer_name} - {skip_reason}")
                        results['layer_results'][layer_name] = {
                            'neurons_pruned': 0,
                            'neurons_remaining': plan['total_neurons'],
                            'action': 'skipped_unsafe',
                            'reason': f'Skipped: {skip_reason}'
                        }
                        continue
                    
                    # Apply pruning based on mode
                    if self.structured_mode:
                        # Structured pruning: remove entire neurons
                        if isinstance(layer_module, nn.Embedding):
                            neurons_to_keep = [
                                i for i in range(layer_module.embedding_dim)
                                if i not in plan['neurons_to_prune_indices']
                            ]
                        else:
                            neurons_to_keep = [
                                i for i in range(layer_module.out_features)
                                if i not in plan['neurons_to_prune_indices']
                            ]
                        
                        new_layer = self._create_pruned_layer(layer_module, neurons_to_keep)
                        self._replace_layer_in_model(model, layer_name, new_layer)
                    else:
                        # Unstructured pruning: zero out individual weights
                        self._apply_unstructured_pruning(layer_module, plan)
                    
                    results['layer_results'][layer_name] = {
                        'neurons_pruned': plan['neurons_to_prune'],
                        'neurons_remaining': plan['neurons_to_keep'],
                        'action': 'executed'
                    }
                    
                    results['neurons_pruned'] += plan['neurons_to_prune']
                    results['layers_affected'] += 1
                    
                    # Track pruning in tracker
                    if self.tracker and hasattr(self.tracker, 'track_pruned_neurons'):
                        self.tracker.track_pruned_neurons(layer_name, plan['neurons_to_prune'])
                
                except Exception as e:
                    self.logger.error(f"Failed to prune layer {layer_name}: {e}")
                    results['layer_results'][layer_name] = {
                        'neurons_pruned': 0,
                        'neurons_remaining': plan['total_neurons'],
                        'action': 'failed',
                        'error': str(e)
                    }
        
        self.logger.info(f"Context-aware pruning completed: {results['neurons_pruned']} neurons pruned from {results['layers_affected']} layers")
        return results
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer module by name."""
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
    
    def _create_pruned_layer(self, original_layer: nn.Module, neurons_to_keep: List[int]) -> nn.Module:
        """Create a new layer with specified neurons kept."""
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        
        if isinstance(original_layer, nn.Linear):
            new_layer = nn.Linear(
                in_features=original_layer.in_features,
                out_features=len(neurons_to_keep),
                bias=original_layer.bias is not None,
                device=device,
                dtype=dtype
            )
            
            # Copy weights
            neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=device)
            new_layer.weight.data = original_layer.weight.data[neurons_to_keep_tensor].clone()
            
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data[neurons_to_keep_tensor].clone()
        
        elif isinstance(original_layer, nn.Embedding):
            new_layer = nn.Embedding(
                num_embeddings=original_layer.num_embeddings,
                embedding_dim=len(neurons_to_keep),
                padding_idx=original_layer.padding_idx,
                device=device,
                dtype=dtype
            )
            
            # Copy weights (for embedding, we keep the embedding_dim dimension)
            neurons_to_keep_tensor = torch.tensor(neurons_to_keep, device=device)
            new_layer.weight.data = original_layer.weight.data[:, neurons_to_keep_tensor].clone()
        
        return new_layer
    
    def _apply_unstructured_pruning(self, layer_module: nn.Module, plan: Dict[str, Any]) -> None:
        """Apply unstructured pruning by zeroing out individual weights."""
        if isinstance(layer_module, nn.Linear):
            # Calculate number of weights to prune
            total_weights = layer_module.weight.numel()
            weights_to_prune = int(total_weights * plan['pruning_ratio'])
            
            # Get weight magnitudes for importance scoring
            weight_magnitudes = torch.abs(layer_module.weight.data).flatten()
            
            # Find least important weights
            _, least_important_indices = torch.topk(weight_magnitudes, weights_to_prune, largest=False)
            
            # Zero out the least important weights
            flat_weight = layer_module.weight.data.flatten()
            flat_weight[least_important_indices] = 0
            layer_module.weight.data = flat_weight.view(layer_module.weight.shape)
            
            # Also prune bias if present
            if layer_module.bias is not None:
                bias_magnitudes = torch.abs(layer_module.bias.data)
                bias_to_prune = int(layer_module.bias.numel() * plan['pruning_ratio'])
                _, least_important_bias = torch.topk(bias_magnitudes, bias_to_prune, largest=False)
                layer_module.bias.data[least_important_bias] = 0
        
        elif isinstance(layer_module, nn.Embedding):
            # For embeddings, prune embedding dimensions (but keep structure)
            total_weights = layer_module.weight.numel()
            weights_to_prune = int(total_weights * plan['pruning_ratio'])
            
            weight_magnitudes = torch.abs(layer_module.weight.data).flatten()
            _, least_important_indices = torch.topk(weight_magnitudes, weights_to_prune, largest=False)
            
            flat_weight = layer_module.weight.data.flatten()
            flat_weight[least_important_indices] = 0
            layer_module.weight.data = flat_weight.view(layer_module.weight.shape)
    
    def _is_attention_layer(self, layer_name: str) -> bool:
        """Check if this is an attention layer that needs special handling."""
        attention_keywords = ['query_key_value', 'attention.dense', 'attn.c_attn', 'attn.c_proj']
        return any(keyword in layer_name.lower() for keyword in attention_keywords)
    
    def _is_mlp_layer(self, layer_name: str) -> bool:
        """Check if this is an MLP layer that needs special handling."""
        mlp_keywords = ['dense_h_to_4h', 'dense_4h_to_h', 'mlp']
        return any(keyword in layer_name.lower() for keyword in mlp_keywords)
    
    def _is_lm_head_layer(self, layer_name: str) -> bool:
        """Check if this is an LM head layer that needs special handling."""
        lm_head_keywords = ['lm_head', 'head', 'embed_out']
        return any(keyword in layer_name.lower() for keyword in lm_head_keywords)
    
    def _is_safe_to_prune(self, layer_name: str) -> bool:
        """Check if a layer is safe to prune without causing dimension/tensor issues."""
        # For unstructured pruning, we can safely prune most linear layers
        # since we're only zeroing out weights, not removing neurons
        unsafe_patterns = [
            'embed_in',         # Input embedding (dimension mismatch with LayerNorm)
        ]
        
        # Skip layers with unsafe patterns
        for pattern in unsafe_patterns:
            if pattern in layer_name:
                return False
        
        # Allow pruning of most linear layers for unstructured pruning
        return True
    
    def _get_transformer_block_layers(self, layer_name: str) -> List[str]:
        """Get all layers in the same transformer block that need consistent pruning."""
        try:
            # Extract block info from layer name (e.g., "gpt_neox.layers.0.attention.dense")
            parts = layer_name.split('.')
            if len(parts) >= 4 and parts[0] == 'gpt_neox' and parts[1] == 'layers':
                block_idx = parts[2]
                block_prefix = f"gpt_neox.layers.{block_idx}"
                
                # Return the two layers that need consistent dimensions for residual connections
                return [
                    f"{block_prefix}.attention.dense",
                    f"{block_prefix}.mlp.dense_4h_to_h"
                ]
        except Exception:
            pass
        return [layer_name]  # Fallback to just the layer itself
    
    def _get_mlp_block_layers(self, model: nn.Module, layer_name: str) -> List[str]:
        """Get all layers in the same MLP block as the given layer."""
        try:
            # Extract the block number from layer name (e.g., "gpt_neox.layers.0.mlp.dense_h_to_4h")
            parts = layer_name.split('.')
            if len(parts) >= 3 and parts[0] == 'gpt_neox' and parts[1] == 'layers':
                block_idx = parts[2]
                mlp_block_prefix = f"gpt_neox.layers.{block_idx}.mlp"
                
                # Find all MLP layers in this block
                mlp_layers = []
                for name, module in model.named_modules():
                    if name.startswith(mlp_block_prefix) and isinstance(module, nn.Linear):
                        mlp_layers.append(name)
                
                return mlp_layers
        except Exception:
            pass
        return [layer_name]  # Fallback to just the layer itself
    
    def _get_attention_layer_info(self, model: nn.Module, layer_name: str) -> Optional[Dict]:
        """Get information about attention layer structure."""
        try:
            # For Pythia-160M, find the attention module structure
            layer_parts = layer_name.split('.')
            current_module = model
            
            # Navigate to the attention module
            for part in layer_parts[:-1]:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    try:
                        idx = int(part)
                        current_module = current_module[idx]
                    except (ValueError, IndexError, TypeError):
                        return None
            
            # Check if this is part of an attention module
            if hasattr(current_module, 'num_heads') or hasattr(current_module, 'attention'):
                return {
                    'is_attention': True,
                    'num_heads': getattr(current_module, 'num_heads', 8),
                    'hidden_size': getattr(current_module, 'hidden_size', 768)
                }
            
            return None
        except Exception:
            return None
    
    def _replace_layer_in_model(self, model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
        """Replace a layer in the model."""
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
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of pruning operations."""
        return {
            'pruning_enabled': self.enable_pruning,
            'current_schedule_step': self.current_schedule_step,
            'current_ratio': self.get_current_pruning_ratio(),
            'layer_ratios': self.layer_pruning_ratios.copy(),
            'pruning_mode': 'structured' if self.structured_mode else 'unstructured',
            'recovery_epochs': self.recovery_epochs
        }
    
    def set_structured_mode(self, structured: bool) -> None:
        """Set structured/unstructured pruning mode."""
        self.structured_mode = structured
        self.unstructured_mode = not structured
        self.logger.info(f"Pruning mode set to: {'structured' if structured else 'unstructured'}")
    
    def set_recovery_epochs(self, epochs: int) -> None:
        """Set number of recovery training epochs."""
        self.recovery_epochs = epochs
        self.logger.info(f"Recovery epochs set to: {epochs}")

