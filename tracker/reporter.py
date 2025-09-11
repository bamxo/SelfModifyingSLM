"""
Neuron Activity Reporter

This module provides functionality for generating reports, visualizations,
and optimization recommendations based on neuron tracking data.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json
import datetime
import sys
from io import StringIO


class ActivityReporter:
    """
    Generates reports and visualizations for neuron activity analysis.
    """
    
    def __init__(self, tracker, analyzer):
        self.tracker = tracker
        self.analyzer = analyzer
    
    def generate_neuron_report(self, show_heatmap=True, save_heatmap=None):
        """
        Generate a comprehensive report of neuron activity and patterns.
        
        Args:
            show_heatmap: Whether to display the activity heatmap
            save_heatmap: Path to save heatmap image (optional)
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE NEURON ACTIVITY REPORT")
        print("=" * 80)
        
        # Get dataset statistics
        dataset_stats = self.analyzer.get_dataset_statistics()
        
        if not dataset_stats:
            print("No neuron statistics available. Run tracking first.")
            return
        
        # Analyze activity patterns
        self._report_activity_analysis(dataset_stats)
        
        # Report layer-wise statistics
        self._report_layer_statistics()
        
        # Generate heatmap if requested
        if show_heatmap or save_heatmap:
            self._generate_activity_heatmap(dataset_stats, show_heatmap, save_heatmap)
    
    def _report_activity_analysis(self, dataset_stats):
        """Report top/bottom neurons by activity."""
        print("\nNEURON ACTIVITY ANALYSIS")
        print("-" * 50)
        
        # Calculate activity scores for each neuron
        neuron_activities = []
        for neuron_id, stats in dataset_stats.items():
            # Use firing frequency as primary activity metric
            activity_score = stats['firing_frequency']
            mean_activation = stats['mean_activation']
            
            layer_name, local_idx = self.tracker.get_neuron_info(neuron_id)
            
            neuron_activities.append({
                'neuron_id': neuron_id,
                'layer_name': layer_name,
                'local_idx': local_idx,
                'activity_score': activity_score,
                'mean_activation': mean_activation,
                'is_dead': stats['is_dead']
            })
        
        # Sort by activity score
        neuron_activities.sort(key=lambda x: x['activity_score'], reverse=True)
        
        # Top 10 most active neurons
        print("\nTOP 10 MOST ACTIVE NEURONS (Key Contributors):")
        print(f"{'Rank':<5} {'Neuron ID':<10} {'Layer':<15} {'Local Idx':<10} {'Frequency':<12} {'Mean Act':<12}")
        print("-" * 75)
        
        for i, neuron in enumerate(neuron_activities[:10]):
            print(f"{i+1:<5} {neuron['neuron_id']:<10} {neuron['layer_name']:<15} "
                  f"{neuron['local_idx']:<10} {neuron['activity_score']:<12.3f} "
                  f"{neuron['mean_activation']:<12.4f}")
        
        # Bottom 10 least active neurons (excluding dead ones for pruning candidates)
        non_dead_neurons = [n for n in neuron_activities if not n['is_dead']]
        non_dead_neurons.sort(key=lambda x: x['activity_score'])
        
        print(f"\nTOP 10 LEAST ACTIVE NEURONS (Pruning Candidates):")
        print(f"{'Rank':<5} {'Neuron ID':<10} {'Layer':<15} {'Local Idx':<10} {'Frequency':<12} {'Mean Act':<12}")
        print("-" * 75)
        
        for i, neuron in enumerate(non_dead_neurons[:10]):
            print(f"{i+1:<5} {neuron['neuron_id']:<10} {neuron['layer_name']:<15} "
                  f"{neuron['local_idx']:<10} {neuron['activity_score']:<12.3f} "
                  f"{neuron['mean_activation']:<12.4f}")
        
        # Activity statistics
        all_frequencies = [n['activity_score'] for n in neuron_activities]
        dead_count = sum(1 for n in neuron_activities if n['is_dead'])
        
        print(f"\nACTIVITY STATISTICS:")
        print(f"  Total neurons analyzed: {len(neuron_activities)}")
        print(f"  Dead neurons: {dead_count} ({dead_count/len(neuron_activities)*100:.1f}%)")
        print(f"  Average firing frequency: {np.mean(all_frequencies):.3f}")
        print(f"  Activity range: {min(all_frequencies):.3f} - {max(all_frequencies):.3f}")
        print(f"  Standard deviation: {np.std(all_frequencies):.3f}")
    
    def _report_layer_statistics(self):
        """Report layer-wise dead and redundant neuron statistics."""
        print(f"\nLAYER-WISE NEURON ANALYSIS")
        print("-" * 50)
        
        # Get layer summary for dead neurons
        layer_summary = self.analyzer.get_layer_summary()
        
        # Get redundancy data if correlation analysis was performed
        redundant_data = {}
        if self.tracker.activation_vectors:  # If correlation data was collected
            try:
                redundant_pairs = self.analyzer.find_redundant_neurons()
                for layer_name, pairs in redundant_pairs.items():
                    redundant_data[layer_name] = len(pairs)
            except:
                pass  # Correlation analysis not available
        
        print(f"{'Layer':<20} {'Total':<8} {'Dead':<8} {'Dead %':<8} {'Redundant':<12} {'Red %':<8}")
        print("-" * 70)
        
        for layer_name, stats in layer_summary.items():
            total_neurons = stats['total_neurons']
            dead_neurons = stats['dead_neurons']
            dead_percentage = stats['dead_percentage']
            
            redundant_pairs = redundant_data.get(layer_name, 0)
            # Estimate redundant neurons (each pair represents 2 potentially redundant neurons)
            redundant_neurons = min(redundant_pairs * 2, total_neurons)
            redundant_percentage = (redundant_neurons / total_neurons * 100) if total_neurons > 0 else 0
            
            print(f"{layer_name:<20} {total_neurons:<8} {dead_neurons:<8} "
                  f"{dead_percentage:<8.1f} {redundant_neurons:<12} {redundant_percentage:<8.1f}")
        
        # Summary statistics
        total_neurons = sum(stats['total_neurons'] for stats in layer_summary.values())
        total_dead = sum(stats['dead_neurons'] for stats in layer_summary.values())
        total_redundant = sum(redundant_data.get(layer, 0) * 2 for layer in layer_summary.keys())
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_neurons:<8} {total_dead:<8} "
              f"{total_dead/total_neurons*100:<8.1f} {total_redundant:<12} "
              f"{total_redundant/total_neurons*100:<8.1f}")
    
    def _generate_activity_heatmap(self, dataset_stats, show_plot=True, save_path=None):
        """Generate a heatmap visualization of neuron activity patterns."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            print("\nHEATMAP VISUALIZATION:")
            print("matplotlib not available. Install with: pip install matplotlib")
            return
        
        print(f"\nGENERATING ACTIVITY HEATMAP")
        print("-" * 30)
        
        # Organize data by layers
        layer_data = {}
        for neuron_id, stats in dataset_stats.items():
            layer_name, local_idx = self.tracker.get_neuron_info(neuron_id)
            
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            layer_data[layer_name][local_idx] = {
                'firing_frequency': stats['firing_frequency'],
                'mean_activation': stats['mean_activation'],
                'is_dead': stats['is_dead']
            }
        
        # Create figure with subplots for each layer
        num_layers = len(layer_data)
        fig, axes = plt.subplots(num_layers, 2, figsize=(12, 3 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        # Custom colormap for activity
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('activity', colors)
        
        for i, (layer_name, neuron_data) in enumerate(layer_data.items()):
            # Prepare data for heatmap
            max_neurons = max(neuron_data.keys()) + 1
            firing_freq_data = np.zeros(max_neurons)
            mean_act_data = np.zeros(max_neurons)
            
            for local_idx, data in neuron_data.items():
                firing_freq_data[local_idx] = data['firing_frequency']
                mean_act_data[local_idx] = abs(data['mean_activation'])  # Use absolute value
            
            # Reshape data for visualization (make it 2D for better visualization)
            neurons_per_row = min(16, max_neurons)
            rows = (max_neurons + neurons_per_row - 1) // neurons_per_row
            
            # Pad data to fit grid
            padded_size = rows * neurons_per_row
            freq_grid = np.zeros(padded_size)
            act_grid = np.zeros(padded_size)
            
            freq_grid[:max_neurons] = firing_freq_data
            act_grid[:max_neurons] = mean_act_data
            
            freq_grid = freq_grid.reshape(rows, neurons_per_row)
            act_grid = act_grid.reshape(rows, neurons_per_row)
            
            # Plot firing frequency heatmap
            im1 = axes[i, 0].imshow(freq_grid, cmap=cmap, vmin=0, vmax=1, aspect='auto')
            axes[i, 0].set_title(f'{layer_name} - Firing Frequency')
            axes[i, 0].set_xlabel('Neuron Index (within row)')
            axes[i, 0].set_ylabel('Row')
            
            # Add colorbar
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Plot mean activation heatmap
            max_act = np.max(act_grid) if np.max(act_grid) > 0 else 1
            im2 = axes[i, 1].imshow(act_grid, cmap='viridis', vmin=0, vmax=max_act, aspect='auto')
            axes[i, 1].set_title(f'{layer_name} - Mean Activation (abs)')
            axes[i, 1].set_xlabel('Neuron Index (within row)')
            axes[i, 1].set_ylabel('Row')
            
            # Add colorbar
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Mark dead neurons with red X
            for local_idx, data in neuron_data.items():
                if data['is_dead']:
                    row = local_idx // neurons_per_row
                    col = local_idx % neurons_per_row
                    axes[i, 0].text(col, row, 'X', ha='center', va='center', 
                                   color='red', fontsize=8, fontweight='bold')
                    axes[i, 1].text(col, row, 'X', ha='center', va='center', 
                                   color='red', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('Neuron Activity Heatmaps by Layer', fontsize=16, y=1.02)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
            print("Heatmap displayed")
        else:
            plt.close()
    
    def save_report_to_file(self, filepath, include_heatmap=True):
        """
        Save a complete report to a text file.
        
        Args:
            filepath: Path to save the report
            include_heatmap: Whether to save heatmap as image
        """
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Generate report
            self.generate_neuron_report(show_heatmap=False, 
                                      save_heatmap=filepath.replace('.txt', '_heatmap.png') if include_heatmap else None)
        finally:
            sys.stdout = old_stdout
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(captured_output.getvalue())
        
        print(f"Report saved to: {filepath}")
        if include_heatmap:
            print(f"Heatmap saved to: {filepath.replace('.txt', '_heatmap.png')}")


class OptimizationRecommender:
    """
    Generates structured optimization recommendations in JSON format.
    """
    
    def __init__(self, tracker, analyzer):
        self.tracker = tracker
        self.analyzer = analyzer
    
    def generate_optimization_recommendations(self, 
                                            pruning_threshold=0.01,
                                            redundancy_threshold=0.9,
                                            saturation_threshold=0.95,
                                            performance_metrics=None):
        """
        Generate structured JSON recommendations for network optimization.
        
        Args:
            pruning_threshold: Firing frequency below which neurons are pruning candidates
            redundancy_threshold: Correlation above which neurons are redundant
            saturation_threshold: Activity level above which layer expansion might help
            performance_metrics: Dict with 'accuracy' and 'loss' for expansion decisions
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "model_info": {
                    "total_neurons": sum(info['neuron_count'] for info in self.tracker.layer_info.values()),
                    "total_layers": len(self.tracker.layer_info),
                    "layer_details": {name: info['neuron_count'] for name, info in self.tracker.layer_info.items()}
                },
                "analysis_parameters": {
                    "pruning_threshold": pruning_threshold,
                    "redundancy_threshold": redundancy_threshold,
                    "saturation_threshold": saturation_threshold,
                    "activation_threshold": self.tracker.activation_threshold
                }
            },
            "prune": [],
            "expand": [],
            "merge": [],
            "statistics": {
                "total_prunable_neurons": 0,
                "total_mergeable_pairs": 0,
                "layers_needing_expansion": 0
            }
        }
        
        # Get dataset statistics
        dataset_stats = self.analyzer.get_dataset_statistics()
        
        if not dataset_stats:
            print("Warning: No dataset statistics available for recommendations")
            return recommendations
        
        # Generate pruning recommendations
        prune_candidates = self._identify_pruning_candidates(dataset_stats, pruning_threshold)
        recommendations["prune"] = prune_candidates
        recommendations["statistics"]["total_prunable_neurons"] = len(prune_candidates)
        
        # Generate merge recommendations (if correlation data available)
        if self.tracker.activation_vectors:
            merge_candidates = self._identify_merge_candidates(redundancy_threshold)
            recommendations["merge"] = merge_candidates
            recommendations["statistics"]["total_mergeable_pairs"] = len(merge_candidates)
        
        # Generate expansion recommendations
        expansion_candidates = self._identify_expansion_candidates(
            dataset_stats, saturation_threshold, performance_metrics
        )
        recommendations["expand"] = expansion_candidates
        recommendations["statistics"]["layers_needing_expansion"] = len(expansion_candidates)
        
        return recommendations
    
    def _identify_pruning_candidates(self, dataset_stats, threshold):
        """Identify neurons that should be pruned."""
        prune_candidates = []
        
        for neuron_id, stats in dataset_stats.items():
            should_prune = (
                stats['is_dead'] or 
                stats['firing_frequency'] < threshold or
                abs(stats['mean_activation']) < threshold
            )
            
            if should_prune:
                layer_name, local_idx = self.tracker.get_neuron_info(neuron_id)
                prune_candidates.append({
                    "neuron_id": neuron_id,
                    "layer_name": layer_name,
                    "local_index": local_idx,
                    "reason": "dead" if stats['is_dead'] else "low_activity",
                    "firing_frequency": stats['firing_frequency'],
                    "mean_activation": stats['mean_activation']
                })
        
        # Sort by priority (dead neurons first, then by activity level)
        prune_candidates.sort(key=lambda x: (
            0 if x['reason'] == 'dead' else 1,
            x['firing_frequency']
        ))
        
        return prune_candidates
    
    def _identify_merge_candidates(self, correlation_threshold):
        """Identify pairs of neurons that could be merged due to redundancy."""
        merge_candidates = []
        
        try:
            redundant_pairs = self.analyzer.find_redundant_neurons(correlation_threshold)
            
            for layer_name, pairs in redundant_pairs.items():
                for neuron_id1, neuron_id2, correlation in pairs:
                    _, local_idx1 = self.tracker.get_neuron_info(neuron_id1)
                    _, local_idx2 = self.tracker.get_neuron_info(neuron_id2)
                    
                    merge_candidates.append({
                        "neuron_pair": [neuron_id1, neuron_id2],
                        "layer_name": layer_name,
                        "local_indices": [local_idx1, local_idx2],
                        "correlation": correlation,
                        "merge_strategy": "average_weights"
                    })
            
            # Sort by correlation strength (highest first)
            merge_candidates.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
        except Exception:
            # Correlation analysis not available
            pass
        
        return merge_candidates
    
    def _identify_expansion_candidates(self, dataset_stats, saturation_threshold, performance_metrics):
        """Identify layers that might benefit from expansion."""
        expansion_candidates = []
        
        # Analyze each layer for saturation
        layer_activities = defaultdict(list)
        
        for neuron_id, stats in dataset_stats.items():
            layer_name, _ = self.tracker.get_neuron_info(neuron_id)
            layer_activities[layer_name].append(stats['firing_frequency'])
        
        for layer_name, activities in layer_activities.items():
            avg_activity = np.mean(activities)
            high_activity_ratio = sum(1 for a in activities if a > saturation_threshold) / len(activities)
            
            needs_expansion = (
                avg_activity > saturation_threshold * 0.9 and
                high_activity_ratio > 0.5
            )
            
            # Additional check: if performance metrics suggest plateauing
            if performance_metrics:
                accuracy = performance_metrics.get('accuracy', 1.0)
                if accuracy < 0.9:  # If accuracy is not great, expansion might help
                    needs_expansion = needs_expansion or (avg_activity > 0.8)
            
            if needs_expansion:
                layer_info = self.tracker.layer_info[layer_name]
                expansion_candidates.append({
                    "layer_name": layer_name,
                    "current_size": layer_info['neuron_count'],
                    "suggested_expansion": max(int(layer_info['neuron_count'] * 0.25), 4),
                    "avg_activity": avg_activity,
                    "saturation_ratio": high_activity_ratio,
                    "reason": "high_saturation"
                })
        
        return expansion_candidates
    
    def save_optimization_recommendations(self, filepath, **kwargs):
        """
        Generate and save optimization recommendations to JSON file.
        
        Args:
            filepath: Path to save the JSON file
            **kwargs: Arguments to pass to generate_optimization_recommendations
        """
        recommendations = self.generate_optimization_recommendations(**kwargs)
        
        with open(filepath, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"Optimization recommendations saved to: {filepath}")
        return recommendations
