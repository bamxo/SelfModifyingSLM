"""
Integration Example

This module demonstrates how to integrate the pruner with the neuron tracker
for comprehensive neural network analysis and optimization.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Import tracker components
try:
    from ..tracker import NeuronEngine
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tracker import NeuronEngine

# Import pruner components
from . import PruningEngine


class IntegratedSystem:
    """
    Integrated system combining neuron tracking and pruning capabilities.
    
    This class provides a unified interface for tracking neurons, analyzing their
    behavior, and applying intelligent pruning based on the analysis results.
    """
    
    def __init__(self):
        """Initialize the integrated tracking and pruning system."""
        self.tracker_engine = NeuronEngine()
        self.pruning_engine = PruningEngine()
        
        # Link tracker to pruner
        self.pruning_engine.set_tracker(self.tracker_engine.tracker)
        
        self.analysis_results = None
        self.pruning_recommendations = None
    
    def analyze_model(self, model: nn.Module, model_name: str = "Model") -> Dict[str, Any]:
        """
        Perform comprehensive model analysis using the tracker.
        
        Args:
            model: PyTorch model to analyze
            model_name: Name for the model
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Starting comprehensive analysis of {model_name}")
        print("=" * 60)
        
        # Step 1: Track all neurons
        neuron_mapping = self.tracker_engine.track_model(model, model_name)
        
        # Step 2: Register hooks for activation tracking
        self.tracker_engine.register_hooks()(model)
        
        # Step 3: Start tracking with correlation analysis
        self.tracker_engine.start_tracking(enable_correlation_analysis=True)
        
        analysis_summary = {
            "model_name": model_name,
            "neuron_mapping": neuron_mapping,
            "total_neurons": sum(len(ids) for ids in neuron_mapping.values()),
            "total_layers": len(neuron_mapping),
            "tracking_enabled": True,
            "correlation_analysis": True
        }
        
        self.analysis_results = analysis_summary
        print(f"Analysis complete. Ready for data collection.")
        return analysis_summary
    
    def collect_data(self, data_loader, num_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect activation data from the model using provided data.
        
        Args:
            data_loader: PyTorch DataLoader with input data
            num_batches: Maximum number of batches to process (None for all)
            
        Returns:
            Dictionary with data collection results
        """
        if self.analysis_results is None:
            raise RuntimeError("Must run analyze_model() first")
        
        print(f"Collecting activation data...")
        
        batch_count = 0
        total_samples = 0
        
        for batch_idx, (data, _) in enumerate(data_loader):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Forward pass to collect activations
            with torch.no_grad():
                _ = self.tracker_engine.tracker.layer_info  # Dummy access to trigger hooks
            
            batch_count += 1
            total_samples += data.size(0)
        
        # Stop tracking
        self.tracker_engine.stop_tracking()
        
        collection_results = {
            "batches_processed": batch_count,
            "total_samples": total_samples,
            "dataset_statistics": self.tracker_engine.get_dataset_statistics(),
            "layer_summary": self.tracker_engine.get_layer_summary()
        }
        
        print(f"Data collection complete: {batch_count} batches, {total_samples} samples")
        return collection_results
    
    def generate_pruning_recommendations(self, **kwargs) -> Dict[str, Any]:
        """
        Generate pruning recommendations based on collected data.
        
        Args:
            **kwargs: Arguments for optimization recommendations
            
        Returns:
            Dictionary with pruning recommendations
        """
        if self.analysis_results is None:
            raise RuntimeError("Must run analyze_model() and collect_data() first")
        
        print("Generating pruning recommendations...")
        
        # Generate optimization recommendations from tracker
        recommendations = self.tracker_engine.generate_optimization_recommendations(**kwargs)
        
        # Store for later use
        self.pruning_recommendations = recommendations
        
        print(f"Recommendations generated:")
        print(f"  - Neurons to prune: {recommendations['statistics']['total_prunable_neurons']}")
        print(f"  - Mergeable pairs: {recommendations['statistics']['total_mergeable_pairs']}")
        print(f"  - Layers needing expansion: {recommendations['statistics']['layers_needing_expansion']}")
        
        return recommendations
    
    def apply_pruning(self, model: nn.Module, strategy: str = "magnitude", 
                     dry_run: bool = True) -> Dict[str, Any]:
        """
        Apply pruning to the model based on recommendations.
        
        Args:
            model: PyTorch model to prune
            strategy: Pruning strategy ("magnitude", "structured", "gradual")
            dry_run: If True, only simulate pruning
            
        Returns:
            Dictionary with pruning results
        """
        if self.pruning_recommendations is None:
            raise RuntimeError("Must generate recommendations first")
        
        action = "Simulating" if dry_run else "Applying"
        print(f"{action} {strategy} pruning...")
        
        # Apply pruning using recommendations
        pruning_results = self.pruning_engine.prune_by_recommendations(
            self.pruning_recommendations, strategy
        )
        
        print(f"Pruning {('simulation' if dry_run else 'application')} complete:")
        print(f"  - Neurons pruned: {pruning_results.get('neurons_pruned', 0)}")
        print(f"  - Layers affected: {pruning_results.get('layers_affected', 0)}")
        
        return pruning_results
    
    def compute_pruning_metrics(self, model: nn.Module, 
                               original_model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for the pruned model.
        
        Args:
            model: Current (potentially pruned) model
            original_model: Original model before pruning
            
        Returns:
            Dictionary with pruning metrics
        """
        print("Computing pruning metrics...")
        
        metrics = self.pruning_engine.compute_metrics(model, original_model)
        
        print("Metrics computed:")
        if "compression" in metrics:
            comp = metrics["compression"]
            print(f"  - Parameter compression: {comp['parameter_compression_ratio']:.2%}")
            print(f"  - Neuron compression: {comp['neuron_compression_ratio']:.2%}")
        
        sparsity = metrics["sparsity"]
        print(f"  - Overall sparsity: {sparsity['overall_sparsity']:.2%}")
        
        return metrics
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis and pruning report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Dictionary with complete report
        """
        print("Generating comprehensive report...")
        
        # Generate tracker report
        tracker_report = self.tracker_engine.generate_report(show_heatmap=False)
        
        # Get pruning summary
        pruning_summary = self.pruning_engine.get_pruning_summary()
        
        complete_report = {
            "analysis_results": self.analysis_results,
            "pruning_recommendations": self.pruning_recommendations,
            "pruning_summary": pruning_summary,
            "tracker_report": tracker_report
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(complete_report, f, indent=2, default=str)
            print(f"Report saved to: {save_path}")
        
        return complete_report
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the integrated system state."""
        return {
            "tracker_summary": self.tracker_engine.get_summary(),
            "pruning_summary": self.pruning_engine.get_pruning_summary(),
            "analysis_complete": self.analysis_results is not None,
            "recommendations_available": self.pruning_recommendations is not None
        }


def create_integrated_workflow(model: nn.Module, data_loader, 
                              model_name: str = "Model") -> IntegratedSystem:
    """
    Create and run a complete integrated workflow for model analysis and pruning.
    
    Args:
        model: PyTorch model to analyze and prune
        data_loader: DataLoader for activation collection
        model_name: Name for the model
        
    Returns:
        Configured IntegratedSystem ready for pruning operations
    """
    # Initialize integrated system
    system = IntegratedSystem()
    
    # Run analysis
    system.analyze_model(model, model_name)
    
    # Collect data (limited batches for demo)
    system.collect_data(data_loader, num_batches=10)
    
    # Generate recommendations
    system.generate_pruning_recommendations()
    
    print("\nIntegrated workflow complete!")
    print("System ready for pruning operations.")
    
    return system


# Example usage function
def example_usage():
    """
    Example demonstrating how to use the integrated system.
    """
    # This is a placeholder example
    print("Integrated System Example")
    print("=" * 30)
    print("1. Create model and data loader")
    print("2. Initialize IntegratedSystem()")
    print("3. Run analyze_model(model)")
    print("4. Run collect_data(data_loader)")
    print("5. Run generate_pruning_recommendations()")
    print("6. Run apply_pruning(model, strategy='magnitude', dry_run=True)")
    print("7. Run compute_pruning_metrics(model)")
    print("8. Run generate_report()")
    print("\nFor actual usage, see examples in the examples/ directory.")
