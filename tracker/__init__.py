"""
Neural Network Neuron Tracker Package

A comprehensive system for tracking, analyzing, and optimizing neural network neurons.
Provides functionality for neuron enumeration, activation analysis, correlation detection,
and optimization recommendations.
"""

from .tracker import NeuronTracker
from .analyzer import ActivationAnalyzer
from .reporter import ActivityReporter, OptimizationRecommender

__version__ = "1.0.0"
__all__ = ["NeuronTracker", "ActivationAnalyzer", "ActivityReporter", "OptimizationRecommender", "NeuronEngine"]


class NeuronEngine:
    """
    Main interface for the neuron tracking system.
    Combines all functionality into a single, easy-to-use class.
    """
    
    def __init__(self):
        self.tracker = NeuronTracker()
        self.analyzer = ActivationAnalyzer(self.tracker)
        self.reporter = ActivityReporter(self.tracker, self.analyzer)
        self.optimizer = OptimizationRecommender(self.tracker, self.analyzer)
    
    def track_model(self, model, model_name="Model"):
        """Track all neurons in the given model."""
        return self.tracker.track_model(model, model_name)
    
    def register_hooks(self):
        """Register activation hooks for tracking."""
        return self.analyzer.register_activation_hooks
    
    def start_tracking(self, enable_correlation_analysis=False):
        """Start tracking neuron activations."""
        return self.analyzer.start_tracking(enable_correlation_analysis)
    
    def stop_tracking(self):
        """Stop tracking neuron activations."""
        return self.analyzer.stop_tracking()
    
    def generate_report(self, show_heatmap=True, save_heatmap=None):
        """Generate comprehensive activity report."""
        return self.reporter.generate_neuron_report(show_heatmap, save_heatmap)
    
    def save_report(self, filepath, include_heatmap=True):
        """Save report to file."""
        return self.reporter.save_report_to_file(filepath, include_heatmap)
    
    def generate_optimization_recommendations(self, **kwargs):
        """Generate optimization recommendations."""
        return self.optimizer.generate_optimization_recommendations(**kwargs)
    
    def save_optimization_recommendations(self, filepath, **kwargs):
        """Save optimization recommendations to JSON."""
        return self.optimizer.save_optimization_recommendations(filepath, **kwargs)
    
    def get_summary(self):
        """Get model summary."""
        return self.tracker.get_model_summary()
    
    def get_dataset_statistics(self):
        """Get dataset-level neuron statistics."""
        return self.analyzer.get_dataset_statistics()
    
    def get_layer_summary(self):
        """Get layer-wise summary."""
        return self.analyzer.get_layer_summary()
    
    def find_redundant_neurons(self, threshold=None):
        """Find redundant neuron pairs."""
        return self.analyzer.find_redundant_neurons(threshold)
    
    def get_dead_neurons(self):
        """Get dead neurons by layer."""
        return self.analyzer.get_dead_neurons()
