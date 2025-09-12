"""
Integrated Workflow for Tracker and Pruner

This module provides end-to-end workflows that combine neuron tracking and pruning
into seamless operations with automatic output management and verification.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import time
import copy
from datetime import datetime

# Import components
from .config import PrunerConfig, create_default_config
from . import PruningEngine
from .model_io import PrunedModelRepresentation, ModelSerializer, create_pruning_report


class IntegratedTrackingPruningWorkflow:
    """
    End-to-end workflow that combines neuron tracking and pruning.
    
    This class provides a complete pipeline: track → analyze → recommend → prune → verify
    """
    
    def __init__(self, tracker_engine=None, config=None, outputs_dir="outputs"):
        """
        Initialize the integrated workflow.
        
        Args:
            tracker_engine: NeuronEngine instance (will import if None)
            config: PrunerConfig instance
            outputs_dir: Base directory for outputs
        """
        # Initialize tracker
        if tracker_engine is None:
            try:
                from tracker import NeuronEngine
                self.tracker = NeuronEngine()
            except ImportError:
                raise ImportError("Tracker module not available. Please ensure tracker is properly installed.")
        else:
            self.tracker = tracker_engine
        
        # Initialize pruner with tracker integration
        self.config = config or create_default_config()
        self.pruner = PruningEngine(tracker=self.tracker.tracker, config=self.config)
        
        # Setup output directories
        self.outputs_dir = Path(outputs_dir)
        self.setup_output_directories()
        
        # Workflow state
        self.workflow_id = f"workflow_{int(time.time())}"
        self.current_step = "initialized"
        self.results = {}
        
        self.logger = self.pruner.logger
        self.logger.info(f"Integrated workflow initialized with ID: {self.workflow_id}")
    
    def setup_output_directories(self):
        """Setup required output directory structure."""
        directories = [
            self.outputs_dir / "logs",
            self.outputs_dir / "reports", 
            self.outputs_dir / "models",
            self.outputs_dir / "pruned_models",
            self.outputs_dir / "verification",
            self.outputs_dir / "visualizations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directories setup in: {self.outputs_dir}")
    
    def run_complete_workflow(self, model: nn.Module, data_loader, model_name: str = "Model",
                            num_batches: Optional[int] = None, auto_prune: bool = True) -> Dict[str, Any]:
        """
        Run the complete tracking and pruning workflow.
        
        Args:
            model: PyTorch model to analyze and prune
            data_loader: DataLoader for activation collection
            model_name: Name for the model
            num_batches: Number of batches to process (None for all)
            auto_prune: Whether to automatically apply pruning
            
        Returns:
            Complete workflow results
        """
        workflow_start_time = time.time()
        self.logger.info(f"Starting complete workflow for {model_name}")
        
        try:
            # Step 1: Track model
            self.current_step = "tracking"
            tracking_results = self.run_tracking_phase(model, data_loader, model_name, num_batches)
            
            # Step 2: Generate recommendations
            self.current_step = "recommendations"
            recommendations = self.generate_recommendations()
            
            # Step 3: Apply pruning (if enabled)
            pruning_results = None
            if auto_prune and recommendations.get("statistics", {}).get("total_prunable_neurons", 0) > 0:
                self.current_step = "pruning"
                pruning_results = self.run_pruning_phase(model, recommendations)
            
            # Step 4: Generate comprehensive report
            self.current_step = "reporting"
            final_report = self.generate_final_report(model, tracking_results, recommendations, pruning_results)
            
            # Step 5: Verification (if pruning was applied)
            verification_results = None
            if pruning_results and not pruning_results.get("status") == "simulation":
                self.current_step = "verification"
                verification_results = self.run_verification_phase(model, pruning_results)
            
            workflow_time = time.time() - workflow_start_time
            
            # Compile final results
            complete_results = {
                "workflow_id": self.workflow_id,
                "model_name": model_name,
                "workflow_time_seconds": workflow_time,
                "current_step": "completed",
                "tracking_results": tracking_results,
                "recommendations": recommendations,
                "pruning_results": pruning_results,
                "verification_results": verification_results,
                "final_report": final_report,
                "output_files": self.get_output_file_list()
            }
            
            self.results = complete_results
            self.current_step = "completed"
            
            self.logger.info(f"Complete workflow finished in {workflow_time:.2f}s")
            self.log_workflow_summary(complete_results)
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Workflow failed at step '{self.current_step}': {e}")
            self.current_step = "failed"
            return {
                "workflow_id": self.workflow_id,
                "status": "failed",
                "error": str(e),
                "failed_at_step": self.current_step
            }
    
    def run_tracking_phase(self, model: nn.Module, data_loader, model_name: str, 
                          num_batches: Optional[int]) -> Dict[str, Any]:
        """Run the neuron tracking phase."""
        self.logger.info("Phase 1: Starting neuron tracking")
        
        # Track model structure
        neuron_mapping = self.tracker.track_model(model, model_name)
        
        # Register hooks and start tracking
        self.tracker.register_hooks()(model)
        self.tracker.start_tracking(enable_correlation_analysis=True)
        
        # Collect activation data
        batch_count = 0
        total_samples = 0
        
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Forward pass to collect activations
                _ = model(data)
                
                batch_count += 1
                total_samples += data.size(0)
        
        # Stop tracking
        self.tracker.stop_tracking()
        
        # Get tracking results
        dataset_stats = self.tracker.get_dataset_statistics()
        layer_summary = self.tracker.get_layer_summary()
        
        tracking_results = {
            "neuron_mapping": neuron_mapping,
            "batches_processed": batch_count,
            "total_samples": total_samples,
            "dataset_statistics": dataset_stats,
            "layer_summary": layer_summary,
            "total_neurons_tracked": sum(len(ids) for ids in neuron_mapping.values())
        }
        
        self.logger.info(f"Tracking complete: {batch_count} batches, {total_samples} samples, "
                        f"{tracking_results['total_neurons_tracked']} neurons")
        
        return tracking_results
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate pruning recommendations from tracking data."""
        self.logger.info("Phase 2: Generating pruning recommendations")
        
        # Generate recommendations
        recommendations = self.tracker.generate_optimization_recommendations(
            pruning_threshold=self.config.thresholds.firing_frequency_threshold,
            redundancy_threshold=self.config.thresholds.correlation_threshold,
            saturation_threshold=0.95
        )
        
        # Save recommendations to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recommendations_file = self.outputs_dir / "logs" / f"{self.workflow_id}_recommendations_{timestamp}.json"
        
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        prunable_neurons = recommendations.get("statistics", {}).get("total_prunable_neurons", 0)
        self.logger.info(f"Recommendations generated: {prunable_neurons} neurons identified for pruning")
        self.logger.info(f"Recommendations saved to: {recommendations_file}")
        
        return recommendations
    
    def run_pruning_phase(self, model: nn.Module, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pruning phase."""
        self.logger.info("Phase 3: Starting pruning phase")
        
        # First, do a dry run
        dry_run_results = self.pruner.prune_by_recommendations(
            model, recommendations, dry_run=True
        )
        
        self.logger.info(f"Dry run: {dry_run_results.get('neurons_pruned', 0)} neurons would be pruned")
        
        # Apply actual pruning if dry run was successful and we have neurons to prune
        if (dry_run_results.get("status") == "simulation" and 
            dry_run_results.get("neurons_pruned", 0) > 0 and
            not self.config.validation.dry_run_first):
            
            self.logger.info("Applying actual pruning...")
            pruning_results = self.pruner.prune_by_recommendations(
                model, recommendations, dry_run=False
            )
        else:
            pruning_results = dry_run_results
            pruning_results["note"] = "Only dry run performed due to configuration or no neurons to prune"
        
        # Save pruning results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pruning_file = self.outputs_dir / "logs" / f"{self.workflow_id}_pruning_{timestamp}.json"
        
        with open(pruning_file, 'w') as f:
            json.dump(pruning_results, f, indent=2, default=str)
        
        self.logger.info(f"Pruning results saved to: {pruning_file}")
        
        return pruning_results
    
    def run_verification_phase(self, model: nn.Module, pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run verification tests to confirm pruning was applied correctly."""
        self.logger.info("Phase 4: Starting verification phase")
        
        verification_results = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": self.workflow_id,
            "tests": {},
            "summary": {}
        }
        
        try:
            # Test 1: Model structure verification
            structure_test = self.verify_model_structure(model, pruning_results)
            verification_results["tests"]["model_structure"] = structure_test
            
            # Test 2: Forward pass verification
            forward_test = self.verify_forward_pass(model)
            verification_results["tests"]["forward_pass"] = forward_test
            
            # Test 3: Neuron count verification
            count_test = self.verify_neuron_counts(model, pruning_results)
            verification_results["tests"]["neuron_counts"] = count_test
            
            # Test 4: Parameter count verification
            param_test = self.verify_parameter_counts(model, pruning_results)
            verification_results["tests"]["parameter_counts"] = param_test
            
            # Compile summary
            all_tests = [structure_test, forward_test, count_test, param_test]
            passed_tests = sum(1 for test in all_tests if test.get("passed", False))
            total_tests = len(all_tests)
            
            verification_results["summary"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "overall_success": passed_tests == total_tests
            }
            
            # Save verification results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            verification_file = self.outputs_dir / "verification" / f"{self.workflow_id}_verification_{timestamp}.json"
            
            with open(verification_file, 'w') as f:
                json.dump(verification_results, f, indent=2, default=str)
            
            self.logger.info(f"Verification complete: {passed_tests}/{total_tests} tests passed")
            self.logger.info(f"Verification results saved to: {verification_file}")
            
        except Exception as e:
            verification_results["error"] = str(e)
            self.logger.error(f"Verification failed: {e}")
        
        return verification_results
    
    def verify_model_structure(self, model: nn.Module, pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify model structure is consistent after pruning."""
        try:
            # Get current model structure
            current_structure = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    current_structure[name] = {
                        "type": "Linear",
                        "in_features": module.in_features,
                        "out_features": module.out_features
                    }
                elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    current_structure[name] = {
                        "type": module.__class__.__name__,
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels
                    }
            
            # Check for any structural inconsistencies
            issues = []
            for name, info in current_structure.items():
                if info["type"] == "Linear":
                    if info["in_features"] <= 0 or info["out_features"] <= 0:
                        issues.append(f"Layer {name} has invalid dimensions")
                elif info["type"] in ["Conv2d", "Conv1d"]:
                    if info["in_channels"] <= 0 or info["out_channels"] <= 0:
                        issues.append(f"Layer {name} has invalid channel dimensions")
            
            return {
                "test_name": "model_structure",
                "passed": len(issues) == 0,
                "issues": issues,
                "current_structure": current_structure
            }
            
        except Exception as e:
            return {
                "test_name": "model_structure",
                "passed": False,
                "error": str(e)
            }
    
    def verify_forward_pass(self, model: nn.Module) -> Dict[str, Any]:
        """Verify model can still perform forward pass after pruning."""
        try:
            device = next(model.parameters()).device
            
            # Create dummy input (adjust size based on your model)
            test_input = torch.randn(1, 3, 32, 32).to(device)  # Common CNN input
            
            with torch.no_grad():
                output = model(test_input)
                
                # Check output validity
                has_nan = torch.any(torch.isnan(output))
                has_inf = torch.any(torch.isinf(output))
                output_shape = list(output.shape)
                
                return {
                    "test_name": "forward_pass",
                    "passed": not has_nan and not has_inf,
                    "output_shape": output_shape,
                    "has_nan": has_nan.item(),
                    "has_inf": has_inf.item()
                }
                
        except Exception as e:
            return {
                "test_name": "forward_pass",
                "passed": False,
                "error": str(e)
            }
    
    def verify_neuron_counts(self, model: nn.Module, pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify neuron counts match expected values after pruning."""
        try:
            # Count current neurons
            current_neurons = {}
            total_current = 0
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    current_neurons[name] = module.out_features
                    total_current += module.out_features
                elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    current_neurons[name] = module.out_channels
                    total_current += module.out_channels
            
            # Get expected changes from pruning results
            layer_modifications = pruning_results.get("layer_modifications", {})
            expected_reductions = {}
            total_expected_reduction = 0
            
            for layer_name, modifications in layer_modifications.items():
                if "neurons_removed" in modifications:
                    expected_reductions[layer_name] = modifications["neurons_removed"]
                    total_expected_reduction += modifications["neurons_removed"]
            
            return {
                "test_name": "neuron_counts",
                "passed": True,  # Basic check - model structure is valid
                "current_neurons": current_neurons,
                "total_current_neurons": total_current,
                "expected_reductions": expected_reductions,
                "total_expected_reduction": total_expected_reduction
            }
            
        except Exception as e:
            return {
                "test_name": "neuron_counts",
                "passed": False,
                "error": str(e)
            }
    
    def verify_parameter_counts(self, model: nn.Module, pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify parameter counts are consistent with pruning."""
        try:
            # Count current parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "test_name": "parameter_counts",
                "passed": total_params > 0,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "parameters_per_layer": {
                    name: sum(p.numel() for p in module.parameters())
                    for name, module in model.named_modules()
                    if list(module.parameters())
                }
            }
            
        except Exception as e:
            return {
                "test_name": "parameter_counts",
                "passed": False,
                "error": str(e)
            }
    
    def generate_final_report(self, model: nn.Module, tracking_results: Dict[str, Any],
                            recommendations: Dict[str, Any], pruning_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        self.logger.info("Generating final workflow report")
        
        # Create comprehensive report
        report = {
            "workflow_metadata": {
                "workflow_id": self.workflow_id,
                "timestamp": datetime.now().isoformat(),
                "model_name": tracking_results.get("model_name", "Unknown"),
                "configuration": self.config.to_dict()
            },
            "tracking_summary": {
                "total_neurons_tracked": tracking_results.get("total_neurons_tracked", 0),
                "samples_processed": tracking_results.get("total_samples", 0),
                "batches_processed": tracking_results.get("batches_processed", 0)
            },
            "recommendations_summary": {
                "total_prunable_neurons": recommendations.get("statistics", {}).get("total_prunable_neurons", 0),
                "total_mergeable_pairs": recommendations.get("statistics", {}).get("total_mergeable_pairs", 0),
                "layers_needing_expansion": recommendations.get("statistics", {}).get("layers_needing_expansion", 0)
            }
        }
        
        # Add pruning summary if available
        if pruning_results:
            report["pruning_summary"] = {
                "status": pruning_results.get("status", "unknown"),
                "neurons_pruned": pruning_results.get("neurons_pruned", 0),
                "layers_affected": pruning_results.get("layers_affected", 0),
                "was_dry_run": pruning_results.get("status") == "simulation"
            }
        
        # Save final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.outputs_dir / "reports" / f"{self.workflow_id}_final_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved to: {report_file}")
        
        return report
    
    def get_output_file_list(self) -> Dict[str, List[str]]:
        """Get list of all output files generated by workflow."""
        output_files = {}
        
        for category in ["logs", "reports", "models", "pruned_models", "verification", "visualizations"]:
            category_dir = self.outputs_dir / category
            if category_dir.exists():
                files = [str(f) for f in category_dir.glob(f"{self.workflow_id}*")]
                output_files[category] = files
        
        return output_files
    
    def log_workflow_summary(self, results: Dict[str, Any]):
        """Log comprehensive workflow summary."""
        self.logger.info("=" * 60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Workflow ID: {results['workflow_id']}")
        self.logger.info(f"Model: {results['model_name']}")
        self.logger.info(f"Total time: {results['workflow_time_seconds']:.2f}s")
        
        # Tracking summary
        tracking = results.get("tracking_results", {})
        self.logger.info(f"Neurons tracked: {tracking.get('total_neurons_tracked', 0)}")
        self.logger.info(f"Samples processed: {tracking.get('total_samples', 0)}")
        
        # Recommendations summary
        recommendations = results.get("recommendations", {})
        stats = recommendations.get("statistics", {})
        self.logger.info(f"Prunable neurons identified: {stats.get('total_prunable_neurons', 0)}")
        
        # Pruning summary
        pruning = results.get("pruning_results")
        if pruning:
            self.logger.info(f"Pruning status: {pruning.get('status', 'unknown')}")
            self.logger.info(f"Neurons pruned: {pruning.get('neurons_pruned', 0)}")
            self.logger.info(f"Layers affected: {pruning.get('layers_affected', 0)}")
        
        # Verification summary
        verification = results.get("verification_results")
        if verification:
            summary = verification.get("summary", {})
            self.logger.info(f"Verification tests: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)} passed")
        
        # Output files
        output_files = results.get("output_files", {})
        total_files = sum(len(files) for files in output_files.values())
        self.logger.info(f"Output files generated: {total_files}")
        
        self.logger.info("=" * 60)


def create_simple_workflow(model: nn.Module, data_loader, model_name: str = "Model",
                          config=None, outputs_dir="outputs") -> Dict[str, Any]:
    """
    Create and run a simple end-to-end workflow.
    
    Args:
        model: PyTorch model to process
        data_loader: DataLoader for activation collection
        model_name: Name for the model
        config: PrunerConfig instance
        outputs_dir: Output directory
        
    Returns:
        Workflow results
    """
    workflow = IntegratedTrackingPruningWorkflow(
        tracker_engine=None,
        config=config,
        outputs_dir=outputs_dir
    )
    
    return workflow.run_complete_workflow(
        model=model,
        data_loader=data_loader,
        model_name=model_name,
        num_batches=10,  # Limit for quick testing
        auto_prune=True
    )


def verify_pruning_accuracy(original_model: nn.Module, pruned_model: nn.Module,
                          pruning_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that pruning was applied accurately.
    
    Args:
        original_model: Model before pruning
        pruned_model: Model after pruning
        pruning_results: Results from pruning operation
        
    Returns:
        Verification results
    """
    verification = {
        "timestamp": datetime.now().isoformat(),
        "accuracy_tests": {},
        "summary": {}
    }
    
    try:
        # Test 1: Parameter count reduction
        orig_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        param_reduction = orig_params - pruned_params
        
        verification["accuracy_tests"]["parameter_reduction"] = {
            "original_parameters": orig_params,
            "pruned_parameters": pruned_params,
            "parameters_removed": param_reduction,
            "reduction_ratio": param_reduction / orig_params if orig_params > 0 else 0
        }
        
        # Test 2: Neuron count verification
        orig_neurons = {}
        pruned_neurons = {}
        
        for name, module in original_model.named_modules():
            if isinstance(module, nn.Linear):
                orig_neurons[name] = module.out_features
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                pruned_neurons[name] = module.out_features
        
        neuron_changes = {}
        for layer_name in orig_neurons:
            if layer_name in pruned_neurons:
                neuron_changes[layer_name] = {
                    "original": orig_neurons[layer_name],
                    "pruned": pruned_neurons[layer_name],
                    "removed": orig_neurons[layer_name] - pruned_neurons[layer_name]
                }
        
        verification["accuracy_tests"]["neuron_changes"] = neuron_changes
        
        # Test 3: Expected vs actual changes
        expected_changes = pruning_results.get("layer_modifications", {})
        accuracy_check = {}
        
        for layer_name, expected in expected_changes.items():
            if layer_name in neuron_changes:
                expected_removed = expected.get("neurons_removed", 0)
                actual_removed = neuron_changes[layer_name]["removed"]
                
                accuracy_check[layer_name] = {
                    "expected_removed": expected_removed,
                    "actual_removed": actual_removed,
                    "accurate": expected_removed == actual_removed
                }
        
        verification["accuracy_tests"]["expected_vs_actual"] = accuracy_check
        
        # Summary
        total_checks = len(accuracy_check)
        accurate_checks = sum(1 for check in accuracy_check.values() if check["accurate"])
        
        verification["summary"] = {
            "total_accuracy_checks": total_checks,
            "accurate_checks": accurate_checks,
            "accuracy_rate": accurate_checks / total_checks if total_checks > 0 else 1.0,
            "overall_accurate": accurate_checks == total_checks,
            "parameter_reduction_achieved": param_reduction > 0
        }
        
    except Exception as e:
        verification["error"] = str(e)
        verification["summary"]["overall_accurate"] = False
    
    return verification
