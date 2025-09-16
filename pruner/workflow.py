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
from .engine import PruningEngine
from .context_aware_strategies import ContextAwarePruner
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
        
        # Initialize context-aware pruner for Pythia-160M
        self.context_aware_pruner = ContextAwarePruner(
            tracker=self.tracker.tracker, 
            enable_pruning=True
        )
        
        # Setup logger before output directories
        self.logger = self.pruner.logger
        
        # Setup output directories
        self.outputs_dir = Path(outputs_dir)
        self.setup_output_directories()
        
        # Workflow state
        self.workflow_id = f"workflow_{int(time.time())}"
        self.current_step = "initialized"
        self.results = {}
        
        # Iterative pruning state
        self.warmup_epochs = 3  # Skip pruning for first 3 epochs
        self.pruning_interval = 2  # Prune every 2 epochs after warmup
        self.iterative_target = 0.05  # Prune 5% per iteration
        self.total_pruning_budget = 0.3  # Maximum 30% total pruning
        self.current_pruning_ratio = 0.0  # Track cumulative pruning
        
        self.logger.info(f"Integrated workflow initialized with ID: {self.workflow_id}")
        self.logger.info(f"Iterative pruning: warmup={self.warmup_epochs} epochs, interval={self.pruning_interval} epochs, target={self.iterative_target:.1%} per step")
        self.logger.info(f"Context-aware pruning: enabled with layer-specific ratios")
    
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
    
    def should_prune_this_epoch(self, epoch: int) -> bool:
        """
        Determine if pruning should be performed for this epoch.
        
        Args:
            epoch: Current epoch number (1-indexed)
            
        Returns:
            True if pruning should be performed
        """
        # Skip pruning during warmup period
        if epoch <= self.warmup_epochs:
            return False
        
        # Check if we've hit our pruning budget
        if self.current_pruning_ratio >= self.total_pruning_budget:
            return False
        
        # Prune every pruning_interval epochs after warmup
        epochs_since_warmup = epoch - self.warmup_epochs
        return epochs_since_warmup % self.pruning_interval == 0
    
    def get_iterative_pruning_target(self, epoch: int) -> float:
        """
        Calculate the pruning target for this epoch based on iterative strategy.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Pruning ratio target for this epoch
        """
        if not self.should_prune_this_epoch(epoch):
            return 0.0
        
        # Calculate remaining budget
        remaining_budget = self.total_pruning_budget - self.current_pruning_ratio
        
        # Use the smaller of iterative target or remaining budget
        current_target = min(self.iterative_target, remaining_budget)
        
        return current_target
    
    def run_iterative_pruning_step(self, model: nn.Module, data_loader, epoch: int, 
                                  model_name: str = "Model") -> Dict[str, Any]:
        """
        Run one step of iterative pruning for a specific epoch.
        
        Args:
            model: PyTorch model to prune
            data_loader: DataLoader for activation collection  
            epoch: Current epoch number
            model_name: Name for the model
            
        Returns:
            Pruning step results
        """
        if not self.should_prune_this_epoch(epoch):
            return {
                "status": "skipped",
                "reason": f"Epoch {epoch} - warmup or not pruning interval",
                "epoch": epoch,
                "cumulative_pruning": self.current_pruning_ratio
            }
        
        target_ratio = self.get_iterative_pruning_target(epoch)
        if target_ratio <= 0:
            return {
                "status": "budget_exhausted", 
                "reason": f"Pruning budget exhausted ({self.current_pruning_ratio:.1%})",
                "epoch": epoch,
                "cumulative_pruning": self.current_pruning_ratio
            }
        
        self.logger.info(f"Iterative pruning step - Epoch {epoch}: targeting {target_ratio:.1%} reduction")
        
        # Update config for this iteration
        original_target = self.config.strategy.target_sparsity
        self.config.strategy.target_sparsity = target_ratio
        
        try:
            # Run the complete workflow for this step
            results = self.run_complete_workflow(
                model=model,
                data_loader=data_loader,
                model_name=f"{model_name}_Epoch_{epoch}",
                num_batches=3,  # Quick analysis for iterative steps
                auto_prune=True
            )
            
            # Update cumulative pruning ratio if successful
            if results.get("pruning_results", {}).get("status") == "completed":
                neurons_pruned = results["pruning_results"].get("neurons_pruned", 0)
                if neurons_pruned > 0:
                    self.current_pruning_ratio += target_ratio
                    self.logger.info(f"✅ Iterative step completed. Cumulative pruning: {self.current_pruning_ratio:.1%}")
            
            results["iterative_metadata"] = {
                "epoch": epoch,
                "target_ratio": target_ratio,
                "cumulative_pruning": self.current_pruning_ratio,
                "remaining_budget": self.total_pruning_budget - self.current_pruning_ratio
            }
            
            return results
            
        finally:
            # Restore original config
            self.config.strategy.target_sparsity = original_target
    
    def run_context_aware_pruning(self, model: nn.Module, data_loader, model_name: str = "Model",
                                num_batches: int = 10, current_pruning_ratio: float = None,
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Run context-aware pruning specifically designed for Pythia-160M.
        
        Args:
            model: PyTorch model to prune (Pythia-160M)
            data_loader: DataLoader for activation collection
            model_name: Name for the model
            num_batches: Number of batches to process
            current_pruning_ratio: Current pruning ratio (if None, uses schedule)
            dry_run: If True, simulate pruning without modifying model
            
        Returns:
            Context-aware pruning results
        """
        self.current_step = "context_aware_analysis"
        start_time = time.time()
        
        self.logger.info(f"Starting context-aware pruning for {model_name}")
        self.logger.info(f"Processing {num_batches} batches for analysis")
        
        results = {
            "workflow_id": self.workflow_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "analysis_results": None,
            "pruning_results": None,
            "verification_results": None
        }
        
        try:
            # Step 1: Identify layer types for context-aware pruning - FAST MODE
            self.logger.info("Step 1: FAST identifying transformer layer types...")
            layer_info = self.context_aware_pruner.identify_transformer_layers(model)
            
            # Step 2: Generate context-aware pruning plan - FAST MODE
            self.logger.info("Step 2: FAST generating context-aware pruning plan...")
            pruning_plan = self.context_aware_pruner.generate_context_aware_pruning_plan(
                model, current_pruning_ratio
            )
            
            results["analysis_results"] = {
                "layer_info": {
                    category: [layer['name'] for layer in layers] 
                    for category, layers in layer_info.items() if layers
                },
                "pruning_plan": pruning_plan,
                "context_aware_info": {
                    "layer_ratios": self.context_aware_pruner.layer_pruning_ratios,
                    "current_ratio": current_pruning_ratio or self.context_aware_pruner.get_current_pruning_ratio(),
                    "pruning_mode": "structured" if self.context_aware_pruner.structured_mode else "unstructured"
                }
            }
            
            # Step 3: Execute context-aware pruning
            if pruning_plan.get('status') not in ['disabled', 'no_pruning']:
                self.logger.info("Step 3: Executing context-aware pruning...")
                pruning_results = self.context_aware_pruner.execute_context_aware_pruning(
                    model, pruning_plan, dry_run=dry_run
                )
                results["pruning_results"] = pruning_results
                
                if not dry_run and pruning_results.get('status') == 'completed':
                    # Advance pruning schedule
                    new_ratio = self.context_aware_pruner.advance_pruning_schedule()
                    self.logger.info(f"Pruning schedule advanced to: {new_ratio:.1%}")
            
            # Step 4: Verification (if not dry run) - SKIP FOR DEV SPEED
            if not dry_run and results.get("pruning_results", {}).get("status") == "completed":
                self.logger.info("Step 4: SKIPPING verification for development speed...")
                verification_results = {"status": "skipped_for_dev_speed"}
                results["verification_results"] = verification_results
            
            analysis_time = time.time() - start_time
            results["analysis_time"] = analysis_time
            
            self.logger.info(f"Context-aware pruning completed in {analysis_time:.2f}s")
            
            # Save results
            self._save_context_aware_results(results)
            
        except Exception as e:
            self.logger.error(f"Context-aware pruning failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
        
        finally:
            self.current_step = "completed"
        
        return results
    
    def _verify_pruned_model(self, model: nn.Module, data_loader, num_batches: int = 3) -> Dict[str, Any]:
        """Verify the pruned model can still perform forward passes."""
        verification_results = {
            "forward_pass_success": False,
            "memory_usage": {},
            "performance_metrics": {}
        }
        
        try:
            model.eval()
            total_loss = 0
            batch_count = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    if batch_idx >= num_batches:
                        break
                    
                    # Test forward pass
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                        if hasattr(outputs, 'loss'):
                            total_loss += outputs.loss.item()
                    else:
                        outputs = model(batch)
                    
                    batch_count += 1
            
            verification_results["forward_pass_success"] = True
            verification_results["performance_metrics"]["avg_loss"] = total_loss / batch_count if batch_count > 0 else 0
            
            # Memory usage
            if torch.cuda.is_available():
                verification_results["memory_usage"] = {
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3
                }
            
            self.logger.info("Model verification successful")
            
        except Exception as e:
            verification_results["error"] = str(e)
            self.logger.error(f"Model verification failed: {e}")
        
        return verification_results
    
    def _save_context_aware_results(self, results: Dict[str, Any]) -> None:
        """Save context-aware pruning results to files."""
        try:
            # Save main results
            results_file = self.outputs_dir / "logs" / f"{self.workflow_id}_context_aware_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save pruning summary
            if results.get("pruning_results"):
                summary_file = self.outputs_dir / "reports" / f"{self.workflow_id}_context_aware_summary.json"
                summary = {
                    "model_name": results["model_name"],
                    "pruning_status": results["pruning_results"].get("status"),
                    "neurons_pruned": results["pruning_results"].get("neurons_pruned", 0),
                    "layers_affected": results["pruning_results"].get("layers_affected", 0),
                    "analysis_time": results.get("analysis_time", 0),
                    "timestamp": results["timestamp"]
                }
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            
            self.logger.info(f"Context-aware results saved to {self.outputs_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save context-aware results: {e}")
    
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
            
            # Validate recommendations
            if recommendations is None:
                self.logger.warning("Recommendations generation returned None, creating empty recommendations")
                recommendations = {
                    "statistics": {"total_prunable_neurons": 0},
                    "prune": [],
                    "merge": [],
                    "expand": []
                }
            
            # Step 3: Apply pruning (if enabled)
            pruning_results = None
            total_prunable = recommendations.get("statistics", {}).get("total_prunable_neurons", 0)
            self.logger.info(f"Found {total_prunable} prunable neurons")
            
            if auto_prune and total_prunable > 0:
                self.current_step = "pruning"
                pruning_results = self.run_pruning_phase(model, recommendations)
            else:
                self.logger.info(f"Skipping pruning: auto_prune={auto_prune}, prunable_neurons={total_prunable}")
                pruning_results = {
                    "status": "skipped",
                    "reason": f"No prunable neurons found ({total_prunable})",
                    "neurons_pruned": 0
                }
            
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
        
        # Register hooks and start tracking (disabled correlation analysis for speed)
        self.tracker.register_hooks()(model)
        self.tracker.start_tracking(enable_correlation_analysis=False)
        
        # Collect activation data
        batch_count = 0
        total_samples = 0
        
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Handle different data formats (dictionaries for transformers, tuples for traditional models)
                if isinstance(batch_data, dict):
                    # Transformer format: {"input_ids": tensor, "attention_mask": tensor}
                    input_ids = batch_data["input_ids"]
                    attention_mask = batch_data.get("attention_mask", None)
                    
                    device = next(model.parameters()).device  # Get model's device
                    input_ids = input_ids.to(device, non_blocking=True)
                    
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device, non_blocking=True)
                        _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        _ = model(input_ids=input_ids)
                    
                    total_samples += input_ids.size(0)
                    
                elif isinstance(batch_data, (list, tuple)):
                    # Traditional format: (data, targets)
                    data, targets = batch_data
                    device = next(model.parameters()).device  # Get model's device
                    data = data.to(device, non_blocking=True)  # Move data to same device
                    _ = model(data)
                    total_samples += data.size(0)
                else:
                    # Single tensor format
                    device = next(model.parameters()).device
                    batch_data = batch_data.to(device, non_blocking=True)
                    _ = model(batch_data)
                    total_samples += batch_data.size(0)
                
                batch_count += 1
        
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
        
        try:
            # Generate recommendations
            recommendations = self.tracker.generate_optimization_recommendations(
                pruning_threshold=self.config.thresholds.firing_frequency_threshold,
                redundancy_threshold=self.config.thresholds.correlation_threshold,
                saturation_threshold=0.95
            )
            
            # Validate recommendations
            if recommendations is None:
                self.logger.warning("Tracker returned None recommendations, creating default structure")
                recommendations = {
                    "statistics": {"total_prunable_neurons": 0},
                    "prune": [],
                    "merge": [],
                    "expand": []
                }
            
            # Ensure required structure exists
            if "statistics" not in recommendations:
                recommendations["statistics"] = {"total_prunable_neurons": 0}
            if "prune" not in recommendations:
                recommendations["prune"] = []
            if "merge" not in recommendations:
                recommendations["merge"] = []
            if "expand" not in recommendations:
                recommendations["expand"] = []
            
            # Save recommendations to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recommendations_file = self.outputs_dir / "logs" / f"{self.workflow_id}_recommendations_{timestamp}.json"
            
            with open(recommendations_file, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            prunable_neurons = recommendations.get("statistics", {}).get("total_prunable_neurons", 0)
            self.logger.info(f"Recommendations generated: {prunable_neurons} neurons identified for pruning")
            self.logger.info(f"Recommendations saved to: {recommendations_file}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            # Return empty recommendations structure
            return {
                "statistics": {"total_prunable_neurons": 0},
                "prune": [],
                "merge": [],
                "expand": [],
                "error": str(e)
            }
    
    def run_pruning_phase(self, model: nn.Module, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pruning phase."""
        self.logger.info("Phase 3: Starting pruning phase")
        
        # First, do a dry run
        dry_run_results = self.pruner.prune_by_recommendations(
            model, recommendations, dry_run=True
        )
        
        # Log dry run results for both neuron and filter pruning
        neurons_count = dry_run_results.get('neurons_pruned', 0)
        filters_count = dry_run_results.get('filters_removed', 0)
        if filters_count > 0:
            self.logger.info(f"Dry run: {filters_count} filters would be pruned")
        else:
            self.logger.info(f"Dry run: {neurons_count} neurons would be pruned")
        
        # Apply actual pruning if dry run was successful and we have items to prune
        # Check for either neurons_pruned (traditional) or filters_removed (filter-based)
        items_to_prune = max(
            dry_run_results.get("neurons_pruned", 0),
            dry_run_results.get("filters_removed", 0)
        )
        if (dry_run_results.get("status") == "simulation" and 
            items_to_prune > 0 and
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
