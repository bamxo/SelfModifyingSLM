# Neural Network Pruner Module

A comprehensive system for pruning neural network neurons based on tracker analysis. This module integrates cleanly with the neuron tracker system to provide intelligent, data-driven pruning capabilities.

## Module Structure

```
pruner/
├── __init__.py          # Main interfaces and PruningEngine
├── core.py              # Core pruning functionality
├── strategies.py        # Different pruning strategies
├── utils.py             # Metrics, validation, and utilities
├── integration.py       # Integration examples with tracker
└── README.md           # This file
```

## Core Components

### Main Classes

- **`PruningEngine`** - Main interface combining all pruning functionality
- **`NeuronPruner`** - Core pruning implementation for individual neurons
- **`MagnitudePruner`** - Magnitude-based pruning strategy
- **`StructuredPruner`** - Structured pruning for channels/filters
- **`GradualPruner`** - Gradual pruning over multiple epochs
- **`PruningMetrics`** - Comprehensive metrics computation
- **`PruningValidator`** - Model validation after pruning
- **`IntegratedSystem`** - Complete workflow with tracker integration

### Key Features

1. **Multiple Pruning Strategies**
   - Magnitude-based pruning
   - Structured pruning (channels/filters)
   - Gradual pruning with configurable schedules

2. **Tracker Integration**
   - Uses neuron tracker recommendations
   - Leverages dead neuron detection
   - Incorporates activity analysis

3. **Comprehensive Metrics**
   - Model size reduction
   - Sparsity analysis
   - Inference speedup estimation
   - Memory savings calculation

4. **Validation & Safety**
   - Model structure validation
   - Forward pass testing
   - Dry-run simulation mode

## Quick Start

### Basic Usage

```python
from pruner import PruningEngine, create_default_config
from tracker import NeuronEngine

# Method 1: Use defaults
pruner = PruningEngine()

# Method 2: Use configuration file
pruner = PruningEngine(config_path="configs/conservative.yaml")

# Method 3: Use custom configuration
config = create_default_config().update(
    strategy={"target_sparsity": 0.4},
    logging={"log_level": "DEBUG"}
)
pruner = PruningEngine(config=config)

# Prune from tracker recommendations
results = pruner.prune_from_json("outputs/logs/epoch_5_recommendations.json")
```

### Configuration-Based Usage

```python
from pruner import PruningEngine, create_conservative_config

# Conservative pruning (safe, minimal compression)
pruner = PruningEngine(config=create_conservative_config())
results = pruner.prune_by_recommendations(model, recommendations)

# Runtime configuration changes
pruner.set_aggressive_mode()  # Switch to aggressive pruning
pruner.update_config(**{"strategy.target_sparsity": 0.6})  # Custom update
```

### Integrated Workflow

```python
from pruner import IntegratedTrackingPruningWorkflow, create_simple_workflow

# Method 1: Complete automated workflow
results = create_simple_workflow(
    model=model,
    data_loader=data_loader,
    model_name="MyModel",
    config=create_conservative_config()
)

# Method 2: Step-by-step workflow
workflow = IntegratedTrackingPruningWorkflow()
results = workflow.run_complete_workflow(
    model=model,
    data_loader=data_loader,
    model_name="MyModel",
    auto_prune=True
)

# Access results
print(f"Neurons pruned: {results['pruning_results']['neurons_pruned']}")
print(f"Verification passed: {results['verification_results']['summary']['overall_success']}")
```

## Pruning Strategies

### 1. Magnitude-Based Pruning
Removes neurons with the lowest weight magnitudes.

```python
results = pruner.prune_by_magnitude(model, sparsity_ratio=0.3, dry_run=True)
```

### 2. Structured Pruning
Removes entire channels or filters for hardware efficiency.

```python
layer_sparsity = {"conv1": 0.25, "fc1": 0.5}
results = pruner.prune_structured(model, layer_sparsity, dry_run=True)
```

### 3. Gradual Pruning
Progressively prunes over multiple epochs.

```python
schedule = pruner.create_gradual_schedule(0.0, 0.5, num_epochs=10)
for epoch in range(10):
    results = pruner.prune_for_epoch(model, schedule["schedule_id"], epoch)
```

## Integration with Tracker

The pruner is designed to work seamlessly with the neuron tracker:

1. **Dead Neuron Removal**: Automatically identifies and removes neurons that never activate
2. **Activity-Based Pruning**: Uses firing frequency and activation patterns
3. **Correlation Analysis**: Identifies redundant neurons for merging/removal
4. **Optimization Recommendations**: Converts tracker insights into actionable pruning plans

## Metrics and Validation

### Comprehensive Metrics
```python
metrics = pruner.compute_metrics(pruned_model, original_model)
# Returns: model size, sparsity, compression ratios, speedup estimates
```

### Model Validation
```python
validation = pruner.validate_pruning(pruned_model, original_model)
# Returns: structural consistency, forward pass validation, warnings/errors
```

## Safety Features

- **Dry Run Mode**: Test pruning without modifying the model
- **Structure Validation**: Ensures model integrity after pruning
- **Backup & Restore**: Save/restore original model state
- **Layer Safety**: Prevents pruning all neurons from a layer

## Integrated Tracker-Pruner Workflow

The pruner provides complete end-to-end integration with the neuron tracker:

### Workflow Steps

1. **Track**: Analyze model and collect neuron activation data
2. **Recommend**: Generate pruning recommendations based on activity
3. **Prune**: Apply pruning using selected strategy
4. **Verify**: Confirm pruning was applied correctly
5. **Report**: Generate comprehensive before/after analysis
6. **Store**: Organize outputs in structured folders

### Output Management

All outputs are automatically organized in the `outputs/` directory:

```
outputs/
├── logs/              # Recommendations and pruning results (JSON)
├── reports/           # Final reports and summaries  
├── models/            # Original model states
├── pruned_models/     # Pruned model states
├── verification/      # Verification test results
└── visualizations/    # Plots and heatmaps
```

### Verification Tests

The workflow includes comprehensive verification:

- **Model Structure**: Validates layer dimensions and connections
- **Forward Pass**: Tests model functionality after pruning
- **Neuron Counts**: Verifies expected vs actual neuron removal
- **Parameter Counts**: Confirms parameter reduction consistency

### Before/After Logging

Detailed logging shows complete before/after state:

```
2025-09-12 10:15:23 - INFO - Original model: 202 neurons across 3 layers
2025-09-12 10:15:24 - INFO - Pruning candidates identified: 1 neurons
2025-09-12 10:15:25 - INFO - Pruning completed: 1 neurons removed from 1 layers
2025-09-12 10:15:26 - INFO - Verification: 4/4 tests passed
2025-09-12 10:15:27 - INFO - Final model: 201 neurons (0.5% reduction)
```

## File Descriptions

- **`__init__.py`**: Exports main classes and provides `PruningEngine` interface
- **`core.py`**: Core `NeuronPruner` with fundamental pruning operations
- **`strategies.py`**: Specialized pruning strategies (magnitude, structured, gradual)
- **`utils.py`**: Metrics computation, validation, and utility functions
- **`workflow.py`**: End-to-end integrated tracker-pruner workflows
- **`integration.py`**: Integration examples and complete workflow demonstrations

## Configuration System

The pruner includes a comprehensive configuration system for easy customization:

### Configuration Files

```yaml
# pruner_config.yaml
thresholds:
  firing_frequency_threshold: 0.01
  max_sparsity_per_layer: 0.9
  
strategy:
  default_strategy: "magnitude"
  target_sparsity: 0.3
  magnitude_weight: 0.5
  activity_weight: 0.5
  
logging:
  log_level: "INFO"
  verbose: true
  
validation:
  dry_run_first: true
  max_pruning_ratio: 0.8
```

### Configuration Options

**Pruning Thresholds:**
- `firing_frequency_threshold`: Minimum neuron activity to keep (0.0-1.0)
- `weight_magnitude_threshold`: Minimum weight magnitude 
- `max_sparsity_per_layer`: Maximum pruning per layer (0.0-1.0)
- `min_neurons_per_layer`: Minimum neurons to keep per layer

**Strategy Parameters:**
- `default_strategy`: "magnitude", "structured", or "gradual"
- `target_sparsity`: Overall pruning target (0.0-1.0) 
- `magnitude_weight`/`activity_weight`: Scoring weights (must sum to 1.0)
- `layer_specific_sparsity`: Per-layer sparsity targets

**Logging Options:**
- `log_level`: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
- `verbose`: Detailed output during operations
- `log_to_file`/`log_to_console`: Output destinations

**Validation Settings:**
- `dry_run_first`: Always simulate before actual pruning
- `validate_inputs`/`validate_outputs`: Enable validation checks
- `max_pruning_ratio`: Safety limit for total pruning (0.0-1.0)

### Preset Configurations

```python
from pruner import create_conservative_config, create_aggressive_config

# Safe pruning with minimal compression
conservative = create_conservative_config()  # 10% target sparsity

# Maximum compression with higher risk
aggressive = create_aggressive_config()      # 70% target sparsity
```

### Environment Overrides

```bash
export PRUNER_STRATEGY_TARGET_SPARSITY=0.5
export PRUNER_LOGGING_LOG_LEVEL=DEBUG
export PRUNER_VALIDATION_DRY_RUN_FIRST=false
```

### Runtime Configuration

```python
pruner = PruningEngine()

# Update individual parameters
pruner.update_config(**{"strategy.target_sparsity": 0.4})

# Switch modes
pruner.set_conservative_mode()
pruner.set_aggressive_mode()

# Save/load configurations
pruner.save_config("my_config.yaml")
pruner.load_config("saved_config.json")
```

## Extension Points

The module is designed for easy extension:

1. **New Strategies**: Add new pruning algorithms in `strategies.py`
2. **Custom Metrics**: Extend `PruningMetrics` for domain-specific metrics
3. **Integration Patterns**: Build on `IntegratedSystem` for custom workflows
4. **Layer Types**: Add support for new layer types in core pruning functions
5. **Configuration**: Extend configuration classes for custom parameters

## Dependencies

- PyTorch (for neural network operations)
- NumPy (for numerical computations)
- The tracker module (for neuron analysis)
- PyYAML (optional, for YAML configuration files)

All pruning operations are performed using standard PyTorch operations without additional dependencies.
