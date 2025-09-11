# Self-Modifying SLM

A neural network pruning and modification engine that can dynamically alter network structure during training.

## Features

- **Neuron Tracker**: Enumerate and track all neurons in any PyTorch nn.Module
- **Activation Monitoring**: Capture and analyze neuron activations with forward hooks
- **Statistical Analysis**: Compute mean, variance, and sparsity statistics per neuron
- **Dataset-Level Analysis**: Comprehensive statistics across entire datasets
- **Dead Neuron Detection**: Identify and report neurons that never activate
- **Correlation Analysis**: Detect redundant neurons via activation correlation
- **Training History**: Track neuron evolution and model performance over epochs
- **JSON Logging**: Export comprehensive training data for analysis
- **Advanced Reporting**: Generate detailed activity reports and pruning recommendations
- **JSON Optimization**: Export structured optimization recommendations for automated tools
- **Visual Analysis**: Create matplotlib heatmaps of neuron activity patterns
- **Memory Efficient**: Store only summary statistics, not raw activation tensors
- **Pruning Engine**: Remove neurons/layers based on various criteria (coming soon)
- **Dynamic Architecture**: Add new neurons/layers during training (coming soon)

## Setup

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure

```
SelfModifyingSLM/
├── tracker/                          # Core neuron tracking system
│   ├── __init__.py                   # Main interface (NeuronEngine)
│   ├── tracker.py                    # Core neuron enumeration and tracking
│   ├── analyzer.py                   # Activation analysis and correlation detection
│   └── reporter.py                   # Report generation and optimization recommendations
├── outputs/                          # Generated analysis outputs
│   ├── logs/                         # JSON data files (*.json)
│   ├── reports/                      # Text reports and summaries (*.txt)
│   └── visualizations/               # Heatmaps and plots (*.png)
├── demos/                            # Demo scripts and validation tests
│   ├── simple_example.py             # Quick start example
│   ├── neuron_tracker_demo.py        # Basic tracking demo
│   ├── activation_tracking_demo.py   # Activation analysis demo
│   ├── correlation_analysis_demo.py  # Correlation analysis demo
│   ├── training_tracker_demo.py      # Training history demo
│   ├── optimization_recommendations_demo.py # JSON optimization demo
│   └── *_validation.py               # Success criteria validation scripts
├── data/                             # Fashion-MNIST dataset
├── fashion_mnist_mlp.py              # Example MLP model
├── test_tracker.py                   # Main verification script
├── requirements.txt                  # Python dependencies
└── README.md                         # This documentation
```

## Usage

### Quick Start

```python
from tracker import NeuronEngine
import torch.nn as nn

# Create any PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Create neuron engine (combines all functionality)
engine = NeuronEngine()

# Track all neurons
neuron_map = engine.track_model(model, "My Model")

# Get model summary
summary = engine.get_summary()
print(f"Total neurons: {summary['total_neurons']}")
```

### Basic Neuron Tracking

```python
from tracker import NeuronTracker
import torch.nn as nn

# Create model and tracker
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
tracker = NeuronTracker()

# Track all neurons
neuron_map = tracker.track_model(model, "My Model")
# Result: {'0': [0, 1, ..., 255], '2': [256, 257, ..., 265]}
print(neuron_map)

# Look up specific neurons
layer, local_idx = tracker.get_neuron_info(neuron_id=128)
```

### Activation Monitoring

```python
# Register hooks to capture activations
tracker.register_activation_hooks(model)

# Start tracking
tracker.start_tracking()

# Run your training/inference
model.eval()
with torch.no_grad():
    for batch in data_loader:
        output = model(batch)  # Hooks automatically capture stats

# Stop tracking and view results
tracker.stop_tracking()
tracker.print_activation_summary()

# Get stats for specific neurons
stats = tracker.get_neuron_stats(neuron_id=42)
print(f"Mean: {stats['mean']}, Variance: {stats['variance']}, Sparsity: {stats['sparsity']}")
```

### Dataset-Level Analysis

```python
# Run comprehensive dataset analysis
tracker.start_tracking()

# Process your entire dataset
for batch in dataset_loader:
    output = model(batch)

tracker.stop_tracking()

# Generate comprehensive report
tracker.print_dataset_summary()

# Get layer-wise statistics programmatically
layer_summary = tracker.get_layer_summary()
for layer_name, stats in layer_summary.items():
    print(f"{layer_name}: {stats['dead_percentage']:.1f}% dead, avg freq {stats['avg_firing_frequency']:.2f}")

# Identify dead neurons
dead_neurons = tracker.get_dead_neurons()
for layer_name, dead_list in dead_neurons.items():
    print(f"{layer_name}: {len(dead_list)} dead neurons")
```

### Correlation Analysis

```python
# Enable correlation analysis for redundancy detection
tracker.start_tracking(enable_correlation_analysis=True)

# Process your dataset
for batch in dataset_loader:
    output = model(batch)

tracker.stop_tracking()

# Find redundant neurons
tracker.print_correlation_analysis()

# Get redundant pairs programmatically
redundant_pairs = tracker.find_redundant_neurons(correlation_threshold=0.9)
for layer_name, pairs in redundant_pairs.items():
    for neuron_id1, neuron_id2, correlation in pairs:
        print(f"Layer {layer_name}: Neurons {neuron_id1}-{neuron_id2} correlation: {correlation:.3f}")

# Get redundancy summary
summary = tracker.get_redundancy_summary()
for layer_name, stats in summary.items():
    print(f"{layer_name}: {stats['redundant_pairs']} pairs, {stats['redundancy_rate']:.1%} redundancy")
```

### Training History Tracking

```python
# Setup model and tracker
model = FashionMLP(hidden_sizes=[64, 32])
tracker = NeuronTracker()
tracker.track_model(model, "Training Experiment")
tracker.register_activation_hooks(model)

# Training loop with neuron tracking
for epoch in range(num_epochs):
    # Your training code here
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, tracker)
    
    # Log epoch statistics
    tracker.log_epoch_statistics(epoch, train_loss, train_acc)

# Save complete training history to JSON
tracker.save_training_history('training_history.json')

# Load and analyze later
tracker.load_training_history('training_history.json')
neuron_evolution = tracker.get_neuron_evolution(neuron_id=42)
print(f"Neuron 42 mean evolution: {neuron_evolution['mean']}")
print(f"Training loss evolution: {neuron_evolution['loss']}")
```

### Advanced Reporting

```python
# Generate comprehensive activity report with heatmap
tracker.generate_neuron_report(show_heatmap=True, save_heatmap="activity_heatmap.png")

# Get pruning recommendations
tracker.generate_pruning_recommendations(top_n=20)

# Save complete report to file
tracker.save_report_to_file("comprehensive_report.txt", include_heatmap=True)

# Programmatic access to rankings
dataset_stats = tracker.get_dataset_statistics()
activities = [(nid, stats['firing_frequency']) for nid, stats in dataset_stats.items()]
activities.sort(key=lambda x: x[1], reverse=True)
print(f"Most active neuron: {activities[0]}")
print(f"Least active neuron: {activities[-1]}")
```

### JSON Optimization Recommendations

```python
# Generate structured optimization recommendations
recommendations = tracker.generate_optimization_recommendations(
    pruning_threshold=0.01,
    redundancy_threshold=0.9,
    saturation_threshold=0.95
)

# Save to JSON file for automated tools
tracker.save_optimization_recommendations('optimization.json')

# Print human-readable summary
tracker.print_optimization_recommendations()

# Programmatic access to recommendations
prune_candidates = recommendations['prune']
merge_candidates = recommendations['merge']
expand_candidates = recommendations['expand']

for candidate in prune_candidates:
    print(f"Prune neuron {candidate['neuron_id']} ({candidate['reason']})")

for candidate in merge_candidates:
    pair = candidate['neuron_pair']
    print(f"Merge neurons {pair[0]} and {pair[1]} (correlation: {candidate['correlation']:.3f})")

for candidate in expand_candidates:
    print(f"Expand {candidate['layer_name']}: {candidate['current_size']} → {candidate['current_size'] + candidate['suggested_expansion']}")
```

### Running the System

```bash
# Test the reorganized tracker (recommended first step)
python test_tracker.py

# Run demos (located in demos/ folder)
cd demos/

# Basic examples
python simple_example.py
python simple_activation_demo.py

# Comprehensive demos
python neuron_tracker_demo.py
python activation_tracking_demo.py
python correlation_analysis_demo.py
python training_tracker_demo.py
python optimization_recommendations_demo.py

# Validation tests
python dataset_success_validation.py
python correlation_success_validation.py
python optimization_success_validation.py
```

## Features Overview

### Universal Compatibility
Works with any PyTorch nn.Module including:
- Linear layers
- Convolutional layers (Conv1d, Conv2d, Conv3d)
- Normalization layers (BatchNorm, LayerNorm)
- RNN layers (LSTM, GRU)
- Custom architectures

### Activation Statistics
For each neuron, the tracker computes:
- **Mean Activation**: Average activation value across batches
- **Variance**: Activation variance across batches  
- **Sparsity**: Percentage of non-zero activations
- **Batch Count**: Number of batches processed

### Dataset-Level Analysis
Comprehensive statistics across entire datasets:
- **Dataset Mean**: Average activation across all samples
- **Standard Deviation**: Activation variability measure
- **Firing Frequency**: Percentage of samples where neuron activates above threshold
- **Dead Neuron Detection**: Identifies neurons that never activate
- **Layer-wise Reports**: Summary statistics grouped by layer

### Correlation Analysis
Detect redundant neurons through activation pattern analysis:
- **Pearson Correlation**: Compute correlations between neuron pairs within layers
- **Redundancy Detection**: Identify highly correlated neurons (correlation > threshold)
- **Memory Efficient**: Collect activation vectors only when needed
- **Reproducible Results**: Consistent correlation analysis across runs
- **Configurable Thresholds**: Adjustable correlation threshold for redundancy detection

### Training History Tracking
Monitor neuron evolution and model performance over training epochs:
- **Epoch-by-Epoch Logging**: Track neuron statistics for each training epoch
- **Performance Metrics**: Log loss and accuracy alongside neuron data
- **JSON Export**: Save complete training history in structured JSON format
- **Neuron Evolution**: Track mean, variance, and firing frequency over time
- **Data Consistency**: Ensure logged metrics match actual training values

### Advanced Reporting & Visualization
Generate comprehensive analysis reports and visual insights:
- **Activity Rankings**: Identify top/bottom active neurons for analysis
- **Pruning Recommendations**: Smart candidate selection for network compression
- **Layer-wise Statistics**: Dead and redundant neuron percentages per layer
- **Activity Heatmaps**: Visual representation of neuron activity patterns
- **Export Capabilities**: Save reports as text files and heatmaps as images

### JSON Optimization System
Structured recommendations for automated network optimization:
- **Pruning Suggestions**: Dead and low-activity neurons with detailed reasoning
- **Merge Recommendations**: Redundant neuron pairs with correlation metrics
- **Expansion Proposals**: Layer growth suggestions based on saturation analysis
- **Machine Readable**: JSON format for integration with optimization pipelines
- **Configurable Thresholds**: Adjustable parameters for different use cases

### Memory Efficiency
- Only stores summary statistics per neuron (4 floats)
- Raw activation tensors are immediately discarded
- Minimal memory overhead regardless of model size
- Supports models with thousands of neurons

### Success Criteria Met

✅ **Hooks run without errors**: Forward hooks successfully capture activations during forward passes  
✅ **Per-neuron statistics**: After processing batches, detailed statistics are available for each tracked neuron  
✅ **Memory management**: No large tensors stored; only efficient summary statistics maintained  

## Architecture

```
SelfModifyingSLM/
├── requirements.txt                 # PyTorch dependencies
├── README.md                       # This documentation
├── neuron_tracker.py               # Core tracker with activation monitoring
├── fashion_mnist_mlp.py            # Example MLP model
├── neuron_tracker_demo.py          # Comprehensive demo
├── activation_tracking_demo.py     # Activation monitoring demo
├── dataset_analysis_demo.py        # Dataset-level analysis demo
├── correlation_analysis_demo.py    # Correlation analysis demo
├── training_tracker_demo.py        # Training with neuron tracking
├── simple_reporting_demo.py        # Advanced reporting demo
├── optimization_recommendations_demo.py # JSON optimization demo
├── dataset_success_validation.py   # Success criteria validation
├── correlation_success_validation.py # Correlation validation
├── training_success_validation.py  # Training tracker validation
├── reporting_success_validation.py # Reporting validation
├── optimization_success_validation.py # Optimization validation
├── simple_activation_demo.py       # Simple activation example
├── simple_example.py               # Quick usage example
└── neuron_mapping_result.json      # Example output
```

## Example Output

### Basic Activation Statistics
```
Activation Statistics Summary:
================================================================================
Neuron ID  Layer           Local Idx  Mean         Variance     Sparsity   Batches 
--------------------------------------------------------------------------------
0          fc1             0          -0.120417    0.307160     1.000      3       
1          fc1             1          0.045632     0.217342     1.000      3       
2          fc1             2          0.141631     0.285130     1.000      3       
...
```

### Dataset-Level Analysis Report
```
DATASET-LEVEL NEURON ANALYSIS REPORT
================================================================================
Total samples processed: 1920
Activation threshold: 0.01

LAYER SUMMARY:
------------------------------------------------------------
Layer           Total    Dead   Dead %   Avg Freq  
------------------------------------------------------------
fc1             128      0      0.0%     0.986     
fc2             64       5      7.8%     0.969     
fc_out          10       0      0.0%     0.958     
------------------------------------------------------------
TOTAL           202      5      2.5%

DEAD NEURONS DETAILS:
----------------------------------------
fc2: 5 dead neurons
  Local indices: [10, 11, 12, 13, 14]
```

### Correlation Analysis Report
```
NEURON CORRELATION ANALYSIS REPORT
================================================================================
Correlation threshold: 0.9
Samples used for correlation: 1920

REDUNDANT NEURON PAIRS BY LAYER:
------------------------------------------------------------
Layer           Pairs    Redundant Neurons              Correlation 
------------------------------------------------------------
fc2             8        (0, 8)                         1.0000      
                         (1, 9)                         0.9999      
                         (2, 10)                        0.9999      
                         (3, 11)                        1.0000      
                         (4, 12)                        0.9999      
                         (5, 13)                        1.0000      
                         (6, 14)                        1.0000      
                         (7, 15)                        1.0000      
------------------------------------------------------------
Total redundant pairs: 8

DETAILED REDUNDANCY ANALYSIS:
--------------------------------------------------
fc2:
  Total neurons: 16
  Redundant pairs: 8
  Redundancy rate: 6.7%
  Correlation range: 1.000 - 1.000
  Average correlation: 1.000
```

### Training History JSON Structure
```json
{
  "metadata": {
    "total_epochs": 3,
    "total_neurons": 106,
    "activation_threshold": 0.01
  },
  "neuron_histories": {
    "106": {
      "mean": [0.4069, 0.3397, 0.9154],
      "variance": [1.9255, 3.4347, 5.2280],
      "frequency": [0.99, 0.992, 0.994],
      "loss": [1.6987, 1.0300, 0.8153],
      "accuracy": [0.3979, 0.6333, 0.6953]
    }
  }
}
```

### Activity Report Example
```
COMPREHENSIVE NEURON ACTIVITY REPORT
================================================================================

NEURON ACTIVITY ANALYSIS
--------------------------------------------------

TOP 10 MOST ACTIVE NEURONS (Key Contributors):
Rank  Neuron ID  Layer           Local Idx  Frequency    Mean Act    
---------------------------------------------------------------------------
1     157        fc1             51         1.000        0.5312      
2     166        fc1             60         1.000        -0.0750     
3     184        fc2             14         1.000        -0.3259     

TOP 10 LEAST ACTIVE NEURONS (Pruning Candidates):
Rank  Neuron ID  Layer           Local Idx  Frequency    Mean Act    
---------------------------------------------------------------------------
1     205        fc_out          3          0.771        0.0054      
2     210        fc_out          8          0.844        -0.0552     
3     204        fc_out          2          0.854        -0.0269     

LAYER-WISE NEURON ANALYSIS
--------------------------------------------------
Layer                Total    Dead     Dead %   Redundant    Red %   
----------------------------------------------------------------------
fc1                  64       0        0.0      0            0.0     
fc2                  32       0        0.0      0            0.0     
fc_out               10       0        0.0      0            0.0     

PRUNING RECOMMENDATIONS
----------------------------------------
TOP 10 PRUNING CANDIDATES:
Rank  Neuron ID  Layer           Local Idx  Status   Score    Frequency 
--------------------------------------------------------------------------------
1     205        fc_out          3          LOW      0.771    0.771     
2     210        fc_out          8          LOW      0.849    0.844     
3     204        fc_out          2          LOW      0.857    0.854     
```

### JSON Optimization Example
```json
{
  "metadata": {
    "timestamp": "2025-09-11T15:23:34.621623",
    "model_info": {
      "total_neurons": 106,
      "total_layers": 3,
      "layer_details": {"fc1": 64, "fc2": 32, "fc3": 10}
    }
  },
  "prune": [
    {
      "neuron_id": 88,
      "layer_name": "fc2",
      "local_index": 24,
      "reason": "dead",
      "firing_frequency": 0.0,
      "mean_activation": 0.0008505678561050445
    }
  ],
  "merge": [
    {
      "neuron_pair": [66, 82],
      "layer_name": "fc2",
      "local_indices": [2, 18],
      "correlation": 0.9999596132524416,
      "merge_strategy": "average_weights"
    }
  ],
  "expand": [
    {
      "layer_name": "fc1",
      "current_size": 64,
      "suggested_expansion": 16,
      "avg_activity": 0.986,
      "reason": "high_saturation"
    }
  ],
  "statistics": {
    "total_prunable_neurons": 18,
    "total_mergeable_pairs": 6,
    "layers_needing_expansion": 2
  }
}
```

This foundation enables sophisticated neural network analysis and modification for research into self-modifying architectures and adaptive neural networks.
