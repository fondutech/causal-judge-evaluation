# Arena Analysis Python Interface

The CJE library provides clean Python interfaces for running arena analysis programmatically, leveraging the enhanced `APIPolicyRunner` with automatic caching, error handling, and progress tracking.

## Quick Start

### 1. Simple Python Script (`arena_analysis_python.py`)

```python
from examples.arena_analysis_python import run_arena_analysis, run_quick_test

# Quick test (fast, minimal samples)
results = run_quick_test()

# Full analysis  
results = run_arena_analysis(
    config_name="arena_analysis",
    estimator="DRCPO"
)
```

### 2. Interactive Analysis (`arena_interactive.py`)

```python
from examples.arena_interactive import ArenaAnalyzer

# Create analyzer
analyzer = ArenaAnalyzer()

# Quick test
analyzer.quick_test()

# Plot results
analyzer.plot_estimates()

# Compare multiple estimators
df = analyzer.compare_estimators(["DRCPO", "IPS", "SNIPS"])

# Save results
analyzer.save_results()
```

### 3. Direct Pipeline Interface

```python
from cje.pipeline import run_pipeline
from pathlib import Path

# Get config directory path
configs_dir = Path.cwd() / "configs"

# Direct pipeline call
results = run_pipeline(
    cfg_path=str(configs_dir),
    cfg_name="arena_analysis"
)

print(f"Analysis type: {results['analysis_type']}")
print(f"Estimates: {results['v_hat']}")
```

## Features

All interfaces automatically use the enhanced `APIPolicyRunner` with:

- **Safe Error Handling**: Graceful fallbacks on API failures using `safe_call`
- **Automatic Caching**: Avoids recomputing expensive log-probs with in-memory cache
- **Progress Tracking**: Shows progress for large batches with `track`
- **Batch Processing**: Efficient handling of multiple requests via `log_prob_batch`

## Jupyter Notebook Usage

Perfect for interactive analysis:

```python
from examples.arena_interactive import ArenaAnalyzer
import matplotlib.pyplot as plt

# Create analyzer
analyzer = ArenaAnalyzer(work_dir="my_arena_results")

# Run analysis
results = analyzer.run_analysis("arena_analysis", estimator="DRCPO")

# Visualize
analyzer.plot_estimates("My Arena Analysis")

# Compare estimators
comparison_df = analyzer.compare_estimators(
    estimators=["DRCPO", "IPS", "SNIPS", "MRDR"],
    config_name="arena_analysis"
)

# Access raw data
estimates = analyzer.get_estimates()
std_errors = analyzer.get_standard_errors()
```

## Configuration

Use any of the existing configs:

- `arena_analysis`: Full analysis (1000 samples)
- `arena_quick`: Quick test (25 samples)  
- Custom configs: Create your own in `configs/`

## Benefits over CLI

- **Interactive**: Step-by-step analysis with immediate feedback
- **Programmatic**: Easy to integrate into larger workflows and notebooks
- **Visualization**: Built-in plotting and comparison tools
- **Data Access**: Direct access to results for further analysis
- **Error Handling**: Graceful failure handling with detailed error context
- **Caching**: Automatic optimization for repeated runs and parameter sweeps

## Command Line Usage

```bash
# Quick test
python examples/arena_analysis_python.py quick

# Full analysis  
python examples/arena_analysis_python.py

# Interactive example
python examples/arena_interactive.py
```

## Architecture

The Python interfaces leverage the existing CJE pipeline infrastructure:

- **`cje.pipeline.run_pipeline()`**: Core orchestration engine
- **Enhanced `APIPolicyRunner`**: Improved with caching, error handling, progress tracking
- **Hydra Configuration**: Flexible config management
- **Rich Outputs**: Structured JSON results with metadata

This approach provides maximum robustness by reusing the production-tested pipeline while adding convenience interfaces for interactive use.

## Example Workflow

```python
from examples.arena_interactive import ArenaAnalyzer

# Initialize
analyzer = ArenaAnalyzer()

# Quick validation
analyzer.quick_test()

# Full analysis
results = analyzer.run_analysis("arena_analysis")

# Compare approaches
comparison = analyzer.compare_estimators(["DRCPO", "IPS", "SNIPS"])

# Visualize and save
analyzer.plot_estimates("Final Results")
analyzer.save_results("my_analysis_results.json")
```

The Python interfaces provide the same robust analysis as the CLI but with much more flexibility for interactive exploration and integration into data science workflows. 