# CJE Interface

High-level tools for using CJE. Most users should start here.

## Overview

This module provides two ways to use CJE:
1. **Python API**: `analyze_dataset()` function
2. **Command-line interface**: `python -m cje` or `cje` command

Both provide the same functionality - choose based on your workflow.

## Quick Start

### Python API

```python
from cje import analyze_dataset

# Simple analysis with defaults
results = analyze_dataset("your_data.jsonl")
print(f"Estimate: {results.estimates[0]:.3f}")
```

### Command Line

```bash
# Simple analysis with defaults
python -m cje analyze your_data.jsonl

# With options
python -m cje analyze data.jsonl --estimator dr-cpo --verbose

# Validate dataset
python -m cje validate data.jsonl --verbose
```

## Python API Reference

### `analyze_dataset()`

The main entry point for CJE analysis. Handles the complete workflow automatically.

```python
def analyze_dataset(
    dataset_path: str,
    estimator: str = "calibrated-ips",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    estimator_config: Optional[Dict[str, Any]] = None,
    fresh_draws_dir: Optional[str] = None,
    verbose: bool = False,
) -> EstimationResult:
```

**Parameters:**
- `dataset_path`: Path to JSONL file with your data
- `estimator`: Which estimator to use (see [Estimator Choice](#estimator-choice))
- `judge_field`: Field in metadata containing judge scores
- `oracle_field`: Field in metadata containing oracle labels
- `estimator_config`: Optional configuration for the estimator
- `fresh_draws_dir`: Directory with fresh draws (for DR estimators)
- `verbose`: Print progress messages

**Returns:**
- `EstimationResult` with estimates, standard errors, diagnostics, and metadata

**Example:**
```python
results = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips",
    verbose=True
)

# Access results
for i, policy in enumerate(results.metadata["target_policies"]):
    print(f"{policy}: {results.estimates[i]:.3f} ± {results.standard_errors[i]:.3f}")

# Check diagnostics
if results.diagnostics:
    print(f"ESS: {results.diagnostics.weight_ess:.1%}")
```

## CLI Reference

### Commands

#### `analyze` - Run CJE analysis

```bash
python -m cje analyze <dataset> [options]
```

**Options:**
- `--estimator {calibrated-ips,raw-ips,stacked-dr,dr-cpo,mrdr,tmle}`: Estimation method
- `--estimator-config JSON`: Configuration as JSON string
- `--judge-field FIELD`: Judge score field name (default: judge_score)
- `--oracle-field FIELD`: Oracle label field name (default: oracle_label)
- `--verbose, -v`: Detailed output
- `--quiet, -q`: Minimal output

**Example:**
```bash
python -m cje analyze data.jsonl --estimator dr-cpo --verbose
```

#### `validate` - Check dataset format

```bash
python -m cje validate <dataset> [options]
```

**Options:**
- `--verbose, -v`: Show detailed statistics

**Example:**
```bash
python -m cje validate data.jsonl --verbose
```

## Estimator Choice

| Estimator | When to Use | Requirements |
|-----------|-------------|--------------|
| `calibrated-ips` | **Default - start here** | Judge scores |
| `raw-ips` | Diagnostic comparison | None |
| `stacked-dr` | **Best DR option** - combines all DR methods | Fresh draws |
| `dr-cpo` | Low overlap (ESS < 10%) | Fresh draws |
| `mrdr` | Research/advanced | Fresh draws |
| `tmle` | Research/advanced | Fresh draws |

## Data Format

Your JSONL file should have entries like:

```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -45.67,
  "target_policy_logprobs": {
    "policy_a": -42.31,
    "policy_b": -44.89
  },
  "metadata": {
    "judge_score": 0.82,
    "oracle_label": 0.90
  }
}
```

Required fields:
- `prompt`, `response`: The conversation
- `base_policy_logprob`: Log probability under logging policy
- `target_policy_logprobs`: Log probabilities under target policies
- `metadata.judge_score`: Score from automated judge (for calibration)

Optional:
- `metadata.oracle_label`: Ground truth labels (for calibration)
- Fresh draws in separate directory (for DR estimators)

## Workflow Examples

### Basic Analysis
```python
# Just get estimates
results = analyze_dataset("data.jsonl")
best_policy_idx = np.argmax(results.estimates)
print(f"Best policy: {results.metadata['target_policies'][best_policy_idx]}")
```

### With Diagnostics
```python
results = analyze_dataset("data.jsonl", verbose=True)

# Check quality
if results.diagnostics.weight_ess < 0.05:
    print("Warning: Very low ESS, consider DR estimator")
    results = analyze_dataset("data.jsonl", estimator="dr-cpo")
```

### Custom Configuration
```python
# Tighter variance control for CalibratedIPS
results = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips",
    estimator_config={"var_cap": 0.5}
)

# More folds for DR
results = analyze_dataset(
    "data.jsonl", 
    estimator="dr-cpo",
    estimator_config={"n_folds": 10},
    fresh_draws_dir="fresh/"
)
```

## Interpreting Results

### Point Estimates
- `results.estimates`: Array of estimates for each target policy
- Higher is better (assuming rewards are on 0-1 scale)
- Compare policies by their estimates

### Uncertainty
- `results.standard_errors`: Standard errors for each estimate
- 95% CI: `estimate ± 1.96 * standard_error`
- Wider CIs with less oracle data (automatic adjustment)

### Diagnostics
- `weight_ess`: Effective sample size (> 0.1 is good)
- `tail_indices`: Heavy tail detection (> 2 is good)
- `calibration_r2`: Judge-oracle calibration quality (> 0.5 is good)

### CF-bits Analysis (Advanced)

For deeper reliability analysis, use CF-bits information accounting:

```python
from cje.cfbits import cfbits_report_fresh_draws, cfbits_report_logging_only

# After fitting your estimator
if has_fresh_draws:
    report = cfbits_report_fresh_draws(estimator, policy)
else:
    report = cfbits_report_logging_only(estimator, policy)

# Check reliability gates
if report["gates"]["state"] == "REFUSE":
    print("Cannot trust this estimate")
elif report["gates"]["state"] == "CRITICAL":
    print("Use with extreme caution")

# View decomposition
print(f"A-ESSF: {report['overlap']['aessf']:.1%}")  # Structural overlap
print(f"IFR: {report['efficiency']['ifr_oua']:.1%}")  # Statistical efficiency
```

CF-bits decomposes uncertainty into:
- **Identification width**: Structural limits from overlap/calibration
- **Sampling width**: Statistical noise from finite samples
- **A-ESSF**: Ceiling on S-indexed calibration methods
- **IFR**: How close to efficiency bound

## Common Issues

### "ESS too low"
- Policies are too different from logging policy
- Solution: Use DR estimator with fresh draws

### "No oracle labels found"
- Dataset missing oracle labels for calibration
- Solution: Add at least some oracle labels, or use raw rewards

### "NaN estimates"
- Catastrophic failure (usually extreme weights)
- Solution: Check data quality, use tighter var_cap

## Advanced Usage

For more control, use the library components directly:

```python
from cje import load_dataset_from_jsonl, calibrate_dataset
from cje import PrecomputedSampler, CalibratedIPS

# Manual workflow
dataset = load_dataset_from_jsonl("data.jsonl")
calibrated_dataset, _ = calibrate_dataset(dataset)
sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(sampler)
results = estimator.fit_and_estimate()
```

See module documentation for component details.