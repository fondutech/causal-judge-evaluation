# Causal Judge Evaluation (CJE)

![CJE Logo](docs/img/CJE_logo.svg)

Production-ready framework for unbiased LLM evaluation using causal inference.

## Overview

CJE provides a clean, production-ready implementation for:
- **Unbiased Estimation**: Corrects for distribution shift between policies
- **Variance Control**: Stacked SIMCal (Score-Indexed Monotone Calibration) prevents weight explosion  
- **Doubly Robust**: Optional DR estimation for better bias-variance tradeoff
- **Optimal Stacking**: Combines multiple weight candidates to minimize OOF variance
- **Production Ready**: Clean API, comprehensive tests, type hints throughout

## Installation

```bash
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
pip install -e .
```

## Quick Start

**Requirements:** Minimum 10 samples with oracle labels for cross-validation. For quick testing without oracle labels, use `raw-ips` estimator.

### Command Line Interface

```bash
# Analyze a dataset with default settings
python -m cje analyze data.jsonl

# Use specific estimator with fresh draws for DR
python -m cje analyze data.jsonl --estimator dr-cpo --fresh-draws-dir responses/

# Validate dataset format
python -m cje validate data.jsonl

# Export results to JSON
python -m cje analyze data.jsonl --output results.json
```

### Python API

```python
# High-level API (recommended)
from cje import analyze_dataset

results = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips"
)
best_idx = results.best_policy()
policies = results.metadata.get("target_policies", [])
if policies:
    print(f"Best policy: {policies[best_idx]}")

# Lower-level API for more control
from cje import (
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS
)

dataset = load_dataset_from_jsonl("data.jsonl")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    enable_cross_fit=True,  # For DR
)
sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(
    sampler,
    calibrator=cal_result.calibrator,  # For DR-aware stacking
)
results = estimator.fit_and_estimate()
```

## Key Features

### 🚀 New: Command Line Interface
- `analyze` - Run CJE analysis on datasets
- `validate` - Check dataset format and completeness
- Export results to JSON/CSV formats
- Support for all estimators and configurations

### 🎯 New: High-Level API
- `analyze_dataset()` - One-line analysis function
- Automatic calibration and fresh draw handling
- Smart defaults with full configurability

### Estimators
- **CalibratedIPS** - Variance-controlled IPS with stacked SIMCal calibration (recommended)
  - Combines {baseline, increasing, decreasing} candidates optimally
  - Automatic DR-aware stacking when calibrator available
- **RawIPS** - Standard importance sampling with clipping
- **DRCPOEstimator** - Doubly robust with cross-fitted isotonic outcome models
- **MRDREstimator** - Policy-specific weighted outcome models for heterogeneous effects
- **TMLEEstimator** - Targeted minimum loss estimation with optimal bias-variance tradeoff

### Data Format
```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -35.704,
  "target_policy_logprobs": {
    "gpt4": -32.456,
    "claude": -33.789
  },
  "metadata": {
    "judge_score": 0.85,
    "oracle_label": 0.90
  }
}
```

**Note:** `prompt_id` is optional and will be auto-generated from the prompt hash if not provided. This ensures consistency across datasets for fresh draws.

### Computing Log Probabilities

**Note:** Teacher forcing (computing log probabilities for a given completion) requires specific API support:
- **Fireworks AI** ✅ (Currently supported - full implementation via `compute_teacher_forced_logprob`)
- **Together AI** 🔄 (Has API support but not yet integrated in CJE)
- **OpenAI** ❌ (Does not support teacher forcing)
- **Anthropic** ❌ (Does not support teacher forcing)

```python
from cje import compute_teacher_forced_logprob

# Works with Fireworks AI models
result = compute_teacher_forced_logprob(
    prompt="What is 2+2?",
    response="The answer is 4.",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)
```

## CLI Usage

### Available Commands

#### `analyze` - Run CJE analysis
```bash
python -m cje analyze <dataset> [options]

Options:
  --estimator {calibrated-ips,raw-ips,dr-cpo,mrdr,tmle}
                        Estimation method (default: calibrated-ips)
  --output PATH         Save results to JSON file
  --fresh-draws-dir DIR Directory containing fresh draw responses (for DR)
  --estimator-config JSON
                        JSON config for estimator (e.g., '{"n_folds": 10}')
  --judge-field FIELD   Metadata field with judge scores (default: judge_score)
  --oracle-field FIELD  Metadata field with oracle labels (default: oracle_label)
  -v, --verbose         Enable verbose output
  -q, --quiet           Suppress non-essential output
```

#### `validate` - Check dataset format
```bash
python -m cje validate <dataset> [options]

Options:
  -v, --verbose         Show detailed validation results
```

### Export Formats

```python
from cje import analyze_dataset, export_results_json, export_results_csv

# Analyze
results = analyze_dataset("data.jsonl")

# Export to JSON (includes full metadata and diagnostics)
export_results_json(results, "results.json")

# Export to CSV (tabular format for analysis)
export_results_csv(results, "results.csv")
```

## Fresh Draws for Doubly Robust Estimation

DR estimators need fresh draws (new responses from target policies). Format as JSONL with these fields:

```json
{
  "prompt_id": "prompt_abc123",  
  "target_policy": "premium",
  "judge_score": 0.85,
  "draw_idx": 0
}
```

The `prompt_id` should match your logged data (auto-generated from prompt hash if not provided).

Load and use:
```python
from cje import load_fresh_draws_from_jsonl

fresh_draws = load_fresh_draws_from_jsonl("fresh_draws.jsonl")
dr_estimator.add_fresh_draws("premium", fresh_draws["premium"])
```

## Documentation

Full documentation available at: https://causal-judge-evaluation.readthedocs.io

- [Getting Started](docs/getting_started.rst)
- [CLI Reference](docs/cli.rst)
- [Data Format Guide](docs/data_format.rst)
- [Estimators Guide](docs/estimators.rst)
- [API Reference](docs/api/)

## Testing

```bash
# Run all tests
poetry run pytest cje/tests/

# Run specific test suites
poetry run pytest cje/tests/test_cli.py -v      # CLI tests
poetry run pytest cje/tests/test_analysis.py -v # High-level API tests
poetry run pytest cje/tests/test_export.py -v   # Export functionality

# Run with coverage
poetry run pytest cje/tests/ --cov=cje --cov-report=html
```

## License

MIT License - see LICENSE file for details.