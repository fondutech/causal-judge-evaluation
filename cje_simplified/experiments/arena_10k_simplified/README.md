# Arena 10K Experiment

CJE pipeline using ChatBot Arena dataset for evaluating LLM policies.

## Quick Start

```bash
# Generate test data (50 samples)
poetry run python test_pipeline.py --n-samples 50

# Or run production pipeline
poetry run python generate_arena_data.py --n-samples 1000
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl --oracle-coverage 0.5
```

## Pipeline Overview

### 1. Data Generation (`generate_arena_data.py`)
- Extracts prompts from ChatBot Arena
- Generates responses from different policies
- Adds judge scores and oracle labels
- Computes log probabilities
- Combines into single JSONL file

### 2. Analysis (`analyze_dataset.py`)
- Calibrates judge scores to oracle labels (based on `--oracle-coverage`)
- Runs CJE estimation with importance weight calibration
- Outputs results and diagnostics

## Directory Structure

```
├── pipeline_steps/         # Individual data generation steps
├── data/                   # Generated data (production)
├── test_e2e_data/         # Test data (50 samples)
├── generate_arena_data.py # Main data generation script
├── analyze_dataset.py     # Analysis and calibration
└── test_pipeline.py       # Quick test of full pipeline
```

## Key Parameters

- `--n-samples`: Number of prompts to use
- `--oracle-coverage`: Fraction of oracle labels for calibration (0.0-1.0)
  - 1.0 = use oracle labels directly as rewards
  - <1.0 = calibrate judge scores using subset
- `--estimator`: Choose between `calibrated-ips` (default) or `raw-ips`

## Policies

- **base**: Standard helpful assistant
- **clone**: Identical to base (control)
- **unhelpful**: Deliberately poor responses

All policies are evaluated on the same base policy responses using importance weighting.