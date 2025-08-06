# Arena 10K Experiment

CJE pipeline using ChatBot Arena dataset for evaluating LLM policies.

## Quick Start

```bash
# Generate test data (50 samples)
poetry run python test_pipeline.py --n-samples 50

# Or run production pipeline (with batch saving for resilience)
poetry run python generate_arena_data.py --n-samples 1000 --batch-size 20
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

### Data Generation
- `--n-samples`: Number of prompts to use
- `--batch-size`: Save progress every N samples (default: 20, set to 0 to disable)
- `--skip-existing`: Skip steps where output files already exist
- `--force`: Force overwrite existing files (opposite of --skip-existing)
- `--max-tokens`: Maximum tokens per response (default: 256)

### Analysis
- `--oracle-coverage`: Fraction of oracle labels for calibration (0.0-1.0)
  - 1.0 = use oracle labels directly as rewards
  - <1.0 = calibrate judge scores using subset
- `--estimator`: Choose between `calibrated-ips` (default) or `raw-ips`

## Policies

Policies are dynamically discovered from `policy_config.py`:
- **base**: Standard helpful assistant (llama4-maverick)
- **clone**: Identical to base (control)
- **unhelpful**: Deliberately poor responses
- **premium**: High-quality model (llama-v3p1-405b-instruct)

All policies are evaluated on the same base policy responses using importance weighting.

## Resilience Features

### Batch Saving
The pipeline saves progress incrementally to handle interruptions:
- Responses and log probabilities are saved every `--batch-size` samples
- Uses atomic writes (temp file + rename) to prevent corruption
- Automatically resumes from last saved position
- Detects and skips corrupted lines during resume

### Resume Example
```bash
# Start generation (interrupt with Ctrl+C anytime)
poetry run python generate_arena_data.py --n-samples 1000 --batch-size 20

# Resume from where you left off (same command)
poetry run python generate_arena_data.py --n-samples 1000 --batch-size 20
# Output: "📂 Resuming from previous run: 340 already completed"
```