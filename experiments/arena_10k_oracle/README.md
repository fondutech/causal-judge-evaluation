# Arena 10K Oracle Experiment

This experiment evaluates causal judge estimation methods using 10,000 prompts from ChatBot Arena.

## Overview

### Phase 1: Dataset Preparation
Generates all data needed for causal judge evaluation:
- P0 (logging policy) responses
- Target policy responses (4 policies including pi_clone)
- Teacher-forced log probabilities for importance weighting
- Oracle labels (ground truth from GPT-4)
- Judge scores with uncertainty estimates

### Phase 2: CJE Ablations
Evaluates different estimation methods:
- IPS, SNIPS, DR variations
- With/without calibration
- Deterministic vs uncertainty-aware scoring

## Target Policies

1. **pi_clone**: Identical to P0 (baseline, expects importance weights ≈ 1.0)
2. **pi_cot**: Chain-of-thought prompting
3. **pi_bigger_model**: Larger model (maverick vs scout)
4. **pi_bad**: Deliberately poor policy

## Quick Start

### Prerequisites
```bash
# Set API keys (required)
export FIREWORKS_API_KEY="your-key"
export OPENAI_API_KEY="sk-your-key"

# Or source from file
source ./set_secrets.sh
```

### 1% Sample Test (Recommended First)
```bash
cd phase1_dataset_preparation
./run_sample_test.sh  # 30-45 minutes, ~$0.60
```

### Full Phase 1 Run
```bash
cd phase1_dataset_preparation
./run_full_pipeline.sh  # 50-75 hours, ~$60
```

### Phase 2 Analysis
```bash
cd phase2_cje_ablations
# Run specific ablations or all experiments
```

## Critical Fix: Teacher Forcing

The teacher forcing bug that caused 0.0 log probabilities has been fixed:
- **Problem**: Token boundaries don't align with text boundaries
- **Solution**: `RobustTeacherForcing` with 3 fallback methods
- **Validation**: Sample run must show NO 0.0 values for non-empty responses

## Cost Estimates

### 1% Sample (100 prompts)
- API calls: ~1,400
- Cost: ~$0.60
- Time: 30-45 minutes

### Full Run (10,000 prompts)
- API calls: ~140,000
- Cost: ~$60
- Time: 50-75 hours

## Directory Structure
```
arena_10k_oracle/
├── data/                    # Input/output data
├── phase1_dataset_preparation/
│   ├── 01-05_*.py          # Pipeline scripts
│   ├── sample_run/         # 1% sample testing tools
│   └── run_*.sh            # Execution scripts
└── phase2_cje_ablations/
    └── configs/            # Experiment configurations
```

## Success Criteria

1. Teacher forcing returns no 0.0 log probabilities for non-empty responses
2. All scripts complete without errors
3. Importance weights for pi_clone ≈ 1.0
4. Judge scores show expected policy ranking

For detailed information about specific phases, see the README files in each subdirectory.