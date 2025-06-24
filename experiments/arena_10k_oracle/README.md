# Arena 10K Oracle Experiment

This experiment evaluates CJE using 10,000 prompts from the ChatBot Arena dataset.

## Overview

The experiment has two phases:
1. **Phase 1: Dataset Preparation** - Generate responses and judge scores
2. **Phase 2: CJE Ablations** - Compare estimators and uncertainty methods

## Quick Start

### Testing with Small Dataset
```bash
# Create a 20-sample test dataset
python create_test_dataset.py

# Run ablation analysis on test data
python run_ablation_analysis.py

# Visualize results
python visualize_ablation_results.py
```

### Full Experiment
```bash
# Phase 1: Prepare dataset
cd phase1_dataset_preparation
source ../../../set_secrets.sh  # Load API keys

# Generate responses and scores
python 02b_generate_target_responses.py  # Generate policy responses
python 04a_deterministic_judge_scores.py  # Score with deterministic judge
python 04b_uncertainty_judge_scores.py    # Score with CI uncertainty

# Phase 2: Run ablations
cd ../phase2_cje_ablations
python run_ablations_full.py
```

## Key Components

### Analysis Tools
- `run_ablation_analysis.py` - Direct estimator comparison using PrecomputedMultiTargetSampler
- `visualize_ablation_results.py` - Generate comparison charts
- `create_test_dataset.py` - Create small test dataset for quick iteration

### Verification Tests
- `verification_tests/test_clone_policy_weights.py` - Verify clone policy has unit weights
- `verification_tests/test_clone_policy_estimation.py` - Verify clone policy returns empirical mean

## Experiment Design

### Policies
- **π₀ (logging)**: Base policy generating initial responses  
- **π_cot**: Chain-of-thought reasoning
- **π_bigger_model**: Larger model
- **π_bad**: Intentionally poor responses (baseline)

### Estimators Compared
- IPS (Inverse Propensity Scoring)
- SNIPS (Self-Normalized IPS)
- CalibratedIPS
- DRCPO (Doubly Robust CPO)
- MRDR (Model-based Reward Doubly Robust)

### Judge Types
- **Deterministic**: Standard scoring (variance = 0)
- **Confidence Interval**: Uncertainty via confidence intervals

## Key Findings

1. **Clone Policy Verification**: Confirmed that when π_target = π_behavior, all importance weights = 1.0
2. **Uncertainty Matters**: CI-based uncertainty improves calibration
3. **Estimator Performance**: DR methods generally outperform IPS-only methods

## File Structure
```
arena_10k_oracle/
├── data/                    # Generated datasets
├── phase1_dataset_preparation/
│   ├── 01_sample_prompts.py
│   ├── 02b_generate_target_responses.py
│   ├── 04a_deterministic_judge_scores.py
│   └── 04b_uncertainty_judge_scores.py
├── phase2_cje_ablations/
│   └── run_ablations_full.py
├── verification_tests/      # Unit tests for core properties
├── run_ablation_analysis.py # Simplified analysis tool
└── create_test_dataset.py   # Test data generator
```

## Notes

- The experiment bypasses the full CJE pipeline for direct estimator comparison
- Uses `PrecomputedMultiTargetSampler` for efficient multi-target evaluation
- Supports both deterministic and uncertainty-aware judging