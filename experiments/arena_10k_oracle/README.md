# Arena 10K Human Oracle Experiment

## Overview

This experiment validates Causal Judge Evaluation (CJE) on real ChatBot Arena prompts with **human oracle labels** (crowdsourced, not AI-generated). The experiment has been implemented with improved infrastructure for robustness and monitoring.

**Current Status**: Target policy ground truth generation added. Two separate data streams for calibration and validation.

## Experiment Design

### Dataset
- **Source**: 10,000 single-turn prompts from ChatBot Arena Conversations
- **Calibration Data**: π₀ responses for judge→human calibration (25% of π₀ dataset)
- **Ground Truth Data**: Target policy responses for validation (500 prompts per policy)

### Policies

| Policy | Model | Temperature | Description |
|--------|-------|-------------|-------------|
| π₀ (logging) | llama4-scout-instruct-basic | T=0.5 | baseline logging policy |
| π_clone | llama4-scout-instruct-basic | T=0.5 | identical to π₀ (sanity check) |
| π_cot | llama4-scout-instruct-basic | T=0.5 | chain-of-thought prompting |
| π_bigger_model | llama4-maverick-instruct-basic | T=0.5 | larger model variant |

### Judge Configuration
- **Model**: llama4-scout-instruct-basic at T=0
- **Scoring**: 0-1 scale for helpfulness/correctness/safety
- **Calibration**: Isotonic regression from judge scores to human labels

### Validation Design
- **CJE Prediction**: Uses π₀ responses + importance weights to predict target policy performance
- **Ground Truth**: Actual human labels on target policy responses
- **Validation**: Compare CJE estimates to ground truth human preferences

## Repository Structure

```
experiments/arena_10k_oracle/
├── README.md                        # This file
├── .gitignore                       # Excludes data files from git
├── configs/
│   └── arena_10k.yaml              # Main experiment configuration
├── scripts/
│   ├── 01_prepare_data.py          # Sample prompts from Arena dataset
│   ├── 02_generate_logs.py         # Generate π₀ responses with teacher forcing
│   ├── 02b_generate_target_ground_truth.py  # Generate target policy responses
│   ├── 03_export_for_labeling.py   # Export for crowdsourcing (both types)
│   ├── 04_add_judge_scores.py      # Score all responses with LLM judge
│   ├── 05_generate_target_policies.py  # Generate target policy responses (TBD)
│   ├── 06_import_labels.py         # Import human labels (TBD)
│   ├── check_fireworks_models.py   # Utility to verify model access
│   └── experiment_status.py        # Monitor pipeline progress
├── data/
│   ├── prompts.jsonl               # 10k sampled prompts
│   ├── p0_replies.jsonl            # π₀ responses with teacher forcing logprobs
│   ├── p0_scored.jsonl             # π₀ responses with judge scores
│   ├── target_ground_truth.jsonl   # Target policy responses (no logprobs)
│   ├── target_ground_truth_scored.jsonl  # Target policies with judge scores
│   └── labeling/
│       ├── calibration_export_surge.csv      # π₀ responses for calibration
│       ├── ground_truth_pi_clone_surge.csv   # Target policy responses
│       ├── ground_truth_pi_cot_surge.csv     # for human validation
│       ├── ground_truth_pi_bigger_model_surge.csv
│       └── ground_truth_combined_surge.csv   # All target policies
└── outputs/                        # (created during CJE estimation)
```

## Key Features

- **Robust checkpointing**: Atomic writes prevent data corruption and duplicates
- **Two-pass generation**: Teacher forcing ensures consistent importance weights
- **Separate validation streams**: Calibration data vs ground truth data
- **Progress tracking**: Resume from exact position if interrupted

## Running the Experiment

### Prerequisites

```bash
# Set API key for Fireworks
export FIREWORKS_API_KEY="your-key"

# Verify model access
cd scripts
python check_fireworks_models.py
```

### Check Current Status

```bash
# See what's been completed
python experiment_status.py
```

### Pipeline Steps

```bash
# Step 1: Sample prompts
python 01_prepare_data.py --samples 10000

# Step 2: Generate π₀ responses (with teacher forcing for CJE)
python 02_generate_logs.py

# Step 2b: Generate target policy responses (for ground truth validation)
python 02b_generate_target_ground_truth.py --samples 500

# Step 3: Export for human labeling (both calibration + ground truth)
python 03_export_for_labeling.py \
  --p0-input ../data/p0_replies.jsonl \
  --target-input ../data/target_ground_truth.jsonl

# Step 4: Score all responses with judge
python 04_add_judge_scores.py

# Step 5: Generate target policy responses (TBD)
# python 05_generate_target_policies.py

# Step 6: Import human labels and run CJE validation
python 06_import_labels.py --labels downloaded_labels.csv
```

## Experiment Validation

### Data Sources
1. **Calibration**: π₀ responses + human labels → Train judge→human mapping
2. **Ground Truth**: Target policy responses + human labels → What CJE should predict  
3. **CJE Estimates**: π₀ responses + importance weights → CJE predictions

### Critical Implementation Details
- **Two-pass generation**: π₀ uses teacher forcing to ensure consistent importance weights
- **Target policies**: Generate responses without logprobs (human labeling only)
- **Validation**: Compare CJE estimates to actual human preferences on target policies

## Cost Estimates

| Dataset | Samples | Labels Needed | Estimated Cost |
|---------|---------|---------------|----------------|
| **Calibration** | 25% of π₀ data | ~750 (250 × 3 votes) | ~$60 |
| **Ground Truth** | 500 per policy × 3 policies | 4,500 (1,500 × 3 votes) | ~$360 |
| **API Costs** | All generations + scoring | | ~$20 |
| **Total** | | | **~$440** |