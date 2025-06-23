# Phase 1: Dataset Preparation

This phase creates a complete dataset for evaluating CJE on ChatBot Arena prompts. All scripts are designed to run sequentially with minimal user intervention.

## Prerequisites

1. **API Keys**: Set the following environment variables:
   ```bash
   export FIREWORKS_API_KEY="your-fireworks-key"
   export OPENAI_API_KEY="your-openai-key"  # For oracle labeling
   ```

2. **Dependencies**: Ensure CJE is properly installed (run `make dev-setup` from repo root)

## Overview

The goal is to generate a dataset containing:
- 10,000 prompts from ChatBot Arena
- Responses from multiple policies (π₀, π_cot, π_bigger_model, π_bad)
- Oracle labels for ground truth
- Judge scores with two uncertainty methods (deterministic and confidence intervals) for ALL policies

## Sequential Execution

**All scripts use consistent paths (`../data/`) and can be run in order:**

### Step 1: Download and Prepare Prompts
```bash
python 01_prepare_data.py
# Output: ../data/arena_prompts_10k.jsonl (10,000 prompts)
```

### Step 2: Generate Responses (Can Run in Parallel)

```bash
# Terminal 1: Generate π₀ (logging policy) responses
python 02a_generate_p0_responses.py
# Output: ../data/p0_replies.jsonl (10,000 responses)

# Terminal 2: Generate target policy responses (runs all 3 policies in parallel by default)
python 02b_generate_target_responses.py
# Output: ../data/target_responses.jsonl (30,000 responses: 10,000 prompts × 3 policies)

# Optional: Run target policies sequentially instead
python 02b_generate_target_responses.py --sequential

# Optional: Run only specific policies
python 02b_generate_target_responses.py --policies pi_cot pi_bigger_model
```

### Step 3: Generate Oracle Labels
```bash
python 03_generate_oracle_labels.py
# Output: ../data/labeling/oracle_labels_calibration_detailed.jsonl (2,500 labels)
#         ../data/labeling/oracle_labels_validation_detailed.jsonl (1,500 labels)
```

### Step 4: Add Judge Scores (Can Run All 4 in Parallel)

**Group 1 - Score π₀ responses:**
```bash
# Terminal 1: Deterministic scoring
python 04a_deterministic_judge_scores.py
# Output: ../data/p0_scored_deterministic.jsonl

# Terminal 2: Uncertainty scoring
python 04b_uncertainty_judge_scores.py  
# Output: ../data/p0_scored_uncertainty.jsonl
```

**Group 2 - Score target responses:**
```bash
# Terminal 3: Deterministic scoring
python 04c_score_targets_deterministic.py
# Output: ../data/targets_scored_deterministic.jsonl

# Terminal 4: Uncertainty scoring
python 04d_score_targets_uncertainty.py
# Output: ../data/targets_scored_uncertainty.jsonl
```

### Step 5: Finalize Dataset
```bash
python 05_finalize_dataset.py
# Output: ../data/dataset_info.json (summary statistics)
```

## Parallel Execution Summary

For maximum efficiency:
1. Run steps 2a and 2b in parallel
2. Run step 3 alone (oracle labeling)
3. Run steps 4a, 4b, 4c, 4d in parallel
4. Run step 5 to finalize

## Checkpointing

Most scripts support automatic checkpointing and can resume if interrupted:
- Scripts 02a, 02b save checkpoints every batch
- Script 03 saves checkpoints every 10 labels
- Simply re-run the same command to resume

## Dataset Structure

```
data/
├── arena_prompts_10k.jsonl              # 10,000 prompts
├── p0_replies.jsonl                     # π₀ responses (10,000)
├── target_responses.jsonl               # Target policy responses (30,000 = 10k × 3)
├── p0_scored_deterministic.jsonl        # π₀ deterministic scores (10,000)
├── p0_scored_uncertainty.jsonl          # π₀ uncertainty scores (10,000)
├── targets_scored_deterministic.jsonl   # Target deterministic scores (30,000)
├── targets_scored_uncertainty.jsonl     # Target uncertainty scores (30,000)
├── dataset_info.json                    # Dataset summary
└── labeling/
    ├── oracle_labels_calibration_detailed.jsonl  # 2,500 labels (25% of π₀)
    ├── oracle_labels_validation_detailed.jsonl   # 1,500 labels (5% of targets)
    └── oracle_labels.csv                        # Combined CSV format
```

## Key Statistics

- **Total prompts**: 10,000
- **π₀ responses**: 10,000 (all scored with both judge types)
- **Target responses**: 30,000 total (10,000 × 3 policies, all scored with both judge types)
- **Calibration set**: 2,500 oracle-labeled π₀ responses (25% sample)
- **Validation set**: 1,500 oracle-labeled target policy responses (5% sample = 500 prompts × 3)
- **Judge scoring methods**: 2 (deterministic, uncertainty)
- **Total oracle labels**: 4,000


## Next Steps

Once the dataset is finalized, proceed to Phase 2 for CJE ablations:
- Compare deterministic vs uncertainty judge scoring
- Analyze if uncertainty improves calibration
- Test different estimators (IPW, SNIPW)
- Examine judge behavior across policy quality levels