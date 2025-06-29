# Fresh Start Guide for Arena 10K Experiment

## Overview

We're rerunning the Arena 10K experiment with the fixed teacher forcing implementation. This guide walks through the complete process.

## Key Changes

1. **Teacher forcing is fixed**: No more 0.0 log probabilities for 708 samples
2. **Robust implementation**: Uses `cje.utils.RobustTeacherForcing`
3. **No fallback values**: Explicit error handling throughout

## Step 0: Clean Start

```bash
cd experiments/arena_10k_oracle

# We've already cleaned:
# - Removed all processed data files
# - Kept only original prompts and target responses
# - Removed old logs and reports
```

## Step 1: Update Teacher Forcing Usage

The current phase1 scripts use `APIPolicyRunner.log_prob()` which may not use the robust implementation. We need to update `02c_compute_target_logprobs.py` to use the robust teacher forcing.

### Option 1: Quick Fix

Update the APIPolicyRunner implementations to use RobustTeacherForcing internally.

### Option 2: Direct Usage

Modify `02c_compute_target_logprobs.py` to directly use RobustTeacherForcing:

```python
from cje.utils import RobustTeacherForcing

# Instead of:
# logp = runner.log_prob(context, response)

# Use:
tf = RobustTeacherForcing(provider="fireworks", model=runner.model_name)
result = tf.compute_log_prob(context, response)
if result.is_valid:
    logp = result.value
else:
    raise ValueError(f"Teacher forcing failed: {result.error}")
```

## Step 2: Run the Pipeline

```bash
cd phase1_dataset_preparation

# 1. Prepare data (no changes needed)
python 01_prepare_data.py

# 2. Generate responses (no changes needed)
python 02a_generate_p0_responses.py
python 02b_generate_target_responses.py

# 3. Compute log probabilities (USES ROBUST TEACHER FORCING)
python 02c_compute_target_logprobs.py

# 4. Oracle labeling (no changes needed)
python 03_generate_oracle_labels.py

# 5. Judge scoring (no changes needed)
python 04a_deterministic_judge_scores.py
python 04b_uncertainty_judge_scores.py
python 04c_score_targets_deterministic.py
python 04d_score_targets_uncertainty.py

# 6. Finalize dataset
python 05_finalize_dataset.py
```

## Step 3: Verify No 0.0 Log Probabilities

After step 3, verify the fix worked:

```python
import json

# Check for suspicious 0.0 values
with open("../data/p0_with_target_logps.jsonl") as f:
    data = [json.loads(line) for line in f]
    
zeros = [d for d in data if any(
    v == 0.0 for v in d.get("target_logps", {}).values()
)]

print(f"Found {len(zeros)} samples with 0.0 log probabilities")
# Should be 0 or very few (only genuinely constrained responses)
```

## Step 4: Run CJE Analysis

```bash
cd ../phase2_cje_ablations

# Run all ablations
python run_ablations.py --config configs/ablations/*.yaml
```

## Expected Results

With the fix:
- No more 708 samples with 0.0 log probability
- Importance weights will be reasonable (not 10^17)
- Model rankings should reflect true performance
- `pi_bad` should NOT be the best model

## Monitoring Progress

Watch for:
- Log probability validation errors in step 3
- Extreme importance weights in step 4
- Suspicious model rankings in final results