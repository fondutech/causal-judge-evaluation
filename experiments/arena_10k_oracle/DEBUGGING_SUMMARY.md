# Arena 10K Oracle Experiment: Debugging Summary

## Overview
This document summarizes the debugging process and key findings from the Arena 10K Oracle experiment, which evaluates CJE using 10,000 ChatBot Arena prompts.

## Critical Issues Found

### 1. Missing Teacher Forcing Step
**Problem**: The experiment wasn't computing log P(p0_response | context, target_policy) for each target policy.

**Impact**: Without these log probabilities, all importance weights defaulted to 1.0, making all policies appear identical in the ablation analysis.

**Solution**: Created `02c_compute_target_logprobs.py` to teacher-force P0 responses through each target policy.

### 2. Invalid API Key
**Problem**: Target uncertainty scores were all 0.5 because of an invalid Fireworks API key (only 17 characters instead of 48).

**Symptoms**:
- All uncertainty scores exactly 0.5 with 0 variance
- 403 Forbidden errors from Fireworks API
- ~30,000 invalid scores already computed

**Solution**: 
- Sourced proper API key from `set_secrets.sh`
- Moved bad data to `*_bad.jsonl` files
- Re-running scoring with valid API key

### 3. Checkpoint System Bug
**Problem**: The checkpoint system was creating duplicates when resuming, leading to 20,000 responses instead of 10,000.

**Root Cause**: `BatchProcessor` was returning checkpoint data as-is without filtering already-processed items.

**Solution**: Fixed in PR with proper duplicate filtering in `BatchProcessor`.

## Current Status (as of last check)

| Task | Progress | Status | ETA |
|------|----------|--------|-----|
| Teacher Forcing | 80/10,000 (0.8%) | 🟢 Running | ~2-3 hours |
| Uncertainty Scoring | 960/30,000 (3.2%) | 🟢 Running | ~5-6 hours |
| Deterministic Scoring | 1,067/30,000 (3.6%) | 🟢 Running | ~5-6 hours |

## Key Architectural Insights

### 1. CJE Pipeline Stages
The full CJE pipeline has these critical stages:
1. **Logging Policy**: Generate responses with π₀
2. **Target Policy**: Compute log probs for target policies ← **MISSING IN ARENA**
3. **Judge**: Score all responses  
4. **Estimation**: Compute importance-weighted estimates

### 2. Teacher Forcing Implementation
```python
# Correct: Teacher force P0 response through target policy
logp_target = target_policy.log_prob(context, p0_response)

# Incorrect: Using target's own response
logp_target = target_policy.log_prob(context, target_response)
```

### 3. Importance Weights Formula
```
weight = exp(log P(p0_response | context, π_target) - log P(p0_response | context, π_0))
```

Without proper teacher forcing, both terms are -10.0, making weight = 1.0.

## Lessons Learned

### 1. Don't Bypass the Pipeline
The CJE pipeline handles many subtle details automatically. Manual implementations must replicate ALL stages.

### 2. Validate Early
Check importance weights aren't all 1.0 before running expensive analyses.

### 3. API Key Management
Always verify API keys are valid before large-scale runs. The 17-character key looked plausible but wasn't valid.

### 4. Checkpoint System Design
The checkpoint system needs careful handling of duplicates, especially when resuming partial runs.

## Next Steps

1. **Wait for current processes to complete** (~6 hours)
2. **Run updated ablation analysis** using `run_ablation_analysis_v2.py`
3. **Verify importance weights vary** across policies
4. **Compare estimator performance** with proper weights
5. **Generate final visualizations** and results

## Files Created/Modified

### New Scripts
- `02c_compute_target_logprobs.py` - Teacher forcing implementation
- `run_ablation_analysis_v2.py` - Updated analysis using teacher forcing data
- `monitor_all_tasks.py` - Real-time progress monitoring
- `complete_uncertainty_scoring.py` - Robust scoring with error handling
- `fix_duplicates.py` - Clean duplicate entries

### Documentation
- `CJE_PIPELINE_INSIGHTS.md` - Architectural insights
- `DEBUGGING_SUMMARY.md` - This file

### Data Files
- `data/p0_with_target_logps.jsonl` - P0 data with target log probs (in progress)
- `data/targets_scored_uncertainty.jsonl` - Re-scored with valid API key (in progress)
- `data/targets_scored_deterministic.jsonl` - Deterministic scores (in progress)

## Commands to Resume

If processes stop, resume with:
```bash
# Teacher forcing
cd phase1_dataset_preparation
source ../../../set_secrets.sh
python 02c_compute_target_logprobs.py

# Uncertainty scoring  
python 04d_score_targets_uncertainty.py --batch-size 16

# Deterministic scoring
python 04c_score_targets_deterministic.py --batch-size 32

# Run analysis when ready
cd ..
python run_ablation_analysis_v2.py
```

## Expected Results

With proper teacher forcing, we expect:
- **Importance weights** to vary significantly across policies
- **π_cot** to perform better than baseline (chain-of-thought helps)
- **π_bigger_model** to perform best (70B vs 8B model)
- **π_bad** to perform worst (high temperature + constraints)
- **Uncertainty scoring** to show higher variance than deterministic
- **DR estimators** to outperform IPS-only methods

## Key Takeaway

**Teacher forcing is not optional** - it's the foundation of importance weighting in off-policy evaluation. Without it, all policies appear identical regardless of their true quality.