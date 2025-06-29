# Phase 1: Dataset Preparation

This phase prepares the Arena 10K dataset with:
1. P0 (base policy) responses
2. Target policy responses (4 policies including pi_clone)
3. Teacher-forced log probabilities
4. Oracle labels
5. Judge scores with uncertainty

## Quick Start

### 1% Sample Test (Recommended First)
```bash
# Run 1% sample to validate pipeline
./run_sample_test.sh

# Or run components separately:
cd sample_run
python preflight_check.py      # Pre-flight validation
./run_sample.sh               # Run sample pipeline
python validate_sample_results.py  # Validate results
```

### Full Run (After Sample Passes)
```bash
# Ensure sample mode is off
unset ARENA_SAMPLE_MODE

# Run full pipeline (50-75 hours)
./run_full_pipeline.sh
```

## Scripts

### Core Pipeline
- `01_prepare_data.py` - Extract Arena prompts
- `02a_generate_p0_responses.py` - Generate base policy responses
- `02b_generate_target_responses.py` - Generate target policy responses
- `02c_compute_target_logprobs.py` - Compute teacher forcing ⚠️ CRITICAL
- `03_generate_oracle_labels.py` - Get ground truth labels
- `04*_score_*.py` - Judge scoring scripts

### Utilities
- `run_full_pipeline.sh` - Automated full run
- `emergency_stop.sh` - Stop pipeline if needed
- `sample_run/` - Complete 1% sample testing suite

## Critical Validation

The most important check is that teacher forcing returns no 0.0 log probabilities for non-empty responses. This bug has been fixed in `cje.utils.RobustTeacherForcing`.

## Expected Outputs

All outputs saved to `../data/` (or `../data/sample_1pct/` for sample run):
- `arena_questions_base.jsonl` - Input prompts
- `p0_replies.jsonl` - P0 responses
- `target_responses.jsonl` - Target policy responses
- `p0_with_target_logps.jsonl` - Teacher forcing results
- `oracle_labels.jsonl` - Oracle labels
- `*_scored.jsonl` - Judge scores