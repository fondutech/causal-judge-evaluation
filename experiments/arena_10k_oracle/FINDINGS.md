# Arena 10K Experiment Findings

This document summarizes key findings, bugs fixed, and improvements made to the Arena 10K experiment.

## Critical Bug Fix: Token Boundary Issue

### The Problem
The most critical issue discovered was a token boundary bug that caused extreme importance weights (up to 1.79e+13 for pi_clone, which should be ~1.0).

When computing teacher-forced log probabilities, the token counting method failed when:
- Prompt ends with punctuation (e.g., "recent.")
- Response starts with a capital letter (e.g., "Here")
- The tokenizer merges these into a single token (e.g., ".Here")

This caused the algorithm to extract log probabilities from the wrong token positions.

### The Solution
Implemented a multi-layered fix in `cje/utils/teacher_forcing.py`:

1. **Edge Case Detection**: Automatically detects problematic token boundaries
2. **Method Switching**: Uses continuation method for edge cases (log P(response|prompt) = log P(full) - log P(prompt))
3. **Validation**: Added extreme weight detection in `02b_compute_logprobs.py` to reject corrupted values
4. **Force Continuation Option**: Added `force_continuation=True` flag to always use the most reliable method

### Results
- Sample 40 recomputed: weight = 1.0 ✓ (was 1.79e+13)
- Extreme weights now automatically rejected
- ESS expected to improve from 0.5% to >50%

## Other Major Improvements

### 1. English-Only Filter
- **Problem**: Non-English prompts caused teacher forcing failures (~8% failure rate)
- **Solution**: Added English-only filtering in `01_prepare_data.py`
- **Result**: Eliminated language-related log probability failures

### 2. Fixed Judge Scoring Bug
- **Problem**: Scripts used `0.0` as default for missing log probabilities
- **Impact**: Failed computations treated as perfect certainty (log(1) = 0)
- **Solution**: Changed defaults to `None` in `03_judge_scores_*.py`

### 3. Pipeline Resumability
- **Problem**: Pipeline would clean data directory on restart, losing progress
- **Solution**: Modified `run_phase1_pipeline.py` to resume by default
- **Impact**: Can safely interrupt and resume long runs without data loss

## Known Limitations

### API Non-Determinism
- **Issue**: Fireworks API returns slightly different log probabilities for identical inputs
- **Impact**: pi_clone weights vary around 1.0 instead of being exactly 1.0
- **Status**: Random noise (~0.87 nats), not systematic bias

## Key Validation Results

### 50-Sample English-Filtered Run
- **Data quality**: 0 missing log probabilities (vs 8% before)
- **pi_clone median weight**: 1.000 (validates implementation)
- **All policies evaluated successfully**

### Importance Weight Analysis
- **pi_clone**: Weights centered around 1.0 as expected
- **pi_cot**: Lower weights due to chain-of-thought differences
- **pi_bigger_model**: Very low weights (different model architecture)
- **pi_bad**: Extreme weights due to temperature difference

## Recommendations for 10K Run

1. **Use validated script**: Run `02b_compute_logprobs.py` which includes extreme weight detection
2. **Monitor extreme_weights.jsonl**: Check for any flagged samples
3. **Expect good ESS**: With validation, ESS should be >50% even for challenging policies
4. **English filter works**: Expect <1% missing log probs
5. **Maximum reliability enabled**: Phase 1 uses `force_continuation=True` - ONLY continuation method with no fallback. Failed samples will have null log probs rather than risk token boundary corruption