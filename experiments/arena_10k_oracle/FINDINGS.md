# Arena 10K Experiment Findings

This document summarizes key findings and improvements made to the Arena 10K experiment.

## Major Improvements

### 1. English-Only Filter
- **Problem**: Non-English prompts caused teacher forcing failures (~8% failure rate)
- **Solution**: Added English-only filtering in `01_prepare_data.py`
- **Result**: Eliminated language-related log probability failures

### 2. Fixed Judge Scoring Bug
- **Problem**: Scripts used `0.0` as default for missing log probabilities
- **Impact**: Failed computations treated as perfect certainty (log(1) = 0)
- **Solution**: Changed defaults to `None` in `03_judge_scores_*.py`
- **Result**: Proper handling of missing values in importance weighting

### 3. Pipeline Resumability
- **Problem**: Pipeline would clean data directory on restart, losing progress
- **Solution**: Modified `run_phase1_pipeline.py` to resume by default
- **Impact**: Can safely interrupt and resume long runs without data loss

### 4. Improved Progress Reporting
- **Problem**: Nested batching created confusing progress messages
- **Solution**: Clear separation between checkpoint batches and API batches
- **Result**: Progress is now clearly understandable

## Known Limitations

### 1. Tokenization Boundary Issues
- **Affected prompts**: ~1% (e.g., prompts ending with punctuation)
- **Example**: "Write a single dot." → "." (tokenization of ".." differs from ". .")
- **Handling**: Teacher forcing correctly detects and rejects these cases

### 2. API Non-Determinism
- **Issue**: Fireworks API returns slightly different log probabilities for identical inputs
- **Impact**: pi_clone weights vary around 1.0 instead of being exactly 1.0
- **Status**: Random noise, not systematic bias (median still ~1.0)

## Key Findings from Test Runs

### 50-Sample English-Filtered Run
- **Data quality**: 0 missing log probabilities (vs 8% before)
- **pi_clone median weight**: 1.000 (validates implementation)
- **ESS**: Low (2-23%) due to small sample size
- **All policies evaluated successfully**

### Importance Weight Analysis
- **pi_clone**: Weights centered around 1.0 as expected
- **pi_cot**: Lower weights due to chain-of-thought differences
- **pi_bigger_model**: Very low weights (different model architecture)
- **pi_bad**: Extreme weights due to temperature difference

## Recommendations for 10K Run

1. **English filter eliminates most failures** - expect <1% missing log probs
2. **System is working correctly** - pi_clone validation confirms this
3. **Extreme weights are expected** - will stabilize with larger sample
4. **Monitor for tokenization edge cases** - ~1% expected

## Technical Details

### Teacher Forcing Methods (in order of preference)
1. **Continuation method**: Most reliable, uses log subtraction
2. **Token counting**: Can fail on tokenization boundaries
3. **Echo-based**: Not implemented for Fireworks

### Data Flow
1. Download English prompts from ChatBot Arena
2. Generate responses for 5 policies (P0 + 4 targets)
3. Compute log probabilities using teacher forcing
4. Score with judges (deterministic + uncertainty)
5. Generate oracle labels (optional)
6. Output Phase 2-ready format directly