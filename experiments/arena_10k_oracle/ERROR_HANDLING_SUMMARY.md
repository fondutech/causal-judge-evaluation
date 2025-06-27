# Log Probability Error Handling Summary

## Current Issues in CJE

### 1. Hardcoded Fallback Values
- `FALLBACK_LOG_PROB = -100.0` in `error_handling.py`
- Used by `safe_call()` when exceptions occur
- `multi_target_sampler.py` and `trajectory_sampler.py` use this fallback
- **Impact**: Silently corrupts importance weights in off-policy evaluation

### 2. Silent Failures
- API errors return fake values instead of failing explicitly
- No visibility into failure rates or patterns
- No automatic retry for transient failures

### 3. Our Arena 10K Findings
- Fixed `api_policy.py` to raise exceptions instead of returning 0.0
- Found 708 samples that would have had corrupted log probs
- Rescored 139 samples that had been "fixed" to -50.0
- Still have 15 legitimate failures marked as -50.0

## Recommended Solution

### 1. Remove Hardcoded Fallbacks
```python
# DELETE THESE:
FALLBACK_LOG_PROB = -100.0  # Dangerous!
FALLBACK_PROBABILITY = 1e-10  # Also dangerous!
```

### 2. Implement Robust Error Handling
- **Explicit failure tracking** - Return `None` or raise exceptions
- **Automatic retries** - Handle transient API failures gracefully
- **Detailed reporting** - Track failure types and patterns
- **Actionable recommendations** - Suggest fixes based on error types

### 3. Three Options for Handling Failures

#### Option A: Fail Fast (Current PR)
```python
if not result.success:
    raise RuntimeError(f"Log prob failed: {result.error_message}")
```
- ✅ No data corruption
- ❌ Stops entire analysis

#### Option B: Mark as None (Recommended)
```python
if not result.success:
    logps.append(None)  # Explicit missing value
```
- ✅ Analysis can continue
- ✅ Clear which samples failed
- ✅ Can filter or impute later

#### Option C: Skip Failed Samples
```python
if not result.success:
    continue  # Skip this sample entirely
```
- ✅ Clean dataset
- ❌ Reduces sample size
- ❌ May introduce selection bias

## Implementation Priority

### Phase 1: Critical Fix (Immediate)
1. Remove `FALLBACK_LOG_PROB` constant
2. Update `multi_target_sampler.py` to not use fallbacks
3. Add basic retry logic for API calls

### Phase 2: Robust Handling (Next Sprint)
1. Implement `LogProbResult` and retry logic
2. Add `FailureTracker` for reporting
3. Create failure analysis tools

### Phase 3: Advanced Features (Future)
1. Smart retry strategies by error type
2. Automatic failover to backup models
3. Real-time failure monitoring dashboard

## Key Takeaways

1. **Never use arbitrary fallback values** - They silently corrupt results
2. **Make failures visible** - Log, track, and report all failures
3. **Provide options** - Let users decide how to handle failures
4. **Retry transient failures** - Many API errors are temporary
5. **Document failure impact** - Help users understand limitations

## Our Experiment Results

Despite these issues, our Arena 10K analysis is valid because:
- We fixed `api_policy.py` before running teacher forcing
- We identified and rescored samples with fake values
- We have only 15 legitimate failures out of 10,000 samples (0.15%)
- No -100.0 fallback values contaminated our data

However, the broader CJE codebase needs these fixes to prevent future issues.