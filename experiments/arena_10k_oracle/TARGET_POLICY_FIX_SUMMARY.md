# Target Policy Stage Fix Summary

## The Bug

**Location**: `cje/pipeline/stages/target_policy.py`, line 132

**Issue**: The code called `sampler.log_prob(contexts)` but this method doesn't exist in `MultiTargetSampler`.

```python
# BROKEN CODE (before fix)
def _compute_logprobs(self, rows, sampler, num_policies):
    contexts = [row["context"] for row in rows]
    logprobs_result = sampler.log_prob(contexts)  # ❌ This method doesn't exist!
```

## The Fix

**Solution**: Extract responses from rows and use the existing `logp_matrix` method.

```python
# FIXED CODE
def _compute_logprobs(self, rows, sampler, num_policies):
    # Validate responses exist
    for i, row in enumerate(rows[:5]):
        if "response" not in row:
            raise ValueError(f"Row {i} missing 'response' field...")
    
    contexts = [row["context"] for row in rows]
    responses = [row["response"] for row in rows]  # ✅ Extract responses
    
    # Use the correct method that exists
    logprobs_matrix = sampler.logp_matrix(contexts, responses)  # ✅ Teacher forcing!
    all_logprobs = logprobs_matrix.tolist()
```

## Why This Matters

Teacher forcing is essential for importance weighting:

```
weight = exp(log P(response | context, π_target) - log P(response | context, π_behavior))
```

Without the responses, we can't compute `P(response | context, π_target)`.

## Impact

1. **Before**: Target policy stage was broken, couldn't compute importance weights
2. **After**: Teacher forcing works correctly, enabling proper off-policy evaluation

## Verification

1. **Unit test**: `tests/test_target_policy_fix.py` - All tests pass ✅
2. **Linting**: `make lint` - Code formatted and type-checked ✅
3. **Demonstration**: Shows correct importance weight computation ✅

## Key Insight from User

The "context" passed to teacher forcing must be the **exact input sequence** fed to the logging policy, including:
- System prompts
- User message templates  
- Additional instructions (e.g., CoT prompts)
- Any formatting specific to the logging policy

In our Arena experiment, P0 used raw contexts with no additional formatting, so using `row["context"]` directly is correct.

## Files Modified

1. `/cje/pipeline/stages/target_policy.py` - Fixed the bug
2. `/tests/test_target_policy_fix.py` - Added comprehensive tests
3. Various documentation files explaining the issue and fix

## Next Steps

1. This fix should be submitted as a PR to the CJE repository
2. The Arena experiment can now use the fixed pipeline if desired
3. Our manual implementation remains valid and instructive