# GitHub Issue: Silent Log Probability Failures Corrupt Off-Policy Evaluation

## Summary

CJE currently uses hardcoded fallback values when log probability computations fail, which silently corrupts importance weights and invalidates off-policy evaluation results.

## Current Behavior

When API calls fail during log probability computation:
- `api_policy.py`: Returns `0.0` (implies probability = 1.0)
- `multi_target_sampler.py`: Uses `FALLBACK_LOG_PROB = -100.0`
- `trajectory_sampler.py`: Also uses `FALLBACK_LOG_PROB = -100.0`

## Impact

These fallback values destroy importance weights:
```python
# Example: Correct vs Corrupted weights
correct_weight = exp(-15 - (-12)) = 0.0498  # Real log probs
corrupted_weight = exp(-100 - (-12)) = 6e-39  # With -100 fallback
error_factor = 8.22e36  # Weight is wrong by 36 orders of magnitude!
```

## Steps to Reproduce

1. Run teacher forcing with an invalid API key or model name
2. Check the resulting log probabilities in the output
3. Observe `0.0` or `-100.0` values instead of proper error handling

## Expected Behavior

Log probability failures should:
1. Raise clear exceptions with context
2. Never return arbitrary values
3. Allow users to handle failures explicitly

## Proposed Solution

See PR #[XXX] which:
- Removes hardcoded fallback values
- Makes `api_policy.py` raise exceptions instead of returning 0.0
- Updates `multi_target_sampler.py` to handle failures explicitly

## Workaround

Until fixed, users should:
1. Check for suspicious log prob values (0.0, -100.0, -50.0)
2. Manually validate API credentials before running
3. Monitor logs for silent failures

## Environment

- Discovered during Arena 10K oracle experiment
- Affects all versions with current error_handling.py
- Critical for any off-policy evaluation use case

## Severity

**CRITICAL** - Silent data corruption affecting core functionality

## Labels

- bug
- critical
- data-corruption
- off-policy-evaluation