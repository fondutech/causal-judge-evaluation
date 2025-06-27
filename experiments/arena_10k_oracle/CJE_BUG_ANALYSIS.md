# CJE Target Policy Bug Analysis

## The Bug

In `cje/pipeline/stages/target_policy.py`, line 132:
```python
logprobs_result = sampler.log_prob(contexts)
```

This method doesn't exist in `MultiTargetSampler`!

## Available Methods in MultiTargetSampler

```python
# What exists:
def logp_many(self, context: str, response: str) -> List[float]
def logp_matrix(self, contexts: List[str], responses: List[str]) -> NDArray

# What's being called (doesn't exist):
def log_prob(self, contexts: List[str]) -> ???
```

## The Fix

The target policy stage should extract responses and use `logp_matrix`:

```python
def _compute_logprobs(self, rows, sampler, num_policies):
    contexts = [row["context"] for row in rows]
    responses = [row["response"] for row in rows]  # ← MISSING!
    
    # Use the actual method that exists
    logprobs_matrix = sampler.logp_matrix(contexts, responses)
    
    # Convert to expected format
    all_logprobs = logprobs_matrix.tolist()
    all_responses = [[None] * num_policies] * len(contexts)
```

## Why This Matters

Without passing the responses, the pipeline can't compute:
```
log P(p0_response | context, target_policy)
```

This is essential for importance weighting:
```
weight = exp(log P(response|context, π_target) - log P(response|context, π_0))
```

## Impact on Arena Experiment

This bug explains why we needed to implement teacher forcing manually. The pipeline's target policy stage is broken and can't compute the necessary log probabilities.

## Verification

Run this to verify the bug:
```python
from cje.loggers.multi_target_sampler import MultiTargetSampler
sampler = MultiTargetSampler([], [])
# This will fail:
# sampler.log_prob(["context1", "context2"])
```

## Recommendation

1. File a bug report for the CJE library
2. Continue using our manual implementation which correctly implements teacher forcing
3. The fix is simple but needs to be tested thoroughly