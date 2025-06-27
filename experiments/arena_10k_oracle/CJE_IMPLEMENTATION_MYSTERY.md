# CJE Implementation Mystery: The Missing Method

## The Mystery

After extensive investigation, I've found a puzzling discrepancy in the CJE codebase:

### 1. What the Code Shows

In `cje/pipeline/stages/target_policy.py`:
```python
def _compute_logprobs(self, rows, sampler, num_policies):
    contexts = [row["context"] for row in rows]
    
    # This line calls a method that doesn't exist!
    logprobs_result = sampler.log_prob(contexts)
```

### 2. What Actually Exists

In `MultiTargetSampler`:
```python
class MultiTargetSampler:
    # These methods exist:
    def logp_many(self, context: str, response: str) -> List[float]
    def logp_matrix(self, contexts: List[str], responses: List[str]) -> NDArray
    
    # This method does NOT exist:
    def log_prob(self, contexts: List[str]) -> ???
```

## Possible Explanations

### 1. Code Evolution
The pipeline might have evolved and this is old code that no longer works correctly.

### 2. Duck Typing
Python's duck typing means `sampler` could be a different type than `MultiTargetSampler` that does have this method.

### 3. Missing Implementation
This could be a bug or incomplete implementation.

### 4. Different Sampler Type
The `make_multi_sampler` function might return a different type of sampler under certain conditions.

## Investigation Results

Looking at `make_multi_sampler`, it always returns a `MultiTargetSampler`:
```python
def make_multi_sampler(target_policies_cfg) -> MultiTargetSampler:
    # ... build runners ...
    return MultiTargetSampler(runners, policy_names, log_ratio_clip)
```

## The Correct Implementation

Based on the available methods, the target policy stage SHOULD be doing:

```python
def _compute_logprobs(self, rows, sampler, num_policies):
    contexts = [row["context"] for row in rows]
    responses = [row["response"] for row in rows]  # ← This is missing!
    
    # Use the actual method that exists
    logprobs_matrix = sampler.logp_matrix(contexts, responses)
    
    # Convert to expected format
    all_logprobs = logprobs_matrix.tolist()
```

## Why This Matters for Arena Experiment

This explains why manual experiments are necessary - the pipeline might have issues with the teacher forcing implementation. Our manual approach is actually more correct:

```python
# Our implementation (correct)
for item in p0_data:
    context = item["context"]
    response = item["response"]  # Use P0's response
    
    for policy_name, runner in target_policies.items():
        logp = runner.log_prob(context, response)  # Teacher forcing
        item[f"logp_{policy_name}"] = logp
```

## Recommendations

1. **File a bug report** - The `target_policy.py` code appears to have a method call that doesn't exist
2. **Use manual implementation** - Our approach is actually more transparent and correct
3. **Check test coverage** - This suggests the pipeline might not have adequate test coverage

## Key Takeaway

This investigation reveals that even well-designed libraries can have implementation gaps. The conceptual architecture is sound, but the actual implementation might have issues. This validates our approach of creating a manual implementation for the Arena experiment.