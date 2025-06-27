# CJE Pipeline Architecture: Deep Dive

## Overview

The CJE (Causal Judge Evaluation) pipeline is a sophisticated system for off-policy evaluation using importance weighting. Based on our debugging of the Arena 10K experiment, here's how the library's implementation fits together.

## Core Pipeline Flow

```
1. Dataset Loading → 2. Logging Policy → 3. Judge Scoring → 4. Oracle Labels (optional) 
→ 5. Calibration → 6. Target Policy Computation → 7. Estimation
```

## Key Architectural Components

### 1. Pipeline Coordinator (`coordinator.py`)

The `CJEPipeline` class orchestrates all stages:

```python
class CJEPipeline:
    def run(self):
        # 1. Load dataset
        rows = self._run_stage("dataset", self._load_dataset)
        
        # 2. Generate logging policy responses
        rows = self._run_stage("logging_policy", self._generate_logging_responses, rows)
        
        # 3. Score with judge
        rows = self._run_stage("judge", self._score_with_judge, rows, contexts_hash)
        
        # 4. Oracle labels (optional)
        if self.config.oracle_config:
            rows = self._run_stage("oracle", self._generate_oracle_labels, rows)
        
        # 5. Calibrate scores
        rows = self._run_stage("calibration", self._calibrate_scores, rows)
        
        # 6. Compute target policy log probabilities ← CRITICAL STEP
        rows = self._run_stage("target_policy", self._compute_target_logprobs, rows)
        
        # 7. Run estimators
        results = self._run_stage("estimation", self._run_estimators, rows)
```

### 2. Target Policy Stage (The Critical Missing Step)

This is what was missing in the Arena experiment:

```python
class TargetPolicyStage:
    def run(self, rows, target_policies_config, contexts_hash):
        # Create multi-target sampler
        sampler = make_multi_sampler(target_policies_config)
        
        # Extract contexts from logged data
        contexts = [row["context"] for row in rows]
        
        # THIS IS THE KEY: Teacher force logged responses through target policies
        # Note: It uses the LOGGED responses, not generating new ones
        logprobs_result = sampler.log_prob(contexts)
        
        # Add to each row
        for i, row in enumerate(rows):
            row["logp_target_all"] = logprobs_result[i]  # [logp_π1, logp_π2, ...]
```

### 3. Multi-Target Sampler Architecture

The `MultiTargetSampler` is brilliantly designed for efficiency:

```python
class MultiTargetSampler:
    def __init__(self, runners: List[PolicyRunner]):
        self.runners = runners  # One per target policy
        self.K = len(runners)
    
    def logp_many(self, context: str, response: str) -> List[float]:
        """Return log π^k(response | context) for every target policy"""
        logps = []
        for runner in self.runners:
            # THIS is teacher forcing: score the GIVEN response
            logp = runner.log_prob(context, response)
            logps.append(logp)
        return logps
```

### 4. The Missing Link in Arena Experiment

The Arena experiment did:
1. ✅ Generated P0 responses
2. ✅ Generated target policy responses
3. ✅ Scored all responses
4. ❌ **MISSED**: Computing log P(p0_response | context, target_policy)

Without step 4, the importance weights are:
```python
weight = exp(log_p_target - log_p_behavior)
       = exp(-10.0 - (-10.0))  # Both default to -10.0
       = exp(0)
       = 1.0
```

### 5. Teacher Forcing Implementation

The actual teacher forcing happens in `APIPolicyRunner`:

```python
class APIPolicyRunner:
    def log_prob(self, context: str, response: str) -> float:
        """Teacher force response through policy"""
        # Use completions API with echo=True
        # This scores the EXACT tokens in 'response'
        return self._teacher_forcing_logprob(context, response)
    
    def _teacher_forcing_logprob(self, context, response):
        # Format conversation with response
        full_prompt = self._format_conversation_with_response(messages, response)
        
        # Use completions API to get token-level log probs
        completion = self.completions_client.create(
            prompt=full_prompt,
            max_tokens=0,  # Don't generate, just score
            echo=True,     # Return log probs for input
            logprobs=0     # Get top token log prob
        )
        
        # Sum log probs for response tokens
        return sum(token_logprobs)
```

## Key Design Insights

### 1. Separation of Concerns
- **Logging Policy Stage**: Generates responses and their log probs
- **Target Policy Stage**: Only computes log probs, doesn't generate
- **Judge Stage**: Scores all responses uniformly

### 2. Efficient Batching
The pipeline processes all contexts through all target policies in one pass:
```python
# Not this (inefficient):
for context in contexts:
    for policy in policies:
        logp = policy.log_prob(context, response)

# But this (efficient):
logprobs_matrix = sampler.logp_matrix(contexts, responses)  # Shape: (n, K)
```

### 3. Caching Strategy
Each stage has content-based caching:
```python
contexts_hash = compute_contexts_hash(rows)
if chunk_exists(work_dir, "target_logprobs", contexts_hash):
    return load_chunk(...)
```

### 4. Numerical Stability
Multiple layers of protection:
1. **Hard clipping**: `log_ratio_clip = 20.0`
2. **Soft stabilization**: Subtract 75th percentile
3. **Type safety**: Explicit float64 casting

### 5. Validation at Every Step
```python
validate_target_policy_computation(rows)
# Checks:
# - All rows have logp_target_all
# - Correct number of policies
# - No infinite/NaN values
```

## Why the Pipeline Design Matters

### 1. Correctness
The pipeline ensures teacher forcing happens correctly by design. You can't skip it.

### 2. Efficiency
- Batched API calls
- Caching prevents redundant computation
- Vectorized operations throughout

### 3. Flexibility
- Easy to add new estimators
- Swappable judge implementations
- Multiple uncertainty quantification methods

### 4. Robustness
- Graceful error handling
- Checkpoint recovery
- Validation prevents silent failures

## Lessons for Manual Experiments

When bypassing the pipeline:

1. **Follow the exact stage sequence** - order matters!
2. **Don't skip teacher forcing** - it's not optional
3. **Use the same data structures** - rows with specific fields
4. **Implement proper validation** - check importance weights
5. **Handle errors gracefully** - use safe_call pattern

## The Beauty of the Design

The CJE pipeline elegantly solves a complex problem:
- **Input**: Contexts + one set of logged responses
- **Output**: Importance-weighted estimates for multiple target policies
- **Key Innovation**: Efficient teacher forcing through MultiTargetSampler

The modular design allows researchers to:
- Swap components (different judges, policies, estimators)
- Add new functionality (uncertainty methods, calibration)
- Debug issues (each stage is isolated)
- Scale experiments (caching, batching, parallelization)

This architecture represents thoughtful engineering that makes complex causal inference accessible and reliable.