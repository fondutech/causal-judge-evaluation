# Judge Uncertainty Methods - Design Clarification

## Current State (Confusing)

The codebase claims to support three uncertainty methods:
1. **deterministic** - Always returns variance=0
2. **structured** - "Model estimates its own uncertainty" (but doesn't actually)
3. **monte_carlo** - Multiple samples with temperature > 0

## The Problem

The "structured" method is poorly defined:
- The default APIJudge doesn't actually ask models for uncertainty
- It just returns whatever the model gives (usually variance=0 with structured output)
- There's no clear implementation of asking for confidence/uncertainty

## Proposed Simplification

### 1. **Deterministic Judge** (variance = 0)
- Single API call, temperature = 0
- Uses simple prompt asking for a score
- Always returns JudgeScore(mean=score/10, variance=0)
- Use case: Fast, cheap scoring without uncertainty

### 2. **Confidence Interval Judge** (variance from CI)
- Single API call, temperature = 0
- Explicitly asks for score + 95% CI in the prompt
- Calculates variance from CI width: var = (range/3.92)²/100
- Use case: Principled uncertainty from judge's confidence interval

### 3. **Monte Carlo Judge** (variance from sampling)
- Multiple API calls (default 5), temperature > 0
- Same prompt each time, variation comes from sampling
- Calculates variance from score distribution
- Use case: Empirical uncertainty estimation

## Implementation Status

✅ **Deterministic**: Working (DeterministicAPIJudge)
✅ **Monte Carlo**: Working (MCAPIJudge)  
❌ **Confidence Interval**: Not properly implemented

The "structured" uncertainty in the codebase doesn't match any of these - it's just the base APIJudge that doesn't do anything special for uncertainty.

## Recommendation

1. Remove the confusing "structured" uncertainty method
2. Replace with explicit "confidence_interval" method
3. Update templates to support CI-based scoring
4. Make the uncertainty method selection clearer in the API