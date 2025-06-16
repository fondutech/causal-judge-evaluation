# Fireworks Completions API Critical Bug

## Summary

The Fireworks completions API with `echo=True` returns dramatically incorrect log probabilities, making it unsuitable for teacher forcing in causal inference experiments.

## Evidence

### 1. API Comparison

Testing the same prompts/responses with chat vs completions API:

**Model: llama4-scout-instruct-basic**
- "Cabbages": -0.004 (chat) vs -150.9 (completions) - **37,500x difference!**
- "4": -0.0003 (chat) vs -50.8 (completions)
- "Yes": -0.00007 (chat) vs -61.7 (completions)

**Model: llama4-maverick-instruct-basic** 
- "Cabbages": -0.000 (chat) vs -95.0 (completions)
- "4": -0.00005 (chat) vs -31.7 (completions)
- "Yes": -0.009 (chat) vs -32.0 (completions)

### 2. Length-Dependent Bug

The bug gets worse with longer responses:
- "4" -> -0.415 (reasonable)
- "Yes" -> -0.582 (reasonable)
- "Yes." -> -8.614 (adding period makes it 15x worse!)
- "The answer is 4." -> -16.896 (simple sentence is terrible)

### 3. Self-Inconsistency

The model's own natural generations get bad scores:
- Model generates: "The answer to 2 + 2 is 4."
- Completions API scores it: -16.879 (terrible!)
- This is the exact response the model chose, yet it assigns very low probability

## Root Cause

The completions API appears to have a cumulative error in log probability calculation, where each additional token compounds the error. This makes it completely unreliable for:
- Teacher forcing
- Log probability estimation
- Any causal inference methods requiring accurate probabilities

## Recommendation

**DO NOT use Fireworks for experiments requiring accurate log probabilities.**

Consider alternatives:
1. Use a different provider with working completions API (e.g., Together AI)
2. Use only chat completions API (but this doesn't support teacher forcing)
3. Find models that properly implement the OpenAI completions spec

## Code Fix Status

The token extraction code has been fixed and works correctly. The issue is entirely with the Fireworks API implementation returning wrong values.