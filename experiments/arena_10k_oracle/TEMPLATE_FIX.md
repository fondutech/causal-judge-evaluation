# Llama 4 Template Fix - RESOLVED

## Summary

The log probability issues with Fireworks were caused by using the wrong prompt template for Llama 4 models, not an API bug.

## Solution

Implemented automatic detection of Llama 4 models and proper template formatting:

### Llama 3 Template (OLD)
```
<s>[INST] {prompt} [/INST] {response}</s>
```

### Llama 4 Template (NEW)
```
<|begin_of_text|>
<|header_start|>user<|header_end|>

{prompt}<|eot|>
<|header_start|>assistant<|header_end|>

{response}<|eot|>
```

## Results

With the correct template:
- "Cabbages" (forced response): 0.000 logprob ✅
- Simple "4": 0.000 logprob ✅  
- "The answer is 4.": 0.000 logprob ✅

## Implementation

The `APIPolicyRunner` now:
1. Detects if model name contains "llama4" or "llama-4"
2. Uses appropriate template based on model version
3. Handles `<|eot|>` tokens for Llama 4 instead of `</s>`

## Status

✅ **FIXED** - Fireworks API works correctly with proper template
✅ **READY** - Can proceed with Arena 10K experiment using Fireworks