# LLaMA.cpp Integration Guide

This experiment uses llama.cpp for deterministic, local teacher forcing instead of API calls.

## Key Benefits

1. **Deterministic**: Fixed seed ensures reproducible log probabilities
2. **No API costs**: All computations run locally
3. **GPU acceleration**: Uses Metal on Apple Silicon (~4-5x speedup)
4. **Perfect control**: No API non-determinism issues

## Model Setup

### Download Model
```bash
mkdir -p models
curl -L -o models/Llama-3.2-3B-Instruct-Q6_K.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf"
```

### Model Details
- **Model**: Llama 3.2 3B Instruct Q6_K
- **Size**: ~2.5GB (quantized)
- **Quality**: Q6_K maintains high quality with 6-bit quantization
- **Context**: 2048 tokens (configurable)

## Performance

On M2 Max (tested):
- **GPU**: ~120 tokens/sec generation
- **CPU**: ~25 tokens/sec generation
- **Log prob computation**: ~200-300 samples/min

## Implementation Details

### Teacher Forcing Method
Uses continuation method for reliability:
```python
log_p(response|prompt) = log_p(prompt + response) - log_p(prompt)
```

### Key Fixes Applied
1. **max_tokens=0**: Prevents unwanted generation
2. **Fixed seed**: Set at model construction for determinism
3. **Temperature support**: Properly uses instance temperature
4. **Robust caching**: Avoids redundant computations

## Validation

Expected results:
- `pi_clone` median weight: ~1.0 (validates correctness)
- ESS > 50% for all policies
- No extreme weights for identical policies

## Troubleshooting

### Out of Memory
Reduce context size in config:
```yaml
model:
  n_ctx: 1024  # Reduce from 2048
```

### Slow Performance
Ensure GPU layers are enabled:
```yaml
model:
  n_gpu_layers: -1  # Use all layers on GPU
```

### Non-deterministic Results
Check that seed is properly set in config and not overridden per-call.

## Differences from API Version

1. **Model**: Uses local Llama 3.2 instead of Fireworks models
2. **Policies**: Simulated via prompts/temperature (no actual different models)
3. **Speed**: Slower but deterministic and free
4. **Quality**: May differ slightly from larger API models