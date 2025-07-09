# Using llama.cpp for Arena 10K Teacher Forcing

This guide explains how to use llama.cpp as a local, deterministic alternative to the Fireworks API for teacher forcing in the Arena 10K experiment.

## Why Use llama.cpp?

**Advantages:**
- **Fully deterministic**: With a fixed seed, results are 100% reproducible
- **No token boundary issues**: The continuation method works reliably
- **No API costs**: Run unlimited experiments locally
- **No rate limits**: Process as fast as your hardware allows
- **Works offline**: No internet connection required
- **Supports quantized models**: Use Q4_K_M, Q5_K_M etc. for efficiency

**Trade-offs:**
- Requires local GPU/CPU resources
- Need to download GGUF model files
- May be slower than API calls (depending on hardware)

## Installation

```bash
# Install llama-cpp-python with GPU support (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For Apple Silicon (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# CPU only
pip install llama-cpp-python
```

## Getting GGUF Models

Download quantized models from HuggingFace:

```bash
# Example: Llama 3 8B Instruct (Q4_K_M quantization)
wget https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf

# Example: Llama 3 70B Instruct (Q4_K_M quantization)
wget https://huggingface.co/TheBloke/Llama-3-70B-Instruct-GGUF/resolve/main/llama-3-70b-instruct.Q4_K_M.gguf
```

## Integration with Arena 10K

### 1. Update Configuration

Edit `experiments/arena_10k_oracle/arena_config.yaml`:

```yaml
# Example configuration for llama.cpp
policies:
  p0:
    provider: "llama_cpp"
    model: "~/models/llama-3-8b-instruct.Q4_K_M.gguf"
    temperature: 0.5
    seed: 42
    n_ctx: 4096
    n_gpu_layers: -1  # Use all GPU layers
    
  pi_clone:
    provider: "llama_cpp"
    model: "~/models/llama-3-8b-instruct.Q4_K_M.gguf"  # Same as p0
    temperature: 0.5
    seed: 42
    n_ctx: 4096
    n_gpu_layers: -1
    
  pi_bigger_model:
    provider: "llama_cpp"
    model: "~/models/llama-3-70b-instruct.Q4_K_M.gguf"  # Larger model
    temperature: 0.5
    seed: 42
    n_ctx: 4096
    n_gpu_layers: 35  # Partial offloading for large models
    
  pi_bad:
    provider: "llama_cpp"
    model: "~/models/llama-3-8b-instruct.Q4_K_M.gguf"
    temperature: 1.0  # Higher temperature
    seed: 42
    n_ctx: 4096
    n_gpu_layers: -1
    system_prompt: "You are an unhelpful assistant."
```

### 2. Run Phase 1 with llama.cpp

The existing scripts work seamlessly with llama.cpp:

```bash
cd experiments/arena_10k_oracle/phase1_dataset_preparation

# Step 1: Sample prompts (unchanged)
python 01_sample_prompts.py --num-samples 100

# Step 2: Compute log probs with llama.cpp
python 02b_compute_logprobs.py

# Step 3: Judge scoring (unchanged)
python 03_judge_scores_deterministic.py
```

### 3. Verify Determinism

```python
# Test script to verify determinism
from cje.utils import RobustTeacherForcing

# Create two instances with same seed
tf1 = RobustTeacherForcing(
    provider="llama_cpp",
    model="~/models/llama-3-8b-instruct.Q4_K_M.gguf",
    temperature=0.5,
    seed=42,
    n_ctx=4096,
)

tf2 = RobustTeacherForcing(
    provider="llama_cpp",
    model="~/models/llama-3-8b-instruct.Q4_K_M.gguf",
    temperature=0.5,
    seed=42,
    n_ctx=4096,
)

# Should get identical results
prompt = "What is the capital of France?"
response = "The capital of France is Paris."

result1 = tf1.compute_log_prob(prompt, response)
result2 = tf2.compute_log_prob(prompt, response)

assert result1.value == result2.value, "Non-deterministic results!"
print(f"✓ Deterministic: {result1.value:.4f}")
```

## Performance Tips

1. **GPU Offloading**: Use `n_gpu_layers=-1` to offload all layers to GPU
2. **Context Window**: Set `n_ctx` appropriately (4096 or 8192 for most tasks)
3. **Quantization**: Q4_K_M offers good balance of quality and speed
4. **Caching**: Models are cached globally, so multiple policies using the same model share memory
5. **Batch Processing**: The implementation caches results to avoid recomputation

## Example: Mixed Provider Setup

You can mix llama.cpp and API providers:

```yaml
policies:
  p0:
    provider: "fireworks"  # Use API for baseline
    model: "accounts/fireworks/models/llama-v3-8b-instruct"
    temperature: 0.5
    
  pi_clone:
    provider: "llama_cpp"  # Use local for deterministic testing
    model: "~/models/llama-3-8b-instruct.Q4_K_M.gguf"
    temperature: 0.5
    seed: 42
```

## Troubleshooting

1. **Import Error**: Make sure llama-cpp-python is installed
2. **Model Not Found**: Use absolute paths or ~ for home directory
3. **Out of Memory**: Reduce `n_gpu_layers` or use smaller quantization
4. **Slow Performance**: Check GPU is being used (watch nvidia-smi)

## Full Example Script

See `experiments/arena_10k_oracle/example_llama_cpp.py` for a complete working example.