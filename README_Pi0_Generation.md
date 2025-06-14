# π₀ Data Generation

**Robust implementation of analysis plan steps 1-3:**
1. Download ChatBot Arena corpus
2. Generate π₀ answers (Llama-3-8B)  
3. Teacher-forced scoring (propensities)

## Features

✅ **Progress Tracking**: Real-time progress bar with ETA estimates using shared CJE utilities  
✅ **Checkpointing**: Resume interrupted jobs from where you left off with reusable utilities  
✅ **Batch Processing**: Configurable batch sizes for efficiency  
✅ **Error Recovery**: Continue processing even if some batches fail  
✅ **Cost Estimation**: Track approximate API costs  
✅ **Shared Infrastructure**: Built on top of `cje.utils.progress` and `cje.utils.checkpointing`  

## Usage

### Basic Usage
```bash
export FIREWORKS_API_KEY="your-key"
python scripts/generate_pi0_data.py --samples 1000
```

### Large Scale with Checkpointing (Recommended)
```bash
# Auto-enables checkpointing for runs >= 100 samples
python scripts/generate_pi0_data.py --samples 10000

# Or specify custom checkpoint file
python scripts/generate_pi0_data.py --samples 10000 --checkpoint my_checkpoint.jsonl
```

### Resume Interrupted Job
```bash
# Just run the same command - it will automatically detect and resume
python scripts/generate_pi0_data.py --samples 10000 --checkpoint my_checkpoint.jsonl
```

### Advanced Options
```bash
# Custom model, batch size, and parameters
python scripts/generate_pi0_data.py \
    --samples 5000 \
    --model accounts/fireworks/models/llama-v3p1-70b-instruct \
    --batch-size 32 \
    --temperature 0.7 \
    --max-tokens 512

# Auto-cleanup checkpoint on successful completion
python scripts/generate_pi0_data.py --samples 1000 --cleanup-checkpoint
```

### AWS Secrets Manager
```bash
# Create secret with multiple API keys
aws secretsmanager create-secret \
    --name "cje/prod/api-keys" \
    --description "Production API keys for CJE" \
    --secret-string '{
        "FIREWORKS_API_KEY": "your-fireworks-key",
        "OPENAI_API_KEY": "your-openai-key",
        "ANTHROPIC_API_KEY": "your-anthropic-key"
    }'

# Run script (automatically retrieves from AWS)
python scripts/generate_pi0_data.py --samples 1000
```

**The script automatically tries environment variable first, then AWS Secrets Manager.**

## Command Line Options

```bash
python scripts/generate_pi0_data.py --help
```

```
usage: generate_pi0_data.py [-h] [--samples SAMPLES] [--output OUTPUT] [--checkpoint CHECKPOINT]
                           [--batch-size BATCH_SIZE] [--model MODEL] [--temperature TEMPERATURE]
                           [--max-tokens MAX_TOKENS]

Generate π₀ data with progress tracking and checkpointing

optional arguments:
  --samples SAMPLES      Number of samples to generate (default: 1000)
  --output OUTPUT        Output file path (default: pi0_data.jsonl)
  --checkpoint CHECKPOINT
                        Checkpoint file for resumable generation (recommended)
  --cleanup-checkpoint  Automatically delete checkpoint file on successful completion
  --batch-size BATCH_SIZE
                        Batch size for API calls (default: 16)
  --model MODEL         Model name for generation
  --temperature TEMPERATURE
                        Sampling temperature (default: 0.4)
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 1024)
```

## Test Run

```bash
# Test with 2 samples
python scripts/generate_pi0_data.py --samples 2
```

**Expected behavior:**
- ✅ **API Key**: Retrieved from environment or AWS Secrets Manager
- ⚠️ **Dataset**: May require HuggingFace authentication (falls back to mock data)
- ✅ **Generation**: Produces responses + exact logprobs with progress tracking
- ✅ **Checkpointing**: Auto-saves progress after each batch
- ✅ **Fallback**: Graceful error handling if API issues occur

## Progress Tracking Output

The script shows detailed progress information:

```
🔬 Generating π₀ responses for 1,000 contexts
📊 Model: accounts/fireworks/models/llama4-scout-instruct-basic
🌡️ Temperature: 0.4, Max tokens: 1024
📦 Batch size: 16

Generating responses: 45%|████▌     | 450/1000 [02:15<02:45, 3.32samples/s]
                     batch_time: 4.8s, avg_per_sample: 0.30s, ETA: 2.8m

✅ Generated 1,000 total responses
⏱️ Total time: 5.2 minutes
📈 Average: 0.31 seconds per sample
💰 Estimated cost: ~$0.40
```

## Interruption and Recovery

If interrupted (Ctrl+C), the script saves progress:

```
⚠️ Interrupted after processing 450 samples
💾 Progress saved to checkpoint: pi0_data_checkpoint.jsonl
💾 Resume with: python scripts/generate_pi0_data.py --samples 1000 --checkpoint pi0_data_checkpoint.jsonl
```

When resumed, it continues from where it left off:

```
🔄 Loaded checkpoint with 450 existing responses
🔬 Generating π₀ responses for 550 contexts
🔄 Resuming from 450 already processed
```

## Output

**File**: `pi0_data.jsonl`

```json
{
  "uid": "arena_12345",
  "context": "User prompt from ChatBot Arena",
  "response": "Generated Llama-3-8B response",
  "logp": -15.42,
  "action": "accounts/fireworks/models/llama-v3-8b-instruct"
}
```

## Cost

| Samples | Cost |
|---------|------|
| 1k | ~$0.40 |
| 10k | ~$4.00 |

## Inspect

```bash
# View sample
head -n 1 pi0_data.jsonl | jq .

# Check logprobs  
jq '.logp' pi0_data.jsonl | head -5

# Load in Python
python -c "
import json
data = [json.loads(l) for l in open('pi0_data.jsonl')]
print(f'{len(data)} samples, avg logp: {sum(d[\"logp\"] for d in data)/len(data):.2f}')
"
```

## Reusable Utilities

This script demonstrates the use of shared CJE utilities that can be used in other parts of the codebase:

### Progress Tracking (`cje.utils.progress`)
```python
from cje.utils.progress import console, track

# Rich console with styling
console.print("✅ [green]Success message[/green]")
console.print("⚠️ [yellow]Warning message[/yellow]")

# Progress tracking for iterables
for item in track(items, description="Processing"):
    # ... process item
```

### Checkpointing (`cje.utils.checkpointing`)
```python
from cje.utils.checkpointing import create_jsonl_checkpoint_manager, BatchProcessor

# Create checkpoint manager for JSONL data
checkpoint_manager = create_jsonl_checkpoint_manager("my_checkpoint.jsonl")

# Create batch processor with checkpointing
processor = BatchProcessor(
    batch_size=16,
    checkpoint_manager=checkpoint_manager,
    continue_on_error=True,
)

# Process with automatic checkpointing and progress tracking
results = processor.process_batches(
    items,
    process_function,
    description="Processing items",
)
```

### Auto-Checkpointing
```python
from cje.utils.checkpointing import auto_enable_checkpointing

# Automatically enable checkpointing for large operations
checkpoint_path = auto_enable_checkpointing("output.jsonl", explicit_checkpoint)

# Clean up checkpoint with optional auto-deletion
cleanup_checkpoint_file(checkpoint_path, auto_cleanup=True)
```

## Troubleshooting

**403 Unauthorized Error:**
- Verify Fireworks API key is valid and complete (40+ characters)
- Check key has access to `llama-v3-8b-instruct` model
- Ensure sufficient account credits

**Dataset 401 Error:**
- ChatBot Arena dataset requires HuggingFace authentication
- Script will use mock data as fallback
- For real data: `huggingface-cli login`

**AWS Secrets Manager:**
- Ensure AWS credentials are configured (`aws configure`)
- Verify secret exists: `aws secretsmanager get-secret-value --secret-id "cje/prod/api-keys"`
- Check IAM permissions for `secretsmanager:GetSecretValue`

**Progress Disabled:**
- Set `CJE_DISABLE_PROGRESS=1` to disable progress bars in CI/CD environments
- All utilities respect this environment variable

**Checkpoint Cleanup:**
- By default, checkpoint files are preserved for safety and debugging
- Use `--cleanup-checkpoint` flag for automatic deletion on success
- Checkpoints are small (same size as output) so cleanup is optional
- Recommended for automated pipelines to avoid file accumulation

**Result**: Clean π₀ dataset with exact propensities, ready for custom analysis. 