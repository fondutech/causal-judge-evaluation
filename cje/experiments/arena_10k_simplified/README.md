# Arena 10K Experiment

CJE pipeline using ChatBot Arena dataset for evaluating LLM policies.

## Quick Start

```bash
# View experiment configuration
poetry run python generate_arena_data.py --show-config

# Generate data with retry logic and progress tracking
poetry run python generate_arena_data.py --n-samples 1000

# Analyze with calibration
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl --oracle-coverage 0.5

# Test the pipeline
poetry run python test_full_pipeline.py --n-samples 10
```

## Pipeline Overview

1. **Data Generation** (`generate_arena_data.py`)
   - Extracts prompts from ChatBot Arena
   - Generates responses from different policies
   - Adds judge scores and oracle labels (with resume capability)
   - Computes log probabilities

2. **Analysis** (`analyze_dataset.py`)
   - Calibrates judge scores to oracle labels
   - Runs CJE estimation (IPS, DR-CPO, MRDR, TMLE)
   - Generates visualizations and diagnostics

## Key Features

### 🔄 Retry Logic & Recovery

**Response Generation:**
- Exponential backoff with jitter
- Smart error classification (retryable vs non-retryable)
- Automatic retry of failed responses
- Resume capability for interrupted runs

**Scoring (Judge/Oracle):**
- Resume from exact interruption point
- Skip already-scored records
- Save progress after each batch (50 scores)
- Clear progress bars with ETA

### 📊 Progress Tracking

All operations show detailed progress:
```
Judge scoring: 100%|████████| 2498/2498 [05:32<00:00, 7.52score/s]
```

With status updates:
```
📊 Status:
  Total records:     5000
  Already scored:    2500  <- Resumes from here!
  Need scoring:      2500
```

### 💾 Incremental Saving

- **Responses**: Save after each batch (default: 20)
- **Scores**: Save after each batch (default: 50)
- **Atomic writes**: Use temp file + rename to prevent corruption
- **Configurable**: Adjust `--batch-size` and `--save-every`

## Common Commands

### Data Generation
```bash
# Standard generation
poetry run python generate_arena_data.py --n-samples 1000

# Resume interrupted run (same command!)
poetry run python generate_arena_data.py --n-samples 1000

# Force regenerate everything
poetry run python generate_arena_data.py --n-samples 1000 --force
```

### Response Recovery
```bash
# Check for failures
grep '"response": null' data/responses/*.jsonl | wc -l

# Regenerate failed responses (just re-run the same command!)
poetry run python pipeline_steps/generate_responses.py \
  --prompts data/prompts.jsonl \
  --output-dir data/responses

# Target specific policies
poetry run python pipeline_steps/generate_responses.py \
  --prompts data/prompts.jsonl \
  --output-dir data/responses \
  --policies clone unhelpful

# Skip failed responses (don't retry)
poetry run python pipeline_steps/generate_responses.py \
  --prompts data/prompts.jsonl \
  --output-dir data/responses \
  --skip-failed
```

### Manual Scoring (if needed)
```bash
# Score with automatic resume
poetry run python pipeline_steps/add_scores_with_resume.py data/responses/base_responses.jsonl --type judge

# Force re-score
poetry run python pipeline_steps/add_scores_with_resume.py data/responses/base_responses.jsonl --type judge --force

# Custom batch size for faster saves
poetry run python pipeline_steps/add_scores_with_resume.py data/responses/base_responses.jsonl --type oracle --batch-size 25
```

### Analysis
```bash
# With specific estimator
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl --estimator mrdr

# Different oracle coverage levels
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl --oracle-coverage 0.25

# Skip visualizations for speed
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl --no-plots
```

## Parameters

### Data Generation
- `--n-samples`: Number of prompts (default: 1000)
- `--max-tokens`: Response length (default: 512)
- `--skip-existing`: Resume from existing files (default: True)
- `--force`: Overwrite everything
- `--show-config`: Display experiment configuration and exit


### Scoring
- `--type`: `judge` or `oracle`
- `--batch-size`: Scores per API call (default: 50)
- `--save-every`: Save frequency (default: 50)
- `--force`: Override existing scores

### Analysis
- `--oracle-coverage`: Fraction for calibration (0.0-1.0)
- `--estimator`: `calibrated-ips`, `dr-cpo`, `mrdr`, `tmle`
- `--no-plots`: Skip visualizations
- `--quiet`: Minimal output

## Configuration

All experiment configuration is centralized in `experiment_config.py`:

### Evaluation Models
- **Judge**: `gpt-4.1-nano-2025-04-14` (~0.5s/call, 13x faster than gpt-5-nano)
- **Oracle**: `gpt-5-2025-08-07` (~2s/call, higher quality)

### Policies
- **base**: Standard assistant (Llama 70B)
- **clone**: Identical to base (control)
- **parallel_universe_prompt**: Alternative prompting style
- **unhelpful**: Adversarial responses
- **premium**: High-quality (Llama 405B)

### Batch Sizes (optimized for resilience)
- Response generation: 20 (saves every ~30-60s)
- Judge scoring: 50 (saves every ~25s)
- Oracle scoring: 50 (saves every ~100s)

## Error Handling

### Retryable Errors (automatic retry)
- HTTP 429 (Rate Limit)
- HTTP 500, 502, 503, 530 (Server Errors)
- Connection errors
- Timeouts

### Non-Retryable Errors (fail immediately)
- HTTP 400 (Bad Request)
- HTTP 401 (Unauthorized)
- HTTP 403 (Forbidden)
- HTTP 404 (Not Found)

## Reward Flow

When running analysis, rewards are assigned based on data availability:

1. **Pre-computed rewards** → Use as-is
2. **100% oracle coverage** → Use oracle labels directly (no calibration)
3. **<100% oracle coverage** → Calibrate judge scores to partial oracle

⚠️ **Important**: With 100% oracle coverage, oracle labels are used directly without calibration to preserve all information.

## Troubleshooting

**High failure rates?**
- Check API quotas and status
- Use smaller batch sizes
- Increase retry parameters
- Try different times (API load varies)

**Interrupted run?**
- Just re-run the same command - all scripts auto-resume
- Check progress with `--dry-run` flags
- No work is lost with incremental saves

**Memory issues?**
- Use `--no-plots` to skip visualizations
- Process smaller batches
- Reduce `--n-samples`

## Files Generated

```
data/
├── prompts.jsonl                 # Extracted prompts
├── responses/
│   ├── base_responses.jsonl      # Policy responses with scores
│   ├── clone_responses.jsonl
│   ├── parallel_universe_prompt_responses.jsonl
│   ├── unhelpful_responses.jsonl
│   └── premium_responses.jsonl
├── logprobs/
│   └── *_logprobs.jsonl          # Log probabilities
├── cje_dataset.jsonl             # Combined dataset for analysis
└── plots/                        # Visualization outputs
    ├── weight_dashboard.png
    ├── calibration_comparison.png
    ├── policy_estimates.png
    └── extreme_weights_analysis.txt
```

## Advanced Usage

### Custom Oracle Coverage Experiments
```bash
for coverage in 0.1 0.25 0.5 0.75 1.0; do
    poetry run python analyze_dataset.py \
      --data data/cje_dataset.jsonl \
      --oracle-coverage $coverage \
      --estimator dr-cpo \
      --plot-dir data/plots/coverage_${coverage}
done
```

### Debugging Failed Responses
```python
import json
from collections import Counter

# Check error distribution
errors = []
with open('data/responses/clone_responses.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if data.get('response') is None:
            errors.append(data.get('error_type', 'unknown'))

print(Counter(errors))
```

## Testing

```bash
# Test full pipeline with small sample
poetry run python test_full_pipeline.py --n-samples 10

# Test resume functionality
poetry run python test_resume_pipeline.py
```

## Implementation Notes

### Modular Architecture
The analysis pipeline follows a clean modular structure:
- **Data Generation** (`data_generation/`): Scripts for generating responses, scores, and log probabilities
- **Analysis Modules** (`analysis/`): 7 focused modules doing one thing well
  - `loading.py`: Load and validate data
  - `calibration.py`: Handle rewards and calibration
  - `estimation.py`: Create and configure estimators
  - `results.py`: Display results and statistics
  - `diagnostics.py`: Display weight and DR diagnostics
  - `visualization.py`: Generate plots and dashboards
  - `export.py`: Export results to JSON/CSV
- **Orchestrator** (`analyze_dataset.py`): Thin coordinator (290 lines) that orchestrates modules
- All scripts support resume from interruption
- Production pipeline (`generate_arena_data.py`) is the single entry point
- Tests verify the actual production code path

### MRDR Omega Weights
The MRDR estimator now uses `omega_mode="w"` by default (changed from "snips") for better stability:
- Avoids extreme weight concentration
- Achieves positive R² values
- Lower RMSE in outcome predictions
- See [`cje/estimators/MRDR_OMEGA_WEIGHTS.md`](../../estimators/MRDR_OMEGA_WEIGHTS.md) for details

### Save Strategy
Default configuration saves after each batch:
- **Responses**: Every 20 items (~30-60 seconds of work)
- **Scores**: Every 50 items (~10 seconds of work)
- **Maximum loss**: One batch worth of work
- **Trade-off**: Balances safety vs I/O overhead

## Further Reading

- [Main CJE Documentation](../../README.md) - Core library details
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines