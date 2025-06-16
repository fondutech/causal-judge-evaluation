# Arena 10K Human Oracle Experiment

## Overview

This experiment validates Causal Judge Evaluation (CJE) on real ChatBot Arena prompts with **human oracle labels** (crowdsourced, not AI-generated). The experiment has been implemented with improved infrastructure for robustness and monitoring.

**Current Status**: Data generation complete (72 samples for testing), ready for human labeling phase.

## Experiment Design

### Dataset
- **Source**: 10,000 single-turn prompts from ChatBot Arena Conversations
- **Current Progress**: 72 samples generated for initial testing
- **Split**: 25% calibration (18 samples), 75% evaluation (54 samples)

### Policies

| Policy | Model | Temperature | Description |
|--------|-------|-------------|-------------|
| π₀ (logging) | llama4-scout-instruct-basic | T=0.5 | baseline logging policy |
| π_hot | llama4-scout-instruct-basic | T=0.9 | higher temperature (creative) |
| π_cot | llama4-scout-instruct-basic | T=0.5 | chain-of-thought prompting |
| π_concise | llama4-scout-instruct-basic | T=0.3 | concise responses (2-3 sentences) |

### Judge Configuration
- **Model**: llama4-scout-instruct-basic at T=0
- **Scoring**: 0-1 scale for helpfulness/correctness/safety
- **Calibration**: Will use isotonic regression to human labels

## Repository Structure

```
experiments/arena_10k_oracle/
├── README.md                        # This file
├── .gitignore                       # Excludes data files from git
├── configs/
│   └── arena_experiment.yaml        # Main experiment configuration
├── scripts/
│   ├── 01_prepare_data.py          # Sample prompts from Arena dataset
│   ├── 02_generate_logs.py         # Generate π₀ responses (improved)
│   ├── 03_add_judge_scores.py      # Score with LLM judge
│   ├── 04_generate_target_policies.py  # Generate target policy responses
│   ├── 05_export_for_labeling.py   # Export for crowdsourcing
│   ├── 06_import_labels.py         # Import human labels (TBD)
│   ├── check_fireworks_models.py   # Utility to verify model access
│   └── experiment_status.py        # Monitor pipeline progress
├── data/
│   ├── prompts.jsonl               # 10k sampled prompts
│   ├── p0_replies.jsonl            # π₀ responses with logprobs
│   ├── p0_scored.jsonl             # π₀ with judge scores
│   ├── all_policies.jsonl          # All policy responses (cleaned)
│   └── labeling/
│       ├── calibration_export_surge.csv    # Ready for human labeling
│       └── p0_scored_with_splits.jsonl     # Full data with splits
└── outputs/                        # (created during CJE estimation)
```

## Implementation Improvements

### 1. Atomic Checkpointing
- `AtomicCheckpointManager` prevents duplicate entries
- Tracks processed items by ID
- Uses temp file + atomic rename pattern

### 2. Per-Sample Progress Tracking
- `CheckpointManager` tracks completion at policy AND sample level
- Can resume from exact position if interrupted
- No lost work on failures

### 3. Better Error Handling
- Automatic retry with exponential backoff
- Continues with next batch on failure
- Clear error messages and progress indicators

### 4. Data Cleanup
- Removed unnecessary metadata (14% size reduction)
- Keep only fields needed for CJE:
  - `prompt_id`, `prompt`
  - `response`, `total_logprob` (π₀)
  - `judge_score_raw`
  - `pi_*_response` fields (for oracle labeling)

## Running the Experiment

### Prerequisites

```bash
# Set API key for Fireworks
export FIREWORKS_API_KEY="your-key"

# Verify model access
cd scripts
python check_fireworks_models.py
```

### Check Current Status

```bash
# See what's been completed
python experiment_status.py
```

### Phase 1: Data Generation ✅ (COMPLETE)

```bash
# Step 1: Sample prompts (10k sampled, 72 used for test)
python 01_prepare_data.py --samples 10000 --output-samples 72

# Step 2: Generate π₀ responses with teacher-forcing logprobs
python 02_generate_logs.py

# Step 3: Add judge scores
python 03_add_judge_scores.py

# Step 4: Generate target policy responses
python 04_generate_target_policies.py

# Step 5: Export for labeling
python 05_export_for_labeling.py
```

### Phase 2: Human Labeling (NEXT)

```bash
# Current status:
# - 18 samples ready for calibration labeling
# - Export file: data/labeling/calibration_export_surge.csv
# - Cost: ~$4.32 (54 labels needed)

# Upload to Surge AI and collect 3 votes per sample
# Then import with:
python 06_import_labels.py --labels path/to/downloaded_labels.csv
```

### Phase 3: CJE Estimation (TODO)

After human labels are collected:
1. Run isotonic calibration
2. Compute importance weights
3. Execute DR-CPO estimation
4. Generate results and diagnostics

## Key Technical Details

### Two-Pass Generation (Critical!)
The pipeline uses `generate_with_consistent_logp` which implements:
1. **Generation pass**: Natural response generation
2. **Teacher forcing pass**: Score the generated text with completions API

This ensures π₀ and target policies use identical scoring methods for valid importance weights.

### Importance Weights
CJE computes: `w = π'(π₀_response|prompt) / π₀(π₀_response|prompt)`

Note: The target policy generation logprobs (`pi_hot_logprob`, etc.) are NOT used - they're artifacts from generation. CJE will score π₀ responses under each target policy.

## Monitoring & Diagnostics

### Weight Quality Metrics
- Effective Sample Size (ESS)
- Clipped weight mass
- Weight distribution plots

### Calibration Quality
- Isotonic fit diagnostics
- Cross-validation stability
- Mean absolute calibration error

## Cost Summary
- π₀ generation: ~$0.03 (72 samples)
- Judge scoring: ~$0.01 (72 samples)
- Target policies: ~$0.09 (72 samples × 3 policies)
- Human labeling: ~$4.32 (54 labels)
- **Total: ~$4.45**

## Troubleshooting

### Common Issues

1. **Timeout errors during generation**
   - Scripts automatically retry with exponential backoff
   - Can resume from checkpoint if interrupted

2. **Duplicate entries in checkpoint**
   - Fixed with atomic checkpointing
   - Old duplicates cleaned automatically

3. **Memory issues with large batches**
   - Reduce `--batch-size` parameter
   - Default is 16 for π₀, 4 for target policies

## Next Steps

1. Upload `calibration_export_surge.csv` to crowdsourcing platform
2. Collect 54 human labels (18 samples × 3 votes)
3. Import labels and run calibration
4. Execute full CJE pipeline
5. Compare estimates to ground truth

## Contact

For questions about this experiment:
- GitHub Issues: https://github.com/anthropics/causal-judge-evaluation/issues
- Documentation: https://cje.readthedocs.io