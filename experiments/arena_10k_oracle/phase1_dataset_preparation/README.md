# Phase 1: Dataset Preparation

Creates complete dataset for evaluating CJE on ChatBot Arena prompts.

## Prerequisites
```bash
export FIREWORKS_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # For oracle labeling
```

## Quick Start
Run scripts sequentially:

```bash
# 1. Download prompts
python 01_prepare_data.py

# 2. Generate responses
python 02a_generate_p0_responses.py
python 02b_generate_target_responses.py  # Can run in parallel with 02a

# 2c. Compute target log probabilities (CRITICAL for importance weighting!)
python 02c_compute_target_logprobs.py

# 3. Generate oracle labels
python 03_generate_oracle_labels.py

# 4. Add judge scores (can run all 4 in parallel)
python 04a_deterministic_judge_scores.py &
python 04b_uncertainty_judge_scores.py &
python 04c_score_targets_deterministic.py &
python 04d_score_targets_uncertainty.py &

# 5. (Optional) Create summary statistics
python 05_finalize_dataset.py
```

## Output Structure
```
data/
├── arena_prompts_10k.jsonl              # Source prompts
├── p0_replies.jsonl                     # π₀ responses with log P(response|prompt,π₀)
├── target_responses.jsonl               # Target policy responses
├── p0_with_target_logps.jsonl           # π₀ responses with log P(response|prompt,πₖ)
├── p0_scored_*.jsonl                    # Scored π₀ (deterministic/uncertainty)
├── targets_scored_*.jsonl               # Scored targets
└── labeling/
    └── oracle_labels_*.jsonl            # Oracle ground truth
```

All scripts support automatic checkpointing and can resume if interrupted.