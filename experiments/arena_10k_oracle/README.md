# Arena 10K Oracle Experiment

This experiment evaluates causal judge estimation methods using 10,000 prompts from ChatBot Arena.

**Important**: We filter for English-only prompts to avoid teacher forcing failures with non-English text.

**Update (2025-01-09)**: Fixed critical token boundary bug that caused extreme importance weights. Phase 1 now uses `force_continuation=True` - ONLY continuation method with no fallback to ensure data integrity.

## Overview

### Phase 1: Dataset Preparation
Generates all data needed for causal judge evaluation:
- Downloads and filters ChatBot Arena prompts (English only)
- P0 (logging policy) responses with log probabilities
- Target policy responses (4 policies including pi_clone)
- Teacher-forced log probabilities for importance weighting
- Oracle labels (optional, requires OpenAI)
- Judge scores with and without uncertainty

### Phase 2: CJE Ablations
Evaluates different estimation methods:
- IPS, SNIPS, DR variations
- With/without calibration
- Deterministic vs uncertainty-aware scoring

## Target Policies

1. **pi_clone**: Identical to P0 (baseline, expects importance weights ≈ 1.0)
2. **pi_cot**: Chain-of-thought prompting
3. **pi_bigger_model**: Larger model (llama4-maverick vs llama4-scout)
4. **pi_bad**: Deliberately poor policy (high temperature, brief responses)

## Quick Start

### Prerequisites
```bash
# Set API keys (required for Fireworks API)
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh

# Or use llama.cpp for local, deterministic computation (see LLAMA_CPP_GUIDE.md)

# Check status
python check_status.py
```

### Test Run (100 prompts)
```bash
cd phase1_dataset_preparation

# Run pipeline with 100 samples (~30 minutes, ~$1)
python run_phase1_pipeline.py 100

# Then run Phase 2 analysis
cd ../phase2_cje_ablations
python run_cje_analysis.py
```

**CRITICAL DATA FORMAT NOTE**: 
Phase 1 outputs must use PrecomputedSampler format:
- `total_logprob` field for base policy (NOT `p0_logp`)
- `target_logps` dict for target policies (NOT individual fields)

If Phase 2 gives "No target_logps found" error, Phase 1 format is wrong.
See phase2_cje_ablations/README.md for details.

### Full Run (10,000 prompts)
```bash
cd phase1_dataset_preparation

# Run pipeline with default 10,000 samples (50-75 hours, ~$60)
python run_phase1_pipeline.py

# Then run Phase 2 analysis
cd ../phase2_cje_ablations
python run_cje_analysis.py
```

## Key Files

### Phase 1 Scripts
- `01_prepare_data.py` - Download and sample Arena prompts
- `02_generate_responses.py` - Generate all responses (P0 + targets)
- `02b_compute_logprobs.py` - Compute log probabilities with extreme weight validation
- `03_judge_scores_deterministic.py` - Score responses deterministically
- `03b_judge_scores_uncertainty.py` - Score responses with uncertainty
- `04_generate_oracle_labels.py` - Generate ground truth labels (optional)
- `05_validate_and_summarize.py` - Validate all data and create summary

### Phase 2 Scripts
- `run_cje_analysis.py` - Main analysis script with IPS/SNIPS and weight diagnostics

## Monitoring Progress

```bash
# From arena_10k_oracle directory
python check_status.py

# Or check specific file
wc -l phase1_dataset_preparation/data/all_responses.jsonl
```

## If Interrupted

The pipeline automatically resumes from the last checkpoint:
```bash
# Just run the same command again - it will continue where it left off
python run_phase1_pipeline.py
```

## Starting Fresh

To completely restart (delete all data):
```bash
cd phase1_dataset_preparation
rm -rf data/ .pipeline_checkpoint.pkl
python run_phase1_pipeline.py
```

## Troubleshooting

- **API errors (403, 520)**: Check API keys are set correctly, wait and retry
- **Missing responses**: Pipeline has retry logic, just run again
- **Extreme weights**: Automatically rejected by validation in 02b_compute_logprobs.py
- `DATA_USAGE_GUIDE.md` - Clear guide on data usage
- `ANALYSIS_SUMMARY.md` - Consolidated findings and recommendations
- `configs/ablations/*.yaml` - Different CJE method configurations

## Critical Notes

⚠️ **Teacher Forcing Bug Fixed**: The robust teacher forcing implementation handles tokenization boundaries correctly, reducing zero log probabilities from 17.7% to <4%.

⚠️ **P0 Log Probabilities**: Script 02b now computes P0 log probs under P0 policy, which is essential for proper importance weighting.

⚠️ **Data Format Critical (Fixed 2025-07-08)**: Phase 1 must produce data in PrecomputedSampler format with `total_logprob` and `target_logps` fields. Previous versions used wrong field names.

⚠️ **Judge Scoring Bug Fixed**: Phase 1 judge scoring scripts were using 0.0 as default for missing log probs (now correctly uses None).

⚠️ **Local Alternative Available**: Use llama.cpp for fully deterministic, offline teacher forcing. See [LLAMA_CPP_GUIDE.md](LLAMA_CPP_GUIDE.md) for setup instructions.

## Cost Estimates

### Test Run (100 prompts)
- API calls: ~1,400
- Cost: ~$1
- Time: ~30 minutes

### Full Run (10,000 prompts)
- API calls: ~140,000
- Cost: ~$60
- Time: 50-75 hours

## Data Flow

```
Prompts → Responses → Log Probs → Judge Scores → CJE Analysis
                                ↓
                          Oracle Labels (optional)
```

## Next Steps

After Phase 1 completes:
1. Review teacher forcing statistics (should have <5% zero log probs)
2. Check importance weight distribution for pi_clone (should be near 1.0)
3. Run Phase 2 CJE ablations using scripts in `phase2_cje_ablations/`