# Arena 10K Oracle Experiment

This experiment evaluates causal judge estimation methods using 10,000 prompts from ChatBot Arena.

**Important**: We filter for English-only prompts to avoid teacher forcing failures with non-English text.

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
# Set API keys (required)
export FIREWORKS_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # Optional, only for oracle labels

# Or source from AWS Secrets Manager
source ../../../set_secrets.sh
```

### Test Run (100 prompts)
```bash
cd phase1_dataset_preparation

# Run pipeline with 100 samples (~30 minutes, ~$1)
python run_phase1_pipeline.py 100

# Then run Phase 2 analysis
cd ../phase2_cje_ablations
python run_cje_simple.py
python analyze_weights.py
```

### Full Run (10,000 prompts)
```bash
cd phase1_dataset_preparation

# Run pipeline with default 10,000 samples (50-75 hours, ~$60)
python run_phase1_pipeline.py

# Then run Phase 2 analysis
cd ../phase2_cje_ablations
python run_cje_simple.py
python analyze_weights.py
```

## Key Files

### Phase 1 Scripts
- `01_prepare_data.py` - Download and sample Arena prompts
- `02_generate_responses.py` - Generate all responses (P0 + targets)
- `02b_compute_logprobs.py` - Compute log probabilities for all policies
- `03_judge_scores_deterministic.py` - Score responses deterministically
- `03b_judge_scores_uncertainty.py` - Score responses with uncertainty
- `04_generate_oracle_labels.py` - Generate ground truth labels (optional)
- `05_validate_and_summarize.py` - Validate all data and create summary

### Phase 2 Scripts
- `run_cje_simple.py` - Simple IPS/SNIPS implementation (working)
- `analyze_weights.py` - Comprehensive weight diagnostics
- `run_ablations.py` - Run all estimators (needs fixes)
- `configs/ablations/*.yaml` - Different CJE method configurations

## Critical Notes

⚠️ **Teacher Forcing Bug Fixed**: The robust teacher forcing implementation handles tokenization boundaries correctly, reducing zero log probabilities from 17.7% to <4%.

⚠️ **P0 Log Probabilities**: Script 02b now computes P0 log probs under P0 policy, which is essential for proper importance weighting.

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