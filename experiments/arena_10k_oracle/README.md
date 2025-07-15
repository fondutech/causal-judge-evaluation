# Arena 10K Oracle Experiment

This experiment evaluates causal judge estimation methods using 10,000 prompts from ChatBot Arena with deterministic llama.cpp teacher forcing.

**Important**: We filter for English-only prompts to avoid teacher forcing failures with non-English text.

**Update (2025-01-10)**: Now uses llama.cpp exclusively for deterministic, reproducible teacher forcing. No more API non-determinism issues!

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

With llama.cpp, we simulate different policies using the same model:

1. **pi_clone**: Identical to P0 (baseline, expects importance weights ≈ 1.0)
2. **pi_bad**: Deliberately unhelpful policy (via system prompt)

## Quick Start

### Prerequisites
```bash
# 1. Install llama-cpp-python
pip install llama-cpp-python

# 2. Download the model
mkdir -p models
curl -L -o models/Llama-3.2-3B-Instruct-Q6_K.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf

# 3. Set API keys (still needed for judge/oracle)
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh

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

✅ **Deterministic Teacher Forcing**: Now uses llama.cpp exclusively for 100% reproducible log probabilities. No more API non-determinism!

✅ **Pi_clone Validation**: With deterministic computation, pi_clone weights should be exactly 1.0 (modulo floating point precision).

⚠️ **Data Format**: Phase 1 produces data in PrecomputedSampler format with `total_logprob` and `target_logps` fields.

⚠️ **Model Required**: You must download the Llama 3.2 3B model (~2.5GB) before running. See prerequisites above.

## Performance & Cost

### Test Run (100 prompts)
- Time: ~1-2 hours (depends on GPU)
- Cost: $0 (local computation) + minimal judge/oracle API costs

### Full Run (10,000 prompts)
- Time: ~100-200 hours (depends on GPU)
- Cost: $0 (local computation) + judge/oracle API costs (~$10-20)

### Performance Tips
- Use GPU acceleration (Metal on Mac, CUDA on Linux)
- Ensure `n_gpu_layers: -1` in config for full GPU usage
- ~120 tokens/sec on M2 Max, ~200+ on RTX 4090

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