# Arena 10K Quick Start

## Before Starting
```bash
# 1. Set API keys (required)
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh

# 2. Check status
python check_status.py
```

## Run Pipeline

### Test Run (5 samples, ~2 min)
```bash
cd phase1_dataset_preparation
python run_phase1_pipeline.py 5
```

### Full Run (10k samples, ~60 hours)
```bash
cd phase1_dataset_preparation
python run_phase1_pipeline.py
```

## Monitor Progress
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
rm -rf data/ .pipeline_checkpoint.pkl
python run_phase1_pipeline.py
```

## Phase 2 Analysis
```bash
cd phase2_cje_ablations

# Simple analysis
python run_cje_simple.py

# Weight diagnostics  
python analyze_weights.py
```

## Troubleshooting

### API errors (403, 520)
- Check API keys are set correctly
- Wait a few minutes and retry (rate limits)

### Missing responses
- The pipeline has retry logic built in
- Just run the pipeline again

### Zero log probabilities
- Expected to be <5% with current fixes
- Review in Phase 2 weight analysis output