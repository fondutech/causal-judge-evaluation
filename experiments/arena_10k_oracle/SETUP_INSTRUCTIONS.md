# Setup Instructions for Arena 10K with llama.cpp

## Prerequisites

You need to complete these steps before running the experiment:

### 1. Download the Model (~2.5GB)
```bash
cd /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/experiments/arena_10k_oracle
./download_model.sh
```

This will download the Llama 3.2 3B Instruct model. It's about 2.5GB, so it may take a few minutes depending on your connection.

### 2. Set API Keys
```bash
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh
```

This sets the OPENAI_API_KEY needed for judge scoring and oracle labels.

### 3. Verify Setup
```bash
python setup_status.py
```

You should see all green checkmarks.

## Running the Experiment

Once setup is complete:

```bash
cd phase1_dataset_preparation
python run_phase1_pipeline.py 10  # Run with 10 samples
```

This will:
1. Download 10 prompts from Arena
2. Generate responses using llama.cpp (deterministic!)
3. Compute log probabilities for all policies
4. Score with GPT-4o-mini judge
5. Generate oracle labels for calibration
6. Create summary statistics

Expected time: ~30-60 minutes for 10 samples (depends on GPU)

## What to Expect

- Pi_clone weights should be exactly 1.0 (validates deterministic computation)
- No extreme weights warnings (unless there's an issue)
- All data saved in `phase1_dataset_preparation/data/` and `data/`
- Ready for Phase 2 analysis when complete