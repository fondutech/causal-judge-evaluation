# Setup Notes for Arena 10K Experiment

## Environment Variables Required

Before running the sample test, ensure these environment variables are set:

```bash
# Fireworks API key (for model inference)
export FIREWORKS_API_KEY="your-fireworks-api-key"

# OpenAI API key (for oracle labeling with GPT-4)
export OPENAI_API_KEY="sk-your-openai-api-key"
```

## Known Issues Fixed

1. **Import Error**: `No module named 'cje.loggers.policy'`
   - This module was removed during cleanup
   - The functionality is now in `cje.loggers.api_policy.py`

2. **Teacher Forcing Bug**: 0.0 log probabilities for non-empty responses
   - Fixed in `cje.utils.RobustTeacherForcing`
   - Uses 3 methods: token_counting, echo_based, continuation

## Quick Test Commands

```bash
# From phase1_dataset_preparation directory:

# Run preflight check
cd sample_run && python preflight_check.py

# Run 1% sample test
./run_sample.sh

# Or use the wrapper from parent directory
cd .. && ./run_sample_test.sh
```