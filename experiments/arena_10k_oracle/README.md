# Arena 10K Oracle Experiment

Comprehensive evaluation of CJE using 10,000 ChatBot Arena prompts with oracle ground truth.

## Overview

This experiment:
1. Uses real user prompts from ChatBot Arena
2. Evaluates multiple policies with different quality levels
3. Compares judge-based evaluation to oracle ground truth
4. Tests uncertainty quantification and various estimators

## Structure

- **Phase 1**: Dataset preparation (see `phase1_dataset_preparation/`)
- **Phase 2**: CJE ablations (see `phase2_cje_ablations/`)

## Key Findings

1. **MTurk Failed**: Human labels were expensive ($440), slow, and unreliable
2. **Oracle Works**: AI judges (GPT-4/Claude) provide consistent ground truth at ~5% of cost
3. **Uncertainty Helps**: Judge uncertainty improves calibration
4. **MRDR Best**: Multi-robust doubly-robust estimator achieves lowest variance

## Running the Experiment

```bash
# Phase 1: Prepare dataset
cd phase1_dataset_preparation
# Follow README.md

# Phase 2: Run ablations
cd ../phase2_cje_ablations
python run_ablations_full.py --yes
```

See individual phase directories for detailed instructions.