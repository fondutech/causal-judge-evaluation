# Arena 10K Oracle Experiment

This experiment evaluates CJE using 10,000 prompts from the ChatBot Arena dataset.

## Overview

The experiment has two phases:
1. **Phase 1: Dataset Preparation** - Generate responses and judge scores
2. **Phase 2: CJE Ablations** - Compare estimators and uncertainty methods

## Starting Fresh

After fixing the teacher forcing bug, we're ready to rerun the experiment with clean data.

### Prerequisites

1. Source API keys:
```bash
source /path/to/set_secrets.sh
```

2. Ensure you have the robust teacher forcing implementation:
```python
from cje.utils import RobustTeacherForcing
```

### Phase 1: Dataset Preparation

```bash
cd phase1_dataset_preparation

# Step 1: Prepare base dataset
python 01_prepare_data.py

# Step 2: Generate responses
python 02a_generate_p0_responses.py
python 02b_generate_target_responses.py
python 02c_compute_target_logprobs.py  # Uses RobustTeacherForcing

# Step 3: Generate oracle labels
python 03_generate_oracle_labels.py

# Step 4: Judge scoring
python 04a_deterministic_judge_scores.py
python 04b_uncertainty_judge_scores.py
python 04c_score_targets_deterministic.py
python 04d_score_targets_uncertainty.py

# Step 5: Finalize dataset
python 05_finalize_dataset.py
```

### Phase 2: CJE Analysis

```bash
cd ../phase2_cje_ablations

# Run ablations with different estimators
python run_ablations.py --config configs/ablations/*.yaml
```

## Data Files

- `data/arena_prompts_10k.jsonl` - Original 10K prompts
- `data/target_responses.jsonl` - Target model responses

All other data files will be generated during the pipeline execution.

## Important Notes

- The teacher forcing bug that affected 708 samples has been fixed
- All log probabilities will be computed using `RobustTeacherForcing`
- No fallback values (0.0, -100.0) are used for failures