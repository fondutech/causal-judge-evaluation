# Arena 10K Human Oracle Experiment

## Overview

This experiment is designed to validate Causal Judge Evaluation (CJE) on 10,000 real ChatBot Arena prompts with fresh **human oracle labels** (not automated model-based oracle). The experiment collects crowdsourced human judgments to serve as ground truth, demonstrating CJE's accuracy, efficiency, and cost-effectiveness compared to traditional evaluation methods.

.. note::
   **Human Oracle vs Automated Oracle**: This experiment uses crowdsourced human labels as ground truth ("human oracle"), not automated oracle mode which uses stronger AI models. The directory name refers to human oracle validation.

## Expected Results

Based on the paper's methodology, this experiment is designed to demonstrate:

- **Accuracy**: CJE estimates within ±2pp of ground truth human evaluations
- **Efficiency**: ~70% reduction in 95% confidence interval width vs. raw IPS
- **Cost**: Total experiment cost ~$1,000 (including human labels)
- **Speed**: <1 GPU-hour total compute (10× faster than decode+judge baseline)

*Note: These are expected outcomes based on the experimental design. Actual results will be reported after running the complete experiment.*

## Experiment Design

### Dataset
- **Source**: 10,000 single-turn prompts from ChatBot Arena Conversations (agie-ai/lmsys-chatbot_arena_conversations)
- **License**: CC-BY-4.0
- **Split**: 25% human label calibration (2,500), 75% evaluation (7,500)

### Policies

**Note**: Model specifications are defined in `configs/arena_10k.yaml`. The scripts use these defaults:

| Policy | Model (from config) | Temperature | Description |
|--------|---------------------|-------------|-------------|
| π₀ (logging) | llama4-scout-instruct-basic | T=0.5 | baseline |
| π_clone | llama4-scout-instruct-basic | T=0.5 | Identical to π₀ (sanity check) |
| π_CoT | llama4-scout-instruct-basic | T=0.5 | Chain-of-thought prompt variant |
| π_bigger_model | llama4-maverick-instruct-basic | T=0.5 | Bigger model variant |

### Judge Configuration
- **Model**: llama4-scout-instruct-basic at T=0 (from config)
- **Rubric**: 0-10 helpfulness/correctness/safety scale
- **Calibration**: Isotonic regression to human labels

### Human Oracle Labels
- **Calibration set**: 2,500 prompts × 3 votes = 7,500 labels
- **Validation set**: 800 prompts/policy × 3 votes = 7,200 labels
- **Platform**: Surge AI (C1-English annotators)
- **Cost**: $0.08/vote

## Repository Structure

```
experiments/arena_10k_oracle/
├── README.md                    # This file
├── configs/
│   └── arena_10k.yaml          # Experiment configuration
├── scripts/
│   ├── 01_prepare_data.py      # Download and preprocess Arena data
│   ├── 02_generate_logs.py     # Generate π₀ responses and scores
│   ├── 03_collect_oracle.py   # Orchestrate human labeling
│   ├── 04_run_targets.py      # Generate target policy responses
│   ├── 05_run_cje.py          # Execute CJE pipeline
│   └── 06_analyze_results.py  # Generate figures and tables
├── data/
│   ├── prompts.jsonl           # 10k sampled prompts
│   ├── p0_replies.jsonl        # Logging policy responses
│   ├── oracle_labels.csv       # Human calibration labels
│   └── validation_labels.csv   # Human validation labels
├── outputs/
│   ├── calibration/            # Isotonic models and diagnostics
│   ├── results/                # CJE estimates and CIs
│   └── figures/                # Paper-ready plots
└── notebooks/
    └── analysis.ipynb          # Interactive analysis and visualization
```

## Quick Start

### Prerequisites

```bash
# Clone CJE repository
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation

# Install dependencies
make dev-setup

# Set API keys (Fireworks for Llama models)
export FIREWORKS_API_KEY="your-key"
```

### Running the Experiment

The experiment has natural breakpoints for human labeling steps:

#### Phase 1: Initial Data Generation (Day 1)

```bash
cd experiments/arena_10k_oracle/scripts

# Step 1: Download and prepare ChatBot Arena data
python 01_prepare_data.py --samples 10000

# Step 2: Generate logging policy responses (π₀)
python 02_generate_logs.py

# Step 3: Add judge scores
python 03_add_judge_scores.py
```

#### Phase 2: Human Labeling (Days 2-3)

```bash
# Step 4a: Export data for human labeling
python 04_export_for_labeling.py --platform surge

# >>> PAUSE HERE <<<
# 1. Upload export file to crowdsourcing platform
# 2. Collect 7,500 human labels (2,500 samples × 3 votes)
# 3. Download results as CSV

# Step 4b: Import labels and calibrate judge
python 04_import_labels.py --labels path/to/human_labels.csv
```

#### Phase 3: Target Policies & Estimation (Day 4)

```bash
# Step 5: Generate target policy responses
python 05_generate_targets.py

# Step 6: Run CJE estimation
python 06_run_cje.py
```

#### Phase 4: Validation (Day 5)

```bash
# Step 7a: Export validation pairs for human evaluation
python 07_export_validation.py

# >>> PAUSE HERE <<<
# 1. Upload to crowdsourcing platform
# 2. Collect validation labels
# 3. Download results

# Step 7b: Import validation and compare to CJE
python 07_analyze_results.py --validation path/to/validation_labels.csv
```

### Alternative: Run Steps Script

```bash
# Run all implemented steps
./run_steps.sh all

# Or run individual steps
./run_steps.sh 1  # Prepare data
./run_steps.sh 2  # Generate responses
./run_steps.sh 3  # Add judge scores
```

## Implementation Details

### Data Preparation

```python
# Download ChatBot Arena conversations
from datasets import load_dataset
ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")

# Extract first-turn prompts
prompts = [conv["conversation"][0]["content"] for conv in ds]

# Sample 10k with seed for reproducibility
import random
random.seed(42)
sampled = random.sample(prompts, 10000)
```

### Oracle Calibration

```python
from sklearn.isotonic import IsotonicRegression

# Fit isotonic map from judge scores to human labels
iso_r = IsotonicRegression(out_of_bounds="clip")
iso_r.fit(judge_scores_oracle, human_labels_oracle)

# Apply to all data
calibrated_rewards = iso_r.predict(judge_scores_all)
```

### CJE Estimation

```python
from cje import CalibratedDR

# Initialize with logged data
cje = CalibratedDR(log_df)

# Estimate for each target policy
results = {}
for policy_name, policy in target_policies.items():
    v_hat, eif = cje.estimate(policy, m_hat, w_cal_fn)
    ci_lower = v_hat - 1.96 * eif.std() / np.sqrt(len(eif))
    ci_upper = v_hat + 1.96 * eif.std() / np.sqrt(len(eif))
    results[policy_name] = {
        'estimate': v_hat,
        'ci': (ci_lower, ci_upper),
        'eif': eif
    }
```

## Expected Results

### Table 1: Expected CJE Performance (Based on Paper)
| Policy | Expected Uplift | Expected CI Width | Notes |
|--------|----------------|-------------------|--------|
| π_clone | ~0pp | ±1.5pp | Sanity check (identical to π₀) |
| π_CoT | +5-7pp | ±1.5pp | Chain-of-thought prompting |
| π_RAG | +2-4pp | ±1.5pp | Retrieval augmentation |
| π_big | +7-9pp | ±1.5pp | Larger model (70B vs 34B) |

### Table 2: Expected Efficiency Gains (Based on Paper)
| Metric | Traditional | CJE (Expected) | Expected Improvement |
|--------|-------------|----------------|---------------------|
| GPU Hours | ~10 | ~1 | 10× reduction |
| API Calls | 40k+ | 10k | 4× reduction |
| 95% CI Width | ±5pp | ±1.5pp | ~70% reduction |
| Total Cost | $2,000+ | <$1,000 | >50% reduction |

*Note: These are projections based on the experimental design. Actual results will vary.*

## Diagnostics to Monitor

During the experiment, track these key metrics:

1. **Effective Sample Size (ESS)**
   - Target: ESS/n > 25%
   - Warning if below 15%

2. **Clipped Weight Mass**
   - Target: < 1%
   - Warning if above 5%

3. **Calibration Quality**
   - Target: Spearman ρ > 0.6
   - Monitor mean absolute calibration error

4. **Cross-Validation Stability**
   - Check fold-wise variation
   - Look for systematic patterns

## Computational Requirements

- **GPU**: 1× H100 or A100 (80GB)
- **Time**: ~1 hour total GPU time
- **Storage**: ~5GB for all intermediate files
- **Memory**: 32GB RAM recommended

## Troubleshooting

### Common Issues

1. **Low ESS Warning**
   - Increase weight clipping threshold
   - Check for overlap violations
   - Consider stratified sampling

2. **Calibration Drift**
   - Collect 150 additional oracle labels
   - Refit isotonic map with additive correction
   - Monitor judge consistency

3. **Memory Issues**
   - Use batch processing for large models
   - Enable gradient checkpointing
   - Reduce batch size in config

## Citation

If you use this experiment, please cite:

```bibtex
@article{landesberg2025cje,
  title={Causal Judge Evaluation: Unbiased, Calibrated \& Cost-Efficient 
         Off-Policy Metrics for LLM Systems},
  author={Landesberg, Eddie},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2025}
}
```

## Contact

For questions about this experiment:
- GitHub Issues: https://github.com/fondutech/causal-judge-evaluation/issues
- Email: eddie@fondutech.com

## License

This experiment code is released under MIT License. The ChatBot Arena dataset is licensed under CC-BY-4.0.