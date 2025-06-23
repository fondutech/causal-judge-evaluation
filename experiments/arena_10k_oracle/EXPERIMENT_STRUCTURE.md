# Experiment Structure

## Two-Phase Architecture

```
┌─────────────────────────────────────┐
│    Arena 10K Oracle Experiment      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│   Phase 1   │  │   Phase 2   │
│   Dataset   │  │     CJE     │
│ Preparation │  │  Ablations  │
└─────────────┘  └─────────────┘
```

## Phase 1: Dataset Preparation

```
                    ┌──────────────────┐
                    │ 1. Extract       │
                    │ Prompts          │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ 2. Generate      │
                    │ Responses        │
                    │ (π₀ + targets)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ 3. Generate      │
                    │ Oracle Labels    │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       ┌──────────────┐          ┌──────────────┐
       │ 4a. Score    │          │ 4b. Score    │
       │ Deterministic│          │ Uncertainty  │
       └──────┬───────┘          └──────┬───────┘
              │                          │
              └────────┬─────────────────┘
                       ▼
              ┌──────────────┐
              │ 5. Finalize  │
              │ Dataset      │
              └──────────────┘
```

**Output**: Complete dataset with:
- 10,000 prompts
- Responses from all policies
- Oracle labels (4,000 total)
- Judge scores (both methods)

## Phase 2: CJE Pipeline Ablations

```
         ┌───────────────────────┐
         │  Prepared Dataset     │
         │  (from Phase 1)       │
         └──────────┬────────────┘
                    │
    ┌───────────────┴───────────────┐
    │        CJE Ablations          │
    └───────────────┬───────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Baseline │  │ Doubly   │  │  Self-   │
│   IPW    │  │ Robust   │  │Normalized│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     ├──────────────┼──────────────┤
     ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Det.    │  │  Unc.    │  │ Compare  │
│ Scores   │  │ Scores   │  │ Results  │
└──────────┘  └──────────┘  └──────────┘
```

**Ablation Matrix**:
- 2 judge methods × 3 estimators = 6 configurations
- Each produces policy value estimates
- Results compared for insights

## File Organization

```
arena_10k_oracle/
├── phase1_dataset_preparation/
│   ├── 01_prepare_data.py
│   ├── 02_generate_logs.py
│   ├── 02b_generate_target_ground_truth.py
│   ├── 03_generate_oracle_labels.py
│   ├── 04a_deterministic_judge_scores.py
│   ├── 04b_uncertainty_judge_scores.py
│   ├── 05_finalize_dataset.py
│   └── add_judge_scores.py (shared implementation)
│
├── phase2_cje_ablations/
│   ├── run_ablations.py
│   ├── prepare_for_cje.py
│   ├── configs/         # Generated YAML configs
│   └── results/         # Ablation comparisons
│
├── data/                # All experiment data
│   ├── arena_prompts_10k.jsonl
│   ├── p0_replies.jsonl
│   ├── target_ground_truth.jsonl
│   ├── p0_scored_deterministic.jsonl
│   ├── p0_scored_uncertainty.jsonl
│   ├── dataset_info.json
│   └── labeling/
│       ├── oracle_labels_calibration_detailed.jsonl
│       ├── oracle_labels_validation_detailed.jsonl
│       └── oracle_labels.csv
│
└── outputs/             # CJE results
    ├── arena_10k_baseline_deterministic/
    ├── arena_10k_baseline_uncertainty/
    └── ... (other ablations)
```

## Key Design Principles

1. **Separation of Concerns**: Data preparation is separate from analysis
2. **Reusability**: Once Phase 1 is complete, Phase 2 can be run many times
3. **Modularity**: Easy to add new ablations without regenerating data
4. **Comparability**: All ablations use the same underlying dataset

