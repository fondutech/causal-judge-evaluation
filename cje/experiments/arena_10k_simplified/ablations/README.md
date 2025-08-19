# CJE Ablations

Modular ablation studies validating CJE's calibration benefits for the paper.

## Structure

```
ablations/
├── core/                    # Shared infrastructure
│   ├── base.py             # BaseAblation class with caching
│   ├── schemas.py          # ExperimentSpec and aggregation
│   └── diagnostics.py      # ESS, Hill, CV metrics
├── oracle_coverage.py       # Q: How much oracle data needed?
├── sample_size.py          # Q: When does DR become necessary?
├── interaction.py          # Q: Coverage × size sweet spots?
└── estimator_comparison.py  # Q: Value of each technique?
```

## Key Findings

1. **Oracle Coverage**: 5-10% oracle labels sufficient for calibration
2. **Sample Size**: DR needed when n < 500 or ESS < 10%
3. **Estimator Progression**: Each technique adds value (IPS → SNIPS → Cal-IPS → DR → Stacked)
4. **SIMCal Impact**: 13.9× ESS improvement, 4.5× error reduction

## Running Experiments

```bash
# Individual ablation
python oracle_coverage.py

# Quick test with small data
python quick_test.py

# Analyze cached results
python analyze_results.py
```

## Implementation Status

✅ **Complete**:
- Core infrastructure (base class, schemas, caching)
- Oracle coverage ablation
- Sample size ablation  
- Interaction effects
- Estimator comparison
- ~200+ experiments cached

## Caching

All results cached in `.ablation_cache/` with SHA-based keys for reproducibility.
Cache survives crashes - just re-run to continue.

## Data

Uses `../data/cje_dataset.jsonl` - the Arena 10k simplified dataset with:
- 994 samples
- Judge scores from GPT-4
- Oracle labels from human annotators
- Log probabilities for multiple policies