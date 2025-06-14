# Arena 10K Fresh Oracle Experiment - Summary

## Purpose

This experiment is designed to validate the Causal Judge Evaluation (CJE) framework on real-world data, with the goal of demonstrating:
1. **Accuracy**: CJE predictions should match ground truth within ±2pp
2. **Efficiency**: Target ~70% confidence interval reduction vs baseline methods
3. **Speed**: Target 10× GPU time reduction vs decode+judge approaches
4. **Practicality**: Complete workflow with human labeling for <$1000

*Note: These are target outcomes based on the paper's methodology. Actual results will be measured and reported after running the complete experiment.*

## Directory Structure

```
experiments/arena_10k_oracle/
├── README.md                    # Detailed instructions
├── EXPERIMENT_SUMMARY.md        # This file
├── configs/
│   └── arena_10k.yaml          # Experiment configuration
├── scripts/
│   ├── 01_prepare_data.py      # Download ChatBot Arena data
│   ├── 02_generate_logs.py     # Generate π₀ responses
│   ├── 03_add_judge_scores.py  # Score with LLM judge
│   ├── 04_export_for_labeling.py # Export for human labeling
│   ├── 04_import_labels.py     # Import labels & calibrate
│   ├── 05_generate_targets.py  # Generate target policies (TBD)
│   ├── 06_run_cje.py          # Run CJE estimation (TBD)
│   ├── 07_export_validation.py # Export for validation (TBD)
│   ├── 07_analyze_results.py  # Final analysis (TBD)
│   └── run_steps.sh           # Helper script
├── data/                       # Generated data (not in git)
└── outputs/                    # Results and figures (not in git)
```

## Key Design Decisions

### 1. **Modular Scripts**
Each step is a standalone script that:
- Can be run independently
- Saves intermediate results
- Supports checkpointing for long runs
- Reports progress and costs

### 2. **Human-in-the-Loop Design**
Natural breakpoints for crowdsourcing:
- After Step 3: Export 2,500 samples for calibration
- After Step 6: Export validation pairs for ground truth

### 3. **Cost Transparency**
Every script reports:
- API calls made
- Estimated costs
- Time taken
- Resources used

### 4. **Reproducibility**
- Fixed random seeds throughout
- Versioned dependencies
- Complete configuration in YAML
- All intermediate data saved

## Workflow

### Phase 1: Data Generation (Automated)
```bash
python 01_prepare_data.py      # ~5 min
python 02_generate_logs.py     # ~30 min, ~$20
python 03_add_judge_scores.py  # ~20 min, ~$10
```
**Output**: 10k scored responses ready for calibration

### Phase 2: Calibration (Human Labels)
```bash
python 04_export_for_labeling.py
# >>> Upload to Surge AI / MTurk <<<
# Wait 1-2 days for 7,500 labels (~$600)
python 04_import_labels.py --labels human_labels.csv
```
**Output**: Calibrated judge scores aligned with human preferences

### Phase 3: Evaluation (Automated)
```bash
python 05_generate_targets.py  # ~45 min, ~$30
python 06_run_cje.py          # ~10 min
```
**Output**: CJE estimates with confidence intervals

### Phase 4: Validation (Human Labels)
```bash
python 07_export_validation.py
# >>> Collect validation labels (~$400) <<<
python 07_analyze_results.py
```
**Output**: Accuracy comparison and final figures

## Expected Timeline

- **Day 1**: Run Phase 1 (3 scripts, ~1 hour compute)
- **Days 2-3**: Collect calibration labels
- **Day 4**: Run Phase 3 (2 scripts, ~1 hour compute)  
- **Day 5**: Collect validation labels
- **Day 6**: Final analysis and paper figures

Total: ~1 week elapsed, ~3 hours compute, ~$1000 total cost

## Integration with Main Codebase

This experiment uses core CJE components:
- `cje.judge` - LLM judge scoring
- `cje.calibration` - Isotonic regression
- `cje.estimators` - Calibrated DR-CPO
- `cje.utils` - Checkpointing, progress tracking

But remains standalone for:
- Clear reproducibility
- Educational value
- Paper artifact submission

## Next Steps

Currently implemented:
- ✅ Steps 1-4: Data prep through calibration
- ⏳ Steps 5-7: Target policies and validation (TBD)

To complete the experiment:
1. Implement remaining scripts (5-7)
2. Run Phase 1 to generate initial data
3. Set up crowdsourcing account
4. Execute full pipeline

## Contact

For questions about this experiment:
- GitHub Issues: [Create an issue](https://github.com/fondutech/causal-judge-evaluation/issues)
- Paper: "Causal Judge Evaluation" (Landesberg 2025)