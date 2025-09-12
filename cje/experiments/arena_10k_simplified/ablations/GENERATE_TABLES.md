# Table Generation Guide

## Quick Start

From the ablations directory, run:

```bash
# Generate main paper tables (LaTeX format)
python reporting/paper_tables.py --format latex

# Generate markdown versions for quick viewing
python reporting/paper_tables.py --format markdown

# Generate appendix tables
python reporting/appendix_tables.py --format markdown

# Generate quadrant-specific leaderboards
python generate_quadrant_leaderboard.py
```

## Output Locations

Tables are saved to:
- `tables/main/` - Main paper tables (Table 1-3)
- `tables/appendix/` - Appendix tables 
- `tables/quadrant/` - Quadrant-specific leaderboards

## Main Tables

### Table 1: Leaderboard
- **File**: `table1_leaderboard.{tex,md}`
- **Content**: Overall estimator rankings with aggregate scores
- **Metrics**: RMSE^d, IntervalScore^OA, CalibScore, SE_GeoMean, Kendall τ, Top-1 accuracy
- **Scoring**: 30% ranking, 25% accuracy, 25% efficiency, 20% calibration

### Table 2: Design Choice Effects
- **Files**: 
  - `table2a_calibration.{tex,md}` - Weight calibration effects
  - `table2b_iic.{tex,md}` - IIC (Influence IC) effects
- **Content**: Paired comparisons showing impact of design choices

### Table 3: Stacking Diagnostics
- **File**: `table3_stacking.{tex,md}`
- **Content**: Stacked-DR specific metrics and diagnostics

## Appendix Tables

### Table A1: Quadrant Leaderboard
- **File**: `table_a1_quadrant_leaderboard.md`
- **Content**: Rankings by sample size × oracle coverage quadrants

### Table A2: SE Progression
- **File**: `table_a2_se_progression.md`
- **Content**: How standard errors change with sample size

### Table A3: Coverage Analysis
- **File**: `table_a3_coverage.md`
- **Content**: Confidence interval coverage by policy

### Table A4: Runtime Performance
- **File**: `table_a4_runtime.md`
- **Content**: Computational efficiency metrics

### Table A5: Oracle Boundary Analysis
- **File**: `table_a5_boundary.md`
- **Content**: Reward calibration support analysis

## Quadrant Leaderboards

Generated separately via `generate_quadrant_leaderboard.py`:
- Small-Low (n≤1000, coverage≤0.10)
- Small-High (n≤1000, coverage>0.10)
- Large-Low (n>1000, coverage≤0.10)
- Large-High (n>1000, coverage>0.10)

## Prerequisites

Before generating tables, ensure:
1. Experiments have been run (`python run.py`)
2. Results are in `results/all_experiments.jsonl`
3. Metrics have been computed (automatic during table generation)

## Customization

### Aggregate Score Weights
Edit `reporting/paper_tables.py` line ~300-340 to adjust scoring weights:
```python
# Current weights:
# - Ranking quality: 30% (Kendall τ + Top-1)
# - Accuracy: 25% (RMSE^d)
# - Efficiency: 25% (SE GeoMean)  
# - Calibration: 20% (CalibScore + IntervalScore^OA)
```

### Adding New Tables
1. Create function in `reporting/paper_tables.py` or `reporting/appendix_tables.py`
2. Add to main() function to include in generation
3. Follow naming convention: `table{N}_{description}.{ext}`

## Troubleshooting

**Empty tables**: Run `python analyze.py` first to compute metrics

**Missing estimators**: Check that experiments completed successfully

**NaN values**: Some metrics may be unavailable for certain estimators (expected)

**Formatting issues**: LaTeX tables may need manual adjustment for publication