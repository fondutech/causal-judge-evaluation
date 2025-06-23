# Judge Uncertainty Methods

## Overview

This experiment extends CJE with judge uncertainty quantification, allowing for more nuanced evaluation that accounts for judge confidence.

## Uncertainty Methods

### 1. Deterministic (Default)
- **Variance**: Always 0
- **Use case**: Fast evaluation when judge confidence isn't needed
- **Template**: `deterministic`
- **Script**: `04a_add_judge_scores_deterministic.py`

### 2. Confidence Interval
- **Variance**: Calculated from 95% CI width
- **Use case**: When judge uncertainty matters for decision-making
- **Template**: `confidence_interval` 
- **Script**: `04b_add_judge_scores_uncertainty.py`
- **How it works**:
  1. Judge provides score + 95% CI (e.g., "Score: 7, CI: [6, 8]")
  2. Variance = (CI_width / 3.92)² / 100
  3. Narrower CI = lower variance = higher confidence

### 3. Monte Carlo (Not used in this experiment)
- **Variance**: From multiple samples
- **Use case**: When you need empirical uncertainty estimates
- **Template**: Regular templates with temperature > 0

## Implementation Details

### Structured Output Schema

```python
class JudgeScoreWithCI(JudgeScore):
    """Judge score with explicit confidence interval."""
    ci_lower: float  # Lower bound (0-10 scale)
    ci_upper: float  # Upper bound (0-10 scale)
    # Variance auto-calculated from CI width
```

### Template Format

The `confidence_interval` template asks for:
```json
{
  "mean": 0.7,      // Score as 0-1
  "ci_lower": 6,    // Lower bound as 0-10
  "ci_upper": 8     // Upper bound as 0-10
}
```

### Mathematical Foundation

For a 95% confidence interval:
- CI = mean ± 1.96σ
- CI_width = 2 × 1.96σ = 3.92σ
- σ = CI_width / 3.92
- Variance = σ²

Scale conversion (judge uses 0-10, CJE uses 0-1):
- σ_01 = σ_10 / 10
- Var_01 = (σ_10)² / 100

## Running Experiments

### Quick Comparison
```bash
# Run both methods
python run_pipeline.py
# Select option 3: Run both scoring methods

# This will:
# 1. Score with deterministic judge
# 2. Score with CI-based uncertainty
# 3. Run CJE with both
# 4. Compare results
```

### Manual Steps
```bash
# Deterministic
python scripts/04a_add_judge_scores_deterministic.py
cje run --cfg-path configs --cfg-name arena_10k_oracle

# With uncertainty
python scripts/04b_add_judge_scores_uncertainty.py
cje run --cfg-path configs --cfg-name arena_10k_oracle_uncertainty
```

## Interpreting Results

### When Uncertainty Helps
- **Ambiguous responses**: Judge is unsure → high variance → less weight in estimation
- **Clear quality differences**: Judge is confident → low variance → more weight
- **Calibration**: Uncertainty can improve calibration curve fitting

### Expected Outcomes
1. **Tighter confidence intervals**: Uncertainty-aware estimates should have narrower CIs for the policy value estimates
2. **Better calibration**: Accounting for judge confidence improves calibration
3. **Robust rankings**: Policy rankings should be consistent but with better uncertainty quantification

## Technical Notes

### CI Judge Implementation
- Located in `cje/judge/ci_judge.py`
- Forces `confidence_interval` template
- Uses `JudgeScoreWithCI` structured output
- Recommends temperature=0 for consistency

### Template System
- Cleaned templates in `cje/prompts/judge_templates.py`
- Removed legacy templates
- Focus on: `deterministic`, `confidence_interval`, `simple`, `comparative`

### Integration Points
1. **JudgeFactory**: `uncertainty_method` parameter controls behavior
2. **Structured Output**: Automatic parsing of JSON responses
3. **Variance Calculation**: Happens in `JudgeScoreWithCI` validator