# Phase 2: CJE Analysis on Arena 10K Data

This directory contains scripts for running Causal Judge Estimation (CJE) analysis on the Arena 10K dataset prepared in Phase 1.

## Overview

Phase 2 performs importance sampling analysis using precomputed data from Phase 1:
- Importance Propensity Scoring (IPS)
- Self-Normalized IPS (SNIPS) 
- Oracle evaluation for comparison

## Prerequisites

1. **Complete Phase 1**: Ensure these data files exist:
   ```bash
   ../data/p0_scored_deterministic.jsonl     # Main file for importance sampling
   ../data/targets_scored_deterministic.jsonl # Oracle evaluation only
   ```

2. **Understand the Data**: Read `DATA_USAGE_GUIDE.md` for details on:
   - Which files to use for what purpose
   - Common mistakes to avoid
   - Data format specifications

3. **API Keys**: Not required (Phase 2 uses precomputed data)

## Quick Start

### Run CJE Analysis
```bash
# Run IPS/SNIPS analysis with importance weight diagnostics
python run_cje_analysis.py
```

This will:
- Load P0 data for importance sampling
- Compute importance weights with clipping for numerical stability
- Calculate IPS and SNIPS estimates
- Compare with oracle values from target responses
- Display weight statistics and warnings

## Key Scripts

### `run_cje_analysis.py` (Primary Script)
The main analysis script that:
- Uses ONLY P0 data for importance sampling (as it should)
- Implements log ratio clipping to handle extreme values
- Computes both IPS and SNIPS estimates
- Loads oracle data for comparison only
- Provides clear diagnostics and warnings


## Data Files Explained

### Primary Data: `p0_scored_deterministic.jsonl`
Contains everything needed for importance sampling:
```json
{
  "prompt_id": "arena_sampled_0",
  "prompt": "...",
  "response": "P0's response",
  "judge_score": 0.925,
  "total_logprob": -86.05,        // log P(response | P0)
  "target_logps": {               // log P(response | each policy)
    "pi_clone": -86.06,
    "pi_cot": -119.03,
    "pi_bigger_model": -292.24,
    "pi_bad": -187.09
  }
}
```

### Oracle Data: `targets_scored_deterministic.jsonl`
Contains each policy's actual responses (for oracle comparison only):
```json
{
  "prompt_id": "arena_sampled_0",
  "policy": "pi_cot",
  "response": "pi_cot's own response",
  "judge_score": 0.890
}
```

## Important Notes

1. **Data Usage**:
   - IPS/SNIPS uses ONLY P0 data
   - Target responses are for oracle comparison only
   - Never try to match responses between files (they're different due to sampling)

2. **Known Issues**:
   - Extreme weights for pi_clone indicate API non-determinism issues
   - Log ratio clipping helps but doesn't fully solve the problem
   - Need to fix Phase 1 log probability computation before 10K run

3. **Expected Results**:
   - pi_clone median weight should be ~1.0
   - SNIPS is more stable than IPS
   - Oracle values show actual policy performance

## Typical Output

```
Importance Weight Statistics
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━┳━━━━━━━━━┓
┃ Policy       ┃    Mean ┃ Median ┃   Min ┃     Max ┃ ESS% ┃ Extreme ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━╇━━━━━━━━━┩
│ pi_clone     │ 9.1e+10 │  1.000 │ 0.000 │ 1.8e+13 │ 0.5% │       7 │
│ pi_cot       │ 3566.95 │  0.000 │ 0.000 │ 690681. │ 0.5% │     173 │
└──────────────┴─────────┴────────┴───────┴─────────┴──────┴─────────┘

Policy Value Estimates
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ Policy          ┃     IPS ┃ SNIPS ┃ Oracle ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ pi_clone        │ 7.7e+10 │ 0.850 │  0.893 │
│ pi_cot          │ 3387.11 │ 0.950 │  0.896 │
└─────────────────┴─────────┴───────┴────────┘
```

## Troubleshooting

### Extreme Weights
- Expected due to API non-determinism in Phase 1
- Log ratio clipping helps but doesn't eliminate the issue
- Fix Phase 1 before running 10K samples

### Missing Files
- Ensure Phase 1 completed successfully
- Check that files are in `../data/` directory

## Archive

Old/debugging scripts have been moved to `archive/` to avoid confusion.
Use only `run_cje_analysis.py` for analysis.