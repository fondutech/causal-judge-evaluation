# Sample Run Tools

This directory contains all tools for running and validating the 1% sample test.

## Quick Start

```bash
# 1. Pre-flight check
python ../preflight_check.py

# 2. Run sample
../run_sample.sh

# 3. Monitor (in another terminal)
python ../monitor_sample_run.py

# 4. Validate results
python ../validate_sample_results.py
python ../analyze_teacher_forcing_stats.py
```

## Files

- `run_sample.sh` - Main sample execution script
- `sample_config.py` - Configuration for sample vs full runs
- `monitor_sample_run.py` - Real-time progress monitoring
- `validate_sample_results.py` - Post-run validation
- `analyze_teacher_forcing_stats.py` - Detailed TF analysis
- `preflight_check.py` - Pre-run dependency checks
- `estimate_costs.py` - Cost estimation tool

See `1_percent_sample_plan.md` for detailed documentation.