# Labeling Data Directory

This directory will contain oracle labels for the Arena 10K experiment.

## Current Status

**Empty** - Ready for oracle label generation.

Run this command to generate oracle labels:
```bash
python scripts/05b_generate_oracle_labels_cje.py --model gpt-4o
# or for o3:
python scripts/05b_generate_oracle_labels_cje.py --model o3-2025-01-17 --temperature 1.0
```

## Deprecated Content

All MTurk-related files have been moved to `deprecated_mturk_attempt/` including:
- Original data exports
- Failed human labels 
- Analysis scripts
- Test data

See `deprecated_mturk_attempt/README.md` for details on why human labeling failed.