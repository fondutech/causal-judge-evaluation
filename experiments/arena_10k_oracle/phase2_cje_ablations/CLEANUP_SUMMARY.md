# Cleanup Summary - phase2_cje_ablations

## Kept Files

### Scripts
- run_direct_ablations.py
- visualize_results.py
- README_WORKFLOW.md

### Data Files
- p0_with_target_logps_fully_fixed.jsonl (our best data with fixed log probs)
- p0_with_target_logps.checkpoint.jsonl (active teacher forcing checkpoint)
- Original scored data files

## Archived to archive_20250626/
- 6 analysis scripts
- 5 checkpoint files

## Removed
- 16 temporary scripts
- 16 temporary data files

## Status
- Teacher forcing: Still running (check process)
- Best available data: p0_with_target_logps_fully_fixed.jsonl (1,824 samples)
- Key fix: Replaced 139 incorrect -50.0 log probs with correct values
