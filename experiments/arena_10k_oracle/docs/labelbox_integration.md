# Labelbox Integration Guide

## Overview

This guide explains how to use the Labelbox integration scripts for human labeling in the Arena 10K Oracle experiment.

## Scripts Created

### 1. Export Script: `03b_export_for_labelbox.py`

Exports data to Labelbox for human labeling with the following features:
- Samples 25% of π₀ data for calibration
- Combines calibration and target policy responses
- Removes policy information for blind rating
- Creates Labelbox-compatible data rows with metadata
- Uploads data in batches to Labelbox dataset

**Usage:**
```bash
python 03b_export_for_labelbox.py --api-key YOUR_API_KEY --dataset-id YOUR_DATASET_ID
```

### 2. Import Script: `06b_import_labelbox_labels.py`

Imports labeled data from Labelbox with:
- Downloads labels using Labelbox SDK
- Matches labels with internal tracking data
- Computes rating statistics (mean, median, std)
- Normalizes scores to [0, 1] range for CJE compatibility
- Generates summary reports

**Usage:**
```bash
python 06b_import_labelbox_labels.py --api-key YOUR_API_KEY --project-id YOUR_PROJECT_ID
```

## Data Format

### Labelbox Data Rows

Each data row contains:
- **row_data**: Formatted conversation text (USER: ... ASSISTANT: ...)
- **global_key**: Unique identifier (arena_task_XXXXXX)
- **media_type**: TEXT
- **metadata_fields**: Non-revealing metadata (task type, lengths)
- **attachments**: Raw prompt and response as separate attachments

### Internal Tracking

Maintains mapping between Labelbox task IDs and:
- Original prompt IDs
- Policy information (pi_0, target)
- Split information (calibration, evaluation)

## Labelbox Project Setup

1. Create a new dataset in Labelbox
2. Upload data using the export script
3. Create a labeling project with:
   - Classification task type
   - Rating scale 0-10
   - Single selection
4. Configure labeling instructions (provided in labelbox_instructions.md)

## Output Files

### From Export Script:
- `labelbox_tracking.jsonl`: Internal tracking data
- `labelbox_instructions.md`: Instructions for labelers

### From Import Script:
- `human_labeled_scores.jsonl`: CJE-compatible labeled data
- `human_labeled_scores.csv`: CSV for easy inspection
- `labeling_report.txt`: Summary statistics

## Cost Estimation

Based on default settings:
- Calibration samples (25% of π₀): 2,500
- Target policy samples (500 × 3 policies): 1,500
- **Total samples: 4,000**
- Votes per sample: 3
- Total votes: 12,000
- Cost per vote: $0.08
- **Total cost: $960**
- Time estimate: ~150 hours @ 45s/vote

Note: The export scripts now sample 500 prompt/response pairs from each target policy (excluding pi_clone), not just from the target ground truth file.

## API Key Note

The API key provided appears to be invalid. You'll need to:
1. Log into your Labelbox account
2. Generate a new API key from Settings > API
3. Use the new key with the scripts

## Testing

Use `test_labelbox_export_local.py` to test the export format locally without API calls.