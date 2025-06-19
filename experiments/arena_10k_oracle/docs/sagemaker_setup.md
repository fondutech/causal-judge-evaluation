# SageMaker Ground Truth Setup Guide

This guide explains how to set up human labeling for the Arena 10K Oracle experiment using AWS SageMaker Ground Truth.

## Prerequisites

1. AWS account with SageMaker Ground Truth access
2. S3 bucket for storing manifest and output files
3. IAM role with appropriate permissions
4. Generated export file from step 3: `03_export_for_labeling.py`

## Files Included

- `sagemaker_manifest.jsonl` - The manifest file containing all tasks with inline data
- `sagemaker_template.html` - Custom HTML template for the labeling interface

## Step-by-Step Setup

### 1. Generate the Manifest

First, ensure you have exported the labeling data:

```bash
python 03_export_for_labeling.py
```

Then create the SageMaker manifest:

```bash
python create_sagemaker_manifest.py
```

This creates `../data/labeling/sagemaker_manifest.jsonl` with 4,000 tasks.

### 2. Upload Files to S3

Upload the manifest file to your S3 bucket:

```bash
aws s3 cp ../data/labeling/sagemaker_manifest.jsonl s3://your-bucket/manifests/manifest.jsonl
```

### 3. Create Labeling Job in SageMaker

1. Go to AWS SageMaker Console
2. Navigate to **Ground Truth** → **Labeling jobs**
3. Click **Create labeling job**

### 4. Configure the Job

#### Basic Configuration
- **Job name**: `arena-10k-oracle-rating`
- **Input dataset location**: `s3://your-bucket/manifests/manifest.jsonl`
- **Output dataset location**: `s3://your-bucket/output/`
- **IAM role**: Select or create a role with S3 access

#### Task Type
- Select **Custom** → **Custom labeling task**

#### Workers
- Choose your workforce (private, vendor, or mechanical turk)
- Configure worker instructions if needed

#### Task Template
1. In the template editor, paste the contents of `sagemaker_template.html`
2. The template provides:
   - Clear instructions for raters
   - User question and AI response display
   - Prominent 0-10 rating slider
   - Visual feedback for selected rating

### 5. IAM Permissions

Ensure your IAM role has:
- `s3:GetObject` permission on the input manifest
- `s3:PutObject` permission on the output location
- SageMaker Ground Truth execution permissions

### 6. Launch and Monitor

1. Review all settings
2. Click **Create** to launch the labeling job
3. Monitor progress in the SageMaker console
4. Download results when complete

## Output Format

The labeling job will produce output in JSON Lines format with:
- Original task data (task_id, prompt, response)
- Worker annotations including the 0-10 rating
- Metadata about worker agreement and confidence

## Data Structure

Each manifest entry contains:
```json
{
  "source": "inline",
  "task_id": "task_000000",
  "prompt": "User question text...",
  "response": "AI assistant response text..."
}
```

The `source: "inline"` field is required by SageMaker Ground Truth for custom labeling jobs.

## Troubleshooting

1. **S3 Access Errors**: Verify IAM role permissions
2. **Template Rendering Issues**: Check for Liquid syntax errors
3. **Missing Ratings**: Ensure the slider value is properly captured in the form submission

## Next Steps

After labeling is complete:
1. Download the output from S3
2. Run `06_import_labels.py` to process the human labels
3. Continue with judge scoring (step 4) and analysis