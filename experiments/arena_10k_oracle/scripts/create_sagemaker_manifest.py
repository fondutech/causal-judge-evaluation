#!/usr/bin/env python3
"""
Create SageMaker Ground Truth manifest file with inline data.
No separate S3 files needed - all data is in the manifest.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console


def create_manifest(input_csv: str, output_manifest: str) -> None:
    """Create SageMaker manifest from CSV export with inline data."""

    # Load the CSV
    console.print(f"📄 Loading export from {input_csv}")
    df = pd.read_csv(input_csv)
    console.print(f"   • Loaded {len(df):,} tasks")

    # Create output directory if needed
    manifest_path = Path(output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"\n📝 Creating SageMaker manifest with inline data...")

    with open(manifest_path, "w") as f:
        for idx, row in df.iterrows():
            # Handle any NaN values
            prompt_text = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
            response_text = str(row["response"]) if pd.notna(row["response"]) else ""

            # Create manifest entry with required source field for Ground Truth
            manifest_entry = {
                "source": "inline",  # Required by SageMaker Ground Truth
                "task_id": row["task_id"],
                "prompt": prompt_text,
                "response": response_text,
            }

            f.write(json.dumps(manifest_entry) + "\n")

    console.print(f"✅ Created manifest: {manifest_path}")
    console.print(f"   • Total entries: {len(df):,}")

    # Show a sample
    console.print(f"\n📋 Sample manifest entry:")
    with open(manifest_path, "r") as f:
        sample_line = f.readline()
        sample = json.loads(sample_line)
        console.print(f"   • Task ID: {sample['task_id']}")
        console.print(f"   • Prompt: {sample['prompt'][:50]}...")
        console.print(f"   • Response: {sample['response'][:50]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SageMaker Ground Truth manifest with inline data"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/labeling/human_labeling_export.csv",
        help="Input CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/labeling/sagemaker_manifest.jsonl",
        help="Output manifest file",
    )

    args = parser.parse_args()

    console.print(f"🔧 [bold blue]Creating SageMaker Manifest[/bold blue]")

    try:
        create_manifest(args.input, args.output)

        console.print(f"\n📚 Next steps:")
        console.print(f"1. Upload manifest to S3:")
        console.print(
            f"   aws s3 cp {args.output} s3://your-bucket/manifests/manifest.jsonl"
        )
        console.print(f"\n2. In SageMaker Ground Truth:")
        console.print(
            f"   - Input manifest S3 URI: s3://your-bucket/manifests/manifest.jsonl"
        )
        console.print(f"   - Output location S3 URI: s3://your-bucket/output/")
        console.print(f"   - Task type: Custom")
        console.print(f"   - Use the custom template (sagemaker_template.html)")
        console.print(f"\n3. Make sure your IAM role has:")
        console.print(f"   - s3:GetObject on the input manifest")
        console.print(f"   - s3:PutObject on the output location")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
