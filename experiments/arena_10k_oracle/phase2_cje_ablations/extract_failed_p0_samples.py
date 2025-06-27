#!/usr/bin/env python3
"""Extract P0 samples that need re-scoring due to -50.0 log probability."""

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def extract_failed_samples():
    """Extract samples with P0 logp = -50.0 for re-scoring."""

    # Load data
    data_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    failed_samples = []
    valid_samples = []

    with open(data_file, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("total_logprob") == -50.0:
                failed_samples.append(item)
            else:
                valid_samples.append(item)

    console.print(
        f"[bold]Found {len(failed_samples)} samples with P0 logp = -50.0[/bold]"
    )
    console.print(f"Valid samples: {len(valid_samples)}")

    # Analyze failed samples
    lengths = [len(item["response"]) for item in failed_samples]

    table = Table(title="Failed Sample Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total failed", str(len(failed_samples)))
    table.add_row("Min response length", str(min(lengths)))
    table.add_row("Max response length", str(max(lengths)))
    table.add_row("Avg response length", f"{sum(lengths)/len(lengths):.0f}")
    table.add_row("Short (<100)", str(sum(1 for l in lengths if l < 100)))
    table.add_row("Medium (100-500)", str(sum(1 for l in lengths if 100 <= l < 500)))
    table.add_row("Long (>500)", str(sum(1 for l in lengths if l >= 500)))

    console.print(table)

    # Save failed samples for re-scoring
    output_file = Path("../data/p0_samples_to_rescore.jsonl")
    with open(output_file, "w") as f:
        for item in failed_samples:
            # Extract just what we need for re-scoring
            rescore_item = {
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "response": item["response"],
                "original_total_logprob": item["total_logprob"],
                "metadata": item.get("metadata", {}),
            }
            f.write(json.dumps(rescore_item) + "\n")

    console.print(f"\n✅ Saved {len(failed_samples)} samples to {output_file}")

    # Also save the valid data separately
    valid_file = Path("../data/p0_with_target_logps_valid_only.jsonl")
    with open(valid_file, "w") as f:
        for item in valid_samples:
            f.write(json.dumps(item) + "\n")

    console.print(f"✅ Saved {len(valid_samples)} valid samples to {valid_file}")

    # Show some examples
    console.print("\n[yellow]Sample failed entries:[/yellow]")
    for i, item in enumerate(failed_samples[:3]):
        console.print(f"\n{i+1}. ID: {item['prompt_id']}")
        console.print(f"   Response length: {len(item['response'])}")
        console.print(f"   Response: {item['response'][:100]}...")


def check_if_already_rescored():
    """Check if we already have rescored data."""
    rescored_file = Path("../data/p0_samples_rescored.jsonl")

    if rescored_file.exists():
        count = sum(1 for _ in open(rescored_file))
        console.print(
            f"\n[yellow]⚠️  Found existing rescored file with {count} samples[/yellow]"
        )
        console.print(f"   Path: {rescored_file}")
        return True
    return False


def main():
    """Run the extraction."""
    console.print("[bold blue]🔍 Extracting Failed P0 Samples[/bold blue]\n")

    extract_failed_samples()

    if check_if_already_rescored():
        console.print("\n[green]You may already have rescored data available![/green]")

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Run P0 scoring on p0_samples_to_rescore.jsonl")
    console.print("2. Merge rescored data back into the main dataset")
    console.print("3. Re-run ablations with complete valid data")


if __name__ == "__main__":
    main()
