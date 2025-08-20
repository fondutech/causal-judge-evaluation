#!/usr/bin/env python3
"""
Master script to run all ablations for the CJE paper.

This runs all ablation experiments in sequence and collects results.
Results are saved to ablations/results/ directory.
"""

import subprocess
import sys
import time
from pathlib import Path
import json
from datetime import datetime


def run_ablation(script_name: str) -> bool:
    """Run a single ablation script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_name], capture_output=True, text=True, check=True
        )
        elapsed = time.time() - start_time
        print(f"✓ {script_name} completed in {elapsed:.1f}s")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {script_name} failed after {elapsed:.1f}s")
        print(f"Error: {e.stderr}")
        return False


def main() -> int:
    """Run all ablations and collect results."""

    # Ablations to run in order
    ablations = [
        "oracle_coverage.py",  # Effect of oracle slice size
        "sample_size.py",  # Effect of dataset size
        "estimator_comparison.py",  # Compare all estimators
        "interaction.py",  # Oracle × sample size interaction
    ]

    print("=" * 80)
    print("CJE ABLATION EXPERIMENTS")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Track results
    from typing import Dict, Any

    results: Dict[str, Any] = {"timestamp": datetime.now().isoformat(), "ablations": {}}

    total_start = time.time()
    successful = 0
    failed = 0

    for ablation in ablations:
        success = run_ablation(ablation)
        if success:
            successful += 1
            results["ablations"][ablation] = "completed"

            # Try to load the results if they exist
            result_file = Path(f"results/{ablation.replace('.py', '_results.json')}")
            if result_file.exists():
                with open(result_file) as f:
                    results["ablations"][ablation] = json.load(f)
        else:
            failed += 1
            results["ablations"][ablation] = "failed"

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"Successful: {successful}/{len(ablations)}")
    print(f"Failed: {failed}/{len(ablations)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")

    # Save master results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_file = results_dir / f"all_ablations_{timestamp}.json"
    with open(master_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMaster results saved to: {master_file}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if successful > 0:
        print(
            """
Based on completed ablations:

1. Oracle Coverage: Shows how calibration quality affects estimates
2. Sample Size: Demonstrates convergence behavior  
3. Estimator Comparison: StackedDR should show best performance
4. Interaction: Reveals when DR methods are most valuable

Check individual result files in results/ for detailed analysis.
"""
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
