#!/usr/bin/env python3
"""
Verify that the experiment is properly configured for reproducibility.
"""

import sys
from pathlib import Path
from experiment_config import (
    validate_environment,
    print_experiment_config,
    EVALUATION_MODELS,
)


def main() -> int:
    print("=" * 60)
    print("Verifying Arena 10K Experiment Setup")
    print("=" * 60)

    # Check environment
    print("\n🔍 Checking environment...")
    if not validate_environment():
        print("\n❌ Please fix environment issues before running experiments")
        return 1

    print("✅ Environment validation passed")

    # Show configuration
    print("\n📋 Current configuration:")
    print_experiment_config()

    # Verify model choices
    print("\n🚀 Model Performance Notes:")
    print(f"  Judge model ({EVALUATION_MODELS['judge']}):")
    print("    - 13x faster than gpt-5-nano")
    print("    - Suitable for thousands of evaluations")
    print(f"  Oracle model ({EVALUATION_MODELS['oracle']}):")
    print("    - Higher quality evaluations")
    print("    - Used only for calibration subset")

    print("\n✅ Setup verified! Ready to run experiments.")
    print("\nNext steps:")
    print("  1. Generate data:  poetry run python generate_arena_data.py")
    print(
        "  2. Analyze:        poetry run python analyze_dataset.py --data data/cje_dataset.jsonl"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
