#!/usr/bin/env python3
"""
Quick check if we're ready to run the sample test.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def check_ready():
    """Check if everything is ready for sample run."""
    print("🔍 Checking readiness for sample run...\n")

    issues = []

    # Check API keys
    if not os.environ.get("FIREWORKS_API_KEY"):
        issues.append("❌ FIREWORKS_API_KEY not set")
    else:
        print("✅ FIREWORKS_API_KEY is set")

    if not os.environ.get("OPENAI_API_KEY"):
        issues.append("❌ OPENAI_API_KEY not set")
    else:
        print("✅ OPENAI_API_KEY is set")

    # Check data file
    data_file = Path(__file__).parent.parent / "data" / "arena_questions_base.jsonl"
    if not data_file.exists():
        issues.append("❌ arena_questions_base.jsonl not found")
    else:
        with open(data_file) as f:
            count = sum(1 for _ in f)
        print(f"✅ Found arena_questions_base.jsonl with {count:,} prompts")

    # Check CJE import
    try:
        from cje.utils import RobustTeacherForcing

        print("✅ CJE package imports successfully")
    except ImportError as e:
        issues.append(f"❌ CJE import error: {e}")

    # Check scripts exist
    scripts = [
        "01_prepare_data.py",
        "02a_generate_p0_responses.py",
        "02b_generate_target_responses.py",
        "02c_compute_target_logprobs.py",
        "03_generate_oracle_labels.py",
        "04a_deterministic_judge_scores.py",
    ]

    missing_scripts = []
    for script in scripts:
        if not (Path(__file__).parent / script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        issues.append(f"❌ Missing scripts: {', '.join(missing_scripts)}")
    else:
        print(f"✅ All {len(scripts)} pipeline scripts found")

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("❌ NOT READY - Issues found:\n")
        for issue in issues:
            print(f"   {issue}")
        print("\nTo fix:")
        print("1. Run: source ./set_secrets.sh")
        print("2. Ensure you're in the correct directory")
        return False
    else:
        print("✅ READY TO RUN!")
        print("\nNext steps:")
        print("1. cd sample_run")
        print("2. ./run_sample.sh")
        print("\nOr just run: ./run_sample_test.sh")
        return True


if __name__ == "__main__":
    ready = check_ready()
    sys.exit(0 if ready else 1)
