#!/usr/bin/env python3
"""Test the modular analysis pipeline.

This tests the refactored modular architecture to ensure:
1. All modules can be imported
2. Basic functionality works
3. The orchestrator runs successfully
"""

import sys
import tempfile
import json
from pathlib import Path


def test_module_imports() -> bool:
    """Test that all analysis modules can be imported."""
    print("Testing module imports...")

    try:
        from analysis import (
            load_data,
            handle_rewards,
            restore_oracle_labels,
            create_estimator,
            display_results,
            display_weight_diagnostics,
            display_dr_diagnostics,
            analyze_extreme_weights_report,
            generate_visualizations,
            export_results,
        )

        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_data_loading() -> bool:
    """Test that data loading works with the test dataset."""
    print("\nTesting data loading...")

    from analysis.loading import load_data

    # Check if test data exists
    test_data_path = "data copy/cje_dataset.jsonl"
    if not Path(test_data_path).exists():
        print(f"⚠️  Test data not found at {test_data_path}")
        return False

    try:
        dataset = load_data(test_data_path, verbose=False)
        print(f"✅ Loaded {dataset.n_samples} samples")
        print(f"   Policies: {dataset.target_policies}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def test_orchestrator_runs() -> bool:
    """Test that the main orchestrator runs without errors."""
    print("\nTesting orchestrator with minimal config...")

    import subprocess

    # Run with minimal config that should complete quickly
    cmd = [
        sys.executable,
        "analyze_dataset.py",
        "--data",
        "data copy/cje_dataset.jsonl",
        "--estimator",
        "calibrated-ips",
        "--oracle-coverage",
        "0.5",
        "--no-plots",
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Orchestrator completed successfully")
            # Check for expected output
            if "Analysis complete!" in result.stdout:
                print("   Found success message in output")
            return True
        else:
            print(f"❌ Orchestrator failed with exit code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Orchestrator timed out (>30s)")
        return False
    except Exception as e:
        print(f"❌ Failed to run orchestrator: {e}")
        return False


def test_export_functionality() -> bool:
    """Test that export functionality works."""
    print("\nTesting export functionality...")

    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        sys.executable,
        "analyze_dataset.py",
        "--data",
        "data copy/cje_dataset.jsonl",
        "--estimator",
        "calibrated-ips",
        "--oracle-coverage",
        "0.5",
        "--no-plots",
        "--quiet",
        "--output",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"❌ Export run failed: {result.returncode}")
            return False

        # Check if file was created and has content
        output_file = Path(output_path)
        if not output_file.exists():
            print("❌ Output file was not created")
            return False

        # Try to load the JSON
        with open(output_path) as f:
            data = json.load(f)

        # Check for expected fields
        expected_fields = ["timestamp", "dataset", "estimation", "best_policy"]
        missing = [f for f in expected_fields if f not in data]

        if missing:
            print(f"❌ Output missing fields: {missing}")
            return False

        print("✅ Export functionality works")
        print(f"   Best policy: {data.get('best_policy')}")

        # Clean up
        output_file.unlink()
        return True

    except Exception as e:
        print(f"❌ Export test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("Testing Modular Analysis Pipeline")
    print("=" * 60)

    tests = [
        test_module_imports,
        test_data_loading,
        test_orchestrator_runs,
        test_export_functionality,
    ]

    passed = 0
    failed = 0

    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
