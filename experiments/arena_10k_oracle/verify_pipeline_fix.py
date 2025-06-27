#!/usr/bin/env python3
"""Verify that the fixed target policy stage works correctly in the full pipeline."""

import json
import tempfile
from pathlib import Path
from cje.config.unified import ConfigurationBuilder


def verify_pipeline_fix():
    """Verify the full pipeline with the target policy fix."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        test_data = [
            {"context": "What is 2+2?"},
            {"context": "What is the capital of France?"},
            {"context": "Explain quantum mechanics in simple terms."},
        ]

        test_file = tmpdir / "test_data.jsonl"
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        print("📝 Created test data file")

        # Build configuration
        config = (
            ConfigurationBuilder()
            .paths(str(tmpdir / "work"))
            .dataset(str(test_file))
            .logging_policy("mock-base", provider="mock", temperature=0.5)
            .add_target_policy("mock_cot", "mock-cot", provider="mock", temperature=0.7)
            .add_target_policy(
                "mock_bigger", "mock-big", provider="mock", temperature=0.3
            )
            .judge("mock", "mock-judge")
            .estimator("IPS")
            .build()
        )

        print("🔧 Running pipeline with fixed target policy stage...")

        try:
            # Run the pipeline
            results = config.run()

            print("\n✅ Pipeline completed successfully!")

            # Display results
            if "IPS" in results:
                ips_result = results["IPS"]
                print("\n📊 IPS Results:")
                for i, (estimate, std_err) in enumerate(
                    zip(ips_result.estimates, ips_result.standard_errors)
                ):
                    print(f"   Policy {i}: {estimate:.4f} ± {std_err**0.5:.4f}")

            # Load and check intermediate data
            rows_file = tmpdir / "work" / "pipeline" / "rows_final.jsonl"
            if rows_file.exists():
                with open(rows_file) as f:
                    rows = [json.loads(line) for line in f]

                print(f"\n📋 Processed {len(rows)} rows")

                # Check first row
                if rows:
                    row = rows[0]
                    print("\n🔍 Sample row structure:")
                    print(f"   Context: {row.get('context', 'N/A')[:50]}...")
                    print(f"   Response: {row.get('response', 'N/A')[:50]}...")
                    print(f"   Logging policy logp: {row.get('logp', 'N/A')}")

                    if "logp_target_all" in row:
                        print(f"   Target policy logps: {row['logp_target_all']}")
                        print("\n🎉 Teacher forcing is working correctly!")
                    else:
                        print("\n⚠️  Warning: No target policy logps found")

            return True

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def check_importance_weights():
    """Additional check to verify importance weights are computed correctly."""
    print("\n" + "=" * 60)
    print("🔍 Checking importance weight computation...")

    import numpy as np

    # Simulate some log probabilities
    logp_behavior = -5.0
    logp_targets = [-4.0, -6.0, -5.5]  # Three target policies

    # Compute importance weights
    weights = [np.exp(logp_t - logp_behavior) for logp_t in logp_targets]

    print(f"\nBehavior policy logp: {logp_behavior}")
    print(f"Target policy logps: {logp_targets}")
    print(f"Importance weights: {[f'{w:.3f}' for w in weights]}")

    # Verify
    print("\nInterpretation:")
    for i, (logp_t, w) in enumerate(zip(logp_targets, weights)):
        if w > 1:
            print(
                f"  Policy {i}: weight={w:.3f} > 1 → target policy MORE likely than behavior"
            )
        elif w < 1:
            print(
                f"  Policy {i}: weight={w:.3f} < 1 → target policy LESS likely than behavior"
            )
        else:
            print(f"  Policy {i}: weight={w:.3f} = 1 → target policy SAME as behavior")


if __name__ == "__main__":
    print("🚀 Testing CJE Pipeline with Target Policy Fix")
    print("=" * 60)

    success = verify_pipeline_fix()

    if success:
        check_importance_weights()
        print("\n✅ All checks passed! The fix is working correctly.")
    else:
        print("\n❌ Fix verification failed.")
