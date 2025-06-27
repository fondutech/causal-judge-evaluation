#!/usr/bin/env python3
"""Test that the fixed target policy stage works correctly."""

import tempfile
from pathlib import Path
from cje.config.unified import simple_config
from cje.testing.mocks.multi_target_sampler import create_mock_multi_sampler


def test_pipeline_with_fix():
    """Test the full pipeline with the target policy fix."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple test config
        config = simple_config(
            dataset_name="test.jsonl",
            dataset_data=[
                {"context": "What is 2+2?"},
                {"context": "What is the capital of France?"},
            ],
            logging_policy_name="mock-base",
            logging_policy_provider="mock",
            judge_model="mock-judge",
            judge_provider="mock",
            target_policies=[
                {"name": "mock_cot", "model_name": "mock-cot", "provider": "mock"},
                {"name": "mock_bigger", "model_name": "mock-big", "provider": "mock"},
            ],
            work_dir=tmpdir,
        )

        print("🔧 Running pipeline with fixed target policy stage...")

        try:
            # Run the pipeline
            results = config.run()

            print("✅ Pipeline completed successfully!")
            print(f"Results: {results}")

            # Check that we got reasonable results
            assert "simple_ips" in results
            assert len(results["simple_ips"].estimates) == 2  # Two target policies

            print("\n🎉 The target policy fix works correctly!")

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    test_pipeline_with_fix()
