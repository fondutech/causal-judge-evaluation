#!/usr/bin/env python3
"""Quick test to verify ablation infrastructure works."""

import logging
from pathlib import Path
from core import ExperimentSpec, create_result
from core.diagnostics import effective_sample_size, hill_alpha, compute_rmse
from core.gates import check_gates, apply_mitigation_ladder
from core.base import BaseAblation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class QuickTestAblation(BaseAblation):
    """Minimal ablation for testing infrastructure."""

    def run_ablation(self):
        """Run a single quick test."""

        # Define minimal experiment
        spec = ExperimentSpec(
            ablation="quick_test",
            dataset_path="../data/cje_dataset.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=0.1,  # 10% oracle
            sample_fraction=0.02,  # 2% of data (~100 samples)
            n_seeds=1,  # Just one seed for quick test
            seed_base=42,
        )

        logger.info("=" * 60)
        logger.info("QUICK TEST ABLATION")
        logger.info("=" * 60)
        logger.info(f"Dataset: {spec.dataset_path}")
        logger.info(f"Estimator: {spec.estimator}")
        logger.info(f"Oracle coverage: {spec.oracle_coverage:.0%}")
        logger.info(f"Sample fraction: {spec.sample_fraction:.0%}")
        logger.info("")

        # Run experiment
        results = self.run_with_seeds(spec)

        # Check results
        if results and results[0]["success"]:
            result = results[0]
            logger.info("✓ Experiment successful!")
            logger.info(f"  Samples used: {result['n_samples']}")
            logger.info(f"  Oracle labels: {result['n_oracle']}")
            logger.info(f"  RMSE vs oracle: {result.get('rmse_vs_oracle', 'N/A'):.4f}")
            logger.info(f"  Mean CI width: {result.get('mean_ci_width', 'N/A'):.4f}")

            # Show some diagnostics
            if result.get("ess_absolute"):
                logger.info("\nEffective Sample Sizes:")
                for policy, ess in result["ess_absolute"].items():
                    ess_pct = result["ess_relative"].get(policy, 0)
                    logger.info(f"  {policy}: {ess:.0f} ({ess_pct:.1f}%)")

            # Show gate status
            if result.get("gate_status"):
                logger.info("\nGate Status:")
                for policy, gates in result["gate_status"].items():
                    status = "✓ PASS" if gates.get("all_pass") else "✗ FAIL"
                    logger.info(f"  {policy}: {status}")
                    if not gates.get("all_pass"):
                        failures = [
                            k for k, v in gates.items() if k != "all_pass" and not v
                        ]
                        logger.info(f"    Failed: {', '.join(failures)}")
        else:
            logger.error("✗ Experiment failed!")
            if results:
                logger.error(f"  Error: {results[0].get('error', 'Unknown')}")

        return results


def test_diagnostics():
    """Test diagnostic functions directly."""
    import numpy as np

    logger.info("\nTesting diagnostic functions...")

    # Create some test weights
    weights_good = np.ones(1000)  # Uniform weights
    weights_bad = np.concatenate([np.ones(990), np.ones(10) * 100])  # Heavy tail

    # Test ESS
    ess_good = effective_sample_size(weights_good)
    ess_bad = effective_sample_size(weights_bad)
    logger.info(f"ESS (uniform): {ess_good:.0f}")
    logger.info(f"ESS (heavy tail): {ess_bad:.0f}")

    # Test Hill alpha
    alpha_good = hill_alpha(weights_good)
    alpha_bad = hill_alpha(weights_bad)
    logger.info(f"Hill α (uniform): {alpha_good:.2f}")
    logger.info(f"Hill α (heavy tail): {alpha_bad:.2f}")

    # Test gates
    gates_good = check_gates(weights_good)
    gates_bad = check_gates(weights_bad)
    logger.info(f"Gates (uniform): {gates_good['all_pass']}")
    logger.info(f"Gates (heavy tail): {gates_bad['all_pass']}")

    # Test mitigation
    if not gates_bad["all_pass"]:
        logger.info("\nTrying mitigations on bad weights...")
        weights_mitigated, info = apply_mitigation_ladder(weights_bad, verbose=True)
        logger.info(f"Mitigation used: {info['method']}")
        logger.info(f"ESS after: {info.get('ess_after', 'N/A'):.0f}")


def main():
    """Run all tests."""

    # Test diagnostics
    test_diagnostics()

    # Test full ablation
    logger.info("\n" + "=" * 60)
    ablation = QuickTestAblation("quick_test")
    results = ablation.run_ablation()

    logger.info("\n" + "=" * 60)
    logger.info("QUICK TEST COMPLETE")
    logger.info("=" * 60)

    # Return success status
    return results[0]["success"] if results else False


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
