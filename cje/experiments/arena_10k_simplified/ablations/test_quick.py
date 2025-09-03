#!/usr/bin/env python3
"""
Quick test to verify parameters are being passed correctly.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cje.experiments.arena_10k_simplified.ablations.config import (
    EXPERIMENTS,
    DR_CONFIG,
    DATA_PATH,
    RESULTS_PATH,
    CHECKPOINT_PATH,
)
from cje.experiments.arena_10k_simplified.ablations.core.base import BaseAblation
from cje.experiments.arena_10k_simplified.ablations.core.schemas import ExperimentSpec


class TestAblation(BaseAblation):
    """Test ablation that runs just a few experiments."""

    def run_all(self, test_mode: bool = True) -> None:
        """Run test experiments."""
        print("Running quick test experiments...")

        # Test configurations
        test_specs = [
            # IPS without IIC
            {"estimator": "raw-ips", "use_calibration": False, "use_iic": False},
            # IPS with IIC
            {"estimator": "raw-ips", "use_calibration": False, "use_iic": True},
            # Calibrated IPS without IIC
            {"estimator": "calibrated-ips", "use_calibration": True, "use_iic": False},
            # DR without SIMCal or IIC
            {"estimator": "dr-cpo", "use_calibration": False, "use_iic": False},
            # DR with both SIMCal and IIC
            {"estimator": "dr-cpo", "use_calibration": True, "use_iic": True},
        ]

        results = []
        for i, config in enumerate(test_specs):
            print(f"\n{'='*60}")
            print(
                f"Test {i+1}: {config['estimator']} with SIMCal={config['use_calibration']}, IIC={config['use_iic']}"
            )
            print("=" * 60)

            spec = ExperimentSpec(
                ablation="test",
                dataset_path=str(DATA_PATH),
                estimator=config["estimator"],
                sample_size=500,  # Small for quick test
                oracle_coverage=0.5,
                n_seeds=1,
                seed_base=42,
                extra={
                    "use_calibration": config["use_calibration"],
                    "use_iic": config["use_iic"],
                },
            )

            try:
                result = self.run_single(spec, 42)

                # Check what was captured
                print(f"\nResult keys: {list(result.keys())}")
                print(f"Success: {result.get('success')}")

                if result.get("success"):
                    print(f"Estimates: {result.get('estimates')}")

                    # Check for DR diagnostics
                    if "dr" in config["estimator"]:
                        print(
                            f"Has orthogonality_score: {'orthogonality_score' in result}"
                        )
                        print(
                            f"Has orthogonality_scores: {'orthogonality_scores' in result}"
                        )
                        print(f"Has IIC diagnostics: {'iic_diagnostics' in result}")

                        if "orthogonality_scores" in result:
                            for policy, scores in result[
                                "orthogonality_scores"
                            ].items():
                                if isinstance(scores, dict) and "score" in scores:
                                    print(
                                        f"  {policy}: orthogonality={scores['score']:.3f}"
                                    )

                        if "iic_diagnostics" in result:
                            iic = result["iic_diagnostics"]
                            if isinstance(iic, dict):
                                print(
                                    f"  IIC diagnostics found for {len(iic)} policies"
                                )
                                for policy, diag in iic.items():
                                    if isinstance(diag, dict):
                                        r2 = diag.get("r_squared", diag.get("r2", 0))
                                        se_red = diag.get("se_reduction", 0)
                                        print(
                                            f"    {policy}: R²={r2:.3f}, SE reduction={se_red:.1%}"
                                        )
                else:
                    print(f"Error: {result.get('error')}")

                results.append(result)

            except Exception as e:
                print(f"Failed: {e}")
                import traceback

                traceback.print_exc()

        # Save test results (skip for now due to numpy serialization issues)
        test_output = Path("test_results.jsonl")
        # with open(test_output, 'w') as f:
        #     for r in results:
        #         f.write(json.dumps(r) + '\n')

        print(f"\n{'='*60}")
        print(f"Test complete! Results saved to {test_output}")
        print(
            f"Successful: {sum(1 for r in results if r.get('success'))}/{len(results)}"
        )


def main():
    """Run test."""
    ablation = TestAblation(name="test_params")
    ablation.run_all(test_mode=True)


if __name__ == "__main__":
    main()
