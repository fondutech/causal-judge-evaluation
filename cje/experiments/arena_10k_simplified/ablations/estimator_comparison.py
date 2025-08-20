#!/usr/bin/env python3
"""Systematic estimator comparison ablation.

This ablation compares estimators systematically to show:
1. Impact of self-normalization (IPS vs SNIPS)
2. Impact of calibration (SNIPS vs Cal-IPS, DR vs Cal-DR)
3. Impact of stacking (individual DR vs stacked)

Estimators compared:
- IPS: Raw importance sampling (unnormalized)
- SNIPS: Self-normalized IPS (Hajek estimator)
- Cal-IPS: Calibrated IPS (SIMCal)
- DR-CPO: Basic doubly robust
- Cal-DR-CPO: Calibrated DR (uses Cal-IPS weights)
- Stacked-DR: Optimal combination of DR methods
- Cal-Stacked-DR: Calibrated stacked DR
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

import sys

sys.path.append(str(Path(__file__).parent.parent))

from core import ExperimentSpec
from core.base import BaseAblation
from core.schemas import aggregate_results

# Import CJE components directly for custom configurations
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import CalibratedIPS
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.stacking import StackedDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EstimatorConfig:
    """Configuration for an estimator variant."""

    name: str
    display_name: str
    estimator_class: str
    use_calibration: bool
    weight_mode: str = "hajek"  # "raw" or "hajek"
    is_dr: bool = False
    is_stacked: bool = False


class EstimatorComparison(BaseAblation):
    """Systematic comparison of estimation methods."""

    def __init__(self) -> None:
        super().__init__(
            name="estimator_comparison",
            cache_dir=Path("../.ablation_cache/estimator_comparison"),
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define estimators to compare
        self.estimator_configs = [
            EstimatorConfig("ips", "IPS", "ips", False, "raw"),
            EstimatorConfig("snips", "SNIPS", "ips", False, "hajek"),
            EstimatorConfig("cal-ips", "Cal-IPS", "calibrated-ips", True, "hajek"),
            EstimatorConfig("dr-cpo", "DR-CPO", "dr-cpo", False, "hajek", is_dr=True),
            EstimatorConfig(
                "cal-dr-cpo", "Cal-DR-CPO", "dr-cpo", True, "hajek", is_dr=True
            ),
            EstimatorConfig(
                "stacked-dr",
                "Stacked-DR",
                "stacked",
                False,
                "hajek",
                is_dr=True,
                is_stacked=True,
            ),
            EstimatorConfig(
                "cal-stacked-dr",
                "Cal-Stacked-DR",
                "stacked",
                True,
                "hajek",
                is_dr=True,
                is_stacked=True,
            ),
        ]

    def create_custom_estimator(
        self, config: EstimatorConfig, sampler: PrecomputedSampler, cal_result: Any
    ) -> Union[CalibratedIPS, DRCPOEstimator, StackedDREstimator]:
        """Create estimator with specific configuration."""

        if config.estimator_class == "ips":
            # Raw or self-normalized IPS
            return CalibratedIPS(sampler, calibrate=False)

        elif config.estimator_class == "calibrated-ips":
            # Calibrated IPS (always uses SIMCal)
            return CalibratedIPS(sampler, calibrate=True)

        elif config.estimator_class == "dr-cpo":
            # DR-CPO with or without calibrated weights
            estimator = DRCPOEstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                n_folds=5,
                use_calibrated_weights=config.use_calibration,
            )
            return estimator

        elif config.estimator_class == "stacked":
            # Stacked DR with or without calibration
            # Note: StackedDR doesn't have use_calibrated_weights param
            # The calibration is controlled by passing calibrator to component estimators
            estimator = StackedDREstimator(
                sampler,
                estimators=["dr-cpo", "tmle", "mrdr"],
                V_folds=5,
                parallel=True,
                # Pass calibrator via kwargs for component estimators
                calibrator=(
                    cal_result.calibrator
                    if cal_result and config.use_calibration
                    else None
                ),
            )
            return estimator

        else:
            raise ValueError(f"Unknown estimator class: {config.estimator_class}")

    def run_single_comparison(
        self, spec: ExperimentSpec, config: EstimatorConfig, seed: int
    ) -> Dict[str, Any]:
        """Run a single estimator configuration."""

        # Check cache
        cache_key = f"{spec.uid()}_{config.name}_{seed}"
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except:
                pass

        # Initialize result
        result = {
            "spec": spec.__dict__,
            "config": config.name,
            "display_name": config.display_name,
            "seed": seed,
            "success": False,
        }

        try:
            # Load and prepare data
            np.random.seed(seed)
            dataset = load_dataset_from_jsonl(spec.dataset_path)

            # Subsample if requested
            if spec.sample_size:
                n = min(spec.sample_size, len(dataset.samples))
                indices = sorted(
                    np.random.choice(len(dataset.samples), n, replace=False)
                )
                dataset.samples = [dataset.samples[i] for i in indices]

            # Mask oracle labels for calibration
            n_samples = len(dataset.samples)
            if spec.oracle_coverage and spec.oracle_coverage < 1.0:
                oracle_indices = [
                    i
                    for i, s in enumerate(dataset.samples)
                    if s.metadata.get("oracle_label") is not None
                ]
                n_keep = max(2, int(len(oracle_indices) * spec.oracle_coverage))
                keep_indices = set(
                    np.random.choice(oracle_indices, n_keep, replace=False)
                )

                for i, sample in enumerate(dataset.samples):
                    if i not in keep_indices and "oracle_label" in sample.metadata:
                        sample.metadata["oracle_label"] = None

            # Calibrate dataset
            calibrated_dataset, cal_result = calibrate_dataset(
                dataset,
                judge_field="judge_score",
                oracle_field="oracle_label",
                enable_cross_fit=True,
                n_folds=5,
            )

            # Create sampler with appropriate weight mode
            sampler = PrecomputedSampler(calibrated_dataset)

            # Override weight computation for IPS vs SNIPS
            if config.weight_mode == "raw" and config.estimator_class == "ips":
                # Monkey-patch to use raw weights instead of Hajek
                # Note: CalibratedIPS.get_raw_weights already passes mode="raw"
                # So we need to intercept and handle that case
                original_method = sampler.compute_importance_weights

                def raw_weights(policy: str, **kwargs: Any) -> Any:
                    # Remove mode from kwargs to avoid duplicate since we'll set it
                    kwargs.pop("mode", None)
                    # Always use raw mode for IPS
                    return original_method(policy, mode="raw", **kwargs)

                sampler.compute_importance_weights = raw_weights  # type: ignore[assignment]

            # Create and run estimator
            estimator = self.create_custom_estimator(config, sampler, cal_result)

            # Add fresh draws for DR methods
            if config.is_dr:
                data_dir = Path(spec.dataset_path).parent
                for policy in sampler.target_policies:
                    try:
                        fresh_draws = load_fresh_draws_auto(
                            data_dir, policy, verbose=False
                        )
                        estimator.add_fresh_draws(policy, fresh_draws)
                    except:
                        pass

            # Run estimation
            import time

            start_time = time.time()
            estimation_result = estimator.fit_and_estimate()
            runtime = time.time() - start_time

            # Extract results
            result["estimates"] = {
                policy: float(estimation_result.estimates[i])
                for i, policy in enumerate(sampler.target_policies)
            }
            result["standard_errors"] = {
                policy: float(estimation_result.standard_errors[i])
                for i, policy in enumerate(sampler.target_policies)
            }
            result["runtime"] = runtime
            result["n_samples"] = n_samples
            result["success"] = True

            # Add estimator-specific diagnostics
            if hasattr(estimator, "weights_per_policy") and config.is_stacked:
                # Stacking weights
                result["stacking_weights"] = {
                    policy: weights.tolist()
                    for policy, weights in estimator.weights_per_policy.items()
                }

            # Compute ESS for IPS methods
            if not config.is_dr:
                ess_values = {}
                for policy in sampler.target_policies:
                    try:
                        weights = sampler.compute_importance_weights(policy)
                        ess = np.sum(weights) ** 2 / np.sum(weights**2)
                        ess_values[policy] = float(ess)
                    except:
                        pass
                result["ess"] = ess_values

        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"Failed {config.name}: {e}")

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
        except:
            pass

        return result

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run systematic comparison across all estimators."""

        # Test scenarios
        scenarios = [
            {"n": 1000, "oracle": 0.05, "label": "Low Oracle (5%)"},
            {"n": 1000, "oracle": 0.10, "label": "Medium Oracle (10%)"},
            {"n": 1000, "oracle": 0.20, "label": "High Oracle (20%)"},
            {"n": 2000, "oracle": 0.10, "label": "Large Sample"},
        ]

        n_seeds = 3  # Multiple seeds for stability

        logger.info("=" * 70)
        logger.info("ESTIMATOR COMPARISON ABLATION")
        logger.info("=" * 70)
        logger.info(f"Estimators: {[c.display_name for c in self.estimator_configs]}")
        logger.info(f"Scenarios: {len(scenarios)}")
        logger.info(f"Seeds: {n_seeds}")
        logger.info("")

        all_results = []

        for scenario in scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(
                f"Scenario: {scenario['label']} (n={scenario['n']}, oracle={scenario['oracle']:.0%})"
            )
            logger.info(f"{'='*60}")

            for config in self.estimator_configs:
                logger.info(f"\n{config.display_name}:")

                # Determine correct data path
                data_path = Path("../data/cje_dataset.jsonl")
                if not data_path.exists():
                    data_path = Path("../../data/cje_dataset.jsonl")

                # Create spec with n_seeds instead of individual seed
                spec = ExperimentSpec(
                    ablation="estimator_comparison",
                    dataset_path=str(data_path),
                    estimator=config.name,
                    oracle_coverage=scenario["oracle"],
                    sample_size=scenario["n"],
                    n_seeds=n_seeds,
                    seed_base=42,
                )

                # Run with multiple seeds using the base class method
                for seed_offset in range(n_seeds):
                    seed = 42 + seed_offset

                    result = self.run_single_comparison(spec, config, seed)
                    result["scenario"] = scenario["label"]
                    all_results.append(result)

                    if result["success"]:
                        # Show mean estimate and SE
                        mean_est = np.mean(list(result["estimates"].values()))
                        mean_se = np.mean(list(result["standard_errors"].values()))
                        logger.info(
                            f"  Seed {seed}: {mean_est:.3f} ± {mean_se:.3f} ({result['runtime']:.1f}s)"
                        )
                    else:
                        logger.info(f"  Seed {seed}: FAILED")

        # Save all results
        output_dir = Path("../ablations/results/estimator_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.jsonl", "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

        logger.info(f"\nSaved {len(all_results)} results to {output_dir}")

        return all_results

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison results."""

        # Group by scenario and estimator
        by_scenario: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for r in results:
            if r.get("success", False):
                scenario = r["scenario"]
                estimator = r["display_name"]

                if scenario not in by_scenario:
                    by_scenario[scenario] = {}
                if estimator not in by_scenario[scenario]:
                    by_scenario[scenario][estimator] = []

                # Compute mean RMSE (would need oracle truth)
                mean_se = np.mean(list(r["standard_errors"].values()))
                by_scenario[scenario][estimator].append(
                    {
                        "se": mean_se,
                        "runtime": r["runtime"],
                        "ess": (
                            np.mean(list(r.get("ess", {}).values()))
                            if "ess" in r
                            else None
                        ),
                    }
                )

        # Compute rankings
        rankings = {}
        for scenario, estimators in by_scenario.items():
            scenario_rankings = []

            for est_name, runs in estimators.items():
                if runs:
                    mean_se = np.mean([r["se"] for r in runs])
                    mean_runtime = np.mean([r["runtime"] for r in runs])
                    mean_ess = np.mean([r["ess"] for r in runs if r["ess"] is not None])

                    scenario_rankings.append(
                        {
                            "estimator": est_name,
                            "mean_se": mean_se,
                            "mean_runtime": mean_runtime,
                            "mean_ess": mean_ess if not np.isnan(mean_ess) else None,
                        }
                    )

            # Sort by SE (lower is better)
            scenario_rankings.sort(key=lambda x: x["mean_se"])
            rankings[scenario] = scenario_rankings

        return rankings

    def create_figure(
        self, results: List[Dict[str, Any]], output_path: Optional[Path] = None
    ) -> Any:
        """Create comparison figure."""

        rankings = self.analyze_results(results)

        if not rankings:
            logger.warning("No results to plot")
            return

        # Create figure with subplots for each scenario
        n_scenarios = len(rankings)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (scenario, ranking) in enumerate(rankings.items()):
            if idx >= 4:
                break

            ax = axes[idx]

            # Extract data
            estimators = [r["estimator"] for r in ranking]
            ses = [r["mean_se"] for r in ranking]

            # Color by type
            colors = []
            for est in estimators:
                if "Cal-" in est:
                    colors.append("green")  # Calibrated
                elif "Stacked" in est:
                    colors.append("purple")  # Stacked
                elif "DR" in est:
                    colors.append("orange")  # DR
                else:
                    colors.append("blue")  # IPS variants

            # Create bar chart
            bars = ax.barh(range(len(estimators)), ses, color=colors)
            ax.set_yticks(range(len(estimators)))
            ax.set_yticklabels(estimators)
            ax.set_xlabel("Standard Error")
            ax.set_title(scenario)
            ax.invert_yaxis()  # Best at top

            # Add values on bars
            for i, (bar, se) in enumerate(zip(bars, ses)):
                ax.text(se, i, f" {se:.4f}", va="center")

        plt.suptitle(
            "Estimator Comparison: Standard Error by Scenario",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {output_path}")

        plt.show()

        return fig


def main() -> List[Dict[str, Any]]:
    """Run estimator comparison."""

    comparison = EstimatorComparison()
    results = comparison.run_ablation()

    # Analyze
    rankings = comparison.analyze_results(results)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    for scenario, ranking in rankings.items():
        logger.info(f"\n{scenario}:")
        logger.info("-" * 40)
        for i, r in enumerate(ranking, 1):
            logger.info(
                f"{i}. {r['estimator']:15s}: SE={r['mean_se']:.4f}, Runtime={r['mean_runtime']:.1f}s"
            )

    # Create figure
    figure_path = Path(
        "../ablations/results/estimator_comparison/comparison_figure.png"
    )
    comparison.create_figure(results, figure_path)

    logger.info("\n" + "=" * 70)
    logger.info("ESTIMATOR COMPARISON COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
