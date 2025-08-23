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
        super().__init__(name="estimator_comparison")

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
            # For non-calibrated Stacked-DR, we should NOT pass calibrator
            # For Cal-Stacked-DR, we pass calibrator and it will be used
            if config.use_calibration and cal_result:
                # Cal-Stacked-DR: pass calibrator for component estimators to use
                estimator = StackedDREstimator(
                    sampler,
                    estimators=["dr-cpo", "tmle", "mrdr"],
                    V_folds=5,
                    parallel=True,
                    calibrator=cal_result.calibrator,
                )
            else:
                # Stacked-DR: no calibrator, components will use raw weights
                estimator = StackedDREstimator(
                    sampler,
                    estimators=["dr-cpo", "tmle", "mrdr"],
                    V_folds=5,
                    parallel=True,
                    # Explicitly no calibrator - components will use raw weights
                )
            return estimator

        else:
            raise ValueError(f"Unknown estimator class: {config.estimator_class}")

    def run_single_comparison(
        self, spec: ExperimentSpec, config: EstimatorConfig, seed: int
    ) -> Dict[str, Any]:
        """Run a single estimator configuration."""

        # Initialize result
        result = {
            "spec": spec if isinstance(spec, dict) else spec.__dict__,
            "config": config.name,
            "display_name": config.display_name,
            "seed": seed,
            "success": False,
        }

        try:
            # Load and prepare data
            np.random.seed(seed)
            dataset_path = (
                spec["dataset_path"] if isinstance(spec, dict) else spec.dataset_path
            )
            dataset = load_dataset_from_jsonl(dataset_path)

            # Subsample if requested
            sample_size = (
                spec.get("sample_size") if isinstance(spec, dict) else spec.sample_size
            )
            if sample_size:
                n = min(sample_size, len(dataset.samples))
                indices = sorted(
                    np.random.choice(len(dataset.samples), n, replace=False)
                )
                dataset.samples = [dataset.samples[i] for i in indices]

            # Mask oracle labels for calibration
            n_samples = len(dataset.samples)
            oracle_coverage = (
                spec.get("oracle_coverage")
                if isinstance(spec, dict)
                else spec.oracle_coverage
            )
            if oracle_coverage and oracle_coverage < 1.0:
                oracle_indices = [
                    i
                    for i, s in enumerate(dataset.samples)
                    if s.metadata.get("oracle_label") is not None
                ]
                n_keep = max(2, int(len(oracle_indices) * oracle_coverage))
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
                data_dir = Path(dataset_path).parent
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

            # Compute ESS for all methods (IPS and DR both use importance weights)
            ess_values = {}
            for policy in sampler.target_policies:
                try:
                    weights = None

                    # For DR estimators, get weights from their internal IPS estimator
                    if hasattr(estimator, "ips_estimator") and hasattr(
                        estimator.ips_estimator, "get_weights"
                    ):
                        weights = estimator.ips_estimator.get_weights(policy)
                    # For IPS-based estimators, get weights directly
                    elif hasattr(estimator, "get_weights"):
                        weights = estimator.get_weights(policy)
                    # Check weight cache
                    elif (
                        hasattr(estimator, "_weights_cache")
                        and policy in estimator._weights_cache
                    ):
                        weights = estimator._weights_cache[policy]

                    # If no weights from estimator, get raw weights from sampler
                    if weights is None:
                        weights = sampler.compute_importance_weights(policy)

                    if weights is not None:
                        ess = np.sum(weights) ** 2 / np.sum(weights**2)
                        ess_values[policy] = float(ess)
                except:
                    pass
            result["ess"] = ess_values

        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"Failed {config.name}: {e}")

        return result

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run systematic comparison across all estimators."""

        # Test scenarios: sample size x oracle coverage
        sample_sizes = [500, 1000, 2500, 5000]
        oracle_coverages = [0.05, 0.10, 0.25, 1.00]

        scenarios = []
        for n in sample_sizes:
            for oracle in oracle_coverages:
                scenarios.append(
                    {"n": n, "oracle": oracle, "label": f"n={n}, oracle={oracle:.0%}"}
                )

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

                # Determine correct data path - use absolute path to be safe
                base_dir = Path(__file__).parent.parent
                data_path = base_dir / "data" / "cje_dataset.jsonl"
                if not data_path.exists():
                    # Fallback to relative path from current directory
                    data_path = Path("../data/cje_dataset.jsonl").resolve()

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
        output_dir = Path("results/estimator_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON."""
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            return obj

        with open(output_dir / "results.jsonl", "w") as f:
            for result in all_results:
                f.write(json.dumps(convert_numpy(result)) + "\n")

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

    def load_oracle_means(self) -> Dict[str, float]:
        """Load oracle truth means for each policy from response files.

        Returns:
            Dictionary mapping policy name to oracle mean
        """
        import json
        from pathlib import Path

        oracle_means = {}

        # Policy name mapping from file names
        response_files = {
            "clone": "../data/responses/clone_responses.jsonl",
            "parallel_universe_prompt": "../data/responses/parallel_universe_prompt_responses.jsonl",
            "premium": "../data/responses/premium_responses.jsonl",
            "unhelpful": "../data/responses/unhelpful_responses.jsonl",
        }

        for policy, file_path in response_files.items():
            file_path = Path(file_path)
            if file_path.exists():
                oracle_values = []
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            data = json.loads(line)
                            oracle_label = data.get("metadata", {}).get("oracle_label")
                            if oracle_label is not None:
                                oracle_values.append(oracle_label)

                    if oracle_values:
                        oracle_means[policy] = np.mean(oracle_values)
                        logger.info(
                            f"Loaded {len(oracle_values)} oracle values for {policy}, mean: {oracle_means[policy]:.3f}"
                        )
                    else:
                        logger.warning(f"No oracle values found in {file_path}")
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
            else:
                logger.warning(f"Response file not found: {file_path}")

        return oracle_means

    def analyze_by_policy(self, results: List[Dict[str, Any]]) -> "pd.DataFrame":
        """Extract per-policy performance metrics from estimator comparison results.

        Creates a DataFrame with one row per (method, policy) combination showing:
        - Standard error
        - ESS (effective sample size)
        - Estimates and oracle truth values if available

        Args:
            results: List of estimator comparison results

        Returns:
            DataFrame with columns: Method, Policy, Estimate, SE, ESS_%, Oracle_Truth, Error
        """
        import pandas as pd

        # Load oracle truth means
        oracle_means = self.load_oracle_means()

        rows = []
        for r in results:
            if r.get("success", False) and "estimates" in r:
                method = r.get("display_name", r.get("config", "unknown"))

                for policy in r.get("estimates", {}).keys():
                    # Get absolute ESS and calculate percentage
                    ess_abs = r.get("ess", {}).get(policy, np.nan)
                    n_samples = r.get("n_samples", 1000)  # Default fallback
                    ess_pct = (
                        (ess_abs / n_samples * 100) if not np.isnan(ess_abs) else np.nan
                    )

                    # Get oracle truth from loaded means
                    oracle_truth = oracle_means.get(policy, np.nan)
                    estimate = r["estimates"].get(policy, np.nan)

                    row = {
                        "Method": method,
                        "Policy": policy,
                        "Estimate": estimate,
                        "SE": r.get("standard_errors", {}).get(policy, np.nan),
                        "ESS_abs": ess_abs,
                        "ESS_%": ess_pct,
                        "Oracle_Truth": oracle_truth,
                    }

                    # Calculate absolute error if both values available
                    if not np.isnan(oracle_truth) and not np.isnan(estimate):
                        row["Abs_Error"] = abs(estimate - oracle_truth)
                    else:
                        row["Abs_Error"] = np.nan

                    rows.append(row)

        return pd.DataFrame(rows)

    def create_policy_heterogeneity_figure(
        self,
        results: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
        scenario_label: Optional[str] = None,
        color_by: str = "se",
    ) -> Any:
        """Create heatmap showing SE or Abs Error by (method × policy) to demonstrate heterogeneity.

        Shows how different policies require different estimation methods based on
        their distribution shift from the base policy. Can color by either standard
        error or absolute error.

        Args:
            results: List of estimator comparison results
            output_path: Optional path to save figure
            scenario_label: Optional label describing the scenario
            color_by: "se" for standard error (default) or "error" for absolute error

        Returns:
            matplotlib figure
        """
        import pandas as pd

        df = self.analyze_by_policy(results)

        if df.empty:
            logger.warning("No policy-specific data to visualize")
            return None

        # Create matrices for different metrics
        se_matrix = df.pivot_table(
            index="Method", columns="Policy", values="SE", aggfunc="mean"
        )

        ess_matrix = df.pivot_table(
            index="Method", columns="Policy", values="ESS_%", aggfunc="mean"
        )

        # Get absolute error matrix
        abs_error_matrix = df.pivot_table(
            index="Method", columns="Policy", values="Abs_Error", aggfunc="mean"
        )

        # Reorder methods to the desired order
        method_order = [
            "IPS",
            "SNIPS",
            "Cal-IPS",
            "DR-CPO",
            "Cal-DR-CPO",
            "Stacked-DR",
            "Cal-Stacked-DR",
        ]
        # Keep only methods that exist in the data
        existing_methods = [m for m in method_order if m in se_matrix.index]

        # Reindex all matrices with the desired order
        se_matrix = se_matrix.reindex(existing_methods)
        ess_matrix = ess_matrix.reindex(existing_methods)
        abs_error_matrix = abs_error_matrix.reindex(existing_methods)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Choose the matrix and scale based on color_by parameter
        if color_by.lower() == "error":
            # Use absolute error for coloring
            color_matrix = abs_error_matrix
            color_vals = color_matrix.values
            color_vals_clean = np.where(
                np.isnan(color_vals) | (color_vals < 0), 0, color_vals
            )  # Replace NaN/negative with 0

            # Fixed scale for absolute error: 0 to 0.05
            vmin = 0.0
            vmax = 0.05

            color_label = "Absolute Error"
            print(
                f"Fixed color scale: Absolute Error from {vmin} to {vmax} (linear scale)"
            )
        else:
            # Default: use standard error for coloring
            color_matrix = se_matrix
            color_vals = color_matrix.values
            color_vals_clean = np.where(
                np.isnan(color_vals) | (color_vals <= 0), 0, color_vals
            )  # Replace NaN/negative with 0

            # Fixed scale for SE: 0 to 0.1
            vmin = 0.0
            vmax = 0.1

            color_label = "Standard Error"
            print(
                f"Fixed color scale: Standard Error from {vmin} to {vmax} (linear scale)"
            )

        # Clip values to the fixed range
        vals_for_color = np.clip(color_vals_clean, vmin, vmax)

        # Create heatmap with fixed linear scale
        im = ax.imshow(
            vals_for_color, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(se_matrix.columns)))
        ax.set_yticks(np.arange(len(se_matrix.index)))
        ax.set_xticklabels(se_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(se_matrix.index)

        # Add colorbar with fixed linear scale
        cbar = plt.colorbar(im, ax=ax)
        if color_by.lower() == "error":
            cbar.set_label(f"{color_label} (0.0 to {vmax})", rotation=270, labelpad=20)
        else:
            cbar.set_label(f"{color_label} (0.0 to {vmax})", rotation=270, labelpad=20)

        # Add text annotations showing SE and ESS%
        for i in range(len(se_matrix.index)):
            for j in range(len(se_matrix.columns)):
                method = se_matrix.index[i]
                policy = se_matrix.columns[j]

                se_val = se_matrix.loc[method, policy]

                # Safely get ESS and error values using loc to avoid index errors
                ess_pct_val = np.nan
                abs_error_val = np.nan

                if (
                    not ess_matrix.empty
                    and method in ess_matrix.index
                    and policy in ess_matrix.columns
                ):
                    ess_pct_val = ess_matrix.loc[method, policy]

                if (
                    not abs_error_matrix.empty
                    and method in abs_error_matrix.index
                    and policy in abs_error_matrix.columns
                ):
                    abs_error_val = abs_error_matrix.loc[method, policy]

                if not np.isnan(se_val):
                    # Handle zero/near-zero SEs (likely numerical precision issues)
                    if se_val < 1e-6:
                        se_display = "<0.001"
                    else:
                        se_display = f"{se_val:.3f}"

                    # Build annotation text with SE, ESS%, and absolute error
                    text_lines = [f"SE: {se_display}"]

                    if not np.isnan(ess_pct_val):
                        text_lines.append(f"ESS: {ess_pct_val:.1f}%")

                    if not np.isnan(abs_error_val):
                        text_lines.append(f"Err: {abs_error_val:.3f}")

                    text = "\n".join(text_lines)

                    # Choose text color based on background brightness
                    # Get the actual value used for coloring
                    if color_by.lower() == "error":
                        color_val = abs_error_val if not np.isnan(abs_error_val) else 0
                        # For error scale (0 to 0.05): midpoint is 0.025
                        threshold = 0.025
                    else:
                        color_val = se_val
                        # For SE scale (0 to 0.1): midpoint is 0.05
                        threshold = 0.05

                    # Values above threshold are reddish (need white text)
                    # Values below threshold are greenish/yellow (need black text)
                    color = "white" if color_val > threshold else "black"

                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                        fontweight="bold",
                    )

        # Extract scenario info from results if not provided
        if scenario_label is None and results:
            # Try to extract from first result
            first_result = results[0]
            if "spec" in first_result:
                sample_size = first_result["spec"].get("sample_size", "unknown")
                oracle_coverage = first_result["spec"].get("oracle_coverage", "unknown")
                if oracle_coverage != "unknown":
                    oracle_pct = (
                        f"{oracle_coverage:.0%}"
                        if oracle_coverage >= 1
                        else f"{oracle_coverage*100:.0f}%"
                    )
                    scenario_label = f"n={sample_size}, oracle={oracle_pct}"

        if scenario_label:
            metric_name = (
                "Absolute Error" if color_by.lower() == "error" else "Standard Error"
            )
            title = f"Policy Heterogeneity ({metric_name}): {scenario_label}"
        else:
            metric_name = (
                "Absolute Error" if color_by.lower() == "error" else "Standard Error"
            )
            title = f"Policy Heterogeneity: {metric_name} by Method and Policy"

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Target Policy", fontsize=12)
        ax.set_ylabel("Estimation Method", fontsize=12)

        plt.tight_layout()

        # Generate default output path if not provided
        if output_path is None:
            # Use consistent naming: _by_se for standard error, _by_abs_error for absolute error
            suffix = "_by_abs_error" if color_by.lower() == "error" else "_by_se"
            if results and "spec" in results[0]:
                sample_size = results[0]["spec"].get("sample_size", "unknown")
                oracle_coverage = results[0]["spec"].get("oracle_coverage", "unknown")
                if oracle_coverage != "unknown":
                    oracle_pct = (
                        int(oracle_coverage * 100)
                        if oracle_coverage < 1
                        else int(oracle_coverage)
                    )
                    output_path = Path(
                        f"results/estimator_comparison/policy_heterogeneity_n{sample_size}_oracle{oracle_pct}pct{suffix}.png"
                    )
                else:
                    output_path = Path(
                        f"results/estimator_comparison/policy_heterogeneity{suffix}.png"
                    )
            else:
                output_path = Path(
                    f"results/estimator_comparison/policy_heterogeneity{suffix}.png"
                )

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved policy heterogeneity figure to {output_path}")

        # Print insights
        metric_name = (
            "Absolute Error" if color_by.lower() == "error" else "Standard Error"
        )
        metric_abbrev = "Err" if color_by.lower() == "error" else "SE"
        logger.info(f"\nPolicy Heterogeneity Insights ({metric_name}):")
        logger.info("-" * 40)

        # Find best method per policy based on the selected metric
        metric_matrix = abs_error_matrix if color_by.lower() == "error" else se_matrix
        for policy in metric_matrix.columns:
            policy_vals = metric_matrix[policy].dropna()
            if not policy_vals.empty:
                best_method = policy_vals.idxmin()
                best_val = policy_vals.min()
                worst_val = policy_vals.max()
                improvement = (
                    (worst_val - best_val) / worst_val * 100 if worst_val > 0 else 0
                )
                logger.info(f"{policy}:")
                logger.info(
                    f"  Best method: {best_method} ({metric_abbrev}={best_val:.3f})"
                )
                logger.info(f"  Improvement over worst: {improvement:.0f}%")

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

    # Create figures
    figure_path = Path("results/estimator_comparison/comparison_figure.png")
    comparison.create_figure(results, figure_path)

    # Create policy heterogeneity analysis
    policy_fig_path = Path("results/estimator_comparison/policy_heterogeneity.png")
    comparison.create_policy_heterogeneity_figure(results, policy_fig_path)

    logger.info("\n" + "=" * 70)
    logger.info("ESTIMATOR COMPARISON COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
