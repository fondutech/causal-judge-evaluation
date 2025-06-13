"""Interactive Arena Analysis Interface.

Use this for notebook-style arena analysis with the enhanced CJE pipeline.
Provides a simple Python API that wraps the pipeline interface.
"""

from typing import Dict, Any, Optional, List
from cje.pipeline import run_pipeline
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ArenaAnalyzer:
    """Interactive arena analysis with the enhanced CJE pipeline."""

    def __init__(self, work_dir: Optional[str] = None):
        """Initialize analyzer.

        Args:
            work_dir: Optional work directory for results
        """
        self.work_dir = Path(work_dir) if work_dir else Path("arena_results")
        self.results: Optional[Dict[str, Any]] = None
        self.oracle_data: Optional[List[Dict[str, Any]]] = None

    def run_analysis(
        self,
        config_name: str = "arena_analysis",
        estimator: str = "DRCPO",
        oracle_fraction: Optional[float] = None,
        **config_overrides: Any,
    ) -> Dict[str, Any]:
        """Run arena analysis with specified parameters.

        Args:
            config_name: Configuration to use
            estimator: Estimator type (DRCPO, IPS, SNIPS, MRDR)
            oracle_fraction: Oracle fraction for ablation study (overrides config)
            **config_overrides: Additional config overrides

        Returns:
            Analysis results dictionary
        """
        print(f"🏟️ Running arena analysis...")
        print(f"   Config: {config_name}")
        print(f"   Estimator: {estimator}")
        if config_overrides:
            print(f"   Overrides: {config_overrides}")

        try:
            # Get absolute path to configs directory
            # Try multiple potential locations
            script_dir = Path(__file__).parent.resolve()
            project_root = script_dir.parent  # Go up from examples/ to project root
            configs_dir = project_root / "configs"

            if not configs_dir.exists():
                # Try current working directory
                configs_dir = Path.cwd() / "configs"

            if not configs_dir.exists():
                raise FileNotFoundError(
                    f"Configs directory not found. Looked in: {project_root / 'configs'} and {Path.cwd() / 'configs'}"
                )

            print(f"   Using configs from: {configs_dir}")

            # Override oracle fraction if specified
            if oracle_fraction is not None:
                config_overrides["oracle.logging_policy_oracle_fraction"] = (
                    oracle_fraction
                )

            # This automatically uses our enhanced APIPolicyRunner
            self.results = run_pipeline(cfg_path=str(configs_dir), cfg_name=config_name)

            print("✅ Analysis complete!")
            self._print_summary()

            # Check if we have oracle evaluation results
            if "oracle_evaluation" in self.results:
                self._print_oracle_evaluation()

            return self.results

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

    def run_oracle_ablation_study(
        self,
        config_name: str = "arena_oracle_demo",
        oracle_fractions: Optional[List[float]] = None,
        estimator: str = "DRCPO",
    ) -> pd.DataFrame:
        """Run ablation study varying oracle fraction available to logging policy.

        This studies how the amount of oracle data available for calibration
        affects CJE's estimation accuracy.

        Args:
            config_name: Configuration to use (should have oracle.enabled=true)
            oracle_fractions: List of oracle fractions to test
            estimator: Estimator to use

        Returns:
            DataFrame with ablation study results
        """
        if oracle_fractions is None:
            oracle_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        print(f"🧪 Running oracle ablation study...")
        print(f"   Config: {config_name}")
        print(f"   Oracle fractions: {oracle_fractions}")
        print(f"   Estimator: {estimator}")

        results_data = []

        for fraction in oracle_fractions:
            print(f"\n📊 Testing oracle fraction: {fraction:.1%}")

            try:
                # Run analysis with this oracle fraction
                result = self.run_analysis(
                    config_name=config_name,
                    estimator=estimator,
                    oracle_fraction=fraction,
                )

                # Extract oracle evaluation metrics if available
                if "oracle_evaluation" in result:
                    oracle_eval = result["oracle_evaluation"]

                    # Handle both new calibration format and legacy format
                    rmse = oracle_eval.get(
                        "calibration_rmse", oracle_eval.get("rmse", float("nan"))
                    )
                    mae = oracle_eval.get(
                        "calibration_mae", oracle_eval.get("mae", float("nan"))
                    )

                    results_data.append(
                        {
                            "oracle_fraction": fraction,
                            "rmse": rmse,
                            "mae": mae,
                            "n_policies": oracle_eval.get("n_policies_evaluated", 0),
                            "n_held_out": oracle_eval.get("n_held_out_samples", 0),
                            "estimator": estimator,
                        }
                    )

                    print(f"   RMSE: {rmse:.4f}")
                    print(f"   MAE: {mae:.4f}")
                else:
                    print(f"   ⚠️ No oracle evaluation available")

            except Exception as e:
                print(f"   ❌ Failed with oracle fraction {fraction}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(results_data)

        if not df.empty:
            print(f"\n📈 Ablation study complete!")
            print(f"   Tested {len(df)} oracle fractions")
            print(
                f"   Best RMSE: {df['rmse'].min():.4f} at fraction {df.loc[df['rmse'].idxmin(), 'oracle_fraction']:.1%}"
            )

            # Plot results
            self.plot_oracle_ablation(df)
        else:
            print(f"\n❌ No successful runs in ablation study")

        return df

    def plot_oracle_ablation(
        self, ablation_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Plot oracle ablation study results.

        Args:
            ablation_df: DataFrame from run_oracle_ablation_study()
            save_path: Optional path to save the plot
        """
        if ablation_df.empty:
            print("No ablation data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot RMSE
        ax1.plot(
            ablation_df["oracle_fraction"],
            ablation_df["rmse"],
            "o-",
            linewidth=2,
            markersize=8,
        )
        ax1.set_xlabel("Oracle Fraction Available to Logging Policy")
        ax1.set_ylabel("RMSE vs. Oracle Truth")
        ax1.set_title("CJE Estimation Error vs. Oracle Fraction")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)

        # Plot MAE
        ax2.plot(
            ablation_df["oracle_fraction"],
            ablation_df["mae"],
            "o-",
            color="orange",
            linewidth=2,
            markersize=8,
        )
        ax2.set_xlabel("Oracle Fraction Available to Logging Policy")
        ax2.set_ylabel("MAE vs. Oracle Truth")
        ax2.set_title("Mean Absolute Error vs. Oracle Fraction")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Ablation plot saved to: {save_path}")

        plt.show()

    def quick_test(self) -> Dict[str, Any]:
        """Run a quick test analysis."""
        print("🚀 Running quick test...")
        return self.run_analysis("arena_quick", estimator="DRCPO")

    def _print_summary(self) -> None:
        """Print analysis summary."""
        if not self.results:
            return

        analysis_type = self.results.get("analysis_type", "unknown")
        interpretation = self.results.get("interpretation", "No interpretation")

        print(f"\n📊 Analysis Summary:")
        print(f"   Type: {analysis_type}")
        print(f"   Result: {interpretation}")

        if "v_hat" in self.results:
            v_hat = self.results["v_hat"]
            se = self.results.get("se", [])
            v_hat_logging = self.results.get("v_hat_logging")
            policy_names = self.results.get("policy_names", [])

            print(f"\n📈 Policy Estimates:")

            # Show logging policy value first
            if v_hat_logging is not None:
                se_logging = self.results.get("se_logging", 0.0)
                if se_logging > 0:
                    print(f"   Logging Policy: {v_hat_logging:.3f} ± {se_logging:.3f}")
                else:
                    print(f"   Logging Policy: {v_hat_logging:.3f}")

            # Show target policy values
            if isinstance(v_hat, list):
                target_names = (
                    policy_names[1:]
                    if len(policy_names) > 1
                    else [f"Policy {i}" for i in range(len(v_hat))]
                )
                for i, estimate in enumerate(v_hat):
                    stderr = se[i] if i < len(se) else 0.0
                    policy_name = (
                        target_names[i] if i < len(target_names) else f"Policy {i}"
                    )
                    print(f"   {policy_name}: {estimate:.3f} ± {stderr:.3f}")

                    # Show improvement over logging policy
                    if v_hat_logging is not None:
                        improvement = estimate - v_hat_logging
                        print(f"      (Δ vs logging: {improvement:+.3f})")
            else:
                stderr = se if isinstance(se, (int, float)) else 0.0
                policy_name = (
                    policy_names[1] if len(policy_names) > 1 else "Target Policy"
                )
                print(f"   {policy_name}: {v_hat:.3f} ± {stderr:.3f}")

                # Show improvement over logging policy
                if v_hat_logging is not None:
                    improvement = v_hat - v_hat_logging
                    print(f"      (Δ vs logging: {improvement:+.3f})")

        # Add weight diagnostics if available
        self._print_weight_diagnostics()

    def _print_oracle_evaluation(self) -> None:
        """Print oracle evaluation results if available."""
        if not self.results or "oracle_evaluation" not in self.results:
            return

        oracle_eval = self.results["oracle_evaluation"]

        print(f"\n🎯 Oracle Calibration Evaluation Results:")
        print(
            f"   RMSE: {oracle_eval.get('calibration_rmse', oracle_eval.get('rmse', 'N/A')):.4f}"
        )
        print(
            f"   MAE: {oracle_eval.get('calibration_mae', oracle_eval.get('mae', 'N/A')):.4f}"
        )

        # Show calibration details if available
        if "n_held_out_samples" in oracle_eval:
            print(f"   Held-out samples: {oracle_eval.get('n_held_out_samples', 0)}")
            print(
                f"   Mean oracle score: {oracle_eval.get('mean_oracle_score', 0.0):.3f}"
            )
            print(
                f"   Mean calibrated score: {oracle_eval.get('mean_calibrated_score', 0.0):.3f}"
            )

            # Show oracle truth comparison with logging policy baseline
            v_hat_logging = self.results.get("v_hat_logging")
            mean_oracle = oracle_eval.get("mean_oracle_score")

            if v_hat_logging is not None and mean_oracle is not None:
                print(f"\n   📊 Oracle Truth Comparison:")
                print(f"     Logging policy (oracle truth): {mean_oracle:.3f}")
                print(f"     Logging policy (DR estimate): {v_hat_logging:.3f}")
                se_logging = self.results.get("se_logging", 0.0)
                if se_logging > 0:
                    print(f"     DR standard error: ±{se_logging:.3f}")
                print(f"     Oracle-DR Gap: {abs(mean_oracle - v_hat_logging):.3f}")

        # Legacy support for old oracle evaluation format
        if oracle_eval.get("n_policies_evaluated", 0) > 0:
            print(
                f"   Policies evaluated: {oracle_eval.get('n_policies_evaluated', 0)}"
            )

            # Show per-policy comparison if available
            oracle_true = oracle_eval.get("oracle_true_values", {})
            cje_estimates = oracle_eval.get("cje_estimates", {})
            absolute_errors = oracle_eval.get("absolute_errors", {})

            if oracle_true and cje_estimates:
                print(f"\n📊 Per-Policy Comparison:")
                for policy in oracle_true.keys():
                    oracle_val = oracle_true.get(policy, 0.0)
                    cje_val = cje_estimates.get(policy, 0.0)
                    abs_error = absolute_errors.get(policy, 0.0)
                    print(
                        f"   {policy}: Oracle={oracle_val:.3f}, CJE={cje_val:.3f}, Error={abs_error:.3f}"
                    )

    def get_estimates(self) -> Optional[List[float]]:
        """Get policy estimates as a list."""
        if not self.results or "v_hat" not in self.results:
            return None

        v_hat = self.results["v_hat"]
        return v_hat if isinstance(v_hat, list) else [v_hat]

    def get_standard_errors(self) -> Optional[List[float]]:
        """Get standard errors as a list."""
        if not self.results or "se" not in self.results:
            return None

        se = self.results["se"]
        return se if isinstance(se, list) else [se]

    def get_oracle_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get oracle evaluation results if available."""
        if not self.results:
            return None
        oracle_eval: Optional[Dict[str, Any]] = self.results.get(
            "oracle_evaluation"
        )  # Explicit type annotation
        return oracle_eval

    def _print_weight_diagnostics(self) -> None:
        """Print importance weight diagnostics."""
        try:
            from cje.utils.weight_diagnostics import (
                analyze_arena_weights,
                create_weight_summary_table,
            )

            arena_data = self._load_arena_data()
            if not arena_data:
                return

            diagnostics = analyze_arena_weights(arena_data)
            if not diagnostics:
                return

            print(f"\n🔍 Importance Weight Diagnostics:")

            # Print summary table
            summary_table = create_weight_summary_table(diagnostics)
            print(summary_table)

            # Check for issues
            critical_policies = [
                name
                for name, diag in diagnostics.items()
                if diag.consistency_flag == "CRITICAL"
            ]
            warning_policies = [
                name
                for name, diag in diagnostics.items()
                if diag.consistency_flag == "WARNING"
            ]

            if critical_policies:
                print(f"\n❌ CRITICAL weight issues: {', '.join(critical_policies)}")
                print(
                    "   → May indicate teacher forcing problems or policy inconsistencies"
                )
            elif warning_policies:
                print(f"\n⚠️  Weight warnings: {', '.join(warning_policies)}")
                print("   → Consider investigating low ESS or extreme weights")
            else:
                print(f"\n✅ All importance weights look healthy")

        except ImportError:
            print("\n⚠️  Weight diagnostics require: pip install numpy")
        except Exception as e:
            print(f"\n⚠️  Could not compute weight diagnostics: {e}")

    def _load_arena_data(self) -> Optional[List[Dict[str, Any]]]:
        """Load arena data for weight analysis."""
        if not self.results:
            return None

        # Try to find arena data file
        work_dir = self.results.get("metadata", {}).get("work_dir")
        if not work_dir:
            return None

        import json
        from pathlib import Path

        # Look for arena data files
        work_path = Path(work_dir)
        arena_files = list(
            work_path.glob("**/quick_judge_scores_cal_with_target_logp.jsonl")
        )
        if not arena_files:
            arena_files = list(work_path.glob("**/*target_logp*.jsonl"))

        if not arena_files:
            return None

        try:
            with open(arena_files[0], "r") as f:
                data: List[Dict[str, Any]] = [
                    json.loads(line) for line in f.readlines()
                ]  # Explicit type annotation
                return data  # Explicitly return the loaded data
        except Exception:
            return None

    def plot_weight_diagnostics(
        self, save_plots: bool = False, output_dir: Optional[str] = None
    ) -> None:
        """Create weight diagnostic plots."""
        try:
            from cje.utils.weight_plots import create_weight_diagnostic_dashboard

            arena_data = self._load_arena_data()
            if not arena_data:
                print("❌ No arena data available for plotting")
                return

            if output_dir is None and save_plots:
                output_dir = str(self.work_dir / "weight_diagnostics")

            print("📊 Creating weight diagnostic plots...")
            saved_files = create_weight_diagnostic_dashboard(
                arena_data,
                output_dir=output_dir if save_plots else None,
                show_plots=not save_plots,
            )

            if saved_files:
                print(f"✅ Weight diagnostics saved!")

        except ImportError:
            print("⚠️  Weight plotting requires: pip install matplotlib numpy")
        except Exception as e:
            print(f"❌ Failed to create weight plots: {e}")

    def quick_weight_check(self, policy_name: Optional[str] = None) -> None:
        """Quick visual check of importance weights."""
        try:
            from cje.utils.weight_plots import quick_weight_check

            arena_data = self._load_arena_data()
            if not arena_data:
                print("❌ No arena data available for weight check")
                return

            quick_weight_check(arena_data, policy_name)

        except ImportError:
            print("⚠️  Weight plotting requires: pip install matplotlib numpy")
        except Exception as e:
            print(f"❌ Failed to create weight check: {e}")

    def diagnose_weights(self) -> Optional[Dict[str, Any]]:
        """Get detailed weight diagnostics."""
        try:
            from cje.utils.weight_diagnostics import analyze_arena_weights

            arena_data = self._load_arena_data()
            if not arena_data:
                return None

            return analyze_arena_weights(arena_data)

        except ImportError:
            print("⚠️  Weight diagnostics require: pip install numpy")
            return None
        except Exception as e:
            print(f"❌ Failed to diagnose weights: {e}")
            return None

    def plot_estimates(self, title: str = "Policy Estimates") -> None:
        """Plot policy estimates with error bars."""
        estimates = self.get_estimates()
        std_errors = self.get_standard_errors()

        if not estimates:
            print("No estimates to plot")
            return

        if not std_errors:
            std_errors = [0.0] * len(estimates)

        fig, ax = plt.subplots(figsize=(8, 6))

        policies = [f"Policy {i}" for i in range(len(estimates))]

        bars = ax.bar(
            policies,
            estimates,
            yerr=std_errors,
            capsize=5,
            alpha=0.7,
            color="skyblue",
            edgecolor="navy",
        )

        ax.set_title(title)
        ax.set_ylabel("Estimate")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, est, se in zip(bars, estimates, std_errors):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + se,
                f"{est:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    def plot_oracle_comparison(self) -> None:
        """Plot comparison between CJE estimates and oracle truth."""
        oracle_eval = self.get_oracle_evaluation()

        if not oracle_eval:
            print("No oracle evaluation results to plot")
            return

        oracle_true = oracle_eval.get("oracle_true_values", {})
        cje_estimates = oracle_eval.get("cje_estimates", {})

        if not oracle_true or not cje_estimates:
            print("Missing oracle comparison data")
            return

        # Extract values for plotting
        policies = list(oracle_true.keys())
        oracle_values = [oracle_true[p] for p in policies]
        cje_values = [cje_estimates[p] for p in policies]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax1.scatter(oracle_values, cje_values, s=100, alpha=0.7)
        ax1.plot(
            [min(oracle_values), max(oracle_values)],
            [min(oracle_values), max(oracle_values)],
            "r--",
            label="Perfect agreement",
        )
        ax1.set_xlabel("Oracle Truth")
        ax1.set_ylabel("CJE Estimate")
        ax1.set_title("CJE vs. Oracle Truth")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bar plot comparison
        x = range(len(policies))
        width = 0.35
        ax2.bar(
            [i - width / 2 for i in x],
            oracle_values,
            width,
            label="Oracle Truth",
            alpha=0.7,
        )
        ax2.bar(
            [i + width / 2 for i in x],
            cje_values,
            width,
            label="CJE Estimate",
            alpha=0.7,
        )
        ax2.set_xlabel("Policy")
        ax2.set_ylabel("Value")
        ax2.set_title("Policy Value Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(policies, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_estimators(
        self,
        estimators: List[str] = ["DRCPO", "IPS", "SNIPS", "MRDR"],
        config_name: str = "arena_analysis",
    ) -> pd.DataFrame:
        """Compare multiple estimators."""

        print(f"🔄 Comparing estimators: {estimators}")

        results_data = []

        for estimator in estimators:
            print(f"   Running {estimator}...")
            try:
                result = self.run_analysis(config_name, estimator=estimator)

                estimates = self.get_estimates()
                std_errors = self.get_standard_errors()

                if estimates:
                    for i, (est, se) in enumerate(
                        zip(estimates, std_errors or [0] * len(estimates))
                    ):
                        results_data.append(
                            {
                                "Estimator": estimator,
                                "Policy": i,
                                "Estimate": est,
                                "StdError": se,
                                "CI_Lower": est - 1.96 * se,
                                "CI_Upper": est + 1.96 * se,
                            }
                        )

            except Exception as e:
                print(f"   ❌ {estimator} failed: {e}")
                continue

        df = pd.DataFrame(results_data)

        if not df.empty:
            print("\n📋 Comparison Results:")
            print(df)

            # Plot comparison
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="Policy", y="Estimate", hue="Estimator")
            plt.title("Estimator Comparison")
            plt.show()

        return df

    def save_results(self, filename: Optional[str] = None) -> None:
        """Save results to JSON file."""
        if not self.results:
            print("No results to save")
            return

        if not filename:
            filename = (
                f"arena_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        filepath = self.work_dir / filename
        self.work_dir.mkdir(exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"💾 Results saved to {filepath}")


# Convenience functions for one-liners
def quick_arena_test() -> Dict[str, Any]:
    """Quick arena test."""
    analyzer = ArenaAnalyzer()
    return analyzer.quick_test()


def run_arena_analysis(
    config_name: str = "arena_analysis", **kwargs: Any
) -> Dict[str, Any]:
    """Run arena analysis with specified config."""
    analyzer = ArenaAnalyzer()
    return analyzer.run_analysis(config_name, **kwargs)


def run_oracle_ablation_study(
    config_name: str = "arena_oracle_demo",
    oracle_fractions: Optional[List[float]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run oracle ablation study."""
    analyzer = ArenaAnalyzer()
    return analyzer.run_oracle_ablation_study(config_name, oracle_fractions, **kwargs)


def compare_estimators(estimators: List[str] = ["DRCPO", "IPS"]) -> pd.DataFrame:
    """Compare multiple estimators."""
    analyzer = ArenaAnalyzer()
    return analyzer.compare_estimators(estimators)


# Example usage for notebooks
if __name__ == "__main__":
    # Example interactive usage
    analyzer = ArenaAnalyzer()

    # Quick test
    print("Running quick test...")
    analyzer.quick_test()

    # Plot results
    analyzer.plot_estimates("Quick Test Results")

    # Compare estimators
    print("\nComparing estimators...")
    df = analyzer.compare_estimators(["DRCPO", "IPS"])

    # Save results
    analyzer.save_results()
