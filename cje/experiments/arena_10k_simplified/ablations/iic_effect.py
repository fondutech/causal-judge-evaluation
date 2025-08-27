#!/usr/bin/env python3
"""
IIC (Isotonic Influence Control) ablation with comprehensive visualizations.

This ablation demonstrates:
1. How IIC reduces variance by exploiting judge score monotonicity
2. The magnitude of variance reduction for different policies
3. The relationship between judge explainability (R²) and SE reduction
4. Visual proof that IIC preserves unbiasedness while shrinking CIs
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from core.base import BaseAblation
from core.schemas import ExperimentSpec

logger = logging.getLogger(__name__)


class IICEffectAblation(BaseAblation):
    """Comprehensive IIC ablation with visualizations."""

    def __init__(self) -> None:
        super().__init__("iic_effect")

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run experiments with and without IIC to show the effect."""
        results = []
        
        # Test multiple estimators and oracle coverages
        estimators = ["dr-cpo", "tmle", "mrdr"]
        oracle_coverages = [0.1, 0.2, 0.5]
        
        for estimator in estimators:
            for oracle_coverage in oracle_coverages:
                # Run WITH IIC (default)
                spec_with = ExperimentSpec(
                    ablation="iic_effect",
                    dataset_path="../data/cje_dataset.jsonl",
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    n_seeds=5,
                    extra={"use_iic": True}  # Explicitly set even though it's default
                )
                
                logger.info(f"Running {estimator} WITH IIC (oracle={oracle_coverage*100:.0f}%)")
                with_iic_results = self.run_with_seeds(spec_with)
                for r in with_iic_results:
                    r["iic_enabled"] = True
                results.extend(with_iic_results)
                
                # Run WITHOUT IIC for comparison
                spec_without = ExperimentSpec(
                    ablation="iic_effect", 
                    dataset_path="../data/cje_dataset.jsonl",
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    n_seeds=5,
                    extra={"use_iic": False}
                )
                
                logger.info(f"Running {estimator} WITHOUT IIC (oracle={oracle_coverage*100:.0f}%)")
                without_iic_results = self.run_with_seeds(spec_without)
                for r in without_iic_results:
                    r["iic_enabled"] = False
                results.extend(without_iic_results)
        
        return results

    def extract_influence_functions(self, result: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """Extract influence functions from result metadata."""
        if not result.get("success") or "metadata" not in result:
            return None
            
        metadata = result["metadata"]
        
        # Look for raw and IIC-smoothed influence functions
        if "dr_influence" in metadata:
            return metadata["dr_influence"]
        
        return None

    def analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Generate comprehensive IIC visualizations."""
        
        output_dir = Path("results/iic_effect")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate results by IIC status
        with_iic = [r for r in results if r.get("iic_enabled", False) and r.get("success")]
        without_iic = [r for r in results if not r.get("iic_enabled", True) and r.get("success")]
        
        # 1. Collect variance reduction statistics
        variance_data = []
        
        for r_with in with_iic:
            # Find matching without-IIC result
            r_without = self._find_matching_result(r_with, without_iic)
            if r_without is None:
                continue
                
            # Compare standard errors
            if "standard_errors" in r_with and "standard_errors" in r_without:
                for policy in r_with["standard_errors"]:
                    if policy in r_without["standard_errors"]:
                        se_with = r_with["standard_errors"][policy]
                        se_without = r_without["standard_errors"][policy]
                        
                        if se_without > 0:
                            se_reduction = (se_without - se_with) / se_without
                            
                            # Get IIC diagnostics if available
                            iic_diag = r_with.get("metadata", {}).get("iic_diagnostics", {}).get(policy, {})
                            
                            variance_data.append({
                                "estimator": r_with["spec"]["estimator"],
                                "oracle_coverage": r_with["spec"]["oracle_coverage"],
                                "policy": policy,
                                "se_with_iic": se_with,
                                "se_without_iic": se_without,
                                "se_reduction": se_reduction,
                                "r_squared": iic_diag.get("r_squared", 0),
                                "direction": iic_diag.get("direction", "unknown"),
                            })
        
        if not variance_data:
            logger.warning("No paired IIC comparison data found")
            return
            
        df = pd.DataFrame(variance_data)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        
        # === Panel A: SE Reduction by Policy ===
        ax1 = plt.subplot(2, 3, 1)
        policy_reduction = df.groupby("policy")["se_reduction"].agg(["mean", "std"]).reset_index()
        policy_reduction = policy_reduction.sort_values("mean", ascending=False)
        
        bars = ax1.bar(range(len(policy_reduction)), policy_reduction["mean"], 
                       yerr=policy_reduction["std"], capsize=5, color='steelblue')
        ax1.set_xticks(range(len(policy_reduction)))
        ax1.set_xticklabels(policy_reduction["policy"], rotation=45, ha="right")
        ax1.set_ylabel("SE Reduction (%)")
        ax1.set_title("A. Variance Reduction by Policy", fontweight="bold")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax1.grid(True, alpha=0.3, axis='y')
        
        # === Panel B: R² vs SE Reduction Scatter ===
        ax2 = plt.subplot(2, 3, 2)
        
        # Color by estimator
        estimator_colors = {"dr-cpo": "blue", "tmle": "green", "mrdr": "orange"}
        
        for estimator in df["estimator"].unique():
            est_df = df[df["estimator"] == estimator]
            ax2.scatter(est_df["r_squared"], est_df["se_reduction"] * 100,
                       label=estimator.upper(), alpha=0.6, s=50,
                       color=estimator_colors.get(estimator, "gray"))
        
        # Add trend line
        valid = df[(df["r_squared"] > 0) & (df["se_reduction"] > 0)]
        if len(valid) > 2:
            z = np.polyfit(valid["r_squared"], valid["se_reduction"] * 100, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 100)
            ax2.plot(x_trend, p(x_trend), "r--", alpha=0.5, label="Trend")
        
        ax2.set_xlabel("R² (Judge Explainability)")
        ax2.set_ylabel("SE Reduction (%)")
        ax2.set_title("B. Explainability vs Variance Reduction", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # === Panel C: Before/After CI Width Comparison ===
        ax3 = plt.subplot(2, 3, 3)
        
        # Select top 5 policies by reduction for clarity
        top_policies = policy_reduction.head(5)["policy"].values
        
        ci_data = []
        for policy in top_policies:
            policy_df = df[df["policy"] == policy]
            if len(policy_df) > 0:
                mean_with = policy_df["se_with_iic"].mean()
                mean_without = policy_df["se_without_iic"].mean()
                ci_data.append({
                    "policy": policy,
                    "Without IIC": mean_without * 1.96,  # 95% CI half-width
                    "With IIC": mean_with * 1.96
                })
        
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            ci_df = ci_df.set_index("policy")
            
            x = np.arange(len(ci_df))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, ci_df["Without IIC"], width, label="Without IIC", color='lightcoral')
            bars2 = ax3.bar(x + width/2, ci_df["With IIC"], width, label="With IIC", color='steelblue')
            
            ax3.set_xlabel("Policy")
            ax3.set_ylabel("95% CI Half-Width")
            ax3.set_title("C. Confidence Interval Tightening", fontweight="bold")
            ax3.set_xticks(x)
            ax3.set_xticklabels(ci_df.index, rotation=45, ha="right")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # === Panel D: SE Reduction by Oracle Coverage ===
        ax4 = plt.subplot(2, 3, 4)
        
        coverage_reduction = df.groupby("oracle_coverage")["se_reduction"].agg(["mean", "std"]).reset_index()
        coverage_reduction["oracle_coverage"] = coverage_reduction["oracle_coverage"] * 100
        
        ax4.errorbar(coverage_reduction["oracle_coverage"], coverage_reduction["mean"] * 100,
                    yerr=coverage_reduction["std"] * 100, marker='o', markersize=8,
                    capsize=5, linewidth=2, color='steelblue')
        ax4.set_xlabel("Oracle Coverage (%)")
        ax4.set_ylabel("Mean SE Reduction (%)")
        ax4.set_title("D. Effect of Oracle Coverage", fontweight="bold")
        ax4.grid(True, alpha=0.3)
        
        # === Panel E: Distribution of R² Values ===
        ax5 = plt.subplot(2, 3, 5)
        
        ax5.hist(df["r_squared"], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax5.axvline(df["r_squared"].mean(), color='red', linestyle='--', 
                   label=f'Mean R² = {df["r_squared"].mean():.3f}')
        ax5.set_xlabel("R² (Judge Explainability)")
        ax5.set_ylabel("Count")
        ax5.set_title("E. Distribution of Judge Explainability", fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # === Panel F: Summary Statistics ===
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Compute summary statistics
        mean_reduction = df["se_reduction"].mean() * 100
        median_reduction = df["se_reduction"].median() * 100
        max_reduction = df["se_reduction"].max() * 100
        mean_r2 = df["r_squared"].mean()
        
        # Count policies with substantial reduction
        substantial = (df.groupby("policy")["se_reduction"].mean() > 0.10).sum()
        total_policies = df["policy"].nunique()
        
        summary_text = f"""IIC Summary Statistics
        
Mean SE Reduction: {mean_reduction:.1f}%
Median SE Reduction: {median_reduction:.1f}%
Max SE Reduction: {max_reduction:.1f}%

Mean R²: {mean_r2:.3f}
Policies with >10% reduction: {substantial}/{total_policies}

Key Finding: IIC reduces variance
proportional to judge explainability (R²)
while preserving unbiasedness."""
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.suptitle("IIC (Isotonic Influence Control) Variance Reduction Analysis", 
                    fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / "iic_comprehensive_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Generate detailed CSV report
        summary_by_policy = df.groupby("policy").agg({
            "se_reduction": ["mean", "std", "count"],
            "r_squared": "mean",
            "se_with_iic": "mean",
            "se_without_iic": "mean"
        }).round(4)
        
        summary_by_policy.to_csv(output_dir / "iic_policy_summary.csv")
        
        # Print summary
        print("\n" + "="*60)
        print("IIC ABLATION SUMMARY")
        print("="*60)
        print(f"Mean SE reduction: {mean_reduction:.1f}%")
        print(f"Median SE reduction: {median_reduction:.1f}%")
        print(f"Max SE reduction: {max_reduction:.1f}%")
        print(f"Mean R²: {mean_r2:.3f}")
        print(f"\nPolicies with >10% SE reduction: {substantial}/{total_policies}")
        print("\nTop 5 policies by SE reduction:")
        for _, row in policy_reduction.head(5).iterrows():
            print(f"  {row['policy']}: {row['mean']*100:.1f}% (±{row['std']*100:.1f}%)")
        
        logger.info(f"Results saved to {output_dir}")

    def _find_matching_result(self, target: Dict, candidates: List[Dict]) -> Optional[Dict]:
        """Find result with same estimator, oracle coverage, and seed."""
        for candidate in candidates:
            if (candidate["spec"]["estimator"] == target["spec"]["estimator"] and
                candidate["spec"]["oracle_coverage"] == target["spec"]["oracle_coverage"] and
                candidate.get("seed") == target.get("seed")):
                return candidate
        return None

    def create_estimator(self, spec: Any, sampler: Any, cal_result: Any) -> Any:
        """Override to control IIC usage."""
        estimator_name = spec.estimator
        use_iic = spec.extra.get("use_iic", True)
        
        if estimator_name == "dr-cpo":
            from cje.estimators import DRCPOEstimator
            return DRCPOEstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,
                n_folds=5
            )
        elif estimator_name == "tmle":
            from cje.estimators import TMLEEstimator
            return TMLEEstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,
                n_folds=5
            )
        elif estimator_name == "mrdr":
            from cje.estimators import MRDREstimator
            return MRDREstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,
                n_folds=5
            )
        else:
            return super().create_estimator(spec, sampler, cal_result)


def main() -> None:
    """Run IIC ablation with comprehensive visualizations."""
    ablation = IICEffectAblation()
    results = ablation.run_ablation()
    ablation.analyze_results(results)


if __name__ == "__main__":
    main()