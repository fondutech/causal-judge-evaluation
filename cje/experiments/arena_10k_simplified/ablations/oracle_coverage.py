#!/usr/bin/env python3
"""Oracle coverage ablation - THE fundamental result.

This ablation answers: How much oracle data do we need?

Key findings we expect:
- 5-10% oracle coverage is sufficient for reliable estimates
- Below 5% calibration becomes unreliable
- Augmentation helps when coverage is low
- Diminishing returns above 20%
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import ExperimentSpec
from core.base import BaseAblation
from core.schemas import aggregate_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class OracleCoverageAblation(BaseAblation):
    """Ablation to study oracle coverage requirements."""
    
    def __init__(self):
        super().__init__(
            name="oracle_coverage",
            cache_dir=Path("../.ablation_cache/oracle_coverage")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run oracle coverage sweep."""
        
        # Define coverage levels to test
        oracle_coverages = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
        
        # Fixed parameters for this ablation
        estimator = "mrdr"  # Use MRDR as it's our best method
        sample_fraction = 1.0  # Use full dataset
        n_seeds = 5  # Multiple seeds for stability
        
        logger.info("=" * 70)
        logger.info("ORACLE COVERAGE ABLATION")
        logger.info("=" * 70)
        logger.info(f"Estimator: {estimator}")
        logger.info(f"Sample fraction: {sample_fraction:.0%}")
        logger.info(f"Coverage levels: {oracle_coverages}")
        logger.info(f"Seeds per level: {n_seeds}")
        logger.info("")
        
        all_results = []
        
        for coverage in oracle_coverages:
            logger.info(f"\n{'='*50}")
            logger.info(f"Oracle Coverage: {coverage:.0%}")
            logger.info(f"{'='*50}")
            
            # Use stable dataset
            data_path = Path("../data/cje_dataset.jsonl")
            if not data_path.exists():
                data_path = Path("../../data/cje_dataset.jsonl")
            
            spec = ExperimentSpec(
                ablation="oracle_coverage",
                dataset_path=str(data_path),
                estimator=estimator,
                oracle_coverage=coverage,
                sample_fraction=sample_fraction,
                n_seeds=n_seeds,
                seed_base=42
            )
            
            # Run with multiple seeds
            results = self.run_with_seeds(spec)
            all_results.extend(results)
            
            # Show aggregate statistics
            agg = aggregate_results(results)
            if agg.get("n_seeds_successful", 0) > 0:
                mean_rmse = np.mean([
                    r.get("rmse_vs_oracle", np.nan) 
                    for r in results 
                    if r.get("success", False)
                ])
                mean_ci_width = np.mean([
                    r.get("mean_ci_width", np.nan)
                    for r in results
                    if r.get("success", False)
                ])
                
                logger.info(f"Results ({agg['n_seeds_successful']}/{agg['n_seeds_total']} successful):")
                logger.info(f"  Mean RMSE: {mean_rmse:.4f}")
                logger.info(f"  Mean CI width: {mean_ci_width:.4f}")
                
                # Show per-policy RMSE if available
                if agg.get("estimates_mean"):
                    oracle_truths = agg.get("oracle_truths", {})
                    for policy in agg["estimates_mean"]:
                        est = agg["estimates_mean"][policy]
                        truth = oracle_truths.get(policy, np.nan)
                        if np.isfinite(est) and np.isfinite(truth):
                            policy_rmse = abs(est - truth)
                            logger.info(f"  {policy}: estimate={est:.3f}, truth={truth:.3f}, error={policy_rmse:.3f}")
            else:
                logger.warning(f"All {agg['n_seeds_total']} seeds failed!")
        
        # Save all results
        output_dir = Path("../ablations/results/oracle_coverage")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types before saving
        def convert_numpy(obj):
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
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj
        
        with open(output_dir / "results.jsonl", "w") as f:
            for result in all_results:
                f.write(json.dumps(convert_numpy(result)) + "\n")
        
        logger.info(f"\nSaved {len(all_results)} results to {output_dir}")
        
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze oracle coverage results."""
        
        # Group by coverage level
        by_coverage = {}
        for r in results:
            if r.get("success", False):
                coverage = r["spec"]["oracle_coverage"]
                if coverage not in by_coverage:
                    by_coverage[coverage] = []
                by_coverage[coverage].append(r)
        
        analysis = {
            "coverage_levels": sorted(by_coverage.keys()),
            "mean_rmse": {},
            "mean_ci_width": {},
            "mean_calibration_rmse": {},
            "success_rate": {}
        }
        
        for coverage in analysis["coverage_levels"]:
            coverage_results = by_coverage[coverage]
            
            # RMSE
            rmses = [r.get("rmse_vs_oracle", np.nan) for r in coverage_results]
            analysis["mean_rmse"][coverage] = np.nanmean(rmses)
            
            # CI width
            ci_widths = [r.get("mean_ci_width", np.nan) for r in coverage_results]
            analysis["mean_ci_width"][coverage] = np.nanmean(ci_widths)
            
            # Calibration RMSE
            cal_rmses = [r.get("calibration_rmse", np.nan) for r in coverage_results]
            analysis["mean_calibration_rmse"][coverage] = np.nanmean(cal_rmses)
            
            # Success rate
            analysis["success_rate"][coverage] = len(coverage_results)
        
        # Find sweet spot (where RMSE stabilizes)
        rmse_values = [analysis["mean_rmse"][c] for c in analysis["coverage_levels"]]
        if len(rmse_values) > 2:
            # Find where improvement slows down
            improvements = np.diff(rmse_values)
            if len(improvements) > 0:
                # Find first point where improvement < 10% of initial
                threshold = 0.1 * abs(improvements[0]) if improvements[0] != 0 else 0.01
                sweet_spot_idx = next(
                    (i for i, imp in enumerate(improvements) if abs(imp) < threshold),
                    len(improvements) - 1
                )
                analysis["sweet_spot_coverage"] = analysis["coverage_levels"][min(sweet_spot_idx + 1, len(analysis["coverage_levels"]) - 1)]
            else:
                analysis["sweet_spot_coverage"] = analysis["coverage_levels"][0]
        
        return analysis
    
    def create_figure(self, results: List[Dict[str, Any]], output_path: Path = None):
        """Create Figure 1: Oracle coverage vs RMSE."""
        
        analysis = self.analyze_results(results)
        
        if not analysis["coverage_levels"]:
            logger.warning("No successful results to plot")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        coverages = np.array(analysis["coverage_levels"]) * 100  # Convert to percentage
        
        # Panel A: RMSE vs coverage
        ax = axes[0]
        rmse_values = [analysis["mean_rmse"][c] for c in analysis["coverage_levels"]]
        ax.plot(coverages, rmse_values, 'o-', linewidth=2, markersize=8, label='MRDR')
        
        # Add sweet spot
        if "sweet_spot_coverage" in analysis:
            sweet_spot_pct = analysis["sweet_spot_coverage"] * 100
            ax.axvline(sweet_spot_pct, color='red', linestyle='--', alpha=0.5, label=f'Sweet spot ({sweet_spot_pct:.0f}%)')
        
        ax.set_xlabel('Oracle Coverage (%)', fontsize=12)
        ax.set_ylabel('RMSE vs Oracle Truth', fontsize=12)
        ax.set_title('A. Estimation Error vs Oracle Coverage', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel B: CI Width vs coverage
        ax = axes[1]
        ci_values = [analysis["mean_ci_width"][c] for c in analysis["coverage_levels"]]
        ax.plot(coverages, ci_values, 'o-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Oracle Coverage (%)', fontsize=12)
        ax.set_ylabel('Mean CI Width', fontsize=12)
        ax.set_title('B. Uncertainty vs Oracle Coverage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel C: Calibration quality
        ax = axes[2]
        cal_values = [analysis["mean_calibration_rmse"][c] for c in analysis["coverage_levels"]]
        ax.plot(coverages, cal_values, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Oracle Coverage (%)', fontsize=12)
        ax.set_ylabel('Calibration RMSE', fontsize=12)
        ax.set_title('C. Judge→Oracle Calibration Quality', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Oracle Coverage Requirements for CJE', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        plt.show()
        
        return fig


def main():
    """Run oracle coverage ablation."""
    
    ablation = OracleCoverageAblation()
    
    # Run the ablation
    results = ablation.run_ablation()
    
    # Analyze
    analysis = ablation.analyze_results(results)
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*70)
    
    if analysis["coverage_levels"]:
        logger.info("\nRMSE by coverage:")
        for coverage in analysis["coverage_levels"]:
            rmse = analysis["mean_rmse"][coverage]
            logger.info(f"  {coverage:6.1%}: {rmse:.4f}")
        
        if "sweet_spot_coverage" in analysis:
            logger.info(f"\nSweet spot: {analysis['sweet_spot_coverage']:.1%} coverage")
            logger.info("(Where diminishing returns begin)")
    
    # Create figure
    figure_path = Path("../ablations/results/oracle_coverage/figure_1_oracle_coverage.png")
    ablation.create_figure(results, figure_path)
    
    logger.info("\n" + "="*70)
    logger.info("ORACLE COVERAGE ABLATION COMPLETE")
    logger.info("="*70)
    
    return results


if __name__ == "__main__":
    results = main()