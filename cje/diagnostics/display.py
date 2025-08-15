"""Display utilities for diagnostic suite."""

from typing import Optional
from .suite import DiagnosticSuite


def format_diagnostic_suite(suite: DiagnosticSuite, verbosity: str = "normal") -> str:
    """Format diagnostic suite for display.

    Args:
        suite: Diagnostic suite to format
        verbosity: Display level - "quiet", "normal", or "detailed"

    Returns:
        Formatted string for display
    """
    if verbosity == "quiet":
        # Just show overall status
        status = "✅" if not suite.has_issues else "⚠️"
        return f"Diagnostics: {status}"

    elif verbosity == "normal":
        # Show summary
        return suite.to_summary()

    elif verbosity == "detailed":
        # Full report
        lines = []
        lines.append("=" * 80)
        lines.append("DETAILED DIAGNOSTIC REPORT")
        lines.append("=" * 80)

        # Full summary
        lines.append(suite.to_summary())

        # Detailed weight diagnostics
        lines.append("\n" + "=" * 60)
        lines.append("WEIGHT DIAGNOSTICS (DETAILED)")
        lines.append("=" * 60)
        for policy, metrics in suite.weight_diagnostics.items():
            lines.append(f"\n{policy}:")
            lines.append(f"  Effective Sample Size: {metrics.ess:.2f}")
            lines.append(f"  Maximum Weight: {metrics.max_weight:.4f}")
            if metrics.hill_index:
                lines.append(f"  Hill Tail Index: {metrics.hill_index:.3f}")
                if metrics.has_heavy_tails:
                    lines.append("  ⚠️ WARNING: Heavy tails detected (α < 2)")
                    lines.append("     Variance may be infinite!")
                elif metrics.has_marginal_tails:
                    lines.append("  ⚠️ WARNING: Marginal tail behavior (2 ≤ α < 2.5)")
            if metrics.cv:
                lines.append(f"  Coefficient of Variation: {metrics.cv:.3f}")
            if metrics.n_unique:
                lines.append(f"  Unique Weight Values: {metrics.n_unique}")

        # Stability details
        if suite.stability:
            lines.append("\n" + "=" * 60)
            lines.append("STABILITY DIAGNOSTICS")
            lines.append("=" * 60)
            lines.append(f"Drift Detected: {suite.stability.has_drift}")
            if suite.stability.max_tau_change:
                lines.append(
                    f"Maximum Kendall τ Change: {suite.stability.max_tau_change:.4f}"
                )
            if suite.stability.drift_policies:
                lines.append(
                    f"Policies with Drift: {', '.join(suite.stability.drift_policies)}"
                )
            if suite.stability.ece:
                lines.append(f"Expected Calibration Error: {suite.stability.ece:.4f}")
            if suite.stability.reliability:
                lines.append(f"Reliability: {suite.stability.reliability:.4f}")
            if suite.stability.resolution:
                lines.append(f"Resolution: {suite.stability.resolution:.4f}")

        # DR quality details
        if suite.dr_quality:
            lines.append("\n" + "=" * 60)
            lines.append("DOUBLY ROBUST QUALITY")
            lines.append("=" * 60)
            lines.append(f"Orthogonality Satisfied: {suite.dr_quality.is_orthogonal}")
            lines.append(
                f"Max Violation: {suite.dr_quality.max_orthogonality_violation:.6f}"
            )
            lines.append("\nOrthogonality Scores by Policy:")
            for policy, score in suite.dr_quality.orthogonality_scores.items():
                status = "✅" if abs(score) < 0.01 else "⚠️"
                lines.append(f"  {status} {policy}: {score:.6f}")

            if suite.dr_quality.dm_contributions:
                lines.append("\nDM-IPS Decomposition:")
                for policy in suite.dr_quality.dm_contributions:
                    dm = suite.dr_quality.dm_contributions[policy]
                    ips = suite.dr_quality.ips_contributions[policy]
                    total = dm + ips
                    if total != 0:
                        dm_pct = 100 * dm / total
                        ips_pct = 100 * ips / total
                        lines.append(
                            f"  {policy}: DM={dm_pct:.1f}%, IPS={ips_pct:.1f}%"
                        )

        # Robust inference details
        if suite.robust_inference:
            lines.append("\n" + "=" * 60)
            lines.append("ROBUST INFERENCE")
            lines.append("=" * 60)
            lines.append(f"Bootstrap Method: {suite.robust_inference.method}")
            lines.append(f"Bootstrap Iterations: {suite.robust_inference.n_bootstrap}")

            if suite.robust_inference.bootstrap_ses:
                lines.append("\nBootstrap Standard Errors:")
                for policy, se in suite.robust_inference.bootstrap_ses.items():
                    lines.append(f"  {policy}: {se:.6f}")

            if suite.robust_inference.fdr_adjusted:
                lines.append(f"\nFDR Control (α={suite.robust_inference.fdr_alpha}):")
                lines.append(
                    f"  Significant: {suite.robust_inference.n_significant_fdr}/{len(suite.robust_inference.fdr_adjusted)}"
                )
                for policy, sig in suite.robust_inference.fdr_adjusted.items():
                    status = "✅" if sig else "❌"
                    lines.append(f"  {status} {policy}")

        # Recommendations
        recs = suite.get_recommendations()
        if recs:
            lines.append("\n" + "=" * 60)
            lines.append("ACTIONABLE RECOMMENDATIONS")
            lines.append("=" * 60)
            for rec in recs:
                lines.append(rec)

        # Metadata
        lines.append("\n" + "=" * 60)
        lines.append("METADATA")
        lines.append("=" * 60)
        if suite.timestamp:
            lines.append(f"Timestamp: {suite.timestamp}")

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown verbosity level: {verbosity}")


def display_diagnostic_suite(
    suite: DiagnosticSuite, verbosity: str = "normal", use_color: bool = True
) -> None:
    """Display diagnostic suite to terminal.

    Args:
        suite: Diagnostic suite to display
        verbosity: Display level - "quiet", "normal", or "detailed"
        use_color: Whether to use terminal colors (if supported)
    """
    formatted = format_diagnostic_suite(suite, verbosity)
    print(formatted)
