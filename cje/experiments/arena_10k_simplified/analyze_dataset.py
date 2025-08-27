#!/usr/bin/env python3
"""CJE Analysis Pipeline - Clean Orchestrator.

This is a thin orchestrator that coordinates the analysis pipeline modules.
Each module does one thing well, following the Unix philosophy.

Usage:
    python analyze_refactored.py --data path/to/dataset.jsonl --estimator calibrated-ips
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# CJE imports
from cje.data.precomputed_sampler import PrecomputedSampler

# Analysis pipeline modules
from analysis import (
    load_data,
    handle_rewards,
    restore_oracle_labels,
    create_estimator,
    display_results,
    display_weight_diagnostics,
    display_dr_diagnostics,
    display_augmentation_diagnostics,
    analyze_extreme_weights_report,
    generate_visualizations,
    export_results,
)

# CF-bits imports
from cje.cfbits.playbooks import cfbits_report_fresh_draws, cfbits_report_logging_only


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CJE analysis on Arena data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        default="data/cje_dataset.jsonl",
        help="Path to CJE dataset",
    )

    # Estimator arguments
    parser.add_argument(
        "--estimator",
        default="calibrated-ips",
        choices=[
            "calibrated-ips",
            "raw-ips",
            "dr-cpo",
            "mrdr",
            "tmle",
            "stacked-dr",
        ],
        help="Estimator to use",
    )
    parser.add_argument(
        "--estimator-config",
        type=json.loads,
        help="JSON config for estimator",
    )

    # Oracle/calibration arguments
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use oracle labels directly as rewards",
    )
    parser.add_argument(
        "--oracle-coverage",
        type=float,
        default=1.0,
        help="Fraction of oracle labels to use for calibration",
    )
    parser.add_argument(
        "--judge-field",
        default="judge_score",
        help="Field name for judge scores",
    )
    parser.add_argument(
        "--oracle-field",
        default="oracle_label",
        help="Field name for oracle labels",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-fitting folds",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results (JSON or CSV)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    # Analysis arguments
    parser.add_argument(
        "--extreme-threshold-low",
        type=float,
        default=1e-10,
        help="Threshold for near-zero weights",
    )
    parser.add_argument(
        "--extreme-threshold-high",
        type=float,
        default=10.0,
        help="Threshold for extreme high weights",
    )
    parser.add_argument(
        "--no-cfbits",
        action="store_true",
        help="Disable CF-bits uncertainty decomposition analysis",
    )
    parser.add_argument(
        "--cfbits-bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap samples for CF-bits confidence intervals (default: 500)",
    )

    # Debug arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> None:
    """Configure logging based on arguments."""
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    """Run the CJE analysis pipeline.

    This orchestrator coordinates the pipeline modules:
    1. Load data
    2. Handle rewards/calibration
    3. Create and fit estimator
    4. Run estimation
    5. Display results
    6. Display diagnostics
    7. Generate visualizations
    8. Export results

    Returns:
        0 on success, 1 on failure
    """
    # Parse arguments
    args = parse_arguments()
    setup_logging(args)

    # Print header
    if not args.quiet:
        print("\n" + "=" * 50)
        print(f"CJE Analysis Pipeline")
        print(f"Estimator: {args.estimator}")
        print(f"Dataset: {args.data}")
        print("=" * 50)

    try:
        # 1. Load data
        dataset = load_data(args.data, verbose=not args.quiet)

        # 2. Handle rewards and calibration
        analysis_config = {
            "n_folds": args.n_folds,
            "oracle_coverage": args.oracle_coverage,
        }
        calibrated_dataset, cal_result = handle_rewards(
            dataset, args, analysis_config, verbose=not args.quiet
        )

        # 3. Create sampler and estimator
        if not args.quiet:
            print("\n3. Setting up estimator...")
        sampler = PrecomputedSampler(calibrated_dataset)
        estimator = create_estimator(args, sampler, calibrated_dataset, cal_result)

        # 4. Fit and run estimation
        if not args.quiet:
            print(f"   ✓ Created {args.estimator} estimator")
            print(f"   Fitting estimator...")
        estimator.fit()

        if not args.quiet:
            print(f"   Running estimation...")
        results = estimator.estimate()

        if not args.quiet:
            print(f"   ✓ Estimation complete")

        # 5. Restore oracle labels for visualization
        # (They were masked during calibration for partial coverage)
        restore_oracle_labels(calibrated_dataset, args)
        # Also restore on original dataset for oracle comparison
        restore_oracle_labels(dataset, args)

        # 6. Display results
        summary_data = display_results(
            results,
            calibrated_dataset,
            sampler,
            estimator,
            args,
            dataset,
        )

        # 7. Display diagnostics
        weight_diagnostics = display_weight_diagnostics(
            estimator, sampler, calibrated_dataset, args
        )

        # Display DR diagnostics if applicable
        if args.estimator in ["dr-cpo", "mrdr", "tmle"]:
            display_dr_diagnostics(results, args)

        # Display augmentation diagnostics (for all estimators)
        display_augmentation_diagnostics(estimator, results, args.oracle_coverage, args)

        # Analyze extreme weights if requested
        if hasattr(estimator, "get_raw_weights"):
            analyze_extreme_weights_report(estimator, sampler, calibrated_dataset, args)

        # Run CF-bits analysis by default (unless disabled)
        cfbits_data = {}
        if not args.no_cfbits:
            if not args.quiet:
                step_num = 9 if args.estimator in ["dr-cpo", "mrdr", "tmle"] else 8
                print(f"\n{step_num}. CF-bits Uncertainty Decomposition:")
                print("   " + "-" * 60)

            # Determine which playbook to use
            is_dr = args.estimator in ["dr-cpo", "mrdr", "tmle", "stacked-dr"]

            # Run CF-bits for each policy
            for policy in sampler.target_policies:
                if not args.quiet:
                    print(f"\n   {policy}:")

                try:
                    if is_dr:
                        # Use fresh draws playbook for DR estimators
                        report = cfbits_report_fresh_draws(
                            estimator=estimator,
                            policy=policy,
                            n_boot=args.cfbits_bootstrap,
                            alpha=0.05,
                        )
                    else:
                        # Use logging-only playbook for IPS estimators
                        report = cfbits_report_logging_only(
                            estimator=estimator,
                            policy=policy,
                            n_boot=args.cfbits_bootstrap,
                            alpha=0.05,
                        )

                    if report:
                        cfbits_data[policy] = report

                        # Display key metrics
                        cfbits = report.get("cfbits", {})
                        if cfbits:
                            bits_tot = cfbits.get("bits_tot", "N/A")
                            w_tot = cfbits.get("w_tot", "N/A")
                            w_id = cfbits.get("w_id", "N/A")
                            w_var = cfbits.get("w_var", "N/A")
                            dominant = cfbits.get("dominant", "unknown")
                            if not args.quiet:
                                if isinstance(bits_tot, (int, float)) and isinstance(
                                    w_tot, (int, float)
                                ):
                                    print(
                                        f"     Total bits: {bits_tot:.2f} (width: {w_tot:.3f}, dominant: {dominant})"
                                    )
                                    if (
                                        args.debug
                                        and isinstance(w_id, (int, float))
                                        and isinstance(w_var, (int, float))
                                    ):
                                        print(
                                            f"       - Wid={w_id:.3f}, Wvar={w_var:.3f}"
                                        )
                                else:
                                    print(
                                        f"     Total bits: {bits_tot} (width: {w_tot}, dominant: {dominant})"
                                    )

                        # Display overlap with confidence interval
                        overlap = report.get("overlap", {})
                        if overlap and not args.quiet:
                            aessf = overlap.get("aessf")
                            aessf_lcb = overlap.get("aessf_lcb")
                            aessf_ucb = overlap.get("aessf_ucb")
                            if aessf:
                                if args.debug and aessf_lcb and aessf_ucb:
                                    print(
                                        f"     A-ESSF: {aessf:.1%} [{aessf_lcb:.1%}, {aessf_ucb:.1%}]"
                                    )
                                else:
                                    print(
                                        f"     A-ESSF: {aessf:.1%} (structural overlap)"
                                    )

                        # Display efficiency metrics for all estimators
                        efficiency = report.get("efficiency", {})
                        sampling = report.get("sampling_width", {})

                        if efficiency:
                            ifr_main = efficiency.get("ifr_main")
                            ifr_oua = efficiency.get("ifr_oua")
                            if ifr_main is not None and not args.quiet:
                                if ifr_oua is not None and ifr_oua != ifr_main:
                                    print(
                                        f"     IFR: {ifr_main:.1%} (main) / {ifr_oua:.1%} (with OUA)"
                                    )
                                else:
                                    print(
                                        f"     IFR: {ifr_main:.1%} (efficiency vs EIF)"
                                    )
                        elif is_dr and sampling:
                            # Fallback for DR when efficiency not in expected place
                            ifr = sampling.get("IFR_main")
                            if ifr is not None and not args.quiet:
                                print(f"     IFR: {ifr:.1%} (efficiency vs EIF)")

                        # Display gates with detailed reasons
                        gates = report.get("gates", {})
                        if gates and not args.quiet:
                            state = gates.get("state", "UNKNOWN")
                            emoji = {
                                "GOOD": "✅",
                                "WARNING": "⚠️",
                                "CRITICAL": "❌",
                                "REFUSE": "🚫",
                            }.get(state, "?")
                            print(f"     Gates: {emoji} {state}")
                            if state != "GOOD":
                                reasons = gates.get("reasons", [])
                                if reasons:
                                    # Show all reasons in debug mode, first reason otherwise
                                    if args.debug and len(reasons) > 1:
                                        for reason in reasons:
                                            print(f"       - {reason}")
                                    else:
                                        print(f"       Issues: {reasons[0]}")

                                # Show suggestions if available
                                suggestions = gates.get("suggestions", {})
                                if suggestions and args.debug:
                                    first_suggestion = (
                                        list(suggestions.values())[0]
                                        if suggestions
                                        else None
                                    )
                                    if first_suggestion:
                                        print(f"       → {first_suggestion}")

                except Exception as e:
                    if args.debug:
                        print(f"     ⚠️ CF-bits failed: {e}")
                    cfbits_data[policy] = {"error": str(e)}

            if not args.quiet and cfbits_data:
                # Summary table if we have multiple policies
                if len(cfbits_data) > 1:
                    print("\n   Summary Table:")
                    print("   " + "-" * 75)
                    print(
                        f"   {'Policy':<30} {'A-ESSF':>8} {'IFR':>8} {'Bits':>7} {'Gates':>10}"
                    )
                    print("   " + "-" * 75)

                    for pol, rep in cfbits_data.items():
                        if isinstance(rep, dict) and "error" not in rep:
                            overlap = rep.get("overlap", {})
                            aessf = overlap.get("aessf", 0) if overlap else 0

                            efficiency = rep.get("efficiency", {})
                            sampling = rep.get("sampling_width", {})
                            ifr = None
                            if efficiency:
                                ifr = efficiency.get("ifr_main") or efficiency.get(
                                    "ifr_oua"
                                )
                            elif sampling:
                                ifr = sampling.get("IFR_main")

                            cfbits = rep.get("cfbits", {})
                            bits = cfbits.get("bits_tot", 0) if cfbits else 0

                            gates = rep.get("gates", {})
                            state = gates.get("state", "?") if gates else "?"

                            # Format display
                            aessf_str = f"{aessf:.1%}" if aessf else "N/A"
                            ifr_str = f"{ifr:.1%}" if ifr is not None else "N/A"
                            bits_str = f"{bits:.2f}" if bits else "N/A"

                            print(
                                f"   {pol:<30} {aessf_str:>8} {ifr_str:>8} {bits_str:>7} {state:>10}"
                            )

                    print("   " + "-" * 75)

                print("\n   💡 CF-bits Interpretation:")
                print("   - Bits: Information gain (each bit = halving of width)")
                print("   - A-ESSF: Structural overlap quality (higher is better)")
                print("   - IFR: Efficiency vs theoretical best (higher is better)")
                print(
                    "   - Gates: Reliability assessment (GOOD > WARNING > CRITICAL > REFUSE)"
                )

        # 8. Generate visualizations
        generate_visualizations(
            results,
            dataset,
            calibrated_dataset,
            estimator,
            sampler,
            args,
            summary_data,
            cal_result,
        )

        # 9. Export results
        export_results(
            results,
            dataset,
            summary_data,
            weight_diagnostics,
            args,
        )

        # Success message
        if not args.quiet:
            steps_completed = 7  # Base steps
            if args.estimator in ["dr-cpo", "mrdr", "tmle"]:
                steps_completed += 1  # DR diagnostics
            if not args.no_cfbits:
                steps_completed += 1  # CF-bits analysis
            if not args.no_plots:
                steps_completed += 1  # Visualizations

            print(f"\n✓ Analysis complete! ({steps_completed} steps)")

        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
