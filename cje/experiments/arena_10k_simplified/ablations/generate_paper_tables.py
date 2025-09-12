#!/usr/bin/env python3
"""
Generate all paper tables for the ablation study.

Usage:
    python generate_paper_tables.py --results results/all_experiments.jsonl --output tables/
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from reporting import (
    generate_leaderboard,
    generate_delta_tables,
    generate_stacking_table,
    generate_quadrant_leaderboard,
    generate_bias_patterns_table,
    generate_overlap_diagnostics_table,
    generate_oracle_adjustment_table,
    generate_boundary_outlier_table,
    generate_runtime_complexity_table,
)


def main():
    """Generate all paper tables."""
    parser = argparse.ArgumentParser(
        description="Generate information-dense tables for paper"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/all_experiments.jsonl"),
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/"),
        help="Output directory for tables",
    )
    parser.add_argument(
        "--format",
        choices=["latex", "markdown", "both"],
        default="latex",
        help="Output format",
    )
    args = parser.parse_args()
    
    # Check results file exists
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = []
    with open(args.results) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
    
    print(f"Loaded {len(results)} experiment results")
    
    # Create output directories
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "main").mkdir(exist_ok=True)
    (args.output / "appendix").mkdir(exist_ok=True)
    
    # Generate main text tables
    print("\n" + "=" * 60)
    print("MAIN TEXT TABLES")
    print("=" * 60)
    
    # Table 1: Leaderboard
    print("\nGenerating Table 1: Estimator Leaderboard...")
    if args.format in ["latex", "both"]:
        leaderboard_latex = generate_leaderboard(results, "latex")
        (args.output / "main" / "table1_leaderboard.tex").write_text(leaderboard_latex)
        print("  ✓ LaTeX version saved")
    
    if args.format in ["markdown", "both"]:
        leaderboard_md = generate_leaderboard(results, "markdown")
        (args.output / "main" / "table1_leaderboard.md").write_text(leaderboard_md)
        print("  ✓ Markdown version saved")
    
    # Table 2: Design Choice Effects
    print("\nGenerating Table 2: Design Choice Effects...")
    if args.format in ["latex", "both"]:
        delta_tables = generate_delta_tables(results, "latex")
        (args.output / "main" / "table2a_calibration.tex").write_text(
            delta_tables["calibration"]
        )
        (args.output / "main" / "table2b_iic.tex").write_text(delta_tables["iic"])
        print("  ✓ LaTeX versions saved (2a: calibration, 2b: IIC)")
    
    if args.format in ["markdown", "both"]:
        delta_tables_md = generate_delta_tables(results, "markdown")
        (args.output / "main" / "table2_deltas.md").write_text(
            "## Panel A: Weight Calibration Effect\n\n" +
            delta_tables_md["calibration"] +
            "\n\n## Panel B: IIC Effect\n\n" +
            delta_tables_md["iic"]
        )
        print("  ✓ Markdown version saved")
    
    # Table 3: Stacking Diagnostics
    print("\nGenerating Table 3: Stacking Efficiency & Stability...")
    try:
        if args.format in ["latex", "both"]:
            stacking_latex = generate_stacking_table(results, "latex")
            (args.output / "main" / "table3_stacking.tex").write_text(stacking_latex)
            print("  ✓ LaTeX version saved")
        
        if args.format in ["markdown", "both"]:
            stacking_md = generate_stacking_table(results, "markdown")
            (args.output / "main" / "table3_stacking.md").write_text(stacking_md)
            print("  ✓ Markdown version saved")
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate stacking table: {e}")
        print("    (This is expected if stacking diagnostics aren't captured yet)")
    
    # Generate appendix tables
    print("\n" + "=" * 60)
    print("APPENDIX TABLES")
    print("=" * 60)
    
    appendix_tables = [
        ("A1", "Quadrant Leaderboard", generate_quadrant_leaderboard),
        ("A2", "Bias Patterns", generate_bias_patterns_table),
        ("A3", "Overlap & Tail Diagnostics", generate_overlap_diagnostics_table),
        ("A4", "Oracle Adjustment Share", generate_oracle_adjustment_table),
        ("A5", "Calibration Boundary Analysis", generate_boundary_outlier_table),
        ("A6", "Runtime & Complexity", generate_runtime_complexity_table),
    ]
    
    for table_num, caption, generator in appendix_tables:
        print(f"\nGenerating Table {table_num}: {caption}...")
        try:
            df = generator(results)
            
            if args.format in ["latex", "both"]:
                # Format as LaTeX
                from reporting.appendix_tables import format_appendix_latex
                latex = format_appendix_latex(df, table_num, caption)
                (args.output / "appendix" / f"table{table_num}.tex").write_text(latex)
                print(f"  ✓ LaTeX version saved")
            
            if args.format in ["markdown", "both"]:
                md = f"## Table {table_num}: {caption}\n\n" + df.to_markdown(index=False)
                (args.output / "appendix" / f"table{table_num}.md").write_text(md)
                print(f"  ✓ Markdown version saved")
                
        except Exception as e:
            print(f"  ⚠ Warning: Could not generate table: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAll tables written to: {args.output}/")
    print(f"  Main text tables: {args.output}/main/")
    print(f"  Appendix tables: {args.output}/appendix/")
    
    # Print quick statistics
    if results:
        estimators = set(r.get("spec", {}).get("estimator") for r in results)
        sample_sizes = set(r.get("spec", {}).get("sample_size") for r in results)
        coverages = set(r.get("spec", {}).get("oracle_coverage") for r in results)
        
        print(f"\nDataset statistics:")
        print(f"  Total experiments: {len(results)}")
        print(f"  Unique estimators: {len(estimators)}")
        print(f"  Sample sizes: {sorted(sample_sizes)}")
        print(f"  Oracle coverages: {sorted(coverages)}")


if __name__ == "__main__":
    main()