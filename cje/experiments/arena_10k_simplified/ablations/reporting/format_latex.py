"""
LaTeX formatting utilities for tables.

Provides clean, journal-ready LaTeX output with booktabs styling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def format_table_m1(
    df: pd.DataFrame,
    caption: str = "Accuracy \\& Uncertainty Metrics",
    label: str = "tab:accuracy",
    highlight_best: bool = True,
) -> str:
    """Format Table M1 as LaTeX.

    Args:
        df: DataFrame from build_table_m1_accuracy_by_regime
        caption: Table caption
        label: LaTeX label
        highlight_best: Whether to bold best and underline second-best

    Returns:
        LaTeX table string
    """
    lines = []

    # Table environment
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Determine column alignment
    n_cols = len(df.columns)
    alignment = "l" + "r" * (n_cols - 1)
    lines.append(f"\\begin{{tabular}}{{{alignment}}}")
    lines.append("\\toprule")

    # Header with indicators
    header_parts = []
    for col in df.columns:
        if col in ["Estimator", "Regime"]:
            header_parts.append(col)
        elif any(x in col for x in ["RMSE", "IS", "Cov-95", "SE", "Regret", "Runtime"]):
            header_parts.append(f"{col} $\\downarrow$")
        elif any(x in col for x in ["τ", "Pairwise", "Top-1"]):
            header_parts.append(f"{col} $\\uparrow$")
        else:
            header_parts.append(col)

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Find best and second-best for each metric (exclude Overall rows)
    best_idx = {}
    second_idx = {}

    if highlight_best:
        # Mask to exclude Overall rows from competition
        mask = (
            (df["Regime"] != "Overall")
            if "Regime" in df.columns
            else pd.Series(True, index=df.index)
        )

        for col in df.columns:
            if col in ["Estimator", "Regime"]:
                continue

            valid_df = df[mask & df[col].notna()]
            if len(valid_df) == 0:
                continue

            # Determine if lower or higher is better
            if any(
                x in col for x in ["RMSE", "IS", "Cov-95", "SE", "Regret", "Runtime"]
            ):
                ascending = True  # Lower is better
            elif any(x in col for x in ["τ", "Pairwise", "Top-1"]):
                ascending = False  # Higher is better
            else:
                ascending = True  # Default to lower is better

            sorted_vals = valid_df[col].sort_values(ascending=ascending)
            if len(sorted_vals) >= 1:
                best_idx[col] = sorted_vals.index[0]
            if len(sorted_vals) >= 2:
                second_idx[col] = sorted_vals.index[1]

    # Format rows
    for idx, row in df.iterrows():
        cells = []

        for col in df.columns:
            val = row[col]

            if col in ["Estimator", "Regime"]:
                cells.append(str(val))
            elif pd.isna(val):
                cells.append("—")
            else:
                # Format number based on column type
                if any(x in col for x in ["RMSE", "IS", "SE", "Regret"]):
                    formatted = f"{val:.4f}"
                elif "Score" in col:
                    formatted = f"{val:.2f}"
                elif "τ" in col:
                    formatted = f"{val:.3f}"
                elif any(x in col for x in ["%", "Pairwise", "Top-1"]):
                    formatted = f"{val:.1f}"
                elif "Runtime" in col:
                    formatted = f"{val:.1f}"
                else:
                    formatted = f"{val:.3f}"

                # Apply highlighting (skip for Overall rows)
                if highlight_best and row.get("Regime", "") != "Overall":
                    if col in best_idx and idx == best_idx[col]:
                        formatted = f"\\textbf{{{formatted}}}"
                    elif col in second_idx and idx == second_idx[col]:
                        formatted = f"\\underline{{{formatted}}}"

                cells.append(formatted)

        lines.append(" & ".join(cells) + " \\\\")

        # Add separator before "Overall" rows
        if "Regime" in df.columns and idx < len(df) - 1:
            next_regime = df.iloc[idx + 1]["Regime"]
            curr_regime = row["Regime"]
            if curr_regime != "Overall" and next_regime == "Overall":
                lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Add footnote
    lines.append("\\footnotesize{")
    lines.append("$\\downarrow$: lower is better, $\\uparrow$: higher is better. ")
    lines.append("\\textbf{Bold}: best, \\underline{underlined}: second-best. ")
    if "Regime" not in df.columns:
        lines.append("Metrics averaged across all regimes.")
    lines.append("}")

    lines.append("\\end{table}")

    return "\n".join(lines)


def format_table_m2_deltas(
    panels: Dict[str, pd.DataFrame],
    caption: str = "Design Choice Effects (Paired Deltas)",
    label: str = "tab:design-deltas",
) -> str:
    """Format Table M2 (design deltas) as LaTeX.

    Args:
        panels: Dict of panel_name -> DataFrame from build_table_m2_design_deltas
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string with panels
    """
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    for panel_name, df in panels.items():
        if df.empty:
            continue

        # Panel header
        if panel_name == "calibration":
            panel_title = "Panel A: Weight Calibration (SIMCal) Effect"
        elif panel_name == "variance_cap":
            panel_title = "Panel B: Variance Cap Sensitivity"
        else:
            panel_title = f"Panel: {panel_name.replace('_', ' ').title()}"

        lines.append(f"\\textbf{{{panel_title}}}")
        lines.append("\\vspace{0.5em}")

        # Determine columns
        if panel_name == "variance_cap":
            # Special formatting for sensitivity panel
            cols = [c for c in df.columns if c != "Estimator"]
            alignment = "l" + "r" * len(cols)
            lines.append(f"\\begin{{tabular}}{{{alignment}}}")
            lines.append("\\toprule")
            lines.append(" & ".join(["Estimator"] + cols) + " \\\\")
            lines.append("\\midrule")

            for _, row in df.iterrows():
                # Ensure Estimator is a string
                estimator = str(row.get("Estimator", row.get("estimator", "")))
                cells = [estimator]
                for col in cols:
                    val = row.get(col)
                    if val is not None:
                        cells.append(str(val))
                    else:
                        cells.append("—")
                lines.append(" & ".join(cells) + " \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\vspace{1em}")

        else:
            # Standard delta panel
            delta_cols = [c for c in df.columns if c.startswith("Δ")]
            n_delta_cols = len(delta_cols)

            alignment = "lr" + "c" * n_delta_cols
            lines.append(f"\\begin{{tabular}}{{{alignment}}}")
            lines.append("\\toprule")

            # Header
            header = ["Estimator", "n"]
            for col in delta_cols:
                metric = col[1:]  # Remove Δ
                if "rmse" in metric.lower():
                    header.append("$\\Delta$RMSE$^d$")
                elif "interval" in metric.lower():
                    header.append("$\\Delta$IS$^{OA}$")
                elif "calib" in metric.lower():
                    header.append("$\\Delta$Calib")
                elif "se" in metric.lower():
                    header.append("$\\Delta$SE")
                elif "tau" in metric.lower():
                    header.append("$\\Delta\\tau$")
                elif "coverage_robust" in metric.lower():
                    header.append("$\\Delta$Coverage (pp)")
                else:
                    header.append(f"$\\Delta${metric}")

            lines.append(" & ".join(header) + " \\\\")
            lines.append("\\midrule")

            # Data rows
            for _, row in df.iterrows():
                # Ensure Estimator is a string
                estimator = str(row.get("Estimator", row.get("estimator", "")))
                n_pairs = str(int(row.get("n_pairs", 0)))
                cells = [estimator, n_pairs]

                for col in delta_cols:
                    if col in row:
                        val_str = row[col]
                        # Values are already formatted with CI and significance
                        cells.append(val_str if isinstance(val_str, str) else "—")
                    else:
                        cells.append("—")

                lines.append(" & ".join(cells) + " \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\vspace{1em}")

    # Footnote
    lines.append("\\footnotesize{")
    lines.append("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001. ")
    lines.append("Values in brackets show 95\\% bootstrap confidence intervals. ")
    if "variance_cap" in panels:
        lines.append(
            "Variance cap sensitivity shows maximum absolute change across $\\rho \\in \\{1, 2\\}$."
        )
    lines.append("}")

    lines.append("\\end{table}")

    return "\n".join(lines)


def format_table_m3_gates(
    df: pd.DataFrame,
    caption: str = "Gate Pass Rates and Diagnostics",
    label: str = "tab:gates",
) -> str:
    """Format Table M3 (gates) as LaTeX.

    Args:
        df: DataFrame from build_table_m3_gates
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Determine columns
    has_regime = "Regime" in df.columns
    gate_cols = [c for c in df.columns if "Pass" in c or "REFUSE" in c]
    diag_cols = [c for c in df.columns if c not in ["Estimator", "Regime"] + gate_cols]

    if has_regime:
        alignment = "ll" + "r" * (len(gate_cols) + len(diag_cols))
    else:
        alignment = "l" + "r" * (len(gate_cols) + len(diag_cols))

    lines.append(f"\\begin{{tabular}}{{{alignment}}}")
    lines.append("\\toprule")

    # Header
    header = ["Estimator"]
    if has_regime:
        header.append("Regime")

    # Gate columns
    for col in gate_cols:
        if "Overlap" in col:
            header.append("Overlap")
        elif "Judge" in col:
            header.append("Judge")
        elif "DR" in col:
            header.append("DR")
        elif "Cap" in col:
            header.append("Cap")
        elif "REFUSE" in col:
            header.append("Refuse")
        else:
            header.append(col.replace(" %", "").replace("Pass", ""))

    # Diagnostic columns
    for col in diag_cols:
        if "ESS" in col:
            header.append("ESS\\%")
        elif "Hill" in col:
            header.append("Hill $\\alpha$")
        else:
            header.append(col)

    lines.append(" & ".join(header) + " \\\\")

    # Subheader for gates vs diagnostics
    if gate_cols and diag_cols:
        subheader = [""] * (1 + (1 if has_regime else 0))
        subheader += [
            "\\multicolumn{" + str(len(gate_cols)) + "}{c}{Gate Pass Rates (\\%)}"
        ]
        if diag_cols:
            subheader += ["\\multicolumn{" + str(len(diag_cols)) + "}{c}{Diagnostics}"]
        lines.append(" & ".join(subheader) + " \\\\")

    lines.append("\\midrule")

    # Data rows
    for idx, row in df.iterrows():
        cells = [row["Estimator"]]
        if has_regime:
            cells.append(row["Regime"])

        # Gate rates
        for col in gate_cols:
            val = row[col]
            if pd.isna(val):
                cells.append("—")
            else:
                # Color-code based on pass rate
                formatted = f"{val:.1f}"
                if "REFUSE" not in col:  # Normal gates (high is good)
                    if val >= 90:
                        formatted = f"\\textcolor{{green!70!black}}{{{formatted}}}"
                    elif val < 50:
                        formatted = f"\\textcolor{{red!70!black}}{{{formatted}}}"
                else:  # Refuse rate (low is good)
                    if val <= 10:
                        formatted = f"\\textcolor{{green!70!black}}{{{formatted}}}"
                    elif val > 50:
                        formatted = f"\\textcolor{{red!70!black}}{{{formatted}}}"
                cells.append(formatted)

        # Diagnostic values
        for col in diag_cols:
            val = row[col]
            if pd.isna(val):
                cells.append("—")
            else:
                if "ESS" in col:
                    cells.append(f"{val:.1f}")
                elif "Hill" in col:
                    cells.append(f"{val:.2f}")
                else:
                    cells.append(f"{val:.3f}")

        lines.append(" & ".join(cells) + " \\\\")

        # Add separator between estimators if by regime
        if has_regime and idx < len(df) - 1:
            next_est = df.iloc[idx + 1]["Estimator"]
            curr_est = row["Estimator"]
            if curr_est != next_est:
                lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Footnote
    lines.append("\\footnotesize{")
    lines.append("Gate pass rates are percentages. ")
    lines.append(
        "\\textcolor{green!70!black}{Green}: good (≥90\\% for pass rates, ≤10\\% for refuse rate), "
    )
    lines.append(
        "\\textcolor{red!70!black}{Red}: poor (<50\\% for pass rates, >50\\% for refuse rate). "
    )
    lines.append("ESS: Effective Sample Size relative to nominal. ")
    lines.append(
        "Hill $\\alpha$: tail index (higher is better, <2 indicates heavy tails)."
    )
    lines.append("}")

    lines.append("\\end{table}")

    return "\n".join(lines)


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text.

    Args:
        text: Raw text string

    Returns:
        LaTeX-escaped string
    """
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
