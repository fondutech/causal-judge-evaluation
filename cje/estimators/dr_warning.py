"""
Utility module for Doubly Robust estimator warnings and validation.
"""

from typing import Optional
from rich.console import Console

console = Console()


def check_dr_requirements(
    estimator_name: str,
    samples_per_policy: int,
    auto_downgrade: bool = False,
    suppress_warning: bool = False,
) -> Optional[str]:
    """
    Check if DR estimator requirements are met and provide appropriate warnings.

    Args:
        estimator_name: Name of the estimator ("dr", "drcpo", "mrdr")
        samples_per_policy: Number of target policy samples configured
        auto_downgrade: If True, suggest downgrading to IPW/SNIPW
        suppress_warning: If True, don't print warning (just return message)

    Returns:
        Warning message if requirements not met, None otherwise
    """
    if samples_per_policy > 0:
        return None

    # Construct warning message
    warning_lines = [
        f"\n⚠️  [bold yellow]WARNING: {estimator_name.upper()} with samples_per_policy=0[/bold yellow]",
        "",
        f"The {estimator_name.upper()} estimator requires target policy samples to compute:",
        "  μ_π(x) = E_π[μ(x,s)] (the baseline term)",
        "",
        "Without target samples (samples_per_policy=0):",
        "  • The baseline term μ_π(x) = 0",
        "  • DR reduces to standard IPW (no variance reduction)",
        "  • Estimates represent differences from outcome model, not absolute values",
        "",
        "Recommended actions:",
        f"  1. Set samples_per_policy ≥ 1 for proper {estimator_name.upper()}",
        "  2. Use 'ipw' or 'snipw' estimators if target sampling is not possible",
        "",
    ]

    if auto_downgrade:
        warning_lines.extend(
            ["[dim]Auto-downgrading to IPW to maintain correctness.[/dim]", ""]
        )

    warning_message = "\n".join(warning_lines)

    if not suppress_warning:
        console.print(warning_message)

    return warning_message


def validate_dr_setup(
    estimator_name: str,
    samples_per_policy: int,
    has_judge_runner: bool,
    score_target_policy_sampled_completions: bool,
) -> None:
    """
    Validate the complete DR estimator setup and provide guidance.

    Args:
        estimator_name: Name of the estimator
        samples_per_policy: Number of target policy samples
        has_judge_runner: Whether a judge runner is available
        score_target_policy_sampled_completions: Whether to score target samples
    """
    # Check basic DR requirements
    check_dr_requirements(estimator_name, samples_per_policy)

    # Additional checks for optimal setup
    if samples_per_policy > 0:
        if not has_judge_runner and score_target_policy_sampled_completions:
            console.print(
                f"\n💡 [dim]Note: No judge_runner provided. "
                f"Target samples won't be scored, which may reduce outcome model accuracy.[/dim]"
            )
        elif has_judge_runner and not score_target_policy_sampled_completions:
            console.print(
                f"\n💡 [dim]Note: Judge available but scoring disabled. "
                f"Set score_target_policy_sampled_completions=True for better accuracy.[/dim]"
            )
