"""Utility to attach sparse *oracle* labels to CJE logs.

This is kept minimal so callers can run:

>>> from cje.oracle_labeling import add_oracle_labels
>>> rows_labeled = add_oracle_labels(rows,
...     provider="openai", model_name="gpt-4o", fraction=0.25, seed=42)

It relies on the JudgeFactory so that any existing judge template can be used.
The default template simply asks for a single 0–10 number as per Arena rubric.
"""

from __future__ import annotations

import random
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional, Set
from datetime import datetime

from .judge import JudgeFactory
from .utils.progress import track, console

__all__ = ["add_oracle_labels", "add_full_oracle_labels_with_holdout"]


DEFAULT_TEMPLATE = "deterministic"  # Uses deterministic scoring for oracle labels


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint data if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint: Dict[str, Any] = json.load(f)
            console.print(
                f"[green]✓ Loaded checkpoint from {checkpoint_path}[/green]\n"
                f"  Completed: {checkpoint['completed_count']}/{checkpoint['total_count']} items\n"
                f"  Last updated: {checkpoint['last_updated']}"
            )
            return checkpoint
    return {
        "completed_indices": set(),
        "completed_count": 0,
        "total_count": 0,
        "last_updated": "",
    }


def _save_checkpoint(
    checkpoint_path: Path,
    completed_indices: Set[int],
    total_count: int,
    rows_out: List[Dict[str, Any]],
) -> None:
    """Save checkpoint data."""
    # Calculate policy statistics
    from collections import defaultdict

    policy_scores = defaultdict(list)

    for idx in completed_indices:
        if idx < len(rows_out):
            row = rows_out[idx]
            if "y_true" in row and row["y_true"] is not None:
                policy = row.get("policy", "unknown")
                policy_scores[policy].append(row["y_true"])

    # Compute statistics per policy
    policy_stats = {}
    for policy, scores in policy_scores.items():
        if scores:
            policy_stats[policy] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
                "mean_10_scale": (sum(scores) / len(scores)) * 10,
            }

    checkpoint = {
        "completed_indices": list(completed_indices),
        "completed_count": len(completed_indices),
        "total_count": total_count,
        "last_updated": datetime.now().isoformat(),
        "policy_statistics": policy_stats,
    }

    # Save checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    # Save partial results
    partial_path = checkpoint_path.with_suffix(".partial.jsonl")
    with open(partial_path, "w") as f:
        for row in rows_out:
            if "y_true" in row or "oracle_full" in row:
                f.write(json.dumps(row) + "\n")

    # Log policy statistics
    if policy_stats:
        console.print("\n📊 [bold]Policy Statistics:[/bold]")
        for policy, stats in sorted(policy_stats.items()):
            console.print(
                f"   {policy}: mean={stats['mean']:.3f} ({stats['mean_10_scale']:.1f}/10), "
                f"n={stats['count']}, range=[{stats['min']:.2f}, {stats['max']:.2f}]"
            )


def add_oracle_labels(
    rows: Sequence[Dict[str, Any]],
    *,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    fraction: float = 0.25,
    seed: int = 42,
    template: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16,
    score_field: str = "y_true",
    checkpoint_dir: Optional[str] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Attach oracle labels to a subset of *rows*.

    Parameters
    ----------
    rows : list of dict
        The log rows (will **not** be mutated; a shallow copy is returned).
    provider, model_name : str
        Passed to JudgeFactory to create the oracle.
    fraction : float
        Fraction of rows to label (uniform sample).
    seed : int
        RNG seed for reproducibility.
    template : Optional[str]
        Judge template name.  If *None*, we fall back to a built-in minimal
        one that prompts for a single 0–10 score.
    temperature, max_tokens : float / int
        Generation parameters for the judge.
    score_field : str
        Where to store the numeric oracle score (defaults to ``"y_true"`` so
        calibration can pick it up automatically).
    checkpoint_dir : Optional[str]
        Directory for saving checkpoints to resume interrupted runs.
    verbose : bool
        If True, print debug information about judge inputs and outputs.

    Returns
    -------
    list of dict
        New list with oracle scores filled in.
    """

    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0,1]")

    rng = random.Random(seed)
    n_rows = len(rows)
    n_oracle = max(1, int(round(n_rows * fraction)))

    # Load checkpoint if available
    checkpoint_path = None
    completed_indices: Set[int] = set()

    if checkpoint_dir:
        checkpoint_path = (
            Path(checkpoint_dir) / f"oracle_checkpoint_{provider}_{model_name}.json"
        )
        checkpoint_data = _load_checkpoint(checkpoint_path)
        completed_indices = set(checkpoint_data.get("completed_indices", []))

    # Select indices, excluding already completed ones
    remaining_indices = [i for i in range(n_rows) if i not in completed_indices]
    n_remaining = max(0, n_oracle - len(completed_indices))

    if n_remaining > 0 and remaining_indices:
        new_indices = rng.sample(
            remaining_indices, min(n_remaining, len(remaining_indices))
        )
    else:
        new_indices = []

    all_indices = sorted(list(completed_indices) + new_indices)

    # Setup logging if verbose
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        logger = logging.getLogger()

    # Prepare judge instance with deterministic uncertainty (no variance)
    judge_template = template or DEFAULT_TEMPLATE
    judge = JudgeFactory.create(
        provider=provider,
        model=model_name,
        template=judge_template,
        temperature=temperature,
        max_tokens=max_tokens,
        uncertainty_method="deterministic",  # Force zero variance for oracle labels
    )

    # Shallow-copy rows so callers' list is untouched
    rows_out: List[Dict[str, Any]] = [row.copy() for row in rows]

    # Score selected rows with progress tracking
    console.print(
        f"\n🏷️  Generating {len(new_indices)} oracle labels ({len(completed_indices)} already completed)"
    )

    errors = 0
    for idx in track(new_indices, description="Generating oracle labels"):
        row = rows_out[idx]

        # Validate required fields
        if "context" not in row:
            console.print(
                f"[yellow]⚠️  Row {idx} missing 'context' field, skipping[/yellow]"
            )
            continue

        context = row.get("context", "")
        if not context or not str(context).strip():
            console.print(f"[yellow]⚠️  Row {idx} has empty context, skipping[/yellow]")
            continue

        response = row.get("response", "")
        if not response or not str(response).strip():
            console.print(f"[yellow]⚠️  Row {idx} has empty response, skipping[/yellow]")
            continue

        try:
            if verbose:
                console.print(f"\n[blue]DEBUG: Scoring row {idx}[/blue]")
                console.print(f"  Policy: {row.get('policy', 'unknown')}")
                console.print(
                    f"  Context: {context[:100]}..."
                    if len(context) > 100
                    else f"  Context: {context}"
                )
                console.print(
                    f"  Response: {response[:150]}..."
                    if len(response) > 150
                    else f"  Response: {response}"
                )

            score = judge.score(context, response)

            if verbose:
                console.print(f"  Score: {float(score.mean)}")
                console.print(f"  10-scale: {float(score.mean) * 10:.1f}/10")

            row[score_field] = float(score.mean)

            # Set oracle metadata fields for DR estimation compatibility
            row["oracle_full"] = float(score.mean)
            row["oracle_available_to_logging"] = True
            row["oracle_holdout_mask"] = False

            completed_indices.add(idx)

            # Save checkpoint every 10 items
            if checkpoint_path and len(completed_indices) % 10 == 0:
                _save_checkpoint(checkpoint_path, completed_indices, n_oracle, rows_out)

        except Exception as e:
            errors += 1
            error_type = type(e).__name__

            # Provide more specific error messages
            if "rate" in str(e).lower() or "429" in str(e):
                console.print(
                    f"[yellow]⚠️  Row {idx}: Rate limit hit - consider reducing batch size[/yellow]"
                )
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                console.print(
                    f"[red]❌ Row {idx}: Authentication error - check your API key[/red]"
                )
                break  # No point continuing with bad auth
            elif "timeout" in str(e).lower():
                console.print(
                    f"[yellow]⚠️  Row {idx}: Request timeout - response might be too long[/yellow]"
                )
            else:
                console.print(f"[yellow]⚠️  Row {idx}: {error_type}: {e}[/yellow]")

            if errors > 5:
                console.print(
                    f"[red]❌ Too many errors ({errors}), stopping oracle generation[/red]"
                )
                console.print(
                    "[yellow]💡 Tip: You can resume from checkpoint by running the script again[/yellow]"
                )
                break

    # Apply labels from previously completed indices by loading partial results
    if checkpoint_path and checkpoint_path.exists():
        partial_path = checkpoint_path.with_suffix(".partial.jsonl")
        if partial_path.exists():
            # Load scores from partial results
            score_map = {}
            with open(partial_path, "r") as f:
                for line in f:
                    row_data = json.loads(line)
                    # Find the index by matching prompt_id or other unique identifier
                    for i, orig_row in enumerate(rows_out):
                        if orig_row.get("prompt_id") == row_data.get(
                            "prompt_id"
                        ) and orig_row.get("policy") == row_data.get("policy"):
                            if score_field in row_data:
                                score_map[i] = row_data[score_field]
                            break

            # Apply recovered scores
            for idx, score in score_map.items():
                if idx < len(rows_out) and idx in completed_indices:
                    rows_out[idx][score_field] = score
                    rows_out[idx]["oracle_full"] = score
                    rows_out[idx]["oracle_available_to_logging"] = True
                    if verbose:
                        console.print(
                            f"[blue]Restored score for row {idx}: {score}[/blue]"
                        )

    # For rows that didn't get oracle labels, set metadata to indicate unavailability
    for i, row in enumerate(rows_out):
        if i not in all_indices:
            row["oracle_available_to_logging"] = False
            row["oracle_holdout_mask"] = True

    # Final checkpoint save
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, completed_indices, n_oracle, rows_out)
        console.print(f"[green]✓ Saved final checkpoint with policy statistics[/green]")

    # Print summary
    console.print(f"\n✅ Generated {len(completed_indices)} oracle labels total")
    if errors > 0:
        console.print(f"[yellow]⚠️  {errors} errors encountered[/yellow]")

    return rows_out


def add_full_oracle_labels_with_holdout(
    rows: Sequence[Dict[str, Any]],
    *,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    logging_policy_oracle_fraction: float = 0.25,
    seed: int = 42,
    template: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> List[Dict[str, Any]]:
    """Generate oracle labels for ALL (context, action) pairs with proper holdout.

    This function implements the experimental design needed for arena analysis:
    1. Generate oracle labels for every (context, action) pair using strong LLM
    2. Store full oracle labels in 'oracle_full' field (for final evaluation)
    3. Make only a fraction available to logging policy in 'y_true' field
    4. Target policies will not see oracle labels during evaluation

    Parameters
    ----------
    rows : list of dict
        The log rows (will **not** be mutated; a shallow copy is returned).
    provider, model_name : str
        Passed to JudgeFactory to create the oracle judge.
    logging_policy_oracle_fraction : float
        Fraction of oracle labels available to logging policy for calibration.
        Remaining oracle labels are held out for final evaluation.
    seed : int
        RNG seed for reproducibility.
    template : Optional[str]
        Judge template name for oracle scoring.
    temperature, max_tokens : float / int
        Generation parameters for the oracle judge.

    Returns
    -------
    list of dict
        New list with oracle labels structured as:
        - 'oracle_full': Oracle score for every row (for evaluation)
        - 'y_true': Oracle score only for logging_policy_oracle_fraction of rows
        - 'oracle_holdout_mask': Boolean indicating if oracle was held out
        - 'oracle_available_to_logging': Boolean indicating if oracle available for calibration
    """

    if not (0.0 < logging_policy_oracle_fraction <= 1.0):
        raise ValueError("logging_policy_oracle_fraction must be in (0,1]")

    # Prepare judge instance for oracle scoring
    judge_template = template or DEFAULT_TEMPLATE
    judge = JudgeFactory.create(
        provider=provider,
        model=model_name,
        template=judge_template,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Shallow-copy rows
    rows_out: List[Dict[str, Any]] = [row.copy() for row in rows]

    console.print(
        f"🏷️  Generating oracle labels for {len(rows_out)} (context, action) pairs..."
    )

    # Step 1: Generate oracle labels for ALL rows with progress tracking
    errors = 0
    for row in track(rows_out, description="Generating oracle labels"):
        # Skip rows with empty or missing responses
        response = row.get("response", "")
        if not response or not response.strip():
            continue

        try:
            oracle_score = judge.score(row["context"], response)
            row["oracle_full"] = float(oracle_score.mean)
        except Exception as e:
            errors += 1
            console.print(f"[yellow]⚠️  Error scoring row: {e}[/yellow]")
            if errors > 10:
                console.print("[red]❌ Too many errors, stopping[/red]")
                break

    console.print(f"✅ Generated {len(rows_out)} oracle labels")
    if errors > 0:
        console.print(f"[yellow]⚠️  {errors} errors encountered[/yellow]")

    # Step 2: Determine which oracle labels are available to logging policy
    rng = random.Random(seed)
    n_rows = len(rows_out)
    n_available_to_logging = max(1, int(round(n_rows * logging_policy_oracle_fraction)))
    available_indices = set(rng.sample(range(n_rows), n_available_to_logging))

    console.print(f"\n📊 Oracle fraction breakdown:")
    console.print(f"   Total oracle labels: {n_rows}")
    console.print(
        f"   Available to logging policy: {n_available_to_logging} ({logging_policy_oracle_fraction:.1%})"
    )
    console.print(
        f"   Held out for evaluation: {n_rows - n_available_to_logging} ({1-logging_policy_oracle_fraction:.1%})"
    )

    # Step 3: Set up holdout structure
    for i, row in enumerate(rows_out):
        if i in available_indices:
            # This oracle label is available to logging policy for calibration
            row["y_true"] = row["oracle_full"]
            row["oracle_available_to_logging"] = True
            row["oracle_holdout_mask"] = False
        else:
            # This oracle label is held out from logging policy
            row["y_true"] = None  # Not available for calibration
            row["oracle_available_to_logging"] = False
            row["oracle_holdout_mask"] = True

    return rows_out


def evaluate_against_oracle_holdout(
    estimated_values: Dict[str, float],
    rows_with_oracle: List[Dict[str, Any]],
    policy_names: List[str],
) -> Dict[str, Any]:
    """Evaluate CJE estimates against held-out oracle labels.

    This compares the CJE policy value estimates against the true oracle
    values computed from held-out oracle labels.

    Parameters
    ----------
    estimated_values : dict
        Policy names mapped to CJE estimated values
    rows_with_oracle : list of dict
        Rows containing oracle_full labels for all (context, action) pairs
    policy_names : list of str
        Names of policies being evaluated

    Returns
    -------
    dict
        Evaluation metrics including:
        - 'oracle_true_values': True oracle values per policy
        - 'cje_estimates': CJE estimates per policy
        - 'absolute_errors': |CJE - Oracle| per policy
        - 'relative_errors': (CJE - Oracle)/Oracle per policy
        - 'rmse': Root mean squared error across policies
        - 'mae': Mean absolute error across policies
    """

    # Compute true oracle values per policy
    # For each policy, compute mean oracle score across all contexts where that policy was evaluated
    oracle_true_values = {}

    for policy_name in policy_names:
        policy_oracle_scores = [
            row["oracle_full"]
            for row in rows_with_oracle
            if row.get("policy_name") == policy_name or row.get("action") == policy_name
        ]

        if policy_oracle_scores:
            oracle_true_values[policy_name] = sum(policy_oracle_scores) / len(
                policy_oracle_scores
            )
        else:
            console.print(
                f"⚠️  Warning: No oracle scores found for policy {policy_name}"
            )
            oracle_true_values[policy_name] = 0.0

    # Compute errors
    absolute_errors = {}
    relative_errors = {}

    for policy_name in policy_names:
        if policy_name in estimated_values and policy_name in oracle_true_values:
            cje_est = estimated_values[policy_name]
            oracle_true = oracle_true_values[policy_name]

            absolute_errors[policy_name] = abs(cje_est - oracle_true)

            if oracle_true != 0:
                relative_errors[policy_name] = (cje_est - oracle_true) / oracle_true
            else:
                relative_errors[policy_name] = float("inf") if cje_est != 0 else 0.0

    # Compute aggregate metrics
    abs_errors_list = list(absolute_errors.values())
    rmse = (
        (sum(e**2 for e in abs_errors_list) / len(abs_errors_list)) ** 0.5
        if abs_errors_list
        else 0.0
    )
    mae = sum(abs_errors_list) / len(abs_errors_list) if abs_errors_list else 0.0

    return {
        "oracle_true_values": oracle_true_values,
        "cje_estimates": estimated_values,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "rmse": rmse,
        "mae": mae,
        "n_policies_evaluated": len(absolute_errors),
    }
