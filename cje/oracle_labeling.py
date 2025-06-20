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
from typing import List, Dict, Any, Sequence, Optional

from .judge import JudgeFactory

__all__ = ["add_oracle_labels", "add_full_oracle_labels_with_holdout"]


DEFAULT_TEMPLATE = "single_score_0_10"


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
    indices = rng.sample(range(n_rows), n_oracle)

    # Prepare judge instance
    judge_template = template or DEFAULT_TEMPLATE
    judge = JudgeFactory.create(
        provider=provider,
        model=model_name,
        template=judge_template,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Shallow-copy rows so callers' list is untouched
    rows_out: List[Dict[str, Any]] = [row.copy() for row in rows]

    # Score selected rows
    for idx in indices:
        row = rows_out[idx]

        # Skip rows with empty or missing responses
        response = row.get("response", "")
        if not response or not response.strip():
            continue

        score = judge.score(row["context"], response)
        row[score_field] = float(score.mean)

        # Set oracle metadata fields for DR estimation compatibility
        row["oracle_full"] = float(score.mean)
        row["oracle_available_to_logging"] = True
        row["oracle_holdout_mask"] = False

    # For rows that didn't get oracle labels, set metadata to indicate unavailability
    for i, row in enumerate(rows_out):
        if i not in indices:
            row["oracle_available_to_logging"] = False
            row["oracle_holdout_mask"] = True

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

    print(f"🏷️ Generating oracle labels for {len(rows_out)} (context, action) pairs...")

    # Step 1: Generate oracle labels for ALL rows
    for i, row in enumerate(rows_out):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(rows_out)} oracle labels generated")

        # Skip rows with empty or missing responses
        response = row.get("response", "")
        if not response or not response.strip():
            continue

        oracle_score = judge.score(row["context"], response)
        row["oracle_full"] = float(oracle_score.mean)

    print(f"✅ Generated {len(rows_out)} oracle labels")

    # Step 2: Determine which oracle labels are available to logging policy
    rng = random.Random(seed)
    n_rows = len(rows_out)
    n_available_to_logging = max(1, int(round(n_rows * logging_policy_oracle_fraction)))
    available_indices = set(rng.sample(range(n_rows), n_available_to_logging))

    print(f"📊 Oracle fraction breakdown:")
    print(f"   Total oracle labels: {n_rows}")
    print(
        f"   Available to logging policy: {n_available_to_logging} ({logging_policy_oracle_fraction:.1%})"
    )
    print(
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
            print(f"⚠️ Warning: No oracle scores found for policy {policy_name}")
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
