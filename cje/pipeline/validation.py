"""
Pipeline validation utilities - ensure data integrity between stages.
"""

import functools
import logging
from typing import List, Dict, Any, Callable, TypeVar, Set, cast

from ..utils.error_handling import ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


def validate_stage_output(
    required_fields: Set[str],
    optional_fields: Set[str] = None,
    at_least_one_of: Set[str] = None,
) -> Callable[[T], T]:
    """
    Decorator to validate stage output before passing to next stage.

    Args:
        required_fields: Fields that must be present in every row
        optional_fields: Fields that may be present (for documentation)
        at_least_one_of: At least one of these fields must be present

    Example:
        @validate_stage_output(
            required_fields={"uid", "context", "response"},
            at_least_one_of={"reward", "y_true"}
        )
        def run(self, rows):
            # process rows
            return rows
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = func(self, *args, **kwargs)

            # Validate if result is a list of dicts (rows)
            if isinstance(result, list) and result and isinstance(result[0], dict):
                stage_name = self.__class__.__name__
                validate_rows(result, stage_name, required_fields, at_least_one_of)
                logger.info(
                    f"{stage_name} output validated: {len(result)} rows with "
                    f"required fields {required_fields}"
                )

            return result

        return cast(T, wrapper)

    return decorator


def validate_rows(
    rows: List[Dict[str, Any]],
    stage_name: str,
    required_fields: Set[str],
    at_least_one_of: Set[str] = None,
) -> None:
    """
    Validate that rows have required structure.

    Args:
        rows: List of row dictionaries to validate
        stage_name: Name of the stage (for error messages)
        required_fields: Fields that must be present
        at_least_one_of: At least one of these fields must be present

    Raises:
        ValidationError: If validation fails
    """
    if not rows:
        raise ValidationError(f"{stage_name} returned empty results")

    # Check first row as sample
    sample_row = rows[0]
    sample_fields = set(sample_row.keys())

    # Check required fields
    missing_required = required_fields - sample_fields
    if missing_required:
        raise ValidationError(
            f"{stage_name} output missing required fields: {missing_required}\n"
            f"Available fields: {sorted(sample_fields)}\n"
            f"This likely means a previous stage failed to add these fields."
        )

    # Check at_least_one_of constraint
    if at_least_one_of:
        if not any(field in sample_fields for field in at_least_one_of):
            raise ValidationError(
                f"{stage_name} output must have at least one of: {at_least_one_of}\n"
                f"Available fields: {sorted(sample_fields)}\n"
                f"Consider:\n"
                f"- Adding rewards to your dataset\n"
                f"- Enabling oracle labeling\n"
                f"- Checking if calibration completed successfully"
            )

    # Validate all rows have same structure
    inconsistent_rows = []
    for i, row in enumerate(rows):
        row_fields = set(row.keys())
        missing = required_fields - row_fields
        if missing:
            inconsistent_rows.append((i, row.get("uid", "unknown"), missing))

    if inconsistent_rows:
        # Only show first 5 inconsistent rows
        examples = inconsistent_rows[:5]
        msg = f"{stage_name} has inconsistent row structure:\n"
        for idx, uid, missing in examples:
            msg += f"  Row {idx} (uid={uid}) missing: {missing}\n"
        if len(inconsistent_rows) > 5:
            msg += f"  ... and {len(inconsistent_rows) - 5} more rows\n"
        raise ValidationError(msg)
