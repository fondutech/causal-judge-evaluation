"""
Data validation utilities for CJE datasets.

This module provides comprehensive validation for CJE data files, helping users
identify and fix data quality issues before running experiments.

Key functions:
- validate_input_data(): Check basic data requirements
- validate_logs(): Check processed log requirements
- check_propensity_quality(): Analyze log probability quality
- check_ground_truth_quality(): Analyze ground truth label quality
- validate_dataset(): Comprehensive validation with user-friendly reports
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation with detailed feedback."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    summary: Dict[str, Any]

    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return len(self.errors) > 0 or len(self.warnings) > 0

    def print_report(self) -> None:
        """Print a user-friendly validation report."""
        print("\n" + "=" * 60)
        print("🔍 CJE Data Validation Report")
        print("=" * 60)

        if self.is_valid and not self.has_issues():
            print("✅ Data validation passed with no issues!")
        else:
            if self.errors:
                print(f"\n❌ {len(self.errors)} Error(s) Found:")
                for error in self.errors:
                    print(f"   • {error}")

            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} Warning(s):")
                for warning in self.warnings:
                    print(f"   • {warning}")

            if self.info:
                print(f"\nℹ️  Additional Information:")
                for info in self.info:
                    print(f"   • {info}")

        if self.summary:
            print(f"\n📊 Data Summary:")
            for key, value in self.summary.items():
                print(f"   • {key}: {value}")

        print("\n" + "=" * 60)


def validate_input_data(file_path: Union[str, Path]) -> List[str]:
    """
    Validate input data requirements for CJE.

    Checks basic requirements like required fields and critical constraints.

    Args:
        file_path: Path to JSONL data file

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    file_path = Path(file_path)

    if not file_path.exists():
        return [f"File not found: {file_path}"]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Row {i}: Invalid JSON - {e}")
                    continue

                # Check basic requirements
                if "context" not in row:
                    errors.append(f"Row {i}: Missing required field 'context'")
                elif not row["context"]:
                    errors.append(f"Row {i}: Empty context field")

                # Check critical constraint from documentation:
                # "If you provide y_true, you must also provide the corresponding response"
                if "y_true" in row and "response" not in row:
                    errors.append(f"Row {i}: Has y_true but missing response")

                # Check for empty required fields
                if "response" in row and not row["response"]:
                    errors.append(f"Row {i}: Empty response field")

    except Exception as e:
        errors.append(f"Error reading file: {e}")

    return errors


def validate_logs(file_path: Union[str, Path]) -> List[str]:
    """
    Validate processed logs for estimator requirements.

    Checks that logs have all fields required by estimators after processing.

    Args:
        file_path: Path to processed logs JSONL file

    Returns:
        List of validation errors (empty if valid)
    """
    # These are required for estimators after CJE processing
    required_for_estimators = ["context", "response", "logp", "reward"]
    errors = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    missing = [f for f in required_for_estimators if f not in row]
                    if missing:
                        errors.append(f"Row {i}: Missing {missing}")
                except json.JSONDecodeError as e:
                    errors.append(f"Row {i}: Invalid JSON - {e}")
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")

    return errors


def check_propensity_quality(file_path: Union[str, Path]) -> List[str]:
    """
    Check log probability quality for potential issues.

    Analyzes logp values for common problems like positive values,
    suspicious ranges, or inconsistent values for duplicate responses.

    Args:
        file_path: Path to data file with logp values

    Returns:
        List of quality issues found
    """
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            response_logps: dict[str, float] = {}

            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    if "logp" in row:
                        logp = row["logp"]

                        # logp should be negative (log probabilities)
                        if logp > 0:
                            issues.append(
                                f"Row {i}: logp should be negative, got {logp}"
                            )

                        # Values shouldn't be suspiciously high (> -1) or low (< -50)
                        if logp > -1:
                            issues.append(f"Row {i}: logp suspiciously high: {logp}")
                        elif logp < -50:
                            issues.append(f"Row {i}: logp suspiciously low: {logp}")

                        # Check for duplicate responses with very different logp values
                        if "response" in row:
                            response = row["response"]
                            if response in response_logps:
                                prev_logp = response_logps[response]
                                if abs(logp - prev_logp) > 5.0:  # Arbitrary threshold
                                    issues.append(
                                        f"Row {i}: Duplicate response with very different logp: "
                                        f"{logp} vs {prev_logp}"
                                    )
                            else:
                                response_logps[response] = logp

                except json.JSONDecodeError:
                    continue  # Skip invalid JSON rows

    except Exception as e:
        issues.append(f"Error checking propensity quality: {e}")

    return issues


def check_ground_truth_quality(file_path: Union[str, Path]) -> List[str]:
    """
    Check ground truth label quality and distribution.

    Analyzes y_true values for sufficient quantity, balance, and data types.

    Args:
        file_path: Path to data file with y_true labels

    Returns:
        List of quality issues found
    """
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            y_true_values = []

            for i, line in enumerate(f):
                try:
                    row = json.loads(line.strip())
                    if "y_true" in row:
                        y_true = row["y_true"]
                        y_true_values.append(y_true)

                        # Check if y_true is numeric
                        if not isinstance(y_true, (int, float)):
                            issues.append(
                                f"Row {i}: y_true should be numeric, got {type(y_true)}"
                            )

                except json.JSONDecodeError:
                    continue

            # Check if we have sufficient examples for calibration
            # Note: For integration purposes, treat very small datasets as warnings not errors
            if len(y_true_values) < 5:
                issues.append(
                    f"Very few ground truth examples: {len(y_true_values)} (minimum 10-20 recommended for production)"
                )
            elif len(y_true_values) < 100:
                issues.append(
                    f"Few ground truth examples: {len(y_true_values)} (≥100 recommended for best calibration)"
                )

            # Check for balanced distribution if binary classification
            if y_true_values:
                unique_values = set(y_true_values)
                if unique_values == {0, 1} or unique_values == {0.0, 1.0}:
                    # Binary classification
                    positive_count = sum(1 for v in y_true_values if v in [1, 1.0])
                    negative_count = len(y_true_values) - positive_count
                    ratio = min(positive_count, negative_count) / max(
                        positive_count, negative_count
                    )
                    if ratio < 0.2:  # Very imbalanced
                        issues.append(
                            f"Highly imbalanced binary labels: {positive_count} positive, "
                            f"{negative_count} negative (ratio: {ratio:.2f})"
                        )

    except Exception as e:
        issues.append(f"Error checking ground truth quality: {e}")

    return issues


def analyze_data_summary(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze data file and provide summary statistics.

    Args:
        file_path: Path to data file

    Returns:
        Dictionary with summary statistics
    """
    # Use explicit counters to avoid type issues
    total_samples = 0
    has_responses = 0
    has_ground_truth = 0
    has_log_probabilities = 0
    has_target_samples = 0
    scenario_type = "unknown"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    total_samples += 1

                    if row.get("response"):
                        has_responses += 1
                    if row.get("y_true") is not None:
                        has_ground_truth += 1
                    if row.get("logp") is not None:
                        has_log_probabilities += 1
                    if row.get("target_samples"):
                        has_target_samples += 1

                except json.JSONDecodeError:
                    continue

        # Determine scenario type
        if has_target_samples > 0:
            scenario_type = "Scenario 3: Pre-computed Policy Data"
        elif has_responses > 0 and has_ground_truth > 0:
            scenario_type = "Scenario 2: Complete Logs with Ground Truth"
        elif has_responses == 0:
            scenario_type = "Scenario 1: Context Only"
        else:
            scenario_type = "Mixed or Custom Scenario"

    except Exception as e:
        logger.error(f"Error analyzing data summary: {e}")

    # Return properly typed summary
    summary: Dict[str, Any] = {
        "total_samples": total_samples,
        "has_responses": has_responses,
        "has_ground_truth": has_ground_truth,
        "has_log_probabilities": has_log_probabilities,
        "has_target_samples": has_target_samples,
        "scenario_type": scenario_type,
    }

    return summary


def validate_dataset(
    file_path: Union[str, Path], scenario: Optional[str] = None
) -> ValidationResult:
    """
    Comprehensive data validation with user-friendly reporting.

    Runs all relevant validation checks and provides a detailed report
    with errors, warnings, and recommendations.

    Args:
        file_path: Path to data file
        scenario: Optional scenario hint ("1", "2", "3" or auto-detect)

    Returns:
        ValidationResult with comprehensive feedback
    """
    file_path = Path(file_path)
    errors = []
    warnings = []
    info = []

    # Basic input validation
    input_errors = validate_input_data(file_path)
    errors.extend(input_errors)

    # Get data summary
    summary = analyze_data_summary(file_path)

    # Scenario-specific validation
    if summary["has_log_probabilities"] > 0:
        propensity_issues = check_propensity_quality(file_path)
        for issue in propensity_issues:
            if "should be negative" in issue:
                errors.append(issue)
            else:
                warnings.append(issue)

    if summary["has_ground_truth"] > 0:
        gt_issues = check_ground_truth_quality(file_path)
        # Treat all ground truth issues as warnings for better user experience
        warnings.extend(gt_issues)

    # Add informational messages
    info.append(f"Detected {summary['scenario_type']}")

    if summary["has_ground_truth"] > 0:
        info.append(
            f"Ground truth available for {summary['has_ground_truth']}/{summary['total_samples']} samples"
        )

    if summary["has_responses"] == 0:
        info.append(
            "No responses in data - CJE will generate them using logging policy"
        )

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid, errors=errors, warnings=warnings, info=info, summary=summary
    )
