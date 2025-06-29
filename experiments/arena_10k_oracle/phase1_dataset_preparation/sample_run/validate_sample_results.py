#!/usr/bin/env python3
"""
Validate the results of the 1% sample run.

This script performs comprehensive validation of the sample run outputs
to ensure the pipeline is ready for the full 10K run.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

console = Console()


class SampleValidator:
    """Validate sample run results."""

    def __init__(self):
        self.sample_dir = Path(__file__).parent.parent / "data" / "sample_1pct"
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    def validate_file_exists(self, filename: str, description: str) -> bool:
        """Check if a file exists and is valid JSONL."""
        filepath = self.sample_dir / filename

        if not filepath.exists():
            self.issues.append(f"Missing file: {filename} ({description})")
            return False

        try:
            count = 0
            with open(filepath) as f:
                for line in f:
                    json.loads(line)
                    count += 1

            self.stats[filename] = count
            console.print(f"✅ {description}: {count} entries")
            return True

        except Exception as e:
            self.issues.append(f"Invalid JSONL in {filename}: {e}")
            return False

    def validate_teacher_forcing(self) -> bool:
        """Validate teacher forcing results - CRITICAL CHECK."""
        console.print("\n🔬 Validating Teacher Forcing Results...")

        filepath = self.sample_dir / "p0_with_target_logps_sample.jsonl"
        if not filepath.exists():
            self.issues.append("Teacher forcing output file not found")
            return False

        # Track statistics
        total_computations = 0
        null_values = 0
        zero_values = 0
        suspicious_zeros = []
        logp_ranges = {"min": float("inf"), "max": float("-inf")}
        method_counts = {"token_counting": 0, "echo_based": 0, "continuation": 0}

        try:
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    response = data.get("response", "")

                    if "target_logps" not in data:
                        self.issues.append(
                            f"Missing target_logps for prompt {data.get('prompt_id')}"
                        )
                        continue

                    for policy, logp in data["target_logps"].items():
                        total_computations += 1

                        if logp is None:
                            null_values += 1
                        elif logp == 0.0:
                            zero_values += 1
                            if response:  # Non-empty response with 0.0 is suspicious
                                suspicious_zeros.append(
                                    {
                                        "prompt_id": data.get("prompt_id"),
                                        "policy": policy,
                                        "response": response[:100],
                                    }
                                )
                        else:
                            # Track value ranges
                            logp_ranges["min"] = min(logp_ranges["min"], logp)
                            logp_ranges["max"] = max(logp_ranges["max"], logp)

            # Report findings
            table = Table(title="Teacher Forcing Validation")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_column("Status", style="green")

            table.add_row(
                "Total computations",
                str(total_computations),
                "✅" if total_computations > 0 else "❌",
            )

            table.add_row(
                "Failed (null)",
                f"{null_values} ({null_values/total_computations*100:.1f}%)",
                "⚠️" if null_values > total_computations * 0.1 else "✅",
            )

            table.add_row(
                "Zero values",
                str(zero_values),
                "✅" if len(suspicious_zeros) == 0 else "❌",
            )

            table.add_row(
                "Log prob range",
                f"[{logp_ranges['min']:.2f}, {logp_ranges['max']:.2f}]",
                "✅" if logp_ranges["min"] < 0 and logp_ranges["max"] < 0 else "❌",
            )

            console.print(table)

            # Critical: Check for suspicious zeros
            if suspicious_zeros:
                self.issues.append(
                    f"Found {len(suspicious_zeros)} suspicious 0.0 values for non-empty responses!"
                )
                console.print(
                    "\n[red]❌ CRITICAL: Suspicious zero values detected![/red]"
                )
                for item in suspicious_zeros[:3]:
                    console.print(f"  - Prompt {item['prompt_id']} ({item['policy']})")
                    console.print(f"    Response: '{item['response']}'")
                return False

            # Warnings for high failure rate
            if null_values > total_computations * 0.1:
                self.warnings.append(
                    f"High teacher forcing failure rate: {null_values/total_computations*100:.1f}%"
                )

            return True

        except Exception as e:
            self.issues.append(f"Error validating teacher forcing: {e}")
            return False

    def validate_data_consistency(self) -> bool:
        """Check data consistency across files."""
        console.print("\n🔍 Checking Data Consistency...")

        # Load prompt IDs from each file
        prompt_ids = {}
        files_to_check = [
            ("arena_questions_base_sample.jsonl", "prompt_id"),
            ("p0_replies_sample.jsonl", "prompt_id"),
            ("p0_with_target_logps_sample.jsonl", "prompt_id"),
            ("oracle_labels_sample.jsonl", "prompt_id"),
        ]

        for filename, id_field in files_to_check:
            filepath = self.sample_dir / filename
            if filepath.exists():
                ids = set()
                with open(filepath) as f:
                    for line in f:
                        data = json.loads(line)
                        ids.add(data.get(id_field))
                prompt_ids[filename] = ids

        # Check consistency
        if len(prompt_ids) >= 2:
            base_ids = list(prompt_ids.values())[0]
            for filename, ids in prompt_ids.items():
                if ids != base_ids:
                    missing = base_ids - ids
                    extra = ids - base_ids
                    if missing:
                        self.warnings.append(
                            f"{filename} missing IDs: {list(missing)[:5]}"
                        )
                    if extra:
                        self.warnings.append(
                            f"{filename} has extra IDs: {list(extra)[:5]}"
                        )

        return True

    def validate_judge_scores(self) -> bool:
        """Validate judge score distributions."""
        console.print("\n⚖️ Validating Judge Scores...")

        judge_files = [
            "p0_scored_sample.jsonl",
            "targets_scored_uncertainty_sample.jsonl",
        ]

        all_scores = []
        for filename in judge_files:
            filepath = self.sample_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    for line in f:
                        data = json.loads(line)
                        if "judge_score" in data:
                            score = data["judge_score"]
                            if isinstance(score, dict):
                                all_scores.append(score.get("mean", 0))

        if all_scores:
            scores_array = np.array(all_scores)
            console.print(
                f"  Score range: [{scores_array.min():.2f}, {scores_array.max():.2f}]"
            )
            console.print(f"  Mean score: {scores_array.mean():.2f}")
            console.print(f"  Std deviation: {scores_array.std():.2f}")

            # Check for reasonable distribution
            if scores_array.std() < 0.1:
                self.warnings.append(
                    "Judge scores have very low variance - may indicate an issue"
                )

        return True

    def generate_report(self) -> Tuple[bool, str]:
        """Generate validation report and return (success, report)."""
        console.print("\n📊 Running Sample Validation...")

        # Check all expected files
        files_to_validate = [
            ("arena_questions_base_sample.jsonl", "Sample prompts"),
            ("p0_replies_sample.jsonl", "P0 responses"),
            ("target_responses_sample.jsonl", "Target policy responses"),
            ("p0_with_target_logps_sample.jsonl", "Teacher forcing results"),
            ("oracle_labels_sample.jsonl", "Oracle labels"),
        ]

        for filename, description in files_to_validate:
            self.validate_file_exists(filename, description)

        # Critical validations
        teacher_forcing_ok = self.validate_teacher_forcing()
        consistency_ok = self.validate_data_consistency()
        judge_scores_ok = self.validate_judge_scores()

        # Generate report
        report = "# Sample Run Validation Report\n\n"

        report += "## File Statistics\n"
        for filename, count in self.stats.items():
            report += f"- {filename}: {count} entries\n"

        report += "\n## Validation Results\n"
        report += (
            f"- Teacher Forcing: {'✅ PASS' if teacher_forcing_ok else '❌ FAIL'}\n"
        )
        report += f"- Data Consistency: {'✅ PASS' if consistency_ok else '❌ FAIL'}\n"
        report += f"- Judge Scores: {'✅ PASS' if judge_scores_ok else '❌ FAIL'}\n"

        if self.issues:
            report += "\n## Critical Issues\n"
            for issue in self.issues:
                report += f"- ❌ {issue}\n"

        if self.warnings:
            report += "\n## Warnings\n"
            for warning in self.warnings:
                report += f"- ⚠️ {warning}\n"

        # Decision
        success = len(self.issues) == 0 and teacher_forcing_ok

        report += "\n## Recommendation\n"
        if success:
            report += "✅ **READY FOR FULL RUN** - All critical checks passed\n"
        else:
            report += "❌ **NOT READY** - Critical issues must be resolved\n"

        return success, report

    def run(self) -> bool:
        """Run validation and return success status."""
        success, report = self.generate_report()

        # Display summary
        if success:
            console.print(
                Panel.fit(
                    "[bold green]✅ Sample validation PASSED[/bold green]\n"
                    "Ready for full 10K run!",
                    title="Validation Complete",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold red]❌ Sample validation FAILED[/bold red]\n"
                    f"Found {len(self.issues)} critical issues",
                    title="Validation Failed",
                )
            )

        # Save report
        report_path = self.sample_dir / "validation_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        console.print(f"\n📄 Full report saved to: {report_path}")

        return success


def main():
    """Main entry point."""
    validator = SampleValidator()
    success = validator.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
