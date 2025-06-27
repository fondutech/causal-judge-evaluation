#!/usr/bin/env python3
"""
Fix log probability error handling to fail explicitly rather than returning invalid values.

Current issues:
1. api_policy.py returns 0.0 on exceptions (log prob = 0 means probability = 1!)
2. Some scripts use -50.0 as a replacement value
3. Failures are logged but not tracked properly

This script identifies all instances and proposes fixes.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple
from rich.console import Console
from rich.syntax import Syntax

console = Console()


def find_problematic_patterns(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find problematic error handling patterns in a Python file."""
    issues = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        # Pattern 1: return 0.0 for log probabilities
        if "return 0.0" in line and any(
            keyword in "".join(lines[max(0, i - 10) : i + 10]).lower()
            for keyword in ["logp", "log_prob", "logprob"]
        ):
            context = "".join(lines[max(0, i - 3) : min(len(lines), i + 2)])
            issues.append((i, "return 0.0", context))

        # Pattern 2: Using -50.0 as replacement
        if "-50.0" in line or "-50." in line:
            context = "".join(lines[max(0, i - 3) : min(len(lines), i + 2)])
            issues.append((i, "hardcoded -50.0", context))

        # Pattern 3: Silent exception handling
        if "except" in line and i + 1 < len(lines):
            next_lines = "".join(lines[i : min(len(lines), i + 5)])
            if "pass" in next_lines or (
                "return" in next_lines and "raise" not in next_lines
            ):
                context = "".join(lines[max(0, i - 2) : min(len(lines), i + 5)])
                issues.append((i, "silent exception", context))

    return issues


def scan_codebase():
    """Scan the codebase for problematic error handling."""
    console.print(
        "[bold]🔍 Scanning for problematic log probability error handling...[/bold]\n"
    )

    # Directories to scan
    dirs_to_scan = [
        Path("/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/cje"),
        Path(
            "/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/experiments/arena_10k_oracle"
        ),
    ]

    all_issues = []

    for base_dir in dirs_to_scan:
        for py_file in base_dir.rglob("*.py"):
            issues = find_problematic_patterns(py_file)
            if issues:
                all_issues.append((py_file, issues))

    # Display findings
    if not all_issues:
        console.print("[green]✅ No problematic patterns found![/green]")
        return

    console.print(f"[red]Found issues in {len(all_issues)} files:[/red]\n")

    for file_path, issues in all_issues:
        rel_path = file_path.relative_to(
            Path("/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation")
        )
        console.print(f"\n[yellow]{rel_path}[/yellow]")

        for line_num, issue_type, context in issues:
            console.print(f"  Line {line_num}: [red]{issue_type}[/red]")
            syntax = Syntax(
                context.strip(),
                "python",
                theme="monokai",
                line_numbers=True,
                start_line=line_num - 2,
            )
            console.print(syntax)
            console.print()


def generate_fixes():
    """Generate proposed fixes for the issues."""
    console.print("\n[bold]🛠️  Proposed Fixes:[/bold]\n")

    # Fix 1: api_policy.py
    console.print("[cyan]1. Fix api_policy.py log_prob method:[/cyan]")
    fix1 = '''
    def log_prob(self, context: str, response: str) -> float:
        """Compute log P(response | context) under this policy."""
        try:
            # ... existing implementation ...
            return result
        except Exception as e:
            # NEVER return a default value - fail explicitly
            raise RuntimeError(
                f"Teacher forcing failed for {self.model_name}: {e}\\n"
                f"Context: {context[:100]}...\\n"
                f"Response: {response[:100]}..."
            ) from e
    '''
    console.print(Syntax(fix1, "python", theme="monokai"))

    # Fix 2: Batch processing
    console.print("\n[cyan]2. Fix batch processing to handle failures:[/cyan]")
    fix2 = '''
    def compute_target_logprobs_batch(batch, runners):
        """Compute log probs with explicit failure tracking."""
        results = []
        failures = []
        
        for item in batch:
            result = item.copy()
            target_logps = {}
            
            for policy_name, runner in runners.items():
                try:
                    logp = runner.log_prob(item["prompt"], item["response"])
                    target_logps[policy_name] = float(logp)
                except Exception as e:
                    # Track failure explicitly
                    failures.append({
                        "prompt_id": item["prompt_id"],
                        "policy": policy_name,
                        "error": str(e),
                        "prompt_preview": item["prompt"][:100],
                        "response_preview": item["response"][:100]
                    })
                    # Mark as failed, not a fake value
                    target_logps[policy_name] = None
            
            result["target_logps"] = target_logps
            results.append(result)
        
        # Report failures
        if failures:
            console.print(f"[red]❌ {len(failures)} log prob calculations failed![/red]")
            # Save failure log
            with open("logprob_failures.json", "w") as f:
                json.dump(failures, f, indent=2)
        
        return results
    '''
    console.print(Syntax(fix2, "python", theme="monokai"))

    # Fix 3: Data validation
    console.print("\n[cyan]3. Add data validation before analysis:[/cyan]")
    fix3 = '''
    def validate_logprob_data(data):
        """Validate that all log probabilities are valid."""
        invalid_samples = []
        
        for item in data:
            # Check P0 log prob
            if item.get("total_logprob") is None:
                invalid_samples.append((item["prompt_id"], "P0", "missing"))
            elif item["total_logprob"] == 0.0:
                invalid_samples.append((item["prompt_id"], "P0", "zero"))
            elif item["total_logprob"] > 0:
                invalid_samples.append((item["prompt_id"], "P0", "positive"))
            
            # Check target log probs
            for policy, logp in item.get("target_logps", {}).items():
                if logp is None:
                    invalid_samples.append((item["prompt_id"], policy, "missing"))
                elif logp == 0.0:
                    invalid_samples.append((item["prompt_id"], policy, "zero"))
                elif logp > 0:
                    invalid_samples.append((item["prompt_id"], policy, "positive"))
        
        if invalid_samples:
            raise ValueError(
                f"Found {len(invalid_samples)} invalid log probabilities!\\n"
                f"First few: {invalid_samples[:5]}"
            )
        
        return True
    '''
    console.print(Syntax(fix3, "python", theme="monokai"))


def main():
    """Run the analysis."""
    console.print("[bold blue]🚨 Log Probability Error Handling Analysis[/bold blue]\n")

    # Scan for issues
    scan_codebase()

    # Generate fixes
    generate_fixes()

    # Recommendations
    console.print("\n[bold]📋 Recommendations:[/bold]")
    console.print(
        "1. [red]NEVER[/red] return default values (0.0, -50.0) for log probabilities"
    )
    console.print(
        "2. [red]ALWAYS[/red] fail explicitly with informative error messages"
    )
    console.print("3. Track failures in a separate log file for debugging")
    console.print("4. Validate data before running any analysis")
    console.print("5. Consider retrying failed API calls with backoff")
    console.print(
        "\n[yellow]Log probability = 0.0 means probability = 1.0 (perfect prediction)![/yellow]"
    )
    console.print(
        "[yellow]This is almost never correct and will corrupt importance weights.[/yellow]"
    )


if __name__ == "__main__":
    main()
