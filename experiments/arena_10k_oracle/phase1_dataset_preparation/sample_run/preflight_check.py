#!/usr/bin/env python3
"""
Pre-flight check before running sample or full pipeline.
Verifies all dependencies, API keys, and configurations.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class PreflightChecker:
    """Check all prerequisites before running the pipeline."""

    def __init__(self, sample_mode: bool = True):
        self.sample_mode = sample_mode
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []

    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version is 3.8+."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor} (need 3.8+)"

    def check_api_keys(self) -> Dict[str, Tuple[bool, str]]:
        """Check required API keys are set."""
        results = {}

        # Fireworks API key
        fireworks_key = os.environ.get("FIREWORKS_API_KEY", "")
        if fireworks_key and len(fireworks_key) > 10:
            results["FIREWORKS_API_KEY"] = (True, "Set (hidden)")
        else:
            results["FIREWORKS_API_KEY"] = (False, "Not set or invalid")

        # OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key and openai_key.startswith("sk-"):
            results["OPENAI_API_KEY"] = (True, "Set (hidden)")
        else:
            results["OPENAI_API_KEY"] = (False, "Not set or invalid")

        return results

    def check_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """Check Python dependencies."""
        results = {}

        try:
            # Check CJE is importable
            sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
            import cje

            results["cje"] = (True, "Importable")

            # Check for RobustTeacherForcing
            from cje.utils import RobustTeacherForcing

            results["RobustTeacherForcing"] = (True, "Available")

        except ImportError as e:
            results["cje"] = (False, f"Import error: {e}")

        # Check other key dependencies
        required_packages = ["rich", "numpy", "fireworks", "openai"]
        for package in required_packages:
            try:
                __import__(package)
                results[package] = (True, "Installed")
            except ImportError:
                results[package] = (False, "Not installed")

        return results

    def check_data_files(self) -> Dict[str, Tuple[bool, str]]:
        """Check required data files exist."""
        results = {}
        data_dir = Path(__file__).parent.parent.parent / "data"

        # Base dataset (required for sample)
        base_file = data_dir / "arena_questions_base.jsonl"
        if base_file.exists():
            try:
                with open(base_file) as f:
                    count = sum(1 for _ in f)
                results["arena_questions_base.jsonl"] = (True, f"{count:,} prompts")
            except:
                results["arena_questions_base.jsonl"] = (False, "Exists but unreadable")
        else:
            results["arena_questions_base.jsonl"] = (
                False,
                "Not found - run 01_prepare_data.py",
            )

        # Sample directory
        if self.sample_mode:
            sample_dir = data_dir / "sample_1pct"
            if not sample_dir.exists():
                sample_dir.mkdir(parents=True, exist_ok=True)
                results["sample_1pct directory"] = (True, "Created")
            else:
                results["sample_1pct directory"] = (True, "Exists")

        return results

    def check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space."""
        try:
            import shutil

            path = Path(__file__).parent
            stat = shutil.disk_usage(path)
            free_gb = stat.free / (1024**3)

            required_gb = 0.1 if self.sample_mode else 1.0

            if free_gb > required_gb:
                return True, f"{free_gb:.1f} GB free"
            else:
                return False, f"{free_gb:.1f} GB free (need {required_gb} GB)"
        except:
            return True, "Unable to check"

    def check_scripts(self) -> Dict[str, Tuple[bool, str]]:
        """Check all required scripts exist."""
        results = {}
        scripts_dir = Path(__file__).parent.parent

        required_scripts = [
            "01_prepare_data.py",
            "02a_generate_p0_responses.py",
            "02b_generate_target_responses.py",
            "02c_compute_target_logprobs.py",
            "03_generate_oracle_labels.py",
            "04a_score_p0_responses.py",
        ]

        for script in required_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                results[script] = (True, "Found")
            else:
                results[script] = (False, "Missing")

        return results

    def run_checks(self) -> bool:
        """Run all preflight checks."""
        console.print(
            Panel.fit(
                f"{'Sample' if self.sample_mode else 'Full'} Run Preflight Check",
                title="🚀 Pre-flight Validation",
            )
        )

        all_checks = []

        # Python version
        console.print("\n🐍 Checking Python version...")
        passed, msg = self.check_python_version()
        all_checks.append(("Python Version", passed, msg))

        # API keys
        console.print("🔑 Checking API keys...")
        for key, (passed, msg) in self.check_api_keys().items():
            all_checks.append((key, passed, msg))

        # Dependencies
        console.print("📦 Checking dependencies...")
        for dep, (passed, msg) in self.check_dependencies().items():
            all_checks.append((f"Package: {dep}", passed, msg))

        # Data files
        console.print("📁 Checking data files...")
        for file, (passed, msg) in self.check_data_files().items():
            all_checks.append((file, passed, msg))

        # Disk space
        console.print("💾 Checking disk space...")
        passed, msg = self.check_disk_space()
        all_checks.append(("Disk Space", passed, msg))

        # Scripts
        console.print("📜 Checking scripts...")
        script_checks = self.check_scripts()
        scripts_ok = all(passed for passed, _ in script_checks.values())
        all_checks.append(
            (
                "Pipeline Scripts",
                scripts_ok,
                f"{sum(1 for p, _ in script_checks.values() if p)}/{len(script_checks)} found",
            )
        )

        # Display results
        console.print("\n" + "=" * 60 + "\n")

        table = Table(title="Preflight Check Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="yellow")

        for check, passed, details in all_checks:
            if passed:
                self.checks_passed += 1
                status = "[green]✅ PASS[/green]"
            else:
                self.checks_failed += 1
                status = "[red]❌ FAIL[/red]"

            table.add_row(check, status, details)

        console.print(table)

        # Summary
        total_checks = self.checks_passed + self.checks_failed
        success_rate = (
            (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        )

        if self.checks_failed == 0:
            console.print(
                Panel.fit(
                    f"[bold green]✅ ALL CHECKS PASSED ({self.checks_passed}/{total_checks})[/bold green]\n"
                    f"Ready to run {'sample' if self.sample_mode else 'full'} pipeline!",
                    title="Ready for Launch",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"[bold red]❌ PREFLIGHT FAILED ({self.checks_failed}/{total_checks} failed)[/bold red]\n"
                    "Please fix the issues above before proceeding.",
                    title="Not Ready",
                )
            )

        # Additional recommendations
        if self.checks_failed == 0:
            console.print("\n📋 Next Steps:")
            if self.sample_mode:
                console.print("1. Run: ./run_sample.sh")
                console.print("2. Monitor: python monitor_sample_run.py")
                console.print("3. Validate: python validate_sample_results.py")
            else:
                console.print("1. Ensure sample run passed validation")
                console.print("2. Schedule 50-75 hours of compute time")
                console.print("3. Set up monitoring and alerts")

        return self.checks_failed == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preflight check for Arena 10K pipeline"
    )
    parser.add_argument(
        "--full", action="store_true", help="Check for full run (default: sample)"
    )
    args = parser.parse_args()

    checker = PreflightChecker(sample_mode=not args.full)
    success = checker.run_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
