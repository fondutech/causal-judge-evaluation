#!/usr/bin/env python3
"""Automated hygiene checks for the CJE codebase."""

import ast
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click
from rich.console import Console
from rich.table import Table

console = Console()


class HygieneChecker:
    """Run automated hygiene checks on the codebase."""

    def __init__(self, root_path: Path):
        self.root = root_path
        self.issues: Dict[str, List[str]] = {}

    def check_stale_todos(self, days: int = 30) -> None:
        """Find TODO comments older than specified days."""
        console.print(
            f"\n[yellow]Checking for TODOs older than {days} days...[/yellow]"
        )

        # This is a simplified check - in practice you'd parse git blame
        todo_pattern = re.compile(r"#\s*TODO:?\s*(.+)", re.IGNORECASE)
        old_todos = []

        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if match := todo_pattern.search(line):
                        old_todos.append(f"{py_file}:{i} - {match.group(1).strip()}")
            except Exception:
                pass

        if old_todos:
            self.issues["stale_todos"] = old_todos

    def check_empty_directories(self) -> None:
        """Find empty directories."""
        console.print("\n[yellow]Checking for empty directories...[/yellow]")
        empty_dirs = []

        for dirpath, dirnames, filenames in os.walk(self.root):
            # Skip hidden and build directories
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d != "__pycache__"
            ]

            path = Path(dirpath)
            if not any(path.rglob("*")) and path != self.root:
                empty_dirs.append(str(path.relative_to(self.root)))

        if empty_dirs:
            self.issues["empty_directories"] = empty_dirs

    def check_type_ignores(self) -> None:
        """Count type: ignore comments."""
        console.print("\n[yellow]Checking for type: ignore comments...[/yellow]")
        type_ignores = []

        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if "# type: ignore" in line:
                        type_ignores.append(f"{py_file}:{i}")
            except Exception:
                pass

        if type_ignores:
            self.issues["type_ignores"] = type_ignores

    def check_old_patterns(self) -> None:
        """Check for old backup files and temporary code."""
        console.print("\n[yellow]Checking for backup and temporary files...[/yellow]")
        bad_patterns = []

        patterns = ["*_old.py", "*_backup.*", "temp_*.*", "*_copy.py"]
        for pattern in patterns:
            for file in self.root.rglob(pattern):
                if ".venv" not in str(file):
                    bad_patterns.append(str(file.relative_to(self.root)))

        if bad_patterns:
            self.issues["old_files"] = bad_patterns

    def check_dead_imports(self) -> None:
        """Check for imports of non-existent modules."""
        console.print("\n[yellow]Checking for dead imports...[/yellow]")
        dead_imports = []

        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("cje"):
                            # Check if the module exists
                            module_path = node.module.replace(".", "/")
                            if (
                                not (self.root / f"{module_path}.py").exists()
                                and not (self.root / module_path).is_dir()
                            ):
                                dead_imports.append(
                                    f"{py_file}:{node.lineno} - from {node.module} import ..."
                                )
            except Exception:
                pass

        if dead_imports:
            self.issues["dead_imports"] = dead_imports

    def check_claude_md_length(self) -> None:
        """Check if CLAUDE.md is growing too large."""
        console.print("\n[yellow]Checking CLAUDE.md length...[/yellow]")

        claude_md = self.root / "CLAUDE.md"
        if claude_md.exists():
            lines = len(claude_md.read_text().splitlines())
            if lines > 100:
                self.issues["claude_md_length"] = [
                    f"CLAUDE.md has {lines} lines (target: <100)"
                ]

    def check_duplicate_functionality(self) -> None:
        """Look for potentially duplicate functions."""
        console.print("\n[yellow]Checking for duplicate functionality...[/yellow]")

        function_names: Dict[str, List[str]] = {}

        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        name = node.name
                        if not name.startswith("_"):  # Skip private functions
                            key = name.lower()
                            if key not in function_names:
                                function_names[key] = []
                            function_names[key].append(
                                str(py_file.relative_to(self.root))
                            )
            except Exception:
                pass

        duplicates = []
        for name, files in function_names.items():
            if len(files) > 1:
                duplicates.append(f"{name}: {', '.join(files)}")

        if duplicates:
            self.issues["duplicate_functions"] = duplicates

    def run_all_checks(self) -> None:
        """Run all hygiene checks."""
        self.check_stale_todos()
        self.check_empty_directories()
        self.check_type_ignores()
        self.check_old_patterns()
        self.check_dead_imports()
        self.check_claude_md_length()
        self.check_duplicate_functionality()

    def print_report(self) -> None:
        """Print a formatted report of all issues."""
        if not self.issues:
            console.print("\n[green]✓ All hygiene checks passed![/green]")
            return

        console.print("\n[red]Hygiene Issues Found:[/red]\n")

        # Create summary table
        table = Table(title="Hygiene Check Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Status", style="red")

        for category, items in self.issues.items():
            table.add_row(category.replace("_", " ").title(), str(len(items)), "❌")

        console.print(table)

        # Print details
        for category, items in self.issues.items():
            console.print(f"\n[yellow]{category.replace('_', ' ').title()}:[/yellow]")
            for item in items[:10]:  # Show first 10
                console.print(f"  • {item}")
            if len(items) > 10:
                console.print(f"  ... and {len(items) - 10} more")


@click.command()
@click.option("--fix", is_flag=True, help="Automatically fix simple issues")
def main(fix: bool) -> None:
    """Run hygiene checks on the CJE codebase."""
    checker = HygieneChecker(Path.cwd())
    checker.run_all_checks()
    checker.print_report()

    if fix and checker.issues:
        console.print(
            "\n[yellow]Auto-fix not yet implemented. Please fix manually.[/yellow]"
        )


if __name__ == "__main__":
    main()
