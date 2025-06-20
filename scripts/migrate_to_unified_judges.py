#!/usr/bin/env python3
"""Migration script to update CJE codebase to unified judge system.

This script performs the following migrations:
1. Updates imports from old to new modules
2. Replaces float-returning judges with JudgeScore-returning ones
3. Updates score access patterns to use compatibility layer
4. Migrates existing JSONL files to unified format
"""

import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import ast


class CodeMigrator:
    """Handles code migration to unified judge system."""

    # Import replacements
    IMPORT_REPLACEMENTS = [
        # Judge imports
        (
            r"from cje\.judge\.judges import Judge",
            "from cje.judge.judges_unified import Judge",
        ),
        (
            r"from cje\.judge import Judge\b",
            "from cje.judge import Judge  # Using unified Judge",
        ),
        # Schema imports
        (
            r"from cje\.judge\.schemas import JudgeScore",
            "from cje.judge.schemas_unified import JudgeScore",
        ),
        (
            r"from cje\.judge\.schemas import (\w+)",
            r"from cje.judge.schemas_unified import \1",
        ),
        # Factory imports
        (
            r"from cje\.judge\.factory import JudgeFactory",
            "from cje.judge.factory_unified import JudgeFactory",
        ),
        (
            r"from cje\.judge import JudgeFactory",
            "from cje.judge import JudgeFactory  # Using unified factory",
        ),
        # API judge imports
        (
            r"from cje\.judge\.api_judge import APIJudge",
            "from cje.judge.api_judge_unified import APIJudge",
        ),
        # Base imports
        (
            r"from cje\.judge\.base import BaseJudge",
            "from cje.judge.base import BaseJudge\nfrom cje.judge.judges_unified import Judge",
        ),
    ]

    # Code pattern replacements
    CODE_REPLACEMENTS = [
        # Judge.score() return type annotations
        (r"def score\(self,([^)]+)\) -> float:", r"def score(self,\1) -> JudgeScore:"),
        (
            r"def score_batch\(self,([^)]+)\) -> List\[float\]:",
            r"def score_batch(self,\1) -> List[JudgeScore]:",
        ),
        # Score access patterns
        (
            r"score = judge\.score\(([^)]+)\)\s*\n",
            r"score_result = judge.score(\1)\nscore = float(score_result.mean)\n",
        ),
        (
            r"scores = judge\.score_batch\(([^)]+)\)\s*\n",
            r"score_results = judge.score_batch(\1)\nscores = [float(s.mean) for s in score_results]\n",
        ),
        # Direct score field access
        (
            r'row\["score_raw"\](?!\s*=)',
            'ScoreCompatibilityLayer.get_score_value(row, "score_raw")',
        ),
        (
            r'row\.get\("score_raw"[^)]*\)',
            'ScoreCompatibilityLayer.get_score_value(row, "score_raw")',
        ),
        # Score assignment
        (
            r'row\["score_raw"\] = score\b',
            'row = update_row_with_score(row, score, "score_raw")',
        ),
    ]

    # Files to skip
    SKIP_PATTERNS = [
        "*.pyc",
        "__pycache__",
        ".git",
        "*.egg-info",
        "migrate_to_unified_judges.py",  # Don't migrate self
        "*_unified.py",  # Skip already unified files
    ]

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.changes: List[Tuple[Path, List[str]]] = []

    def should_skip(self, path: Path) -> bool:
        """Check if file should be skipped."""
        for pattern in self.SKIP_PATTERNS:
            if path.match(pattern):
                return True
        return False

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single Python file.

        Returns True if changes were made.
        """
        if self.should_skip(file_path):
            return False

        try:
            content = file_path.read_text()
            original_content = content
            file_changes = []

            # Apply import replacements
            for old_pattern, new_pattern in self.IMPORT_REPLACEMENTS:
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_pattern, content)
                    file_changes.append(f"Updated import: {old_pattern}")

            # Apply code replacements
            for old_pattern, new_pattern in self.CODE_REPLACEMENTS:
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_pattern, content)
                    file_changes.append(f"Updated code pattern: {old_pattern[:50]}...")

            # Add necessary imports if score compatibility layer is used
            if (
                "ScoreCompatibilityLayer" in content
                and "from cje.utils.score_storage import" not in content
            ):
                # Add import at the top after other imports
                lines = content.split("\n")
                import_added = False
                for i, line in enumerate(lines):
                    if line.startswith("from cje.") and not import_added:
                        lines.insert(
                            i + 1,
                            "from cje.utils.score_storage import ScoreCompatibilityLayer, update_row_with_score",
                        )
                        import_added = True
                        file_changes.append("Added score storage imports")
                        break
                content = "\n".join(lines)

            # Check if changes were made
            if content != original_content:
                if not self.dry_run:
                    # Backup original
                    backup_path = file_path.with_suffix(".py.bak")
                    shutil.copy2(file_path, backup_path)

                    # Write migrated content
                    file_path.write_text(content)

                self.changes.append((file_path, file_changes))
                return True

        except Exception as e:
            print(f"Error migrating {file_path}: {e}")

        return False

    def migrate_directory(self, directory: Path) -> None:
        """Migrate all Python files in directory recursively."""
        py_files = list(directory.rglob("*.py"))

        print(f"Found {len(py_files)} Python files to check")

        migrated_count = 0
        for py_file in py_files:
            if self.migrate_file(py_file):
                migrated_count += 1
                print(f"{'[DRY RUN] ' if self.dry_run else ''}Migrated: {py_file}")

        print(f"\nMigrated {migrated_count} files")

    def print_summary(self) -> None:
        """Print migration summary."""
        if not self.changes:
            print("\nNo changes needed!")
            return

        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)

        for file_path, changes in self.changes:
            print(f"\n{file_path}:")
            for change in changes:
                print(f"  - {change}")


class DataMigrator:
    """Handles JSONL data file migration."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def migrate_jsonl_files(self, directory: Path) -> None:
        """Migrate all JSONL files to unified format."""
        from cje.utils.score_storage import migrate_jsonl_to_unified

        jsonl_files = list(directory.rglob("*scores*.jsonl"))
        print(f"\nFound {len(jsonl_files)} score JSONL files")

        for jsonl_file in jsonl_files:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Migrating: {jsonl_file}")

            if not self.dry_run:
                # Create backup
                backup_path = jsonl_file.with_suffix(".jsonl.bak")
                shutil.copy2(jsonl_file, backup_path)

                # Migrate
                try:
                    migrate_jsonl_to_unified(jsonl_file)
                except Exception as e:
                    print(f"  Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate CJE codebase to unified judge system"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--code-only", action="store_true", help="Only migrate code, not data files"
    )
    parser.add_argument(
        "--data-only", action="store_true", help="Only migrate data files, not code"
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to migrate (default: current directory)",
    )

    args = parser.parse_args()

    if args.code_only and args.data_only:
        print("Error: Cannot specify both --code-only and --data-only")
        sys.exit(1)

    print(f"Migration target: {args.path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Code migration
    if not args.data_only:
        print("=== CODE MIGRATION ===")
        code_migrator = CodeMigrator(dry_run=args.dry_run)
        code_migrator.migrate_directory(args.path)
        code_migrator.print_summary()

    # Data migration
    if not args.code_only:
        print("\n=== DATA MIGRATION ===")
        data_migrator = DataMigrator(dry_run=args.dry_run)
        data_migrator.migrate_jsonl_files(args.path)

    if not args.dry_run:
        print("\n✅ Migration complete!")
        print("Backup files created with .bak extension")
        print("\nNext steps:")
        print("1. Run tests to ensure everything works")
        print("2. Review changes in version control")
        print("3. Delete .bak files once confirmed working")
    else:
        print("\n✅ Dry run complete!")
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
