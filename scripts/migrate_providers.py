#!/usr/bin/env python3
"""
Migration script to consolidate the duplicate provider hierarchy.

This script shows the exact changes needed to migrate from the dual
provider system to a single consolidated provider system.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


class ProviderMigration:
    """Handle the provider consolidation migration."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.root = Path.cwd() / "cje" / "judge" / "providers"
        self.changes: List[Tuple[str, str, str]] = []

    def analyze_current_state(self) -> Dict[str, List[str]]:
        """Analyze the current provider structure."""
        regular_providers = []
        structured_providers = []

        for file in self.root.glob("*.py"):
            if file.name.startswith("structured_"):
                structured_providers.append(file.name)
            elif file.name not in [
                "__init__.py",
                "base.py",
                "structured_base.py",
                "provider_registry.py",
                "schemas.py",
            ]:
                regular_providers.append(file.name)

        return {"regular": regular_providers, "structured": structured_providers}

    def show_migration_plan(self) -> None:
        """Display the migration plan."""
        console.print(
            "\n[bold blue]Provider Consolidation Migration Plan[/bold blue]\n"
        )

        # Step 1: File changes
        console.print("[yellow]1. File Structure Changes:[/yellow]")
        changes = [
            ("DELETE", "base.py", "Regular provider base (unused)"),
            ("DELETE", "openai.py", "Regular OpenAI provider (unused)"),
            ("DELETE", "anthropic.py", "Regular Anthropic provider (unused)"),
            ("DELETE", "fireworks.py", "Regular Fireworks provider (unused)"),
            ("DELETE", "together.py", "Regular Together provider (unused)"),
            ("DELETE", "groq.py", "Regular Groq provider (unused)"),
            (
                "DELETE",
                "structured_base.py",
                "Will be replaced by base_consolidated.py",
            ),
            ("RENAME", "structured_openai.py → openai.py", "Drop 'structured' prefix"),
            (
                "RENAME",
                "structured_anthropic.py → anthropic.py",
                "Drop 'structured' prefix",
            ),
            (
                "RENAME",
                "structured_fireworks.py → fireworks.py",
                "Drop 'structured' prefix",
            ),
            (
                "RENAME",
                "structured_together.py → together.py",
                "Drop 'structured' prefix",
            ),
            ("RENAME", "structured_groq.py → groq.py", "Drop 'structured' prefix"),
            ("UPDATE", "provider_registry.py", "Remove provider_cls field"),
            ("UPDATE", "__init__.py", "Update imports"),
        ]

        for action, file, desc in changes:
            if action == "DELETE":
                console.print(f"  ❌ {file}: {desc}")
            elif action == "RENAME":
                console.print(f"  📝 {file}: {desc}")
            elif action == "UPDATE":
                console.print(f"  ✏️  {file}: {desc}")

        # Step 2: Code changes
        console.print("\n[yellow]2. Code Changes:[/yellow]")

        # Show example of new base class
        console.print("\n[green]New Base Class (base.py):[/green]")
        new_base = '''class Provider(ABC):
    """Single provider interface for all LLM interactions."""
    
    def get_structured_model(self, model_name: str, schema: Type[BaseModel], ...) -> Any:
        """Get model configured for structured output."""
        chat_model = self.get_chat_model(model_name, ...)
        return chat_model.with_structured_output(schema, ...)'''

        console.print(
            Panel(
                Syntax(new_base, "python", theme="monokai"),
                title="Consolidated Base Class",
            )
        )

        # Show example provider
        console.print("\n[green]Example Provider (openai.py):[/green]")
        new_provider = '''class OpenAIProvider(Provider):
    """OpenAI provider with structured output support."""
    
    def get_chat_model(self, model_name: str, ...) -> ChatOpenAI:
        return ChatOpenAI(model=model_name, ...)
    
    def get_recommended_method(self) -> str:
        return "json_mode"'''

        console.print(
            Panel(
                Syntax(new_provider, "python", theme="monokai"),
                title="Simplified Provider",
            )
        )

        # Step 3: Usage changes
        console.print("\n[yellow]3. Usage Changes:[/yellow]")

        console.print("\n[red]Before:[/red]")
        old_usage = """from .structured_openai import StructuredOpenAIProvider
provider = StructuredOpenAIProvider(api_key=key)"""
        console.print(Syntax(old_usage, "python", theme="monokai"))

        console.print("\n[green]After:[/green]")
        new_usage = """from .openai import OpenAIProvider
provider = OpenAIProvider(api_key=key)"""
        console.print(Syntax(new_usage, "python", theme="monokai"))

        # Step 4: Benefits
        console.print("\n[yellow]4. Benefits:[/yellow]")
        benefits = [
            "✅ Single provider hierarchy (no confusion)",
            "✅ Cleaner imports (no 'Structured' prefix)",
            "✅ Less code to maintain (~50% reduction)",
            "✅ Clearer purpose (all providers support structured output)",
            "✅ Simplified registry (no unused provider_cls field)",
        ]
        for benefit in benefits:
            console.print(f"  {benefit}")

    def check_usage_safety(self) -> bool:
        """Verify that regular providers are not used anywhere."""
        console.print("\n[yellow]Checking for regular provider usage...[/yellow]")

        # Check for imports of regular providers
        regular_imports = [
            "from cje.judge.providers.openai import OpenAIProvider",
            "from cje.judge.providers.anthropic import AnthropicProvider",
            "from cje.judge.providers.fireworks import FireworksProvider",
            "from cje.judge.providers.together import TogetherProvider",
            "from cje.judge.providers.groq import GroqProvider",
        ]

        found_usage = False
        for py_file in Path.cwd().rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                for import_stmt in regular_imports:
                    if import_stmt in content:
                        console.print(f"  ❌ Found usage in {py_file}: {import_stmt}")
                        found_usage = True
            except Exception:
                pass

        if not found_usage:
            console.print("  ✅ No usage of regular providers found - safe to remove!")

        return not found_usage

    def generate_sed_commands(self) -> None:
        """Generate sed commands for automatic migration."""
        console.print("\n[yellow]5. Automated Migration Commands:[/yellow]")

        commands = [
            "# Update imports in APIJudge",
            "sed -i '' 's/StructuredOpenAIProvider/OpenAIProvider/g' cje/judge/api_judge.py",
            "sed -i '' 's/StructuredAnthropicProvider/AnthropicProvider/g' cje/judge/api_judge.py",
            "sed -i '' 's/StructuredFireworksProvider/FireworksProvider/g' cje/judge/api_judge.py",
            "sed -i '' 's/StructuredTogetherProvider/TogetherProvider/g' cje/judge/api_judge.py",
            "sed -i '' 's/StructuredGroqProvider/GroqProvider/g' cje/judge/api_judge.py",
            "",
            "# Update imports in __init__.py",
            "sed -i '' 's/from .structured_/from ./g' cje/judge/providers/__init__.py",
            "",
            "# Update registry",
            "sed -i '' 's/structured_cls=/provider_cls=/g' cje/judge/providers/provider_registry.py",
        ]

        for cmd in commands:
            if cmd.startswith("#"):
                console.print(f"\n{cmd}")
            elif cmd:
                console.print(f"  $ {cmd}")


@click.command()
@click.option(
    "--execute", is_flag=True, help="Execute the migration (default is dry run)"
)
def main(execute: bool) -> None:
    """Run the provider consolidation migration."""
    migration = ProviderMigration(dry_run=not execute)

    # Analyze current state
    current = migration.analyze_current_state()
    console.print(f"\nFound {len(current['regular'])} regular providers (unused)")
    console.print(f"Found {len(current['structured'])} structured providers (in use)")

    # Show migration plan
    migration.show_migration_plan()

    # Check safety
    safe = migration.check_usage_safety()

    if not safe:
        console.print("\n[red]⚠️  WARNING: Found usage of regular providers![/red]")
        console.print("Please review and update the code before proceeding.")
        return

    # Generate automation commands
    migration.generate_sed_commands()

    if execute:
        console.print("\n[green]🚀 Executing migration...[/green]")
        console.print("[red]Not implemented yet - please run commands manually[/red]")
    else:
        console.print(
            "\n[cyan]This was a dry run. Use --execute to perform the migration.[/cyan]"
        )


if __name__ == "__main__":
    main()
