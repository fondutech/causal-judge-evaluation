"""
Unified Template CLI for CJE

A powerful command-line interface for managing and exploring all CJE templates
in one place. No more separate systems!
"""

import typer
from rich import print
from rich.table import Table
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.columns import Columns
import json
from typing import List, Dict, Any, Union

from ..prompts.manager import TemplateManager

app = typer.Typer(help="🎯 Unified template management for CJE")
console = Console()


@app.command()
def list(
    template_type: str = typer.Option(
        "all", help="Filter by type: 'policy', 'judge', or 'all'"
    ),
    category: str = typer.Option("", help="Filter by category"),
    search: str = typer.Option("", help="Search templates by name or description"),
) -> None:
    """📋 List available templates with powerful filtering."""

    # Apply search filter first
    if search:
        template_names_result = TemplateManager.search_templates(search)
        print(f"🔍 Search results for '{search}':")
    else:
        template_names_result = TemplateManager.list_templates(template_type)  # type: ignore

    # Handle the different return types from list_templates
    if isinstance(template_names_result, dict):
        # When template_type is "all", we get a dict
        template_names = []
        for template_list in template_names_result.values():
            template_names.extend(template_list)
    else:
        # When template_type is specific, we get a list
        template_names = template_names_result

    # Apply category filter
    if category:
        by_category = TemplateManager.list_by_category(category)
        if category in by_category:
            template_names = [
                name for name in template_names if name in by_category[category]
            ]
        else:
            print(f"[red]❌ Unknown category: {category}[/red]")
            print(
                f"Available categories: {', '.join(TemplateManager.get_categories().keys())}"
            )
            raise typer.Exit(1)

    if not template_names:
        print("[yellow]No templates found matching your criteria[/yellow]")
        return

    # Group by type for display
    policies = []
    judges = []

    for name in template_names:
        template = TemplateManager.get_template(name)
        if template.get("type") == "policy":
            policies.append(name)
        elif template.get("type") == "judge":
            judges.append(name)

    # Display policy templates
    if policies and template_type in ["all", "policy"]:
        policy_table = Table(title="🤖 Policy Templates", show_header=True)
        policy_table.add_column("Name", style="cyan", width=20)
        policy_table.add_column("Category", style="blue", width=12)
        policy_table.add_column("Description", style="green")

        for name in sorted(policies):
            template = TemplateManager.get_template(name)
            policy_table.add_row(
                name,
                template.get("category", "unknown"),
                template.get("description", "No description"),
            )

        console.print(policy_table)
        console.print()

    # Display judge templates
    if judges and template_type in ["all", "judge"]:
        judge_table = Table(title="⚖️ Judge Templates", show_header=True)
        judge_table.add_column("Name", style="cyan", width=20)
        judge_table.add_column("Category", style="blue", width=12)
        judge_table.add_column("Description", style="green")

        for name in sorted(judges):
            template = TemplateManager.get_template(name)
            judge_table.add_row(
                name,
                template.get("category", "unknown"),
                template.get("description", "No description"),
            )

        console.print(judge_table)


@app.command()
def show(
    template_name: str = typer.Argument(..., help="Name of the template to inspect")
) -> None:
    """🔍 Show detailed information about a template."""

    template = TemplateManager.get_template(template_name)
    if not template:
        print(f"[red]❌ Template '{template_name}' not found[/red]")
        raise typer.Exit(1)

    template_type = template.get("type", "unknown")
    category = template.get("category", "unknown")
    description = template.get("description", "No description")

    # Header
    type_emoji = (
        "🤖" if template_type == "policy" else "⚖️" if template_type == "judge" else "❓"
    )
    print(f"{type_emoji} [bold cyan]{template_name}[/bold cyan] ({template_type})")
    print(f"[blue]Category:[/blue] {category}")
    print(f"[green]Description:[/green] {description}")
    print()

    # Show template content
    if template_type == "policy":
        # System prompt
        if template.get("system_prompt"):
            print("[bold]System Prompt:[/bold]")
            syntax = Syntax(
                template["system_prompt"], "jinja2", theme="monokai", line_numbers=True
            )
            console.print(Panel(syntax, title="System Prompt", border_style="blue"))
            print()

        # User template
        if template.get("user_template"):
            print("[bold]User Template:[/bold]")
            syntax = Syntax(
                template["user_template"], "jinja2", theme="monokai", line_numbers=True
            )
            console.print(Panel(syntax, title="User Template", border_style="green"))
            print()

    elif template_type == "judge":
        print("[bold]Judge Template:[/bold]")
        syntax = Syntax(
            template["template"], "jinja2", theme="monokai", line_numbers=True
        )
        console.print(Panel(syntax, title="Judge Template", border_style="yellow"))
        print()

    # Show variables
    variables = TemplateManager.extract_variables(template_name)
    if variables:
        print("[bold]Template Variables:[/bold]")
        var_columns = []
        for var in sorted(variables):
            var_columns.append(f"[yellow]{{{{ {var} }}}}[/yellow]")

        console.print(Columns(var_columns, equal=True, expand=True))
        print()

    # Show default variables
    if template.get("variables"):
        print("[bold]Default Values:[/bold]")
        for key, value in template["variables"].items():
            # Truncate long values
            display_value = str(value)
            if len(display_value) > 60:
                display_value = display_value[:57] + "..."
            print(f"  [cyan]{key}:[/cyan] {display_value}")


@app.command()
def render(
    template_name: str = typer.Argument(..., help="Template to render"),
    context: str = typer.Option("What is machine learning?", help="Context/question"),
    response: str = typer.Option(
        "ML is a subset of AI that learns from data",
        help="Response (for judge templates)",
    ),
    variables: str = typer.Option("", help="JSON string of template variables"),
    output_format: str = typer.Option(
        "pretty", help="Output format: 'pretty' or 'json'"
    ),
) -> None:
    """🎨 Render a template with sample data."""

    template = TemplateManager.get_template(template_name)
    if not template:
        print(f"[red]❌ Template '{template_name}' not found[/red]")
        raise typer.Exit(1)

    # Parse variables
    template_variables: Dict[str, Any] = {}
    if variables:
        try:
            template_variables = json.loads(variables)
        except json.JSONDecodeError as e:
            print(f"[red]❌ Invalid JSON in variables: {e}[/red]")
            raise typer.Exit(1)

    try:
        template_type = template.get("type")

        if template_type == "policy":
            result = TemplateManager.render_policy(
                template_name, context, template_variables
            )

            if output_format == "json":
                print(json.dumps(result, indent=2))
            else:
                print(
                    f"🤖 [bold cyan]Rendered Policy Template: {template_name}[/bold cyan]"
                )
                print()

                if "system_prompt" in result:
                    console.print(
                        Panel(
                            result["system_prompt"],
                            title="System Prompt",
                            border_style="blue",
                        )
                    )
                    print()

                if "user_message" in result:
                    console.print(
                        Panel(
                            result["user_message"],
                            title="User Message",
                            border_style="green",
                        )
                    )

        elif template_type == "judge":
            result_str = TemplateManager.render_judge(
                template_name, context, response, template_variables
            )

            if output_format == "json":
                print(json.dumps({"rendered_template": result_str}, indent=2))
            else:
                print(
                    f"⚖️ [bold cyan]Rendered Judge Template: {template_name}[/bold cyan]"
                )
                print()
                console.print(
                    Panel(result_str, title="Judge Prompt", border_style="yellow")
                )

        else:
            print(f"[red]❌ Unknown template type: {template_type}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        print(f"[red]❌ Error rendering template: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def categories() -> None:
    """📂 Show all template categories."""

    categories = TemplateManager.get_categories()
    by_category = TemplateManager.list_by_category()

    table = Table(title="📂 Template Categories", show_header=True)
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Description", style="green", width=40)
    table.add_column("Templates", style="blue")

    for category, description in categories.items():
        template_count = len(by_category.get(category, []))
        category_templates = by_category.get(category, [])
        template_names = ", ".join(category_templates[:3])
        if template_count > 3:
            template_names += f" (+{template_count - 3} more)"

        table.add_row(category, description, template_names)

    console.print(table)


@app.command()
def validate(
    template_name: str = typer.Argument(..., help="Template to validate")
) -> None:
    """✅ Validate a template."""

    is_valid = TemplateManager.validate_template(template_name)

    if is_valid:
        template = TemplateManager.get_template(template_name)
        template_type = template.get("type", "unknown")
        print(
            f"[green]✅ Template '{template_name}' ({template_type}) is valid[/green]"
        )
    else:
        print(f"[red]❌ Template '{template_name}' is invalid or not found[/red]")
        raise typer.Exit(1)


@app.command()
def search(query: str = typer.Argument(..., help="Search query")) -> None:
    """🔍 Search templates by name or description."""

    matches = TemplateManager.search_templates(query)

    if not matches:
        print(f"[yellow]No templates found matching '{query}'[/yellow]")
        return

    print(f"🔍 [bold]Search results for '{query}':[/bold]")
    print()

    for name in matches:
        template = TemplateManager.get_template(name)
        template_type = template.get("type", "unknown")
        description = template.get("description", "No description")

        type_emoji = "🤖" if template_type == "policy" else "⚖️"
        print(f"{type_emoji} [cyan]{name}[/cyan] ({template_type})")
        print(f"   {description}")
        print()


@app.command()
def create(
    name: str = typer.Argument(..., help="Name for the new template"),
    template_type: str = typer.Option(..., help="Template type: 'policy' or 'judge'"),
    interactive: bool = typer.Option(True, help="Use interactive mode"),
) -> None:
    """🎨 Create a new template interactively."""

    if template_type not in ["policy", "judge"]:
        print(f"[red]❌ Invalid template type: {template_type}[/red]")
        print("Valid types: 'policy', 'judge'")
        raise typer.Exit(1)

    # Get all templates to check for duplicates
    all_templates_result = TemplateManager.list_templates("all")
    if isinstance(all_templates_result, dict):
        all_template_names = []
        for template_list in all_templates_result.values():
            all_template_names.extend(template_list)
    else:
        all_template_names = all_templates_result

    if name in all_template_names:
        print(f"[red]❌ Template '{name}' already exists[/red]")
        raise typer.Exit(1)

    if not interactive:
        print("[yellow]Non-interactive mode not yet implemented[/yellow]")
        return

    print(f"🎨 [bold]Creating new {template_type} template: {name}[/bold]")
    print()

    # Collect template data
    template_data: Dict[str, Any] = {}

    # Description
    description = typer.prompt("Description")
    template_data["description"] = description

    # Category
    categories_dict = TemplateManager.get_categories()
    categories_list = [str(key) for key in categories_dict.keys()]
    print(f"Available categories: {', '.join(categories_list)}")
    category = typer.prompt("Category", default="basic")
    template_data["category"] = category

    if template_type == "policy":
        # System prompt
        print("\nEnter system prompt (press Ctrl+D when done):")
        system_lines = []
        try:
            while True:
                line = input()
                system_lines.append(line)
        except EOFError:
            pass
        template_data["system_prompt"] = "\n".join(system_lines)

        # User template
        print("\nEnter user template (press Ctrl+D when done):")
        user_lines = []
        try:
            while True:
                line = input()
                user_lines.append(line)
        except EOFError:
            pass
        template_data["user_template"] = "\n".join(user_lines)

    elif template_type == "judge":
        # Judge template
        print("\nEnter judge template (press Ctrl+D when done):")
        template_lines = []
        try:
            while True:
                line = input()
                template_lines.append(line)
        except EOFError:
            pass
        template_data["template"] = "\n".join(template_lines)

    # Variables
    add_variables = typer.confirm("Add default variables?", default=False)
    if add_variables:
        variables = {}
        while True:
            var_name = typer.prompt("Variable name (empty to finish)", default="")
            if not var_name:
                break
            var_value = typer.prompt(f"Default value for {var_name}")
            variables[var_name] = var_value

        if variables:
            template_data["variables"] = variables

    # Register the template
    try:
        TemplateManager.register_template(name, template_type, template_data)  # type: ignore
        print(f"[green]✅ Template '{name}' created successfully![/green]")
    except Exception as e:
        print(f"[red]❌ Error creating template: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """📊 Show template statistics."""

    all_templates_result = TemplateManager.list_templates("all")
    policy_templates_result = TemplateManager.list_templates("policy")
    judge_templates_result = TemplateManager.list_templates("judge")
    by_category = TemplateManager.list_by_category()

    # Handle the different return types
    if isinstance(all_templates_result, dict):
        total_count = sum(
            len(template_list) for template_list in all_templates_result.values()
        )
    else:
        total_count = len(all_templates_result)

    # Ensure policy and judge templates are lists
    policy_templates: List[str] = []
    judge_templates: List[str] = []

    if not isinstance(policy_templates_result, dict):
        policy_templates = policy_templates_result

    if not isinstance(judge_templates_result, dict):
        judge_templates = judge_templates_result

    print("📊 [bold]Template Statistics[/bold]")
    print()

    # Overall stats
    stats_table = Table(show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="green")

    stats_table.add_row("Total Templates", str(total_count))
    stats_table.add_row("Policy Templates", str(len(policy_templates)))
    stats_table.add_row("Judge Templates", str(len(judge_templates)))
    stats_table.add_row("Categories", str(len(by_category)))

    console.print(stats_table)
    print()

    # Category breakdown
    category_table = Table(title="Templates by Category", show_header=True)
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Count", style="green")
    category_table.add_column("Templates", style="blue")

    for category, templates in by_category.items():
        template_names = ", ".join(templates[:3])
        if len(templates) > 3:
            template_names += f" (+{len(templates) - 3} more)"

        category_table.add_row(category, str(len(templates)), template_names)

    console.print(category_table)


if __name__ == "__main__":
    app()
