"""
Unified Jinja2 template engine for CJE.

This provides a consistent template rendering system that:
1. Uses Jinja2 for all templates
2. Handles variable substitution properly
3. Supports both judge and policy templates
"""

from typing import Dict, Any, Optional
import jinja2
from jinja2 import Environment, Template, meta


class TemplateEngine:
    """Unified template engine using Jinja2."""

    def __init__(self) -> None:
        # Configure Jinja2 environment
        self.env = Environment(
            # Use standard Jinja2 syntax: {{ variable }}
            variable_start_string="{{",
            variable_end_string="}}",
            # Trim whitespace for cleaner output
            trim_blocks=True,
            lstrip_blocks=True,
            # Enable auto-escaping for safety
            autoescape=False,  # We're not generating HTML
        )

    def render(
        self, template_str: str, variables: Dict[str, Any], strict: bool = False
    ) -> str:
        """
        Render a template with given variables.

        Args:
            template_str: The template string with {{ variable }} syntax
            variables: Dictionary of variable values
            strict: If True, raise error on undefined variables

        Returns:
            Rendered template string
        """
        # Create template
        template = self.env.from_string(template_str)

        if strict:
            # Check for undefined variables
            ast = self.env.parse(template_str)
            undefined_vars = meta.find_undeclared_variables(ast)
            missing = undefined_vars - set(variables.keys())
            if missing:
                raise ValueError(f"Missing template variables: {missing}")

        # Render template
        return str(template.render(**variables))

    def get_variables(self, template_str: str) -> set[str]:
        """
        Extract all variables used in a template.

        Args:
            template_str: The template string

        Returns:
            Set of variable names
        """
        ast = self.env.parse(template_str)
        return set(meta.find_undeclared_variables(ast))

    def prepare_for_langchain(
        self, template_str: str, static_vars: Dict[str, Any]
    ) -> str:
        """
        Prepare a template for LangChain by:
        1. Substituting static variables
        2. Converting remaining {{ var }} to { var } for LangChain

        Args:
            template_str: Original template with {{ var }} syntax
            static_vars: Variables to substitute now (not at runtime)

        Returns:
            Template ready for LangChain with { var } syntax
        """
        # First, render static variables using Jinja2
        template = self.env.from_string(template_str)

        # Get all variables in template
        all_vars = self.get_variables(template_str)

        # Separate static and runtime variables
        runtime_vars = all_vars - set(static_vars.keys())

        # Render with static variables, keeping runtime variables as placeholders
        # We'll use a custom undefined handler
        from jinja2 import Undefined

        class KeepUndefined(Undefined):
            def __str__(self) -> str:
                return str("{{" + str(self._undefined_name) + "}}")

        temp_env = Environment(undefined=KeepUndefined)
        temp_template = temp_env.from_string(template_str)
        partially_rendered = temp_template.render(**static_vars)

        # Now convert remaining {{ var }} to { var } for LangChain
        import re

        langchain_ready = re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{\1}", partially_rendered)

        return str(langchain_ready)


# Global instance
template_engine = TemplateEngine()


def render_template(template_str: str, **kwargs: Any) -> str:
    """Convenience function to render a template."""
    return template_engine.render(template_str, kwargs)


def prepare_judge_template(template_str: str, config_vars: Dict[str, Any]) -> str:
    """
    Prepare a judge template for use with LangChain.

    Config variables (like min_score, max_score) are substituted immediately.
    Runtime variables (like context, response) are converted to LangChain format.
    """
    return template_engine.prepare_for_langchain(template_str, config_vars)
