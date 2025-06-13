"""
Unified Template Manager for CJE

This module provides a single, powerful interface for managing all prompt templates
in CJE, whether they're for LLM policies or judges. No more separate systems!
"""

from typing import Dict, Any, List, Optional, Union, Literal, Tuple, cast
import jinja2
import re
from .unified_templates import UNIFIED_TEMPLATES, TEMPLATE_CATEGORIES


TemplateType = Literal["policy", "judge", "all"]


class TemplateManager:
    """Unified template manager for all CJE prompt templates."""

    @classmethod
    def list_templates(
        cls, template_type: TemplateType = "all"
    ) -> Union[List[str], Dict[str, List[str]]]:
        """List available templates by type."""
        if template_type == "all":
            # Return organized by type
            result: Dict[str, List[str]] = {}
            for template_name_key, template_info in UNIFIED_TEMPLATES.items():
                template_name = cast(str, template_name_key)
                template_type_key = template_info.get("type", "unknown")
                if template_type_key not in result:
                    result[template_type_key] = []
                result[template_type_key].append(template_name)
            return result

        return [
            cast(str, name)
            for name, info in UNIFIED_TEMPLATES.items()
            if info.get("type") == template_type
        ]

    @classmethod
    def list_by_category(cls, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List templates organized by category."""
        if category:
            return {
                category: [
                    cast(str, name)
                    for name, info in UNIFIED_TEMPLATES.items()
                    if info.get("category") == category
                ]
            }

        result = {}
        for cat in TEMPLATE_CATEGORIES.keys():
            result[cat] = [
                cast(str, name)
                for name, info in UNIFIED_TEMPLATES.items()
                if info.get("category") == cat
            ]
        return result

    @classmethod
    def get_template(cls, name: str) -> Dict[str, Any]:
        """Get complete template information."""
        return UNIFIED_TEMPLATES.get(name, {})

    @classmethod
    def get_categories(cls) -> Dict[str, str]:
        """Get all template categories and their descriptions."""
        return TEMPLATE_CATEGORIES.copy()

    @classmethod
    def validate_template(cls, name: str) -> bool:
        """Validate that a template exists and is well-formed."""
        template = cls.get_template(name)
        if not template:
            return False

        template_type = template.get("type")
        if template_type == "policy":
            return all(key in template for key in ["system_prompt", "user_template"])
        elif template_type == "judge":
            return "template" in template

        return False

    @classmethod
    def validate_syntax(cls, template_str: str) -> Tuple[bool, Optional[str]]:
        """Validate Jinja2 template syntax."""
        try:
            jinja2.Template(template_str)
            return True, None
        except jinja2.TemplateSyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Template error: {e}"

    @classmethod
    def render_policy(
        cls,
        template_name: str,
        context: str,
        variables: Optional[Dict[str, Any]] = None,
        **extra_variables: Any,
    ) -> Dict[str, str]:
        """Render a policy template."""
        template = cls.get_template(template_name)
        if not template or template.get("type") != "policy":
            raise ValueError(f"Policy template '{template_name}' not found")

        # Merge variables
        merged_vars = {"context": context}
        if "variables" in template:
            merged_vars.update(template["variables"])
        if variables:
            merged_vars.update(variables)
        merged_vars.update(extra_variables)

        # Render system prompt
        system_template = jinja2.Template(template["system_prompt"])
        system_prompt = system_template.render(**merged_vars)

        # Render user template
        user_template = jinja2.Template(template["user_template"])
        user_message = user_template.render(**merged_vars)

        return {"system_prompt": system_prompt, "user_message": user_message}

    @classmethod
    def render_judge(
        cls,
        template_name: str,
        context: str,
        response: str,
        variables: Optional[Dict[str, Any]] = None,
        **extra_variables: Any,
    ) -> str:
        """Render a judge template."""
        template = cls.get_template(template_name)
        if not template or template.get("type") != "judge":
            raise ValueError(f"Judge template '{template_name}' not found")

        # Merge variables
        merged_vars = {"context": context, "response": response}
        if "variables" in template:
            merged_vars.update(template["variables"])
        if variables:
            merged_vars.update(variables)
        merged_vars.update(extra_variables)

        # Render template
        judge_template = jinja2.Template(template["template"])
        result = judge_template.render(**merged_vars)
        return str(result)

    @classmethod
    def extract_variables(cls, template_name: str) -> List[str]:
        """Extract all variable names from a template."""
        template = cls.get_template(template_name)
        if not template:
            return []

        text = ""
        template_type = template.get("type")

        if template_type == "policy":
            text += template.get("system_prompt", "") + " "
            text += template.get("user_template", "")
        elif template_type == "judge":
            text = template.get("template", "")

        # Extract Jinja2 variables
        variables = re.findall(r"\{\{\s*(\w+)\s*\}\}", text)
        return list(set(variables))

    @classmethod
    def register_template(
        cls, name: str, template_type: TemplateType, template_data: Dict[str, Any]
    ) -> None:
        """Register a new template."""
        if template_type not in ["policy", "judge"]:
            raise ValueError(f"Invalid template type: {template_type}")

        # Validate required fields
        if template_type == "policy":
            required = ["system_prompt", "user_template"]
            missing = [key for key in required if key not in template_data]
            if missing:
                raise ValueError(
                    f"Missing required fields for policy template: {missing}"
                )

            # Validate syntax
            for key in ["system_prompt", "user_template"]:
                is_valid, error = cls.validate_syntax(template_data[key])
                if not is_valid:
                    raise ValueError(f"Invalid {key} syntax: {error}")

        elif template_type == "judge":
            if "template" not in template_data:
                raise ValueError("Judge template must have 'template' field")

            is_valid, error = cls.validate_syntax(template_data["template"])
            if not is_valid:
                raise ValueError(f"Invalid template syntax: {error}")

        # Add type and register
        template_data["type"] = template_type
        UNIFIED_TEMPLATES[name] = template_data

    @classmethod
    def search_templates(cls, query: str) -> List[str]:
        """Search templates by name or description."""
        query_lower = query.lower()
        matches: List[str] = []

        for name_key in UNIFIED_TEMPLATES:
            name = cast(str, name_key)
            template = UNIFIED_TEMPLATES[name]
            if query_lower in name.lower():
                matches.append(name)
            elif query_lower in template.get("description", "").lower():
                matches.append(name)

        return matches

    @classmethod
    def get_template_info(cls, name: str) -> Dict[str, Any]:
        """Get human-readable template information."""
        template = cls.get_template(name)
        if not template:
            return {}

        info = {
            "name": name,
            "type": template.get("type"),
            "category": template.get("category"),
            "description": template.get("description"),
            "variables": cls.extract_variables(name),
        }

        if template.get("variables"):
            info["default_variables"] = template["variables"]

        return info

    @classmethod
    def create_template_variant(
        cls, base_template: str, variant_name: str, modifications: Dict[str, Any]
    ) -> None:
        """Create a variant of an existing template."""
        base = cls.get_template(base_template)
        if not base:
            raise ValueError(f"Base template '{base_template}' not found")

        # Create variant by merging modifications
        variant = base.copy()
        variant.update(modifications)

        # Update description to indicate it's a variant
        if "description" not in modifications:
            variant["description"] = (
                f"Variant of {base_template}: {base.get('description', '')}"
            )

        UNIFIED_TEMPLATES[variant_name] = variant


# Convenience functions for common operations
def render_policy(template_name: str, context: str, **variables: Any) -> Dict[str, str]:
    """Quick policy template rendering."""
    return TemplateManager.render_policy(template_name, context, variables)


def render_judge(
    template_name: str, context: str, response: str, **variables: Any
) -> str:
    """Quick judge template rendering."""
    return TemplateManager.render_judge(template_name, context, response, variables)


def list_templates(
    template_type: TemplateType = "all",
) -> Union[List[str], Dict[str, List[str]]]:
    """Quick template listing."""
    return TemplateManager.list_templates(template_type)


__all__ = [
    "TemplateManager",
    "render_policy",
    "render_judge",
    "list_templates",
    "TemplateType",
]
