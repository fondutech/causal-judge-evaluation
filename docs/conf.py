# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(".."))

# Set up mock imports for problematic dependencies during docs build
from typing import Any
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name: str) -> MagicMock:
        return MagicMock()

    def __getitem__(self, key: Any) -> MagicMock:
        return MagicMock()

    __all__: list[str] = []


# Mock modules that might cause import issues during docs build
MOCK_MODULES = [
    # Core scientific packages that might have C extensions
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pyarrow",
    # ML/AI packages
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "xgboost",
    "sentence_transformers",
    # API clients
    "openai",
    "anthropic",
    "google.generativeai",
    "langchain_core",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_together",
    "boto3",
    "tiktoken",
    # Other potentially problematic imports
    "hydra",
    "omegaconf",
    "rich",
    "typer",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# Now try to import the package after mocking dependencies
try:
    import cje
except ImportError as e:
    print(f"Warning: Could not import cje package even after mocking: {e}")
    # Add additional paths that might help
    sys.path.insert(0, os.path.abspath("../cje"))
    sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CJE-Core"
copyright = "2024, Eddie Landesberg"
author = "Eddie Landesberg"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # Third-party extensions
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Auto-generate summaries
autosummary_generate = True
autosummary_imported_members = False
autodoc_mock_imports = MOCK_MODULES

# Handle import failures gracefully
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
autodoc_class_signature = "mixed"

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Type hints configuration
typehints_defaults = "comma"
typehints_use_signature = True
typehints_use_signature_return = True
simplify_optional_unions = True

# MyST parser configuration (for Markdown files)
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "substitution",
    "colon_fence",
    "attrs_inline",
]

# Intersphinx configuration (links to other projects)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "analytics_anonymize_ip": True,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Additional HTML context
html_context = {
    "display_github": True,
    "github_user": "eddielandesberg",
    "github_repo": "causal-judge-evaluation",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "",
    "printindex": "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "CJE-Core.tex", "CJE-Core Documentation", "Eddie Landesberg", "manual"),
]
