# CJE-Core Documentation Makefile

.PHONY: docs docs-serve docs-watch docs-clean docs-linkcheck docs-pdf help

# Documentation commands
docs:  ## Build documentation
	cd docs && python build.py build

docs-clean:  ## Clean and rebuild documentation
	cd docs && python build.py build --clean

docs-serve:  ## Serve documentation locally
	cd docs && python build.py serve

docs-watch:  ## Watch for changes and auto-rebuild
	cd docs && python build.py watch

docs-linkcheck:  ## Check external links
	cd docs && python build.py linkcheck

docs-pdf:  ## Build PDF documentation
	cd docs && python build.py pdf

docs-autodoc:  ## Generate autodoc stubs
	cd docs && python build.py autodoc

# Development commands
dev-setup:  ## Set up development environment
	poetry install
	pre-commit install

test:  ## Run tests
	poetry run pytest

lint:  ## Run linting
	poetry run black cje/ tests/
	poetry run mypy cje/

format:  ## Format code
	poetry run black cje/ tests/

mypy:  ## Run type checking
	poetry run mypy cje/

# Help
help:  ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help 