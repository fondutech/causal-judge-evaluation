#!/bin/bash
# Simple linting script for CJE project
# Runs black and mypy on the codebase

set -e

echo "🔧 Running black formatter..."
black cje/ tests/ --check

echo "🔍 Running mypy type checker on core files..."
mypy cje/validation.py cje/cli/run_experiment.py tests/test_pipeline_integration.py --ignore-missing-imports

echo "✅ All linting checks passed!" 