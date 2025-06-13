#!/bin/bash
# Simple linting script for CJE project
# Runs black and mypy on the codebase

set -e

echo "🔧 Running black formatter..."
black cje/ tests/ --check

echo "🔍 Running mypy type checker on core files..."
mypy cje/ tests/ --ignore-missing-imports

echo "✅ All linting checks passed!" 