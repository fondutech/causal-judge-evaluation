#!/bin/bash
# Pre-commit wrapper script that uses standard PyPI
# This avoids issues with custom CodeArtifact indexes

set -e

echo "🔧 Running pre-commit hooks with standard PyPI..."

# Use local pip configuration to override global settings
export PIP_CONFIG_FILE="$(pwd)/pip.conf"
export PIP_INDEX_URL=https://pypi.org/simple/

pre-commit "$@"

echo "✅ Pre-commit hooks completed!" 