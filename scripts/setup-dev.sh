#!/bin/bash
# Development environment setup script for CJE project

set -e

echo "🚀 Setting up CJE development environment..."

# Create local pip configuration to use standard PyPI
echo "📦 Configuring pip to use standard PyPI..."
cat > pip.conf << EOF
[global]
index-url = https://pypi.org/simple/
EOF

# Install pre-commit hooks with correct configuration
echo "🔧 Installing pre-commit hooks..."
export PIP_CONFIG_FILE="$(pwd)/pip.conf"
export PIP_INDEX_URL=https://pypi.org/simple/
pre-commit install

echo "✅ Development environment setup complete!"
echo ""
echo "Usage:"
echo "  ./scripts/lint.sh          - Check code formatting and types"
echo "  ./scripts/format.sh        - Auto-fix code formatting"
echo "  ./scripts/pre-commit-run.sh - Run pre-commit hooks manually"
echo ""
echo "Pre-commit hooks will now run automatically on git commit." 