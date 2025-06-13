#!/bin/bash
# Auto-formatting script for CJE project
# Runs black formatter to fix formatting issues

set -e

echo "🔧 Running black formatter (auto-fix mode)..."
black cje/ tests/

echo "✅ Code formatting complete!" 