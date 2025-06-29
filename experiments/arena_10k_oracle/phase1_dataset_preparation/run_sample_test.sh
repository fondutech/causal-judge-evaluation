#!/bin/bash
# Convenience wrapper to run the sample test

echo "🧪 Running 1% Sample Test"
echo "========================"

# Source secrets if available
if [ -f "../../../set_secrets.sh" ]; then
    source ../../../set_secrets.sh
    echo "✅ Loaded API keys from set_secrets.sh"
fi
echo ""
echo "This will:"
echo "1. Test with 100 prompts (1% of 10,000)"
echo "2. Validate teacher forcing implementation"
echo "3. Estimate costs and time for full run"
echo ""

# Run from sample_run directory
cd sample_run && ./run_sample.sh