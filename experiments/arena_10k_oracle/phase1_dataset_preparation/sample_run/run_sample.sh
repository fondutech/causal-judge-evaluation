#!/bin/bash
# Run Phase 1 with 1% sample for validation

echo "🚀 Starting Phase 1 Sample Run (1% of data)"
echo "=========================================="

# Source secrets if available
if [ -f "../../../../../set_secrets.sh" ]; then
    source ../../../../../set_secrets.sh
    echo "✅ Loaded API keys from set_secrets.sh"
fi

# Set sample mode environment variables
export ARENA_SAMPLE_MODE=true
export ARENA_SAMPLE_SIZE=100
export ARENA_SAMPLE_SEED=42

# Create sample directory
mkdir -p ../../data/sample_1pct

# Track timing
START_TIME=$(date +%s)

echo -e "\n📊 Step 1: Preparing sample data..."
python ../01_prepare_data.py
if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi

echo -e "\n🤖 Step 2a: Generating P0 responses..."
python ../02a_generate_p0_responses.py
if [ $? -ne 0 ]; then
    echo "❌ P0 generation failed!"
    exit 1
fi

echo -e "\n🎯 Step 2b: Generating target policy responses..."
python ../02b_generate_target_responses.py
if [ $? -ne 0 ]; then
    echo "❌ Target generation failed!"
    exit 1
fi

echo -e "\n📐 Step 2c: Computing teacher forcing log probabilities..."
python ../02c_compute_target_logprobs.py
if [ $? -ne 0 ]; then
    echo "❌ Teacher forcing failed!"
    exit 1
fi

echo -e "\n🏷️ Step 3: Generating oracle labels..."
python ../03_generate_oracle_labels.py
if [ $? -ne 0 ]; then
    echo "❌ Oracle labeling failed!"
    exit 1
fi

echo -e "\n⚖️ Step 4: Computing judge scores..."
# Run judge scoring scripts
for script in 04a_deterministic_judge_scores.py 04b_uncertainty_judge_scores.py; do
    if [ -f "../$script" ]; then
        echo "  Running $script..."
        python "../$script"
        if [ $? -ne 0 ]; then
            echo "❌ Judge scoring failed at $script!"
            exit 1
        fi
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo -e "\n✅ Sample run completed successfully!"
echo "⏱️  Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
echo "📊 Next steps:"
echo "1. Check ../../data/sample_1pct/ for all outputs"
echo "2. Review teacher forcing statistics"
echo "3. Validate no 0.0 log probs for non-empty responses"
echo "4. If all checks pass, run full pipeline with:"
echo "   unset ARENA_SAMPLE_MODE && ../run_full_pipeline.sh"