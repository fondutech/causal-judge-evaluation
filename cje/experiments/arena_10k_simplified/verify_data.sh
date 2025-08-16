#!/bin/bash
# Verify that critical data files exist and are tracked in git

echo "Verifying critical arena_10k data files..."
echo "=========================================="

# Check if data copy directory exists
if [ ! -d "data copy" ]; then
    echo "❌ ERROR: 'data copy' directory not found!"
    exit 1
fi

# List of critical files
declare -a critical_files=(
    "data copy/cje_dataset.jsonl"
    "data copy/prompts.jsonl"
    "data copy/logprobs/base_logprobs.jsonl"
    "data copy/logprobs/clone_logprobs.jsonl"
    "data copy/logprobs/parallel_universe_prompt_logprobs.jsonl"
    "data copy/logprobs/premium_logprobs.jsonl"
    "data copy/logprobs/unhelpful_logprobs.jsonl"
    "data copy/responses/base_responses.jsonl"
    "data copy/responses/clone_responses.jsonl"
    "data copy/responses/parallel_universe_prompt_responses.jsonl"
    "data copy/responses/premium_responses.jsonl"
    "data copy/responses/unhelpful_responses.jsonl"
)

all_good=true

for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        # Check if file is in git
        if git ls-files --error-unmatch "$file" > /dev/null 2>&1; then
            echo "✅ $file (tracked in git)"
        else
            echo "⚠️  $file (exists but NOT in git)"
            all_good=false
        fi
    else
        echo "❌ $file (MISSING)"
        all_good=false
    fi
done

echo "=========================================="

if [ "$all_good" = true ]; then
    echo "✅ All critical data files are present and tracked in git"
    echo ""
    echo "Dataset stats:"
    echo "  - Main dataset: $(wc -l < "data copy/cje_dataset.jsonl") samples"
    echo "  - Prompts: $(wc -l < "data copy/prompts.jsonl") prompts"
    echo "  - Logprobs: $(ls "data copy/logprobs/"*.jsonl | wc -l) policy files"
    echo "  - Responses: $(ls "data copy/responses/"*.jsonl | wc -l) policy files"
else
    echo "⚠️  WARNING: Some data files are missing or not tracked in git"
    echo "Run 'git add \"data copy\"' to add missing files"
    exit 1
fi