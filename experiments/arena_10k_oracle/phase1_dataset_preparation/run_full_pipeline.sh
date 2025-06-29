#!/bin/bash
# Run the full Phase 1 pipeline (10,000 prompts)
# Only run this after successful sample validation!

echo "🚀 Starting Full Phase 1 Pipeline (10,000 prompts)"
echo "================================================"

# Safety check
if [ "$ARENA_SAMPLE_MODE" = "true" ]; then
    echo "❌ ERROR: ARENA_SAMPLE_MODE is still set to true!"
    echo "This script is for the full run. Use ./run_sample.sh for sample mode."
    exit 1
fi

# Check if sample validation passed
SAMPLE_REPORT="../data/sample_1pct/validation_report.md"
if [ -f "$SAMPLE_REPORT" ]; then
    if grep -q "READY FOR FULL RUN" "$SAMPLE_REPORT"; then
        echo "✅ Sample validation passed"
    else
        echo "⚠️  Warning: Sample validation may not have passed!"
        echo "Check $SAMPLE_REPORT before proceeding."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⚠️  Warning: No sample validation report found!"
    echo "Have you run ./run_sample.sh first?"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Save PID for emergency stop
echo $$ > phase1.pid

# Track timing
START_TIME=$(date +%s)
LOG_FILE="phase1_full_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging to: $LOG_FILE"
echo ""

# Function to run a step with error handling
run_step() {
    local step_name=$1
    local script_name=$2
    
    echo -e "\n=====================================\n" | tee -a "$LOG_FILE"
    echo "🔄 Running: $step_name" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo -e "=====================================\n" | tee -a "$LOG_FILE"
    
    python "$script_name" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ $step_name failed!" | tee -a "$LOG_FILE"
        echo "Check the log file: $LOG_FILE" | tee -a "$LOG_FILE"
        
        # Calculate elapsed time
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "Failed after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m" | tee -a "$LOG_FILE"
        
        rm -f phase1.pid
        exit 1
    fi
    
    echo "✅ $step_name completed" | tee -a "$LOG_FILE"
}

# Run all steps
echo "Starting pipeline at $(date)" | tee "$LOG_FILE"

# Step 1: Data preparation (should already be done)
if [ ! -f "../data/arena_questions_base.jsonl" ]; then
    run_step "Step 1: Data Preparation" "01_prepare_data.py"
else
    echo "✅ Step 1: Data already prepared" | tee -a "$LOG_FILE"
fi

# Step 2a: Generate P0 responses
run_step "Step 2a: P0 Response Generation" "02a_generate_p0_responses.py"

# Step 2b: Generate target responses
run_step "Step 2b: Target Response Generation" "02b_generate_target_responses.py"

# Step 2c: Compute teacher forcing (CRITICAL)
echo "⚠️  CRITICAL STEP: Teacher Forcing" | tee -a "$LOG_FILE"
run_step "Step 2c: Teacher Forcing Computation" "02c_compute_target_logprobs.py"

# Analyze teacher forcing results immediately
echo -e "\n🔍 Analyzing teacher forcing results..." | tee -a "$LOG_FILE"
python analyze_teacher_forcing_stats.py --full 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Teacher forcing analysis found issues!" | tee -a "$LOG_FILE"
    echo "Pipeline stopped for safety. Check the analysis output." | tee -a "$LOG_FILE"
    rm -f phase1.pid
    exit 1
fi

# Step 3: Generate oracle labels
run_step "Step 3: Oracle Label Generation" "03_generate_oracle_labels.py"

# Step 4: Judge scoring (multiple scripts)
echo -e "\n🔄 Running judge scoring scripts..." | tee -a "$LOG_FILE"

for script in 04a_deterministic_judge_scores.py 04b_uncertainty_judge_scores.py 04c_score_targets_deterministic.py 04d_score_targets_uncertainty.py 04e_score_missing_targets.py; do
    if [ -f "$script" ]; then
        run_step "Judge Scoring: $script" "$script"
    fi
done

# Calculate total elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$((ELAPSED % 3600 / 60))

echo -e "\n=====================================" | tee -a "$LOG_FILE"
echo "✅ PIPELINE COMPLETED SUCCESSFULLY! ✅" | tee -a "$LOG_FILE"
echo "=====================================" | tee -a "$LOG_FILE"
echo "Total time: ${HOURS}h ${MINUTES}m" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"

# Final validation
echo -e "\n📊 Final validation..." | tee -a "$LOG_FILE"
echo "File counts:" | tee -a "$LOG_FILE"
for f in ../data/*.jsonl; do
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        echo "  $(basename $f): $count entries" | tee -a "$LOG_FILE"
    fi
done

# Backup reminder
echo -e "\n💾 Don't forget to backup your data!" | tee -a "$LOG_FILE"
echo "Suggested command:" | tee -a "$LOG_FILE"
echo "  tar -czf arena_10k_phase1_$(date +%Y%m%d).tar.gz ../data/*.jsonl" | tee -a "$LOG_FILE"

# Cleanup
rm -f phase1.pid

echo -e "\n🎯 Ready for Phase 2 experiments!" | tee -a "$LOG_FILE"