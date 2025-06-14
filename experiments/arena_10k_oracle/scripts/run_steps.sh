#!/bin/bash
# Run Arena 10K experiment steps individually
# Each step can be run independently or in sequence

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Experiment directory
EXPERIMENT_DIR="$(dirname "$0")/.."
DATA_DIR="$EXPERIMENT_DIR/data"

echo -e "${BLUE}Arena 10K Fresh Oracle Experiment${NC}"
echo "=================================="

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Function to run a step
run_step() {
    local step_num=$1
    local step_name=$2
    local script=$3
    shift 3
    local args=$@
    
    echo -e "\n${BLUE}Step $step_num: $step_name${NC}"
    echo "Running: python $script $args"
    
    if [ -f "$script" ]; then
        python "$script" $args
        echo -e "${GREEN}✓ Step $step_num complete${NC}"
    else
        echo -e "${YELLOW}⚠ Script not found: $script${NC}"
        return 1
    fi
}

# Parse command line arguments
STEP=${1:-all}

case $STEP in
    1|prepare)
        run_step 1 "Prepare Data" \
            "01_prepare_data.py" \
            --samples 10000 \
            --output "$DATA_DIR/prompts.jsonl"
        ;;
        
    2|generate)
        run_step 2 "Generate Logging Policy Responses" \
            "02_generate_logs.py" \
            --input "$DATA_DIR/prompts.jsonl" \
            --output "$DATA_DIR/p0_replies.jsonl" \
            --checkpoint "$DATA_DIR/p0_checkpoint.jsonl"
        ;;
        
    3|score)
        run_step 3 "Add Judge Scores" \
            "03_add_judge_scores.py" \
            --input "$DATA_DIR/p0_replies.jsonl" \
            --output "$DATA_DIR/p0_scored.jsonl" \
            --checkpoint "$DATA_DIR/judge_checkpoint.jsonl"
        ;;
        
    4|calibrate)
        echo -e "${BLUE}Step 4: Oracle Calibration${NC}"
        echo "This step has two parts:"
        echo ""
        echo "4a) Export for labeling:"
        echo "    python 04_export_for_labeling.py --platform surge"
        echo ""
        echo "4b) After collecting labels, import and calibrate:"
        echo "    python 04_import_labels.py --labels path/to/labels.csv"
        ;;
        
    5|targets)
        echo -e "${YELLOW}Step 5: Generate Target Policies - To be implemented${NC}"
        echo "This step will generate responses for:"
        echo "  - π_clone (sanity check)"
        echo "  - π_cot (chain-of-thought)"
        echo "  - π_rag (retrieval-augmented)"
        echo "  - π_big (70B model)"
        ;;
        
    6|estimate)
        echo -e "${YELLOW}Step 6: Run CJE Estimation - To be implemented${NC}"
        echo "This will run the causal estimation pipeline"
        ;;
        
    7|validate)
        echo -e "${YELLOW}Step 7: Gold Validation - To be implemented${NC}"
        echo "This will compare CJE estimates to ground truth"
        ;;
        
    all)
        echo "Running all implemented steps..."
        
        # Run implemented steps
        "$0" 1
        "$0" 2
        "$0" 3
        
        echo -e "\n${YELLOW}Steps 4-7 require additional implementation${NC}"
        echo "See README.md for full experiment details"
        ;;
        
    *)
        echo "Usage: $0 [step|all]"
        echo ""
        echo "Steps:"
        echo "  1|prepare   - Download and prepare ChatBot Arena data"
        echo "  2|generate  - Generate logging policy responses"
        echo "  3|score     - Add judge scores to responses"
        echo "  4|calibrate - Calibrate judge with oracle labels (TBD)"
        echo "  5|targets   - Generate target policy responses (TBD)"
        echo "  6|estimate  - Run CJE estimation (TBD)"
        echo "  7|validate  - Validate against ground truth (TBD)"
        echo "  all         - Run all implemented steps"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"