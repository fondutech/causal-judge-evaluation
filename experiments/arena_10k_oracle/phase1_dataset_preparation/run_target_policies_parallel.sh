#!/bin/bash
# Run all three target policies in parallel

echo "🚀 Starting parallel generation of target policy responses..."
echo "This will run 3 processes in parallel. Check individual log files for progress."
echo ""

# Create log directory
mkdir -p logs

# Run all three policies in parallel
echo "Starting pi_cot..."
python 02b_generate_target_responses_parallel.py --policy pi_cot > logs/pi_cot.log 2>&1 &
PID1=$!

echo "Starting pi_bigger_model..."
python 02b_generate_target_responses_parallel.py --policy pi_bigger_model > logs/pi_bigger_model.log 2>&1 &
PID2=$!

echo "Starting pi_bad..."
python 02b_generate_target_responses_parallel.py --policy pi_bad > logs/pi_bad.log 2>&1 &
PID3=$!

echo ""
echo "All policies started. PIDs: $PID1, $PID2, $PID3"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/pi_cot.log"
echo "  tail -f logs/pi_bigger_model.log"
echo "  tail -f logs/pi_bad.log"
echo ""
echo "Waiting for all policies to complete..."

# Wait for all background processes to complete
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?
wait $PID3
STATUS3=$?

echo ""
echo "All policies completed!"
echo "Exit codes: pi_cot=$STATUS1, pi_bigger_model=$STATUS2, pi_bad=$STATUS3"

# Check if all succeeded
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ] && [ $STATUS3 -eq 0 ]; then
    echo ""
    echo "✅ All policies generated successfully!"
    echo ""
    echo "Merging outputs..."
    python 02b_generate_target_responses_parallel.py --policy all
    echo ""
    echo "🎉 Complete! Output saved to ../data/target_responses.jsonl"
else
    echo ""
    echo "❌ One or more policies failed. Check the log files for details."
    exit 1
fi