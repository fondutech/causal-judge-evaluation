#!/bin/bash
# Emergency stop script for the pipeline

echo "🛑 EMERGENCY STOP INITIATED"
echo "=========================="

# Check if PID file exists
if [ -f phase1.pid ]; then
    PID=$(cat phase1.pid)
    echo "Found process ID: $PID"
    
    # Check if process is running
    if ps -p $PID > /dev/null; then
        echo "Stopping process $PID..."
        kill -TERM $PID
        sleep 2
        
        # Force kill if still running
        if ps -p $PID > /dev/null; then
            echo "Force killing process $PID..."
            kill -KILL $PID
        fi
        
        echo "✅ Process stopped"
    else
        echo "Process $PID is not running"
    fi
    
    rm -f phase1.pid
else
    echo "No PID file found. Looking for running Python processes..."
    
    # Find all relevant Python processes
    PIDS=$(ps aux | grep -E "(02[abc]_|03_|04[a-e]_|run_sample\.sh)" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo "No pipeline processes found running"
    else
        echo "Found processes: $PIDS"
        echo "Stopping all pipeline processes..."
        
        for pid in $PIDS; do
            echo "  Stopping PID $pid"
            kill -TERM $pid
        done
        
        sleep 2
        
        # Check if any still running
        REMAINING=$(ps aux | grep -E "(02[abc]_|03_|04[a-e]_|run_sample\.sh)" | grep -v grep | wc -l)
        if [ $REMAINING -gt 0 ]; then
            echo "Force killing remaining processes..."
            for pid in $PIDS; do
                kill -KILL $pid 2>/dev/null
            done
        fi
        
        echo "✅ All processes stopped"
    fi
fi

# Save current state
echo ""
echo "📊 Current state saved. You can resume from checkpoints."
echo ""
echo "To check what was completed:"
echo "  ls -la ../data/*.jsonl"
echo "  ls -la ../data/sample_1pct/*.jsonl"
echo ""
echo "To resume, run the specific step that was interrupted."