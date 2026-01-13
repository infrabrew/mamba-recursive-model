#!/bin/bash
# List running training jobs and their logs
# Usage: ./list_training_jobs.sh

echo "════════════════════════════════════════════════════════════"
echo "Training Jobs Status"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check for running training processes
TRAINING_PROCS=$(ps aux | grep "[p]ython.*train.py" | grep -v grep)

if [ -z "$TRAINING_PROCS" ]; then
    echo "No training jobs currently running"
else
    echo "Running Training Processes:"
    echo "───────────────────────────────────────────────────────────"
    echo "$TRAINING_PROCS"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Recent Training Logs"
echo "════════════════════════════════════════════════════════════"
echo ""

# List recent log files
if [ -d "logs" ]; then
    LOG_FILES=$(ls -t logs/training_*.log 2>/dev/null)

    if [ -z "$LOG_FILES" ]; then
        echo "No log files found"
    else
        echo "Log File                           | Size    | Modified"
        echo "───────────────────────────────────────────────────────────"

        for log in $LOG_FILES; do
            SIZE=$(du -h "$log" | cut -f1)
            MODIFIED=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$log" 2>/dev/null || stat -c "%y" "$log" 2>/dev/null | cut -d'.' -f1)
            BASENAME=$(basename "$log")
            printf "%-35s | %-7s | %s\n" "$BASENAME" "$SIZE" "$MODIFIED"

            # Show if corresponding PID exists
            PID_FILE="logs/${BASENAME%.log}.pid"
            if [ -f "$PID_FILE" ]; then
                PID=$(cat "$PID_FILE")
                if ps -p $PID > /dev/null 2>&1; then
                    echo "  └─ PID $PID (RUNNING)"
                else
                    echo "  └─ PID $PID (stopped)"
                fi
            fi
        done | head -20
    fi
else
    echo "logs/ directory not found"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Quick Commands"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "View latest log:"
echo "  tail -f \$(ls -t logs/training_*.log | head -1)"
echo ""
echo "View with colors:"
echo "  ./view_training_log.sh"
echo ""
echo "Stop training:"
echo "  kill \$(cat logs/training_TIMESTAMP.pid)"
echo ""
echo "Monitor GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
