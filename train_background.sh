#!/bin/bash
# Background training script with live logging
# Usage: ./train_background.sh [train.py arguments]

# Default log file
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════"
echo "Starting Background Training"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Or use the colorized viewer:"
echo "  ./view_training_log.sh $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Run training in background with output to log file
nohup python3 train.py "$@" > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!

# Save PID to file
echo $PID > "${LOG_DIR}/training_${TIMESTAMP}.pid"

echo "Training started!"
echo "Process ID: $PID"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo "  # or"
echo "  kill \$(cat ${LOG_DIR}/training_${TIMESTAMP}.pid)"
echo ""
echo "To view live progress:"
echo "  tail -f $LOG_FILE"
echo ""

# Wait a moment and check if process is still running
sleep 2
if ps -p $PID > /dev/null; then
    echo "✓ Training process is running successfully"

    # Show initial output
    echo ""
    echo "═══ Initial Output ═══"
    head -20 "$LOG_FILE"
    echo ""
    echo "═══ Use 'tail -f $LOG_FILE' to see more ═══"
else
    echo "✗ Training process failed to start or exited quickly"
    echo "Check the log file for errors:"
    echo "  cat $LOG_FILE"
    exit 1
fi
