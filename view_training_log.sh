#!/bin/bash
# Colorized training log viewer with auto-refresh
# Usage: ./view_training_log.sh [log_file]

# Get log file from argument or use latest
if [ -n "$1" ]; then
    LOG_FILE="$1"
else
    # Find most recent log file
    LOG_FILE=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "No log files found in logs/ directory"
        exit 1
    fi
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "Training Log Viewer"
echo "════════════════════════════════════════════════════════════"
echo "Log file: $LOG_FILE"
echo ""
echo "Press Ctrl+C to exit"
echo "════════════════════════════════════════════════════════════"
echo ""

# Follow log file with color highlighting
tail -f "$LOG_FILE" | while IFS= read -r line; do
    # Color error lines red
    if [[ $line == *"Error"* ]] || [[ $line == *"ERROR"* ]] || [[ $line == *"Failed"* ]]; then
        echo -e "\033[0;31m$line\033[0m"
    # Color warning lines yellow
    elif [[ $line == *"Warning"* ]] || [[ $line == *"WARNING"* ]]; then
        echo -e "\033[0;33m$line\033[0m"
    # Color success/complete lines green
    elif [[ $line == *"complete"* ]] || [[ $line == *"Complete"* ]] || [[ $line == *"success"* ]]; then
        echo -e "\033[0;32m$line\033[0m"
    # Color epoch/step progress lines blue
    elif [[ $line == *"Epoch"* ]] || [[ $line == *"Step"* ]] || [[ $line == *"loss"* ]]; then
        echo -e "\033[0;34m$line\033[0m"
    # Color GPU/CUDA lines cyan
    elif [[ $line == *"GPU"* ]] || [[ $line == *"CUDA"* ]] || [[ $line == *"VRAM"* ]]; then
        echo -e "\033[0;36m$line\033[0m"
    else
        echo "$line"
    fi
done
