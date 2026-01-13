#!/bin/bash
# Stop training job gracefully or forcefully
# Usage: ./stop_training.sh [PID or 'latest']

if [ -z "$1" ]; then
    echo "Usage: $0 <PID|latest|all>"
    echo ""
    echo "Examples:"
    echo "  $0 12345        # Stop specific PID"
    echo "  $0 latest       # Stop most recent training"
    echo "  $0 all          # Stop all training jobs"
    echo ""
    echo "Running training jobs:"
    ps aux | grep "[p]ython.*train.py" | grep -v grep
    exit 1
fi

if [ "$1" == "all" ]; then
    echo "Stopping all training jobs..."
    pkill -f "python.*train.py"
    echo "✓ All training jobs stopped"
    exit 0
fi

if [ "$1" == "latest" ]; then
    # Find most recent PID file
    PID_FILE=$(ls -t logs/training_*.pid 2>/dev/null | head -1)
    if [ -z "$PID_FILE" ]; then
        echo "No PID files found"
        exit 1
    fi
    PID=$(cat "$PID_FILE")
    echo "Found latest training job: PID $PID"
else
    PID=$1
fi

# Check if process exists
if ! ps -p $PID > /dev/null 2>&1; then
    echo "Process $PID is not running"
    exit 1
fi

echo "Stopping training process $PID..."

# Try graceful shutdown first (SIGTERM)
kill $PID

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training stopped gracefully"
        exit 0
    fi
    sleep 1
done

# If still running, force kill (SIGKILL)
echo "Process still running, forcing termination..."
kill -9 $PID

if ! ps -p $PID > /dev/null 2>&1; then
    echo "✓ Training terminated forcefully"
else
    echo "✗ Failed to stop process"
    exit 1
fi
