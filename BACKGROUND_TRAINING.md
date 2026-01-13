# Background Training Guide

Run training in the background with live logging to monitor progress using `tail -f`.

## Quick Start

### Start Background Training

```bash
# Basic usage
./train_background.sh --model_size small --vram 16gb --data_dir synthetic_data

# With custom config
./train_background.sh --config configs/config_16gb_safe.json --data_dir data

# Medium model on 24GB GPU
./train_background.sh --model_size medium --vram 24gb --data_dir synthetic_data
```

### Monitor Live Progress

```bash
# View latest log with tail
tail -f logs/training_TIMESTAMP.log

# Or use the colorized viewer
./view_training_log.sh

# View specific log
./view_training_log.sh logs/training_20240115_143022.log
```

### Stop Training

```bash
# Stop latest training
./stop_training.sh latest

# Stop specific PID
./stop_training.sh 12345

# Stop all training jobs
./stop_training.sh all
```

## Scripts Overview

### train_background.sh

Starts training in background with logging.

**Features**:
- Runs training as background process
- Logs all output to timestamped file
- Saves PID for easy stopping
- Shows initial output
- Provides monitoring commands

**Usage**:
```bash
./train_background.sh [train.py arguments]
```

**Examples**:
```bash
# Small model, 16GB GPU
./train_background.sh --model_size small --vram 16gb --data_dir data

# Custom config
./train_background.sh --config my_config.json --data_dir data

# Resume training
./train_background.sh --resume checkpoints/step_50000
```

**Output**:
```
Starting Background Training
════════════════════════════════════════════════════════════
Log file: logs/training_20240115_143022.log

Monitor progress with:
  tail -f logs/training_20240115_143022.log

Training started!
Process ID: 12345

To stop training:
  kill 12345
```

### view_training_log.sh

View training logs with colorized output.

**Features**:
- Color-coded output (errors=red, warnings=yellow, progress=blue)
- Auto-refreshing (follows log file)
- Can view specific log or latest automatically

**Usage**:
```bash
# View latest log
./view_training_log.sh

# View specific log
./view_training_log.sh logs/training_20240115_143022.log
```

**Color Coding**:
- 🔴 Red: Errors, failures
- 🟡 Yellow: Warnings
- 🟢 Green: Completions, success
- 🔵 Blue: Training progress (epochs, steps, loss)
- 🔷 Cyan: GPU/CUDA info

### list_training_jobs.sh

List all training jobs and their logs.

**Features**:
- Shows running training processes
- Lists recent log files with sizes
- Shows which jobs are still running
- Provides quick command examples

**Usage**:
```bash
./list_training_jobs.sh
```

**Output**:
```
Training Jobs Status
════════════════════════════════════════════════════════════

Running Training Processes:
───────────────────────────────────────────────────────────
user  12345  ...  python3 train.py --model_size small --vram 16gb

Recent Training Logs
════════════════════════════════════════════════════════════

Log File                           | Size    | Modified
───────────────────────────────────────────────────────────
training_20240115_143022.log       | 2.3M    | 2024-01-15 14:35
  └─ PID 12345 (RUNNING)
training_20240115_120000.log       | 156M    | 2024-01-15 13:45
  └─ PID 11111 (stopped)
```

### stop_training.sh

Stop training jobs gracefully or forcefully.

**Features**:
- Graceful shutdown (SIGTERM) with fallback to force kill (SIGKILL)
- Can stop by PID, latest, or all
- Shows confirmation

**Usage**:
```bash
# Stop latest training
./stop_training.sh latest

# Stop specific PID
./stop_training.sh 12345

# Stop all training jobs
./stop_training.sh all
```

## Common Workflows

### Workflow 1: Start and Monitor Training

```bash
# Terminal 1: Start training
./train_background.sh --model_size medium --vram 16gb --data_dir synthetic_data

# Terminal 2: Monitor live logs
tail -f logs/training_TIMESTAMP.log

# Terminal 3: Monitor GPU usage
watch -n 1 nvidia-smi
```

### Workflow 2: Multiple Training Runs

```bash
# Start multiple training jobs with different configs
./train_background.sh --config configs/config_small.json --output_dir run1
./train_background.sh --config configs/config_medium.json --output_dir run2

# List all jobs
./list_training_jobs.sh

# Monitor specific run
tail -f logs/training_20240115_143022.log
```

### Workflow 3: Long Training Session

```bash
# Start training (can close terminal after)
./train_background.sh --model_size large --vram 24gb --data_dir data

# Later, check status
./list_training_jobs.sh

# Monitor progress
./view_training_log.sh

# Stop if needed
./stop_training.sh latest
```

## Log File Format

Log files are saved to: `logs/training_TIMESTAMP.log`

Timestamp format: `YYYYMMDD_HHMMSS`

Example: `logs/training_20240115_143022.log`

Each log contains:
- Configuration summary
- Model architecture
- Training progress (epochs, steps, loss)
- Evaluation results
- Checkpoint saves
- Any errors or warnings

## Monitoring Commands

### View Live Progress

```bash
# Simple tail
tail -f logs/training_TIMESTAMP.log

# Colored viewer
./view_training_log.sh

# Last 100 lines
tail -100 logs/training_TIMESTAMP.log

# Search for errors
grep -i error logs/training_TIMESTAMP.log
```

### Check Training Status

```bash
# List all jobs
./list_training_jobs.sh

# Check if specific PID is running
ps -p 12345

# See all training processes
ps aux | grep "python.*train.py"
```

### Monitor GPU

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Specific GPU (if multiple)
nvidia-smi -i 0
```

### Extract Specific Information

```bash
# Find all loss values
grep -i "loss" logs/training_TIMESTAMP.log

# Find evaluation results
grep -i "eval" logs/training_TIMESTAMP.log

# Find checkpoint saves
grep -i "checkpoint" logs/training_TIMESTAMP.log

# Count epochs completed
grep -c "Epoch" logs/training_TIMESTAMP.log
```

## Advanced Usage

### Custom Log Location

```bash
# Modify train_background.sh to use custom location
LOG_DIR="/path/to/my/logs"
```

### Multiple GPUs

```bash
# Train on specific GPU
CUDA_VISIBLE_DEVICES=0 ./train_background.sh --model_size medium --vram 24gb

# Train on multiple GPUs (if distributed training implemented)
CUDA_VISIBLE_DEVICES=0,1 ./train_background.sh --model_size large --vram 48gb
```

### Remote Training

```bash
# SSH to server and start training
ssh server 'cd /path/to/mamba_trainer && ./train_background.sh --model_size medium --vram 24gb'

# Monitor remotely
ssh server 'tail -f /path/to/mamba_trainer/logs/training_*.log'

# Or use tmux/screen for persistent sessions
ssh server
tmux new -s training
./train_background.sh --model_size medium --vram 24gb
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Automated Restarts

Create a restart script:
```bash
#!/bin/bash
# auto_restart_training.sh

while true; do
    ./train_background.sh --model_size medium --vram 16gb --data_dir data

    # Wait for process to finish
    sleep 60

    # Check if training completed successfully
    if grep -q "Training complete" logs/training_*.log | tail -1; then
        echo "Training completed successfully!"
        break
    else
        echo "Training failed, restarting..."
        sleep 10
    fi
done
```

## Troubleshooting

### Log File Not Updating

**Problem**: Log file exists but not showing new output

**Solution**:
- Check if process is still running: `./list_training_jobs.sh`
- Python output buffering might be cached
- The script now enables line buffering automatically

### Process Not Running

**Problem**: Training process exits immediately

**Solution**:
```bash
# Check log file for errors
cat logs/training_TIMESTAMP.log

# Look for Python errors
grep -i "error\|exception\|traceback" logs/training_TIMESTAMP.log
```

### Can't Stop Training

**Problem**: `stop_training.sh` doesn't work

**Solution**:
```bash
# Find process manually
ps aux | grep "python.*train.py"

# Force kill
kill -9 <PID>

# Or kill all Python processes (careful!)
pkill -9 -f "python.*train.py"
```

### Out of Disk Space

**Problem**: Log files filling up disk

**Solution**:
```bash
# Check disk space
df -h

# Remove old logs
rm logs/training_OLD_*.log

# Or compress old logs
gzip logs/training_2024*.log
```

### Multiple Trainings Conflicting

**Problem**: Multiple training runs using same GPU

**Solution**:
```bash
# List all training processes
./list_training_jobs.sh

# Stop unnecessary ones
./stop_training.sh all

# Or use specific GPUs
CUDA_VISIBLE_DEVICES=0 ./train_background.sh ...  # GPU 0
CUDA_VISIBLE_DEVICES=1 ./train_background.sh ...  # GPU 1
```

## Tips and Best Practices

1. **Always check logs first**: Before starting new training, check if previous one is still running
2. **Monitor GPU usage**: Use `nvidia-smi` to ensure GPU is being utilized
3. **Save PIDs**: The scripts save PID files for easy process management
4. **Use tmux/screen**: For remote training, use terminal multiplexers
5. **Backup checkpoints**: Regularly backup `checkpoints/` directory
6. **Clean old logs**: Remove or archive old training logs periodically
7. **Test configurations**: Test with fast preset before long training runs
8. **Set up alerts**: Use scripts to alert when training completes or fails

## Example Training Session

```bash
# 1. Start training in background
./train_background.sh --model_size medium --vram 16gb --data_dir synthetic_data
# Output: Process ID: 12345
# Output: Log file: logs/training_20240115_143022.log

# 2. Monitor progress (in another terminal)
tail -f logs/training_20240115_143022.log

# 3. Check GPU usage (in another terminal)
watch -n 1 nvidia-smi

# 4. Check all jobs
./list_training_jobs.sh

# 5. View with colors
./view_training_log.sh

# 6. When done or to stop
./stop_training.sh latest

# 7. Review final log
cat logs/training_20240115_143022.log | tail -100
```

## Integration with Other Tools

### TensorBoard

```bash
# Install tensorboard
pip install tensorboard

# Run tensorboard (if logging to tensorboard directory)
tensorboard --logdir=runs/

# Access at http://localhost:6006
```

### Weights & Biases

```bash
# If using W&B (add to train.py)
import wandb
wandb.init(project="mamba-training")
wandb.log({"loss": loss})
```

### Slack/Email Notifications

Add to end of training script:
```bash
# Send notification when training completes
curl -X POST https://hooks.slack.com/... -d '{"text":"Training complete!"}'
```

## Summary

**Start training in background**:
```bash
./train_background.sh --model_size medium --vram 16gb --data_dir data
```

**Monitor live**:
```bash
tail -f logs/training_TIMESTAMP.log
```

**Stop training**:
```bash
./stop_training.sh latest
```

That's it! Your training runs in the background and you can monitor it with `tail -f`.
