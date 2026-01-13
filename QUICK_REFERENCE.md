# Quick Reference Card

## Training Commands

### Basic Training

```bash
# Standard presets
python train.py --model_size small --vram 8gb --data_dir data
python train.py --model_size medium --vram 16gb --data_dir data
python train.py --model_size large --vram 24gb --data_dir data
```

### Custom VRAM (New!)

```bash
# Auto-detect GPU memory
python train.py --vram_auto --model_size medium --data_dir data

# Custom VRAM value
python train.py --vram 24 --model_size large --data_dir data
python train.py --vram 20.5 --model_size medium --data_dir data

# Custom safety margin
python train.py --vram 24 --vram_safety_margin 1024 --data_dir data
```

### Background Training

```bash
# Start in background
./train_background.sh --model_size medium --vram_auto --data_dir data

# Monitor live
tail -f logs/training_*.log

# Colorized viewer
./view_training_log.sh

# Stop training
./stop_training.sh latest
```

## Data Preparation

```bash
# Generate synthetic data (already done - 3,600 files)
python generate_synthetic_data.py --python 1000 --go 100 --cpp 500 --ada 2000

# Inspect data
python prepare_data.py --data_dir synthetic_data --show_stats

# View samples
python prepare_data.py --data_dir data --show_samples 5
```

## Dataset Training

```bash
# HuggingFace dataset
python train.py --dataset wikitext --vram 16gb

# Local files + dataset
python train.py --data_dir data --dataset wikitext --vram 24gb

# Custom dataset column
python train.py --dataset my_dataset --text_column content --vram 16gb
```

## Text Generation

```bash
# Basic generation
python generate.py --checkpoint checkpoints/final --prompt "Hello"

# Custom parameters
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top_k 50 \
  --max_tokens 200
```

## Model Export

```bash
# Export to HuggingFace (for vLLM)
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf

# Use with vLLM
vllm serve models/mamba-hf --dtype float16

# Export to GGUF (for Ollama/LM Studio)
python llama.cpp/convert.py models/mamba-hf \
  --outfile models/mamba-fp16.gguf \
  --outtype f16
```

## Memory Configuration

### By GPU Size

| GPU VRAM | Command |
|----------|---------|
| 8GB | `--model_size small --vram 8gb` |
| 12GB | `--model_size small --vram 12gb` |
| 16GB | `--model_size medium --vram 16gb` or `--vram_auto` |
| 24GB | `--model_size large --vram 24gb` or `--vram_auto` |
| 32GB+ | `--model_size large --vram_auto` |

### Memory Issues

```bash
# Ultra-safe config
python train.py --config configs/config_16gb_safe.json --data_dir data

# Reduce VRAM target
python train.py --vram 14 --model_size small --data_dir data

# Increase safety margin
python train.py --vram 16 --vram_safety_margin 1024 --data_dir data
```

## Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f logs/training_*.log

# List jobs
./list_training_jobs.sh

# Check progress
grep -i "epoch\|step\|loss" logs/training_*.log | tail -20
```

## File Locations

```
mamba_trainer/
├── train.py                    # Main training script
├── generate.py                 # Text generation
├── prepare_data.py             # Data inspection
├── generate_synthetic_data.py  # Create synthetic data
│
├── train_background.sh         # Background training
├── view_training_log.sh        # Log viewer
├── list_training_jobs.sh       # Job manager
├── stop_training.sh            # Stop jobs
│
├── configs/                    # Configuration files
│   ├── config_16gb_safe.json
│   ├── example_8gb.json
│   └── example_24gb.json
│
├── synthetic_data/             # 3,600 generated files
│   ├── python/  (1,000 ML/DL scripts)
│   ├── go/      (100 files)
│   ├── cpp/     (500 files)
│   └── ada/     (2,000 files)
│
├── logs/                       # Training logs
├── checkpoints/                # Model checkpoints
│
└── Documentation
    ├── GETTING_STARTED.md      # Beginner guide
    ├── USAGE_GUIDE.md          # Detailed examples
    ├── QUICKSTART.md           # Quick reference
    ├── README.md               # Full documentation
    ├── CUSTOM_VRAM.md          # Custom VRAM guide
    ├── MEMORY_OPTIMIZATION.md  # Memory troubleshooting
    ├── BACKGROUND_TRAINING.md  # Background training guide
    ├── DEPLOYMENT_GUIDE.md     # Deployment options
    └── DEPLOYMENT_FULL_PRECISION.md  # Full precision guide
```

## Common Workflows

### Start Training Now

```bash
# Your 16GB GPU
python train.py --model_size medium --vram_auto --data_dir synthetic_data

# Or in background
./train_background.sh --vram_auto --model_size medium --data_dir synthetic_data
```

### Train + Monitor

```bash
# Terminal 1
./train_background.sh --vram_auto --data_dir synthetic_data

# Terminal 2
tail -f logs/training_*.log

# Terminal 3
watch -n 1 nvidia-smi
```

### Generate After Training

```bash
# Wait for training to complete, then:
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Write a Python function" \
  --temperature 0.7 \
  --max_tokens 200
```

### Deploy Model

```bash
# Export to HuggingFace
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf

# Serve with vLLM
vllm serve models/mamba-hf --dtype float16 --port 8000

# Or export to GGUF for Ollama
python llama.cpp/convert.py models/mamba-hf \
  --outfile models/mamba-fp16.gguf --outtype f16

ollama create mamba -f Modelfile
ollama run mamba
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Out of memory | `--vram_auto` or `--vram_safety_margin 1024` |
| Training slow | `--training_preset fast` |
| GPU not detected | `nvidia-smi` to check, `export CUDA_VISIBLE_DEVICES=0` |
| No data found | `python prepare_data.py --show_stats` |
| Process stuck | Check logs: `tail logs/training_*.log` |

## Parameter Reference

### Model Sizes
- `small`: ~50M params, fast training
- `medium`: ~150M params, balanced
- `large`: ~400M params, best quality

### VRAM Options
- `8gb`, `12gb`, `16gb`, `24gb`, `32gb`, `48gb` (presets)
- `--vram 20.5` (custom value in GB)
- `--vram_auto` (auto-detect)

### Training Presets
- `default`: Balanced (LR 3e-4, 100K steps)
- `fast`: Quick experiments (LR 5e-4, 50K steps)
- `careful`: Best quality (LR 1e-4, 200K steps)

### Generation Parameters
- `--temperature`: 0.1-2.0 (lower=focused, higher=creative)
- `--top_k`: 1-1000 (limits vocabulary)
- `--max_tokens`: Output length

## Help Commands

```bash
# Script help
python train.py --help
python generate.py --help
python prepare_data.py --help

# Test setup
python test_setup.py

# View documentation
cat GETTING_STARTED.md
cat CUSTOM_VRAM.md
cat MEMORY_OPTIMIZATION.md
```

## One-Liners

```bash
# Train with auto-optimization
python train.py --vram_auto --data_dir synthetic_data

# Background training with monitoring
./train_background.sh --vram_auto --data_dir data && tail -f logs/training_*.log

# Generate 10 samples
for i in {1..10}; do python generate.py --checkpoint checkpoints/final --prompt "Sample $i:" --max_tokens 50; done

# Check latest log
tail -50 $(ls -t logs/training_*.log | head -1)

# Stop all training
./stop_training.sh all
```

---

**Ready to start?**
```bash
python train.py --vram_auto --model_size medium --data_dir synthetic_data
```
